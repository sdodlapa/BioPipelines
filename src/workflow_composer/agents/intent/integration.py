"""
Intent System Integration
=========================

This module provides integration helpers for the enhanced intent system.

It bridges the DialogueManager with the BioPipelines facade.

Now uses the UnifiedIntentParser with LLM arbiter for improved accuracy:
- Pattern matching for simple queries (~80%, <15ms)
- LLM arbiter for complex/ambiguous queries (~20%, ~500ms)
- Handles negation, context, and adversarial queries
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Tuple

from .parser import IntentParser, IntentType, IntentResult, Entity, EntityType
from .context import ConversationContext, ContextMemory, EntityTracker
from .dialogue import (
    DialogueManager,
    DialogueResult,
    ConversationState,
    ConversationPhase,
    TaskState,
)

logger = logging.getLogger(__name__)


# Try to import the arbiter (new hierarchical parser)
try:
    from .unified_parser import UnifiedIntentParser
    from .arbiter import ArbiterStrategy, ArbiterResult
    ARBITER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Arbiter not available, using legacy IntentParser: {e}")
    ARBITER_AVAILABLE = False


class ChatIntegration:
    """
    Integration layer between DialogueManager and BioPipelines facade.
    
    Now uses UnifiedIntentParser with LLM arbiter for better accuracy on
    complex queries while maintaining fast response for simple ones.
    
    Usage:
        from workflow_composer.agents.intent.integration import ChatIntegration
        
        intent_system = ChatIntegration()
        result = intent_system.process_message(message, session_id)
        
        if result.should_execute_tool:
            tool_result = execute_tool(result.tool_name, result.tool_args)
            follow_up = intent_system.handle_tool_result(
                session_id, result.tool_name, tool_result
            )
    """
    
    def __init__(
        self, 
        llm_client=None,
        use_arbiter: bool = True,
        arbiter_strategy: Optional[str] = "smart",
    ):
        """
        Initialize the integration layer.
        
        Args:
            llm_client: Optional LLM client for complex intent parsing
            use_arbiter: Whether to use the hierarchical parser with LLM arbiter
            arbiter_strategy: Strategy for LLM invocation:
                - "smart": Only invoke LLM for complex/ambiguous queries (recommended)
                - "always": Always invoke LLM (most accurate, expensive)
                - "never": Never use LLM (fast, lower accuracy)
        """
        self.llm_client = llm_client
        self.use_arbiter = use_arbiter and ARBITER_AVAILABLE
        self.use_cascade = use_arbiter  # Enable cascade when using arbiter
        
        # Create the appropriate parser
        if self.use_arbiter:
            try:
                # Create UnifiedIntentParser with cascade routing
                self.unified_parser = UnifiedIntentParser(
                    use_cascade=self.use_cascade,
                    arbiter_strategy=arbiter_strategy or "smart",
                )
                # Store the parser for DialogueManager  
                self.parser = self.unified_parser
                logger.info(f"ChatIntegration using UnifiedIntentParser with {arbiter_strategy} strategy")
            except Exception as e:
                logger.warning(f"Failed to initialize UnifiedIntentParser: {e}")
                self.unified_parser = None
                self.parser = IntentParser(llm_client=llm_client)
                self.use_arbiter = False
        else:
            self.unified_parser = None
            self.parser = IntentParser(llm_client=llm_client)
        
        # Session-based dialogue managers
        self._sessions: Dict[str, DialogueManager] = {}
        
        # Track arbiter metrics
        self._arbiter_stats = {
            "total_queries": 0,
            "llm_invoked": 0,
            "unanimous_decisions": 0,
        }
    
    def get_dialogue_manager(self, session_id: str) -> DialogueManager:
        """Get or create dialogue manager for session."""
        if session_id not in self._sessions:
            context = ConversationContext()
            self._sessions[session_id] = DialogueManager(
                intent_parser=self.parser,
                context=context,
                use_arbiter=self.use_arbiter,
                use_cascade=self.use_cascade if hasattr(self, 'use_cascade') else True,
            )
        return self._sessions[session_id]
    
    def parse_with_arbiter(self, message: str, context: Optional[Dict] = None) -> Optional[Any]:
        """
        Parse using the unified parser with LLM arbiter.
        
        This is the recommended way to parse when you need the full
        arbiter capabilities (negation handling, disagreement resolution).
        
        Args:
            message: User message to parse
            context: Optional context dictionary
            
        Returns:
            UnifiedParseResult if arbiter available, None otherwise
        """
        if not self.use_arbiter or not self.unified_parser:
            return None
        
        try:
            result = self.unified_parser.parse(message, context)
            
            # Update stats
            self._arbiter_stats["total_queries"] += 1
            if hasattr(result, 'llm_invoked') and result.llm_invoked:
                self._arbiter_stats["llm_invoked"] += 1
            if hasattr(result, 'method') and result.method == "unanimous":
                self._arbiter_stats["unanimous_decisions"] += 1
                
            logger.debug(
                f"Arbiter: intent={result.primary_intent.name}, "
                f"conf={result.confidence:.2f}, "
                f"method={getattr(result, 'method', 'pattern')}, "
                f"llm={getattr(result, 'llm_invoked', False)}"
            )
            return result
        except Exception as e:
            logger.warning(f"Arbiter parsing failed: {e}")
            return None
    
    def get_arbiter_stats(self) -> Dict[str, Any]:
        """
        Get arbiter usage statistics.
        
        Returns:
            Dictionary with:
            - total_queries: Total queries processed
            - llm_invoked: Number of queries that needed LLM
            - llm_rate: Percentage of queries using LLM
            - unanimous_rate: Percentage of unanimous decisions
        """
        total = self._arbiter_stats["total_queries"]
        if total == 0:
            return {
                "total_queries": 0,
                "llm_invoked": 0,
                "llm_rate": 0.0,
                "unanimous_rate": 0.0,
                "arbiter_enabled": self.use_arbiter,
            }
        
        return {
            "total_queries": total,
            "llm_invoked": self._arbiter_stats["llm_invoked"],
            "llm_rate": self._arbiter_stats["llm_invoked"] / total * 100,
            "unanimous_rate": self._arbiter_stats["unanimous_decisions"] / total * 100,
            "arbiter_enabled": self.use_arbiter,
        }
    
    def process_message(
        self, 
        message: str, 
        session_id: str = "default"
    ) -> DialogueResult:
        """
        Process a user message and return the dialogue result.
        
        Uses the UnifiedIntentParser with LLM arbiter when available,
        falling back to pattern-only parsing if needed.
        
        Args:
            message: User's message
            session_id: Session identifier
            
        Returns:
            DialogueResult with intent, tool to call, and any prompts
        """
        # First, try arbiter-based parsing for complex queries
        arbiter_result = self.parse_with_arbiter(message)
        
        # Get dialogue manager (uses pattern parser internally)
        dm = self.get_dialogue_manager(session_id)
        
        # If arbiter gave us a result, we can use it to enhance the dialogue result
        if arbiter_result:
            # Store arbiter result in context for reference
            dm.context.update_state("last_arbiter_result", {
                "intent": arbiter_result.final_intent,
                "confidence": arbiter_result.confidence,
                "method": arbiter_result.method,
                "llm_used": arbiter_result.llm_invoked,
            })
        
        return dm.process_message(message)
    
    def handle_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: Any
    ) -> Optional[DialogueResult]:
        """
        Handle the result of a tool execution.
        
        For multi-step tasks, this may return a follow-up action.
        
        Args:
            session_id: Session identifier
            tool_name: Name of the tool that was executed
            result: Result from the tool
            
        Returns:
            Follow-up DialogueResult if needed, None otherwise
        """
        dm = self.get_dialogue_manager(session_id)
        return dm.handle_tool_result(tool_name, result)
    
    def get_context_for_llm(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation context for LLM prompting.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with context summary, recent messages, entities
        """
        dm = self.get_dialogue_manager(session_id)
        return dm.get_context_for_llm()
    
    def set_context_state(
        self, 
        session_id: str, 
        key: str, 
        value: Any
    ):
        """
        Set a context state variable (e.g., data_path, current_workflow).
        
        Args:
            session_id: Session identifier
            key: State key
            value: State value
        """
        dm = self.get_dialogue_manager(session_id)
        dm.context.set_state(key, value)
    
    def add_entity(
        self, 
        session_id: str, 
        entity: Entity
    ):
        """
        Add an entity to the conversation context.
        
        Useful when tools discover entities that should be tracked.
        
        Args:
            session_id: Session identifier
            entity: Entity to add
        """
        dm = self.get_dialogue_manager(session_id)
        dm.context.memory.add_entity(entity)
    
    def clear_session(self, session_id: str):
        """Clear session data."""
        if session_id in self._sessions:
            del self._sessions[session_id]


# =============================================================================
# TOOL ROUTING HELPER
# =============================================================================

def route_to_tool(result: DialogueResult) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Extract tool name and arguments from a dialogue result.
    
    Args:
        result: DialogueResult from dialogue manager
        
    Returns:
        Tuple of (tool_name, tool_args) or None if no tool should be called
    """
    if result.should_execute_tool:
        return (result.tool_name, result.tool_args)
    return None


def format_clarification_response(result: DialogueResult) -> str:
    """
    Format a clarification/error message from dialogue result.
    
    Args:
        result: DialogueResult that needs clarification
        
    Returns:
        Formatted message string
    """
    if result.message:
        return result.message
    
    if result.missing_slots:
        from .dialogue import SlotFiller
        return SlotFiller.get_prompts_for_slots(result.missing_slots)
    
    return "I need more information to help you. Could you be more specific?"


# =============================================================================
# EXAMPLE INTEGRATION
# =============================================================================

def example_integration():
    """
    Example of how to integrate with BioPipelines facade.
    
    This shows the pattern for using DialogueManager alongside
    the BioPipelines.chat() method.
    """
    
    # Initialize
    integration = ChatIntegration()
    session_id = "user_123"
    
    # Simulate conversation
    messages = [
        "Check if we have brain RNA-seq data, if not search online",
        "Download the first one",  # Uses coreference
        "Create a workflow for it",
        "yes",  # Confirmation
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        
        result = integration.process_message(msg, session_id)
        
        if result.action == "execute_tool":
            print(f"  → Calling tool: {result.tool_name}")
            print(f"  → Args: {result.tool_args}")
            
            # Simulate tool result
            if result.tool_name == "scan_data":
                # No local data found, check for follow-up
                mock_result = {"samples": []}
                follow_up = integration.handle_tool_result(
                    session_id, result.tool_name, mock_result
                )
                if follow_up and follow_up.should_execute_tool:
                    print(f"  → Follow-up: {follow_up.tool_name}")
        
        elif result.action == "slot_fill":
            print(f"  → Need info: {result.message}")
        
        elif result.action == "greeting":
            print(f"  → Response: {result.message[:100]}...")
        
        elif result.action == "llm_response":
            print(f"  → Defer to LLM")
            context = integration.get_context_for_llm(session_id)
            print(f"  → Context: {context['summary'][:100]}...")


if __name__ == "__main__":
    example_integration()
