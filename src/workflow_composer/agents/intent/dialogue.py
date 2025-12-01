"""
Dialogue Manager
================

Manages conversation flow and multi-turn task execution.

Features:
- Task state tracking (multi-step workflows)
- Intent history for coherent responses
- Slot filling for incomplete requests
- Confirmation handling
- LLM Arbiter integration for intelligent intent parsing
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable, Union

from .parser import IntentParser, IntentResult, IntentType, Entity, EntityType
from .context import ConversationContext

logger = logging.getLogger(__name__)


# =============================================================================
# CONVERSATION STATE
# =============================================================================

class ConversationPhase(Enum):
    """Current phase of conversation."""
    IDLE = auto()                # No active task
    INFORMATION_GATHERING = auto()  # Collecting info for task
    CONFIRMING = auto()          # Awaiting user confirmation
    EXECUTING = auto()           # Task in progress
    REVIEWING = auto()           # Reviewing results
    ERROR_HANDLING = auto()      # Handling an error


class TaskStatus(Enum):
    """Status of a multi-step task."""
    PENDING = auto()
    IN_PROGRESS = auto()
    WAITING_INPUT = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class TaskState:
    """State of a multi-step task."""
    task_id: str
    task_type: str  # e.g., "workflow_execution", "data_download"
    status: TaskStatus = TaskStatus.PENDING
    
    # Steps tracking
    total_steps: int = 1
    current_step: int = 0
    step_descriptions: List[str] = field(default_factory=list)
    
    # Data collected
    slots: Dict[str, Any] = field(default_factory=dict)
    required_slots: List[str] = field(default_factory=list)
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def is_complete(self) -> bool:
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
    
    @property
    def missing_slots(self) -> List[str]:
        return [s for s in self.required_slots if s not in self.slots or self.slots[s] is None]
    
    def advance_step(self):
        """Move to next step."""
        self.current_step = min(self.current_step + 1, self.total_steps)
    
    def complete(self, results: Dict = None):
        """Mark task as complete."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        if results:
            self.results.update(results)
    
    def fail(self, error: str):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()


@dataclass
class ConversationState:
    """Overall conversation state."""
    phase: ConversationPhase = ConversationPhase.IDLE
    active_task: Optional[TaskState] = None
    
    # Intent history
    intent_history: List[IntentResult] = field(default_factory=list)
    
    # Turn tracking
    turn_count: int = 0
    last_user_intent: Optional[IntentType] = None
    last_assistant_action: Optional[str] = None
    
    # Expectations
    expected_intent_types: List[IntentType] = field(default_factory=list)
    
    def add_intent(self, result: IntentResult):
        """Record an intent."""
        self.intent_history.append(result)
        if len(self.intent_history) > 20:
            self.intent_history = self.intent_history[-20:]
        self.last_user_intent = result.primary_intent
    
    def get_recent_intents(self, n: int = 5) -> List[IntentType]:
        """Get last n intents."""
        return [r.primary_intent for r in self.intent_history[-n:]]
    
    def is_expecting(self, intent_type: IntentType) -> bool:
        """Check if this intent type is expected."""
        return intent_type in self.expected_intent_types


# =============================================================================
# TASK TEMPLATES
# =============================================================================

# Templates for multi-step tasks
TASK_TEMPLATES = {
    "search_and_download": {
        "steps": ["search", "select", "download"],
        "required_slots": ["query", "source"],
        "optional_slots": ["organism", "assay_type", "tissue"],
    },
    "workflow_execution": {
        "steps": ["validate_data", "generate_workflow", "configure", "submit"],
        "required_slots": ["data_path", "workflow_type"],
        "optional_slots": ["parameters", "output_dir"],
    },
    "data_analysis": {
        "steps": ["locate_data", "select_analysis", "run", "interpret"],
        "required_slots": ["data_path", "analysis_type"],
        "optional_slots": ["parameters"],
    },
    "error_recovery": {
        "steps": ["diagnose", "propose_fix", "apply_fix", "verify"],
        "required_slots": ["job_id"],
        "optional_slots": ["log_path"],
    },
}


# =============================================================================
# SLOT FILLER
# =============================================================================

class SlotFiller:
    """
    Fill missing slots by asking clarifying questions.
    """
    
    SLOT_PROMPTS = {
        "query": "What would you like to search for?",
        "source": "Which database should I search? (GEO, ENCODE, TCGA)",
        "organism": "What organism is this for? (human, mouse, etc.)",
        "assay_type": "What type of data? (RNA-seq, ChIP-seq, etc.)",
        "tissue": "What tissue or cell type?",
        "data_path": "Where is your data located?",
        "workflow_type": "What type of analysis workflow? (RNA-seq, ChIP-seq, etc.)",
        "job_id": "Which job ID?",
        "dataset_id": "Which dataset ID? (e.g., GSE12345, ENCSR000AAA)",
    }
    
    @classmethod
    def get_prompt(cls, slot_name: str) -> str:
        """Get prompt for a missing slot."""
        return cls.SLOT_PROMPTS.get(slot_name, f"Please provide {slot_name}:")
    
    @classmethod
    def get_prompts_for_slots(cls, slots: List[str]) -> str:
        """Get combined prompt for multiple missing slots."""
        if not slots:
            return ""
        if len(slots) == 1:
            return cls.get_prompt(slots[0])
        
        prompts = [f"• {cls.get_prompt(s)}" for s in slots[:3]]
        return "I need a few more details:\n" + "\n".join(prompts)


# =============================================================================
# DIALOGUE MANAGER
# =============================================================================

class DialogueManager:
    """
    Manages conversation flow and coordinates between components.
    
    Responsibilities:
    - Intent parsing with context
    - Coreference resolution
    - Task state management
    - Slot filling
    - Response generation coordination
    - LLM Arbiter for ambiguous intents (optional)
    """
    
    def __init__(
        self, 
        intent_parser: IntentParser = None,
        context: ConversationContext = None,
        use_arbiter: bool = True,
        use_cascade: bool = True,
    ):
        """
        Initialize DialogueManager with intent parsing and context.
        
        Args:
            intent_parser: Parser for intent extraction. If None and use_arbiter=True,
                          creates UnifiedIntentParser with arbiter. Otherwise uses IntentParser.
            context: Conversation context for state management
            use_arbiter: Whether to use LLM arbiter for ambiguous intents (default: True)
            use_cascade: Whether to use cascading provider router for LLM calls (default: True)
        """
        self.use_arbiter = use_arbiter
        self.use_cascade = use_cascade
        
        if intent_parser is not None:
            self.parser = intent_parser
        elif use_arbiter:
            try:
                from .unified_parser import UnifiedIntentParser
                self.parser = UnifiedIntentParser(use_cascade=use_cascade)
                logger.info("DialogueManager using UnifiedIntentParser with arbiter")
            except ImportError:
                logger.warning("UnifiedIntentParser not available, falling back to IntentParser")
                self.parser = IntentParser()
        else:
            self.parser = IntentParser()
            
        self.context = context or ConversationContext()
        self.state = ConversationState()
        
        # Intent -> tool mapping
        self._intent_tool_map = self._build_intent_tool_map()
    
    def _build_intent_tool_map(self) -> Dict[IntentType, str]:
        """Map intents to tool names."""
        return {
            IntentType.DATA_SCAN: "scan_data",
            IntentType.DATA_SEARCH: "search_databases",
            IntentType.DATA_DOWNLOAD: "download_dataset",
            IntentType.DATA_VALIDATE: "validate_dataset",
            IntentType.DATA_CLEANUP: "cleanup_data",
            IntentType.DATA_DESCRIBE: "describe_files",
            
            IntentType.WORKFLOW_CREATE: "generate_workflow",
            IntentType.WORKFLOW_LIST: "list_workflows",
            IntentType.WORKFLOW_VISUALIZE: "visualize_workflow",
            
            IntentType.REFERENCE_CHECK: "check_references",
            IntentType.REFERENCE_DOWNLOAD: "download_reference",
            IntentType.REFERENCE_INDEX: "build_index",
            
            IntentType.JOB_SUBMIT: "submit_job",
            IntentType.JOB_STATUS: "get_job_status",
            IntentType.JOB_LOGS: "get_logs",
            IntentType.JOB_CANCEL: "cancel_job",
            IntentType.JOB_RESUBMIT: "resubmit_job",
            IntentType.JOB_LIST: "list_jobs",
            IntentType.JOB_WATCH: "watch_job",
            
            IntentType.ANALYSIS_INTERPRET: "analyze_results",
            IntentType.ANALYSIS_COMPARE: "compare_samples",
            
            IntentType.DIAGNOSE_ERROR: "diagnose_error",
            IntentType.DIAGNOSE_RECOVER: "recover_error",
            
            IntentType.SYSTEM_STATUS: "check_system_health",
            IntentType.SYSTEM_RESTART: "restart_vllm",
            
            IntentType.EDUCATION_EXPLAIN: "explain_concept",
            IntentType.EDUCATION_HELP: "get_help",
        }
    
    def process_message(self, message: str) -> "DialogueResult":
        """
        Process a user message and determine appropriate action.
        
        This is the main entry point for the dialogue manager.
        
        Args:
            message: User's message
            
        Returns:
            DialogueResult with intent, tool to call, and any prompts
        """
        self.state.turn_count += 1
        
        # Step 1: Resolve coreferences
        resolved_message, resolutions = self.context.resolve_references(message)
        
        # Step 2: Parse intent
        intent_result = self.parser.parse(
            resolved_message, 
            context={"state": self.state, "entities": self.context.get_salient_entities()}
        )
        
        # Add resolved entities
        for surface, entity in resolutions.items():
            intent_result.entities.append(entity)
        
        # Step 3: Record in context
        self.context.add_turn("user", message, intent_result.entities, intent_result.primary_intent)
        self.state.add_intent(intent_result)
        
        # Step 4: Handle based on phase and intent
        if self.state.phase == ConversationPhase.CONFIRMING:
            return self._handle_confirmation(intent_result)
        
        if intent_result.primary_intent == IntentType.META_CONFIRM:
            return self._handle_unexpected_confirmation()
        
        if intent_result.primary_intent == IntentType.META_CANCEL:
            return self._handle_cancellation()
        
        # Step 5: Check for composite intents
        if intent_result.primary_intent.name.startswith("COMPOSITE_"):
            return self._handle_composite_intent(intent_result)
        
        # Step 6: Build tool call
        return self._build_tool_call(intent_result)
    
    def _handle_confirmation(self, intent_result: IntentResult) -> "DialogueResult":
        """Handle response to confirmation prompt."""
        pending = self.context.get_pending_confirmation()
        
        if intent_result.primary_intent == IntentType.META_CONFIRM:
            self.state.phase = ConversationPhase.EXECUTING
            self.context.clear_pending_confirmation()
            
            # Execute pending action
            return DialogueResult(
                intent=intent_result,
                action="execute_pending",
                tool_name=pending.get("action") if pending else None,
                tool_args=pending.get("data", {}) if pending else {},
                message="Proceeding...",
            )
        
        elif intent_result.primary_intent == IntentType.META_CANCEL:
            self.state.phase = ConversationPhase.IDLE
            self.context.clear_pending_confirmation()
            
            return DialogueResult(
                intent=intent_result,
                action="cancelled",
                message="Cancelled.",
            )
        
        else:
            # User provided new intent instead of confirming
            self.state.phase = ConversationPhase.IDLE
            self.context.clear_pending_confirmation()
            return self._build_tool_call(intent_result)
    
    def _handle_unexpected_confirmation(self) -> "DialogueResult":
        """Handle confirmation when not expected."""
        return DialogueResult(
            intent=IntentResult(
                primary_intent=IntentType.META_CONFIRM,
                confidence=1.0,
                entities=[],
            ),
            action="clarify",
            message="I'm not sure what you're confirming. What would you like me to do?",
        )
    
    def _handle_cancellation(self) -> "DialogueResult":
        """Handle cancellation request."""
        if self.state.active_task:
            self.state.active_task.status = TaskStatus.CANCELLED
            self.state.active_task = None
        
        self.state.phase = ConversationPhase.IDLE
        self.context.clear_pending_confirmation()
        
        return DialogueResult(
            intent=IntentResult(
                primary_intent=IntentType.META_CANCEL,
                confidence=1.0,
                entities=[],
            ),
            action="cancelled",
            message="Cancelled. What would you like to do instead?",
        )
    
    def _handle_composite_intent(self, intent_result: IntentResult) -> "DialogueResult":
        """Handle composite (multi-step) intents."""
        intent = intent_result.primary_intent
        
        if intent == IntentType.COMPOSITE_CHECK_THEN_SEARCH:
            # Create a task for check-then-search
            task = TaskState(
                task_id=f"check_search_{self.state.turn_count}",
                task_type="check_then_search",
                total_steps=2,
                step_descriptions=["Check local data", "Search online if not found"],
                required_slots=["query"],
            )
            
            # Extract query from slots
            query = intent_result.slots.get("query", "")
            if query:
                task.slots["query"] = query
            
            self.state.active_task = task
            self.state.phase = ConversationPhase.EXECUTING
            
            # Return first action (scan local)
            return DialogueResult(
                intent=intent_result,
                action="execute_tool",
                tool_name="scan_data",
                tool_args={"query": query},
                message=f"First, I'll check if we have {query} locally...",
                follow_up_action="search_databases",
                follow_up_condition="no_results",
            )
        
        # Default: treat as the primary sub-intent
        return self._build_tool_call(intent_result)
    
    def _build_tool_call(self, intent_result: IntentResult) -> "DialogueResult":
        """Build tool call from intent."""
        intent = intent_result.primary_intent
        
        # Check for meta intents that don't map to tools
        if intent.name.startswith("META_"):
            return self._handle_meta_intent(intent_result)
        
        if intent == IntentType.META_UNKNOWN:
            return DialogueResult(
                intent=intent_result,
                action="clarify",
                message=intent_result.clarification_prompt or 
                        "I'm not sure what you mean. Could you rephrase?",
                needs_clarification=True,
            )
        
        # Get tool name
        tool_name = self._intent_tool_map.get(intent)
        if not tool_name:
            return DialogueResult(
                intent=intent_result,
                action="llm_response",
                message=f"I understand you want to {intent.name}, but I don't have a tool for that yet.",
            )
        
        # Build tool arguments
        tool_args = self._build_tool_args(intent, intent_result)
        
        # Check for missing required slots
        missing = self._check_required_slots(tool_name, tool_args)
        if missing:
            self.state.phase = ConversationPhase.INFORMATION_GATHERING
            return DialogueResult(
                intent=intent_result,
                action="slot_fill",
                tool_name=tool_name,
                tool_args=tool_args,
                message=SlotFiller.get_prompts_for_slots(missing),
                needs_clarification=True,
                missing_slots=missing,
            )
        
        # Ready to execute
        return DialogueResult(
            intent=intent_result,
            action="execute_tool",
            tool_name=tool_name,
            tool_args=tool_args,
        )
    
    def _build_tool_args(
        self, 
        intent: IntentType, 
        result: IntentResult
    ) -> Dict[str, Any]:
        """Build tool arguments from intent result."""
        args = dict(result.slots)
        
        # Add from entities
        for entity in result.entities:
            if entity.type == EntityType.DIRECTORY_PATH:
                args.setdefault("path", entity.value)
            elif entity.type == EntityType.FILE_PATH:
                args.setdefault("file_path", entity.value)
            elif entity.type == EntityType.DATASET_ID:
                args.setdefault("dataset_id", entity.canonical)
            elif entity.type == EntityType.JOB_ID:
                args.setdefault("job_id", entity.value)
            elif entity.type == EntityType.ORGANISM:
                args.setdefault("organism", entity.canonical)
            elif entity.type == EntityType.TISSUE:
                args.setdefault("tissue", entity.canonical)
            elif entity.type == EntityType.ASSAY_TYPE:
                args.setdefault("assay_type", entity.canonical)
            elif entity.type == EntityType.DISEASE:
                args.setdefault("disease", entity.canonical)
                args.setdefault("cancer_type", entity.canonical)
        
        # Build query string for search intents
        if intent in [IntentType.DATA_SEARCH]:
            if "query" not in args:
                query_parts = []
                for key in ["organism", "tissue", "assay_type", "disease"]:
                    if key in args:
                        query_parts.append(args[key])
                if query_parts:
                    args["query"] = " ".join(query_parts)
        
        # Add context-based defaults
        context_path = self.context.get_state("data_path")
        if context_path and "path" not in args:
            args.setdefault("path", context_path)
        
        return args
    
    def _check_required_slots(self, tool_name: str, args: Dict) -> List[str]:
        """Check for missing required slots."""
        required = {
            "search_databases": ["query"],
            "download_dataset": ["dataset_id"],
            "generate_workflow": ["workflow_type"],
            "submit_job": [],  # Can infer from context
            "get_job_status": [],  # Can check all jobs
            "explain_concept": ["concept"],
        }
        
        tool_required = required.get(tool_name, [])
        return [s for s in tool_required if s not in args or not args[s]]
    
    def _handle_meta_intent(self, result: IntentResult) -> "DialogueResult":
        """Handle meta/conversational intents."""
        intent = result.primary_intent
        
        if intent == IntentType.META_GREETING:
            return DialogueResult(
                intent=result,
                action="greeting",
                message="Hello! I'm your BioPipelines assistant. How can I help you today?\n\nI can help you:\n• Find and download genomics data\n• Generate analysis workflows\n• Run jobs on the cluster\n• Troubleshoot errors\n\nJust tell me what you need!",
            )
        
        if intent == IntentType.META_THANKS:
            return DialogueResult(
                intent=result,
                action="acknowledge",
                message="You're welcome! Let me know if you need anything else.",
            )
        
        if intent == IntentType.EDUCATION_HELP:
            return DialogueResult(
                intent=result,
                action="execute_tool",
                tool_name="get_help",
                tool_args={},
            )
        
        return DialogueResult(
            intent=result,
            action="llm_response",
        )
    
    def handle_tool_result(self, tool_name: str, result: Any) -> Optional["DialogueResult"]:
        """
        Handle result from tool execution for multi-step tasks.
        
        Args:
            tool_name: Name of tool that was executed
            result: Result from tool
            
        Returns:
            DialogueResult for follow-up action, or None if complete
        """
        task = self.state.active_task
        if not task:
            return None
        
        # Update task progress
        task.advance_step()
        
        # Check for follow-up actions
        if task.task_type == "check_then_search":
            if task.current_step == 1:
                # After scanning, check if we found data
                found_data = bool(result.data.get("samples", [])) if hasattr(result, "data") else False
                
                if not found_data:
                    # Proceed to search
                    query = task.slots.get("query", "")
                    return DialogueResult(
                        intent=IntentResult(
                            primary_intent=IntentType.DATA_SEARCH,
                            confidence=1.0,
                            entities=[],
                        ),
                        action="execute_tool",
                        tool_name="search_databases",
                        tool_args={"query": query},
                        message=f"No local data found. Searching online for {query}...",
                    )
                else:
                    task.complete()
                    self.state.phase = ConversationPhase.IDLE
        
        return None
    
    def get_context_for_llm(self) -> Dict[str, Any]:
        """Get context for LLM responses."""
        return {
            "summary": self.context.get_context_summary(),
            "messages": self.context.get_messages_for_llm(10),
            "salient_entities": [
                {"type": e.type.name, "value": e.canonical}
                for e in self.context.get_salient_entities(5)
            ],
            "phase": self.state.phase.name,
            "recent_intents": [i.name for i in self.state.get_recent_intents(3)],
        }


# =============================================================================
# DIALOGUE RESULT
# =============================================================================

@dataclass
class DialogueResult:
    """Result of dialogue processing."""
    intent: IntentResult
    action: str  # "execute_tool", "slot_fill", "clarify", "llm_response", etc.
    
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    
    message: Optional[str] = None
    needs_clarification: bool = False
    missing_slots: List[str] = field(default_factory=list)
    
    # For multi-step
    follow_up_action: Optional[str] = None
    follow_up_condition: Optional[str] = None
    
    @property
    def should_execute_tool(self) -> bool:
        return self.action == "execute_tool" and self.tool_name is not None
    
    @property
    def should_use_llm(self) -> bool:
        return self.action == "llm_response"
