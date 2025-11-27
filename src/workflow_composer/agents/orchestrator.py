"""
Agent Orchestrator for BioPipelines
=====================================

Lightweight orchestration layer that coordinates specialized agents:
- Router: Intent detection and tool routing
- CodingAgent: Error diagnosis and code fixes
- AgentTools: Data discovery and workflow execution

NO external frameworks required - just uses our existing components.

Architecture:
    User Query → Orchestrator
                     ↓
    ┌────────────────┼────────────────┐
    ↓                ↓                ↓
  Router         CodingAgent      AgentTools
  (routing)      (errors/code)    (execution)
    ↓                ↓                ↓
    └────────────────┼────────────────┘
                     ↓
               Final Response
"""

import os
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized agents."""
    ROUTER = "router"           # Query routing and planning
    DATA = "data"               # Data discovery and management
    WORKFLOW = "workflow"       # Workflow generation
    CODING = "coding"           # Error diagnosis and code fixes
    EXECUTOR = "executor"       # Job submission and monitoring


@dataclass
class AgentTask:
    """A task to be executed by an agent."""
    agent_type: AgentType
    action: str
    params: Dict[str, Any]
    description: str


@dataclass
class TaskResult:
    """Result from an agent task."""
    success: bool
    agent_type: AgentType
    action: str
    data: Dict[str, Any]
    message: str
    error: Optional[str] = None


class AgentOrchestrator:
    """
    Coordinates multiple specialized agents for complex tasks.
    
    This is a lightweight orchestrator that:
    1. Uses the existing AgentRouter for intent detection
    2. Routes to specialized handlers (CodingAgent, AgentTools)
    3. Synthesizes responses
    4. Handles multi-step workflows
    
    No external frameworks (LangChain, etc.) required.
    """
    
    def __init__(
        self,
        vllm_url: Optional[str] = None,
        vllm_coder_url: Optional[str] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            vllm_url: URL of main vLLM server (supervisor model)
            vllm_coder_url: URL of coding model vLLM server (optional)
        """
        self.vllm_url = vllm_url or os.environ.get("VLLM_URL", "http://localhost:8000/v1")
        self.vllm_coder_url = vllm_coder_url or os.environ.get("VLLM_CODER_URL")
        
        # Lazy-loaded agents
        self._router = None
        self._coding_agent = None
        self._tools = None
        self._bridge = None
    
    @property
    def router(self):
        """Get or create the router."""
        if self._router is None:
            from .router import AgentRouter
            self._router = AgentRouter(
                local_url=self.vllm_url,
                use_local=True,
                use_cloud=True,
            )
        return self._router
    
    @property
    def coding_agent(self):
        """Get or create the coding agent."""
        if self._coding_agent is None:
            from .coding_agent import CodingAgent
            self._coding_agent = CodingAgent(
                coder_url=self.vllm_coder_url,
                fallback_url=self.vllm_url,
            )
        return self._coding_agent
    
    @property
    def tools(self):
        """Get or create the tools handler."""
        if self._tools is None:
            from .tools import AgentTools
            self._tools = AgentTools()
        return self._tools
    
    @property
    def bridge(self):
        """Get or create the bridge."""
        if self._bridge is None:
            from .bridge import AgentBridge
            self._bridge = AgentBridge()
            # Configure for local vLLM if available
            if self.router.is_local_available():
                self._bridge.router = self.router
        return self._bridge
    
    async def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Process a user query through the agent system.
        
        Args:
            query: User's natural language query
            context: Current conversation/application context
            stream: Whether to stream intermediate updates
            
        Returns:
            Final response string, or async generator if streaming
        """
        if stream:
            return self._process_stream(query, context)
        else:
            return await self._process_direct(query, context)
    
    async def _process_direct(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process query and return final response."""
        context = context or {}
        
        # Step 1: Route the query
        route_result = await self.router.route(query, context)
        
        # Step 2: Handle based on routing result
        if route_result.tool:
            # Execute the tool
            tool_result = await self._execute_tool(
                route_result.tool,
                route_result.arguments,
                context,
            )
            
            # Check if we need coding agent for errors
            if not tool_result.success and tool_result.error:
                diagnosis = await self._diagnose_if_needed(tool_result)
                if diagnosis:
                    return self._format_error_response(tool_result, diagnosis)
            
            return tool_result.message
        
        elif route_result.response:
            # Direct conversational response
            return route_result.response
        
        else:
            # Couldn't determine intent
            return "I'm not sure what you're asking. Could you please rephrase your request?"
    
    async def _process_stream(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Process query with streaming updates."""
        context = context or {}
        
        yield "🤔 Analyzing your request...\n\n"
        
        # Route
        route_result = await self.router.route(query, context)
        
        if route_result.tool:
            yield f"⚙️ Using: `{route_result.tool}`\n"
            if route_result.arguments:
                yield f"   Parameters: {route_result.arguments}\n\n"
            
            # Execute
            tool_result = await self._execute_tool(
                route_result.tool,
                route_result.arguments,
                context,
            )
            
            if tool_result.success:
                yield f"✅ Success!\n\n{tool_result.message}"
            else:
                yield f"❌ Error: {tool_result.error}\n\n"
                
                # Try diagnosis
                yield "🔧 Diagnosing error...\n"
                diagnosis = await self._diagnose_if_needed(tool_result)
                if diagnosis:
                    yield f"\n**Diagnosis:**\n{diagnosis.explanation}\n"
                    if diagnosis.suggested_fix:
                        yield f"\n**Suggested Fix:**\n```\n{diagnosis.suggested_fix}\n```\n"
                else:
                    yield tool_result.message
        
        elif route_result.response:
            yield route_result.response
        
        else:
            yield "I'm not sure what you're asking. Could you please rephrase your request?"
    
    async def _execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Dict[str, Any],
    ) -> TaskResult:
        """Execute a tool and return the result."""
        try:
            # Get the tool method
            tool_method = getattr(self.tools, tool_name, None)
            if not tool_method:
                return TaskResult(
                    success=False,
                    agent_type=AgentType.EXECUTOR,
                    action=tool_name,
                    data={},
                    message=f"Unknown tool: {tool_name}",
                    error=f"Tool '{tool_name}' not found",
                )
            
            # Execute
            result = tool_method(**arguments)
            
            return TaskResult(
                success=result.success,
                agent_type=AgentType.EXECUTOR,
                action=tool_name,
                data=result.data or {},
                message=result.message,
                error=result.error,
            )
            
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            return TaskResult(
                success=False,
                agent_type=AgentType.EXECUTOR,
                action=tool_name,
                data={},
                message=f"Tool execution failed: {e}",
                error=str(e),
            )
    
    async def _diagnose_if_needed(self, result: TaskResult):
        """Diagnose an error if coding agent is available."""
        if not result.error:
            return None
        
        try:
            diagnosis = self.coding_agent.diagnose_error(
                error_log=result.error,
                context={"action": result.action},
                use_llm=True,
            )
            if diagnosis.confidence > 0.5:
                return diagnosis
        except Exception as e:
            logger.debug(f"Diagnosis failed: {e}")
        
        return None
    
    def _format_error_response(self, result: TaskResult, diagnosis) -> str:
        """Format an error response with diagnosis."""
        parts = [
            f"❌ **Error in {result.action}**",
            "",
            f"**What happened:** {diagnosis.explanation}",
            "",
        ]
        
        if diagnosis.suggested_fix:
            parts.extend([
                "**Suggested fix:**",
                f"```",
                diagnosis.suggested_fix,
                "```",
                "",
            ])
        
        if diagnosis.auto_fixable:
            parts.append("💡 This error can be fixed automatically. Would you like me to apply the fix?")
        
        return "\n".join(parts)
    
    async def diagnose_job(
        self,
        job_id: str,
        auto_fix: bool = False,
    ) -> Dict[str, Any]:
        """
        Diagnose a failed job.
        
        Args:
            job_id: The SLURM job ID
            auto_fix: Whether to automatically apply fixes
            
        Returns:
            Diagnosis result with optional fix
        """
        # Get job logs
        logs_result = await self._execute_tool(
            "get_logs",
            {"job_id": job_id},
            {},
        )
        
        if not logs_result.success:
            return {
                "success": False,
                "error": f"Could not get logs for job {job_id}",
            }
        
        # Diagnose
        diagnosis = self.coding_agent.diagnose_error(
            error_log=logs_result.data.get("content", logs_result.message),
            context={"job_id": job_id},
        )
        
        result = {
            "success": True,
            "job_id": job_id,
            "error_type": diagnosis.error_type.value,
            "root_cause": diagnosis.root_cause,
            "explanation": diagnosis.explanation,
            "suggested_fix": diagnosis.suggested_fix,
            "auto_fixable": diagnosis.auto_fixable,
            "confidence": diagnosis.confidence,
        }
        
        # Auto-fix if requested
        if auto_fix and diagnosis.auto_fixable and diagnosis.suggested_fix:
            # Would apply fix here
            result["fix_applied"] = False
            result["fix_message"] = "Auto-fix not yet implemented"
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        status = {
            "vllm_url": self.vllm_url,
            "vllm_coder_url": self.vllm_coder_url,
            "router": self.router.get_status() if self._router else None,
            "coding_agent": self.coding_agent.get_status() if self._coding_agent else None,
        }
        return status


# =============================================================================
# Synchronous Wrapper for Gradio
# =============================================================================

class SyncOrchestrator:
    """
    Synchronous wrapper for use with Gradio.
    
    Gradio doesn't handle async well in all cases, so this provides
    a sync interface to the async orchestrator.
    """
    
    def __init__(self, orchestrator: AgentOrchestrator = None):
        self._orchestrator = orchestrator
    
    @property
    def orchestrator(self) -> AgentOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = AgentOrchestrator()
        return self._orchestrator
    
    def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process a query synchronously."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.orchestrator.process(query, context, stream=False)
        )
    
    def diagnose_job(self, job_id: str, auto_fix: bool = False) -> Dict[str, Any]:
        """Diagnose a job synchronously."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.orchestrator.diagnose_job(job_id, auto_fix)
        )


# =============================================================================
# Convenience Functions
# =============================================================================

_orchestrator = None
_sync_orchestrator = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def get_sync_orchestrator() -> SyncOrchestrator:
    """Get or create the global sync orchestrator instance."""
    global _sync_orchestrator
    if _sync_orchestrator is None:
        _sync_orchestrator = SyncOrchestrator(get_orchestrator())
    return _sync_orchestrator


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        orchestrator = AgentOrchestrator()
        
        print("\n🎯 Agent Orchestrator Status:")
        print(json.dumps(orchestrator.get_status(), indent=2, default=str))
        
        print("\n📝 Testing queries...")
        
        test_queries = [
            "scan data in /scratch/data/raw",
            "what went wrong with job 12345?",
            "create an RNA-seq workflow for human samples",
            "hello, how are you?",
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            response = await orchestrator.process(query)
            print(f"✅ Response: {response[:200]}...")
    
    asyncio.run(test())
