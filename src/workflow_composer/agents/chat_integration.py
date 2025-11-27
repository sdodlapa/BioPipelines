"""
Gradio Chat Integration
========================

Integrates the agent system with the Gradio UI.

Features:
- ReAct agent for complex multi-step queries
- Memory-augmented responses
- Self-healing for failed jobs
- Streaming support for real-time feedback
- Autonomous agent integration (Phase 4)
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List, Generator, Tuple, Callable
from functools import lru_cache

logger = logging.getLogger(__name__)

# Import agent components
try:
    from .react_agent import ReactAgent, SimpleAgent, ToolResult as ReactToolResult
    from .memory import AgentMemory
    from .self_healing import SelfHealer, JobMonitor, get_self_healer
    from .tools import AgentTools, ToolName
    from .router import AgentRouter, AGENT_TOOLS
    from .coding_agent import CodingAgent, get_coding_agent
    from .orchestrator import AgentOrchestrator, SyncOrchestrator, get_sync_orchestrator
    from .context import ConversationContext
    AGENTS_AVAILABLE = True
    AgentRouter_type = AgentRouter
except ImportError as e:
    logger.warning(f"Some agent components not available: {e}")
    AGENTS_AVAILABLE = False
    AgentRouter_type = None  # type: ignore

# Import autonomous agent (Phase 4)
try:
    from .autonomous import (
        AutonomousAgent,
        create_agent,
        HealthChecker,
        HealthStatus,
        RecoveryManager,
        JobMonitor as AutonomousJobMonitor,
    )
    AUTONOMOUS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Autonomous agent not available: {e}")
    AUTONOMOUS_AVAILABLE = False
    AutonomousAgent = None
    create_agent = None
    HealthChecker = None


# =============================================================================
# Chat Handler
# =============================================================================

class AgentChatHandler:
    """
    Chat handler that integrates all agent components.
    
    This replaces the simple regex-based tool matching with a full
    agent system featuring:
    - LLM-based intent routing
    - Multi-step ReAct reasoning for complex queries
    - Memory for learning from interactions
    - Self-healing for failed jobs
    - Streaming responses
    
    Example:
        handler = AgentChatHandler(vllm_url="http://localhost:8000/v1")
        
        for response in handler.chat("scan my data in /path"):
            print(response)
    """
    
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000/v1",
        model: str = "MiniMaxAI/MiniMax-M2-Lite",
        app_state: Any = None,
        enable_memory: bool = True,
        enable_react: bool = True,
        enable_self_healing: bool = True,
        enable_autonomous: bool = True,
        autonomy_level: str = "assisted",
        memory_path: str = ".agent_memory.db",
    ):
        """
        Initialize the chat handler.
        
        Args:
            vllm_url: URL for vLLM server
            model: Model name to use
            app_state: Gradio app state with tools
            enable_memory: Use agent memory
            enable_react: Use ReAct for complex queries
            enable_self_healing: Enable auto-healing for failed jobs
            enable_autonomous: Enable autonomous agent mode
            autonomy_level: Level of autonomy (readonly/monitored/assisted/supervised/autonomous)
            memory_path: Path to memory database
        """
        self.vllm_url = vllm_url
        self.model = model
        self.app_state = app_state
        
        # Initialize client
        self._client = None
        
        # Components (lazy loaded)
        self._memory: Optional[AgentMemory] = None
        self._react_agent: Optional[ReactAgent] = None
        self._simple_agent: Optional[SimpleAgent] = None
        self._self_healer: Optional[SelfHealer] = None
        self._router: Optional[AgentRouter] = None
        self._tools: Optional[Dict[str, Callable]] = None
        self._autonomous_agent: Optional["AutonomousAgent"] = None
        
        # Feature flags
        self.enable_memory = enable_memory
        self.enable_react = enable_react
        self.enable_self_healing = enable_self_healing
        self.enable_autonomous = enable_autonomous and AUTONOMOUS_AVAILABLE
        self.autonomy_level = autonomy_level
        self.memory_path = memory_path
        
        # Tracking
        self._initialized = False
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.vllm_url,
                    api_key="not-needed-for-vllm"
                )
            except ImportError:
                logger.error("openai package not installed")
                raise
        return self._client
    
    @property
    def memory(self) -> Optional[AgentMemory]:
        """Lazy load memory."""
        if self._memory is None and self.enable_memory:
            try:
                self._memory = AgentMemory(db_path=self.memory_path)
            except Exception as e:
                logger.warning(f"Could not initialize memory: {e}")
        return self._memory
    
    @property
    def autonomous_agent(self) -> Optional["AutonomousAgent"]:
        """Lazy load autonomous agent."""
        if self._autonomous_agent is None and self.enable_autonomous:
            try:
                self._autonomous_agent = create_agent(level=self.autonomy_level)
            except Exception as e:
                logger.warning(f"Could not initialize autonomous agent: {e}")
        return self._autonomous_agent
    
    @property
    def tools(self) -> Dict[str, Callable]:
        """Get tool functions."""
        if self._tools is None:
            self._tools = self._build_tools()
        return self._tools
    
    @property
    def react_agent(self) -> Optional[ReactAgent]:
        """Lazy load ReAct agent."""
        if self._react_agent is None and self.enable_react:
            self._react_agent = ReactAgent(
                tools=self.tools,
                llm_client=self.client,
                model=self.model,
                max_steps=5,
            )
        return self._react_agent
    
    @property
    def simple_agent(self) -> SimpleAgent:
        """Lazy load simple agent."""
        if self._simple_agent is None:
            self._simple_agent = SimpleAgent(
                tools=self.tools,
                llm_client=self.client,
                model=self.model,
            )
        return self._simple_agent
    
    @property
    def router(self):
        """Lazy load router."""
        if self._router is None and AGENTS_AVAILABLE:
            self._router = AgentRouter_type(
                local_url=self.vllm_url,
                use_local=True,
            )
        return self._router
    
    def _build_tools(self) -> Dict[str, Callable]:
        """Build tool dictionary from AgentTools."""
        tools = {}
        
        if not self.app_state:
            return tools
        
        # Get AgentTools instance
        try:
            agent_tools = AgentTools(self.app_state)
            
            # Map tool names to functions
            tools["scan_data"] = lambda path: agent_tools.scan_data(path)
            tools["search_encode"] = lambda query: agent_tools.search_encode(query)
            tools["search_geo"] = lambda query: agent_tools.search_geo(query)
            tools["check_references"] = lambda organism: agent_tools.check_references(organism)
            tools["submit_job"] = lambda workflow, profile="slurm": agent_tools.submit_job(workflow, profile)
            tools["get_job_status"] = lambda job_id: agent_tools.get_job_status(job_id)
            tools["get_job_logs"] = lambda job_id: agent_tools.get_job_logs(job_id)
            tools["cancel_job"] = lambda job_id: agent_tools.cancel_job(job_id)
            tools["generate_workflow"] = lambda description: agent_tools.generate_workflow(description)
            tools["list_jobs"] = lambda: agent_tools.list_jobs()
            tools["help"] = lambda topic="": agent_tools.get_help(topic)
            
        except Exception as e:
            logger.warning(f"Could not build tools: {e}")
        
        return tools
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """
        Process a chat message (sync generator for Gradio).
        
        Args:
            message: User message
            history: Chat history
            context: Additional context
            
        Yields:
            Response chunks for streaming
        """
        if not message.strip():
            return
        
        # Try async if available, otherwise sync
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context (unusual for Gradio)
                # Fall back to sync
                yield from self._chat_sync(message, history, context)
            else:
                # Run async in event loop
                async_gen = self._chat_async(message, history, context)
                while True:
                    try:
                        result = loop.run_until_complete(async_gen.__anext__())
                        yield result
                    except StopAsyncIteration:
                        break
        except RuntimeError:
            # No event loop, use sync
            yield from self._chat_sync(message, history, context)
    
    def _chat_sync(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Synchronous chat implementation."""
        
        # Check for simple patterns first
        message_lower = message.lower().strip()
        
        # Simple commands
        if message_lower in ["help", "?", "commands"]:
            yield self._get_help()
            return
        
        # ===== AUTONOMOUS AGENT COMMANDS (Phase 4) =====
        if self.enable_autonomous and self.autonomous_agent:
            # Health check
            if message_lower in ["health", "status", "health check", "check health", "system health"]:
                yield "🏥 **System Health Check**\n\n"
                try:
                    health = asyncio.get_event_loop().run_until_complete(
                        self.autonomous_agent.check_health()
                    )
                    yield self._format_health_status(health)
                except Exception as e:
                    yield f"❌ Health check failed: {e}"
                return
            
            # Recovery commands
            if message_lower.startswith("recover") or message_lower.startswith("fix server"):
                yield "🔧 **Initiating Recovery**\n\n"
                try:
                    if "vllm" in message_lower or "server" in message_lower:
                        result = asyncio.get_event_loop().run_until_complete(
                            self.autonomous_agent.recovery.handle_server_failure("vllm")
                        )
                    else:
                        # Check health and report
                        health = asyncio.get_event_loop().run_until_complete(
                            self.autonomous_agent.check_health()
                        )
                        yield self._format_health_status(health)
                        return
                    
                    status = "✅" if result.success else "❌"
                    yield f"{status} {result.message}\n"
                    if result.details:
                        yield f"\nDetails: {result.details}"
                except Exception as e:
                    yield f"❌ Recovery failed: {e}"
                return
            
            # Agent control
            if message_lower == "start agent":
                try:
                    asyncio.get_event_loop().run_until_complete(
                        self.autonomous_agent.start_loop()
                    )
                    yield "✅ **Autonomous agent started**\n\n"
                    yield "I'll now monitor jobs and apply fixes automatically.\n"
                    yield f"Autonomy level: **{self.autonomy_level}**"
                except Exception as e:
                    yield f"❌ Could not start agent: {e}"
                return
            
            if message_lower == "stop agent":
                try:
                    asyncio.get_event_loop().run_until_complete(
                        self.autonomous_agent.stop_loop()
                    )
                    yield "⏹️ **Autonomous agent stopped**"
                except Exception as e:
                    yield f"❌ Could not stop agent: {e}"
                return
            
            # Watch job command
            if message_lower.startswith("watch job ") or message_lower.startswith("monitor job "):
                job_id = message_lower.split()[-1]
                try:
                    self.autonomous_agent.watch_job(
                        job_id=job_id,
                        on_complete=lambda e: logger.info(f"Job {e.job_id} completed"),
                    )
                    yield f"👁️ **Watching job {job_id}**\n\n"
                    yield "I'll notify you of status changes and attempt auto-recovery on failure."
                except Exception as e:
                    yield f"❌ Could not watch job: {e}"
                return
        
        # Try simple agent for basic commands
        if self._is_simple_command(message_lower):
            yield f"🔧 Processing: {message}\n\n"
            try:
                result = asyncio.get_event_loop().run_until_complete(
                    self.simple_agent.run(message, context)
                )
                yield result
            except Exception as e:
                yield f"❌ Error: {e}"
            return
        
        # Complex queries - use LLM chat
        yield "🤔 Thinking...\n\n"
        
        try:
            # Build messages
            messages = [
                {"role": "system", "content": self._get_system_prompt(context)},
                {"role": "user", "content": message}
            ]
            
            # Add memory context if available
            if self.memory:
                memory_ctx = asyncio.get_event_loop().run_until_complete(
                    self.memory.get_context(message)
                )
                if memory_ctx:
                    messages[0]["content"] += f"\n\nRelevant context:\n{memory_ctx}"
            
            # Stream response
            response_text = ""
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=2048,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response_text += content
                    yield content
            
            # Store in memory
            if self.memory:
                asyncio.get_event_loop().run_until_complete(
                    self.memory.add(
                        content=f"User: {message}\nAssistant: {response_text}",
                        memory_type="conversation",
                    )
                )
                
        except Exception as e:
            yield f"\n\n❌ Error: {e}"
    
    def _format_health_status(self, health) -> str:
        """Format health status for chat display."""
        status_icons = {
            "healthy": "🟢",
            "degraded": "🟡", 
            "unhealthy": "🔴",
            "unknown": "⚪",
        }
        
        icon = status_icons.get(health.status.value, "⚪")
        lines = [f"{icon} **Overall: {health.status.value.upper()}**\n"]
        
        for comp in health.components:
            comp_icon = status_icons.get(comp.status.value, "⚪")
            timing = f" ({comp.response_time_ms:.0f}ms)" if comp.response_time_ms else ""
            lines.append(f"  {comp_icon} **{comp.name}**: {comp.message}{timing}")
        
        lines.append(f"\n_Checked at {health.checked_at.strftime('%H:%M:%S')}_")
        return "\n".join(lines)
    
    async def _chat_async(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Async chat implementation."""
        # Simple commands
        if message.lower().strip() in ["help", "?", "commands"]:
            yield self._get_help()
            return
        
        # Use ReAct for complex queries if enabled
        if self.enable_react and self._needs_multi_step(message):
            async for chunk in self.react_agent.run_streaming(message, context):
                yield chunk
            return
        
        # Simple command
        if self._is_simple_command(message.lower()):
            yield f"🔧 Processing: {message}\n\n"
            result = await self.simple_agent.run(message, context)
            yield result
            return
        
        # LLM chat with memory
        yield "🤔 Thinking...\n\n"
        
        messages = [
            {"role": "system", "content": self._get_system_prompt(context)},
            {"role": "user", "content": message}
        ]
        
        if self.memory:
            memory_ctx = await self.memory.get_context(message)
            if memory_ctx:
                messages[0]["content"] += f"\n\nRelevant context:\n{memory_ctx}"
        
        # Stream response
        response_text = ""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=2048,
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_text += content
                yield content
        
        # Store in memory
        if self.memory:
            await self.memory.add(
                content=f"User: {message}\nAssistant: {response_text}",
                memory_type="conversation",
            )
    
    def _is_simple_command(self, message: str) -> bool:
        """Check if this is a simple one-shot command."""
        simple_patterns = [
            "scan ", "search ", "list ", "show ", "get ",
            "submit ", "run ", "cancel ", "check ",
        ]
        return any(message.startswith(p) for p in simple_patterns)
    
    def _needs_multi_step(self, message: str) -> bool:
        """Check if message needs multi-step reasoning."""
        complex_patterns = [
            "and then", "after that", "first", "analyze",
            "compare", "find and", "scan and create",
            "download and run", "process all",
        ]
        return any(p in message.lower() for p in complex_patterns)
    
    def _get_system_prompt(self, context: Optional[Dict] = None) -> str:
        """Build system prompt."""
        base = """You are BioPipelines AI Assistant, an expert in bioinformatics workflows.

You help users:
- Discover and scan data (FASTQ, BAM, etc.)
- Search databases (ENCODE, GEO, SRA)
- Generate analysis workflows (RNA-seq, ChIP-seq, etc.)
- Submit and monitor SLURM jobs
- Diagnose and fix errors

Be concise, helpful, and proactive with suggestions.
"""
        
        if context:
            if context.get("data_loaded"):
                base += f"\nUser has loaded {context.get('sample_count', 0)} samples from {context.get('data_path', 'unknown')}."
            if context.get("active_job"):
                base += f"\nActive job: {context['active_job']}"
        
        return base
    
    def _get_help(self) -> str:
        """Get help text."""
        help_text = """# BioPipelines Commands

## 📁 Data Discovery
- **"scan /path/to/data"** - Find FASTQ/BAM files
- **"search ENCODE for ChIP-seq K562"** - Search databases
- **"check reference for human"** - Verify genome references

## 🔧 Workflow Generation
- **"create RNA-seq workflow"** - Generate pipeline
- **"create workflow for this data"** - After scanning

## 🚀 Execution
- **"run the workflow"** - Submit to SLURM
- **"show jobs"** - List running jobs
- **"get logs for 12345"** - View job output
- **"cancel 12345"** - Stop a job

## 📊 Analysis
- **"analyze my data"** - Full analysis flow
- **"compare samples A vs B"** - Differential analysis
"""
        
        # Add autonomous commands if available
        if self.enable_autonomous:
            help_text += """
## 🤖 Autonomous Agent
- **"health"** / **"status"** - Check system health
- **"start agent"** - Start autonomous monitoring
- **"stop agent"** - Stop autonomous mode
- **"watch job 12345"** - Monitor job with auto-recovery
- **"recover vllm"** - Restart vLLM server
- **"fix server"** - Attempt server recovery
"""
        
        help_text += """
💡 **Tip:** You can chain commands: "scan my data and create a workflow"
"""
        return help_text


# =============================================================================
# Factory Functions
# =============================================================================

_handler_instance: Optional[AgentChatHandler] = None


def get_chat_handler(
    app_state: Any = None,
    vllm_url: str = None,
    model: str = None,
) -> AgentChatHandler:
    """
    Get or create the global chat handler.
    
    Args:
        app_state: Gradio app state
        vllm_url: vLLM server URL
        model: Model name
        
    Returns:
        AgentChatHandler instance
    """
    global _handler_instance
    
    # Get defaults from environment
    if vllm_url is None:
        vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
    if model is None:
        model = os.environ.get("VLLM_MODEL", "MiniMaxAI/MiniMax-M2-Lite")
    
    if _handler_instance is None:
        _handler_instance = AgentChatHandler(
            vllm_url=vllm_url,
            model=model,
            app_state=app_state,
        )
    elif app_state is not None:
        _handler_instance.app_state = app_state
    
    return _handler_instance


def create_gradio_chat_fn(
    app_state: Any = None,
    vllm_url: str = None,
) -> Callable:
    """
    Create a Gradio-compatible chat function.
    
    This wraps the AgentChatHandler for use with gr.Chatbot.
    
    Example:
        chat_fn = create_gradio_chat_fn(app_state)
        
        chatbot.submit(chat_fn, inputs=[...], outputs=[...])
    """
    handler = get_chat_handler(app_state, vllm_url)
    
    def gradio_chat(
        message: str,
        history: List[Dict[str, str]],
        provider: str,
        state: Any,
    ) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
        """Gradio chat handler with streaming."""
        if not message.strip():
            yield "", history
            return
        
        # Update handler's app_state
        handler.app_state = state
        
        # Add user message
        history = history + [{"role": "user", "content": message}]
        yield "", history
        
        # Get context from conversation
        context = {}
        if hasattr(state, 'conversation_context') and state.conversation_context:
            ctx = state.conversation_context
            context = {
                "data_loaded": ctx.data_path is not None,
                "sample_count": len(ctx.samples) if ctx.samples else 0,
                "data_path": ctx.data_path,
            }
        
        # Stream response
        response_text = ""
        history = history + [{"role": "assistant", "content": ""}]
        
        for chunk in handler.chat(message, history[:-2], context):
            response_text += chunk
            history[-1]["content"] = response_text
            yield "", history
    
    return gradio_chat


# =============================================================================
# Streaming Chat Response
# =============================================================================

def enhanced_chat_with_composer(
    message: str,
    history: List[Dict[str, str]],
    provider: str,
    app_state: Any,
) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
    """
    Enhanced chat function that uses the agent system.
    
    Drop-in replacement for the original chat_with_composer.
    Falls back to simple chat if agents not available.
    """
    if not AGENTS_AVAILABLE:
        yield "", history + [{"role": "assistant", "content": "⚠️ Agent system not available. Using basic mode."}]
        return
    
    # Use the Gradio chat function
    chat_fn = create_gradio_chat_fn(app_state)
    yield from chat_fn(message, history, provider, app_state)
