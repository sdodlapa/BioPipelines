"""
Agents Module
=============

AI-powered agents for workflow generation and monitoring.

Available components:
- AgentTools: Tools that can be invoked during chat conversations
- ConversationContext: Tracks conversation state for multi-turn dialogue
- AgentRouter: LLM-based intent routing with function calling
- AgentBridge: Bridges LLM routing with tool execution
- CodingAgent: Specialized agent for error diagnosis and code fixes
- AgentOrchestrator: Coordinates multiple agents for complex tasks
- AgentMemory: Vector-based RAG memory for learning from interactions
- ReactAgent: Multi-step ReAct reasoning agent
"""

from .tools import AgentTools, ToolResult, ToolName, process_tool_request
from .context import ConversationContext
from .router import AgentRouter, RouteResult, RoutingStrategy, AGENT_TOOLS, route_message
from .bridge import AgentBridge, get_agent_bridge, process_with_agent
from .coding_agent import (
    CodingAgent, 
    DiagnosisResult, 
    CodeFix, 
    ErrorType,
    get_coding_agent,
    diagnose_job_error,
)
from .orchestrator import (
    AgentOrchestrator,
    SyncOrchestrator,
    AgentType,
    AgentTask,
    TaskResult,
    get_orchestrator,
    get_sync_orchestrator,
)
from .memory import (
    AgentMemory,
    MemoryEntry,
    SearchResult,
    EmbeddingModel,
)
from .react_agent import (
    ReactAgent,
    SimpleAgent,
    AgentStep,
    AgentState,
    ToolResult as ReactToolResult,
)
from .self_healing import (
    SelfHealer,
    JobMonitor,
    HealingAttempt,
    HealingAction,
    HealingStatus,
    JobInfo,
    get_self_healer,
    start_job_monitor,
    stop_job_monitor,
)
from .chat_integration import (
    AgentChatHandler,
    get_chat_handler,
    create_gradio_chat_fn,
    enhanced_chat_with_composer,
)
from .multi_model import (
    MultiModelDeployment,
    ModelConfig,
    ModelRole,
    get_deployment,
    get_model_url,
    configure_from_env,
    QUAD_H100_CONFIG,
    DUAL_H100_CONFIG,
    SINGLE_T4_CONFIG,
)

__all__ = [
    # Tools
    "AgentTools",
    "ToolResult",
    "ToolName",
    "process_tool_request",
    # Context
    "ConversationContext",
    # Router
    "AgentRouter",
    "RouteResult",
    "RoutingStrategy",
    "AGENT_TOOLS",
    "route_message",
    # Bridge
    "AgentBridge",
    "get_agent_bridge",
    "process_with_agent",
    # Coding Agent
    "CodingAgent",
    "DiagnosisResult",
    "CodeFix",
    "ErrorType",
    "get_coding_agent",
    "diagnose_job_error",
    # Orchestrator
    "AgentOrchestrator",
    "SyncOrchestrator",
    "AgentType",
    "AgentTask",
    "TaskResult",
    "get_orchestrator",
    "get_sync_orchestrator",
    # Memory (new)
    "AgentMemory",
    "MemoryEntry",
    "SearchResult",
    "EmbeddingModel",
    # ReAct Agent (new)
    "ReactAgent",
    "SimpleAgent",
    "AgentStep",
    "AgentState",
    "ReactToolResult",
    # Self-Healing (new)
    "SelfHealer",
    "JobMonitor",
    "HealingAttempt",
    "HealingAction",
    "HealingStatus",
    "JobInfo",
    "get_self_healer",
    "start_job_monitor",
    "stop_job_monitor",
    # Gradio Integration (new)
    "AgentChatHandler",
    "get_chat_handler",
    "create_gradio_chat_fn",
    "enhanced_chat_with_composer",
    # Multi-Model Deployment (new)
    "MultiModelDeployment",
    "ModelConfig",
    "ModelRole",
    "get_deployment",
    "get_model_url",
    "configure_from_env",
    "QUAD_H100_CONFIG",
    "DUAL_H100_CONFIG",
    "SINGLE_T4_CONFIG",
]
