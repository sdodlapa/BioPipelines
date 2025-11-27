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
    # Coding Agent (new)
    "CodingAgent",
    "DiagnosisResult",
    "CodeFix",
    "ErrorType",
    "get_coding_agent",
    "diagnose_job_error",
    # Orchestrator (new)
    "AgentOrchestrator",
    "SyncOrchestrator",
    "AgentType",
    "AgentTask",
    "TaskResult",
    "get_orchestrator",
    "get_sync_orchestrator",
]
