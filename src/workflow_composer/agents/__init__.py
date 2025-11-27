"""
Agents Module
=============

AI-powered agents for workflow generation and monitoring.

Available components:
- AgentTools: Tools that can be invoked during chat conversations
- ConversationContext: Tracks conversation state for multi-turn dialogue
- AgentRouter: LLM-based intent routing with function calling
- AgentBridge: Bridges LLM routing with tool execution
"""

from .tools import AgentTools, ToolResult, ToolName, process_tool_request
from .context import ConversationContext
from .router import AgentRouter, RouteResult, RoutingStrategy, AGENT_TOOLS, route_message
from .bridge import AgentBridge, get_agent_bridge, process_with_agent

__all__ = [
    # Tools
    "AgentTools",
    "ToolResult",
    "ToolName",
    "process_tool_request",
    # Context
    "ConversationContext",
    # Router (new)
    "AgentRouter",
    "RouteResult",
    "RoutingStrategy",
    "AGENT_TOOLS",
    "route_message",
    # Bridge (new)
    "AgentBridge",
    "get_agent_bridge",
    "process_with_agent",
]
