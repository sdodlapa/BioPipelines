"""
Web UI module for BioPipelines Workflow Composer.

Provides a chat-first Gradio interface with:
- Unified chat handler with tool integration
- Session management
- LLM fallback chain (local vLLM → GitHub Models → Gemini → OpenAI)

Usage:
    # Start the web UI
    python -m workflow_composer.web.app --share
    
    # Or via start_server.sh
    ./scripts/start_server.sh --cloud   # Cloud LLM only
    ./scripts/start_server.sh --gpu     # Local vLLM on GPU
"""

# Import Gradio app (main interface)
try:
    from .app import create_app, main
except ImportError:
    create_app = None
    main = None

# Import chat handler
try:
    from .chat_handler import UnifiedChatHandler, get_chat_handler
except ImportError:
    UnifiedChatHandler = None
    get_chat_handler = None

# Import utilities
try:
    from .utils import detect_vllm_endpoint, get_default_port, use_local_llm
except ImportError:
    detect_vllm_endpoint = None
    get_default_port = None
    use_local_llm = None

__all__ = [
    'create_app',
    'main',
    'UnifiedChatHandler',
    'get_chat_handler',
    'detect_vllm_endpoint',
    'get_default_port',
    'use_local_llm',
]
