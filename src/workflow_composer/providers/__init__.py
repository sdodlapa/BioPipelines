"""
Unified Provider Framework for BioPipelines
============================================

This package provides a single, unified interface to all LLM providers.
It consolidates the previously scattered implementations into one clean layer.

Architecture:
    providers/
    ├── base.py           # BaseProvider abstract class
    ├── registry.py       # Central provider & model registry
    ├── router.py         # Smart routing with fallback
    ├── lightning.py      # Lightning.ai (FREE tier)
    ├── gemini.py         # Google Gemini (FREE tier)
    ├── openai.py         # OpenAI (paid)
    ├── anthropic.py      # Anthropic Claude (paid)
    ├── ollama.py         # Local Ollama (free)
    ├── vllm.py           # Local vLLM server (free)
    └── utils/
        ├── health.py     # Health checking
        └── metrics.py    # Usage tracking

Provider Priority (waterfall on failure):
    1. Lightning.ai   - 30M free tokens/month
    2. Gemini         - Free tier with rate limits
    3. OpenAI         - Paid, most reliable
    4. Anthropic      - Paid, high quality
    5. Ollama         - Local, free
    6. vLLM           - Local GPU, free

Usage:
    from workflow_composer.providers import (
        get_provider,
        get_best_provider,
        complete,
        chat,
        ProviderRouter,
    )
    
    # Simple completion (auto-selects best provider)
    response = complete("Explain RNA-seq analysis")
    
    # Chat with messages
    response = chat([
        Message.system("You are a bioinformatics expert."),
        Message.user("What causes OOM errors in STAR?"),
    ])
    
    # Force specific provider
    response = complete("Debug this error", provider="gemini")
    
    # Get provider instance for advanced use
    provider = get_provider("lightning")
    response = provider.complete("...")
    
    # Smart routing with automatic fallback
    router = ProviderRouter()
    response = router.complete("...", fallback=True)
"""

__version__ = "2.0.0"

# Core types
from .base import (
    BaseProvider,
    Message,
    Role,
    ProviderResponse,
    ProviderError,
)

# Registry
from .registry import (
    ProviderRegistry,
    ProviderConfig,
    ModelConfig,
    ProviderType,
    get_registry,
)

# Router
from .router import (
    ProviderRouter,
    get_router,
    get_best_provider,
)

# Individual providers
from .lightning import LightningProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .ollama import OllamaProvider
from .vllm import VLLMProvider

# Utilities
from .utils import (
    check_provider,
    check_all_providers,
    HealthStatus,
    UsageMetrics,
    get_usage_tracker,
)

# Factory functions
from .factory import (
    get_provider,
    get_router as get_factory_router,
    complete,
    chat,
    stream,
    check_providers,
    get_available_providers,
    list_providers,
    register_provider,
    print_provider_status,
)


__all__ = [
    # Version
    "__version__",
    
    # Core types
    "BaseProvider",
    "Message",
    "Role",
    "ProviderResponse",
    "ProviderError",
    
    # Registry
    "ProviderRegistry",
    "ProviderConfig",
    "ModelConfig",
    "ProviderType",
    "get_registry",
    
    # Router
    "ProviderRouter",
    "get_router",
    "get_best_provider",
    
    # Providers
    "LightningProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "VLLMProvider",
    
    # Factory
    "get_provider",
    "complete",
    "chat",
    "stream",
    "check_providers",
    "get_available_providers",
    "list_providers",
    "register_provider",
    "print_provider_status",
    
    # Utilities
    "check_provider",
    "check_all_providers",
    "HealthStatus",
    "UsageMetrics",
    "get_usage_tracker",
]
