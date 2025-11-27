"""
Model Registry and Provider Framework for BioPipelines.

DEPRECATED: This module is deprecated. Use workflow_composer.providers instead.

This module now re-exports from workflow_composer.providers for backwards
compatibility. New code should import directly from providers:

    # Old (deprecated):
    from workflow_composer.models import get_model_client, check_all_providers
    
    # New (recommended):
    from workflow_composer.providers import get_provider, check_providers

Provider Priority:
    1. Lightning.ai (free tier)
    2. Google Gemini (free tier)
    3. OpenAI (subscription)
    4. Anthropic (subscription)
    5. Local Ollama/vLLM (free)
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "workflow_composer.models is deprecated. Use workflow_composer.providers instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from providers for backwards compatibility
from ..providers import (
    # Core types
    BaseProvider,
    Message,
    Role,
    ProviderResponse as ModelResponse,
    ProviderError,
    
    # Registry
    ProviderRegistry as ModelRegistry,
    ProviderConfig,
    ModelConfig,
    get_registry,
    
    # Router
    ProviderRouter as ModelRouter,
    get_router,
    get_best_provider,
    
    # Providers
    LightningProvider,
    GeminiProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    VLLMProvider,
    
    # Factory functions
    get_provider as get_model_client,
    complete,
    chat,
    check_providers as check_all_providers,
    
    # Utilities
    check_provider,
    HealthStatus,
)

# Re-export metrics from local utils (these aren't in providers)
try:
    from .utils.metrics import UsageMetrics, get_usage_tracker
except ImportError:
    UsageMetrics = None
    get_usage_tracker = None

__all__ = [
    # Deprecated aliases (map to providers equivalents)
    "ModelRegistry",  # -> ProviderRegistry
    "ModelRouter",    # -> ProviderRouter
    "ModelConfig",
    "ProviderConfig",
    "get_registry",
    "get_model_client",  # -> get_provider
    "get_best_provider",
    "check_all_providers",  # -> check_providers
    
    # Core types
    "BaseProvider",
    "Message",
    "Role",
    "ModelResponse",  # -> ProviderResponse
    "ProviderError",
    
    # Providers
    "LightningProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "VLLMProvider",
    
    # Factory
    "complete",
    "chat",
    
    # Health
    "check_provider",
    "HealthStatus",
    
    # Metrics
    "UsageMetrics",
    "get_usage_tracker",
]

__version__ = "2.0.0"  # Bumped for deprecation
