"""
Model Registry and Provider Framework for BioPipelines.

This module provides a unified interface for accessing various LLM providers,
with automatic fallback and smart routing capabilities.

Provider Priority:
    1. Lightning.ai (free tier)
    2. Google Gemini (free tier)
    3. GitHub Copilot (subscription)
    4. OpenAI (subscription)
    5+ Local vLLM models (backup)
"""

from .registry import (
    ModelRegistry,
    ModelConfig,
    ProviderConfig,
    get_registry,
)
from .router import (
    ModelRouter,
    get_model_client,
    get_best_provider,
)
from .providers import (
    BaseProvider,
    LightningProvider,
    GeminiProvider,
    OpenAIProvider,
    VLLMProvider,
)
from .utils.health import (
    check_provider,
    check_all_providers,
    HealthStatus,
)
from .utils.metrics import (
    UsageMetrics,
    get_usage_tracker,
)

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelConfig",
    "ProviderConfig",
    "get_registry",
    # Router
    "ModelRouter",
    "get_model_client",
    "get_best_provider",
    # Providers
    "BaseProvider",
    "LightningProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "VLLMProvider",
    # Health
    "check_provider",
    "check_all_providers",
    "HealthStatus",
    # Metrics
    "UsageMetrics",
    "get_usage_tracker",
]

__version__ = "1.0.0"
