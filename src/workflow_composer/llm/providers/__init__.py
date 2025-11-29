"""
LLM Providers Package
=====================

Unified interfaces to local and cloud LLM providers.

This package provides:
- LocalProvider: Access to vLLM and Ollama
- CloudProvider: Access to Lightning.ai, OpenAI, Anthropic

Usage:
    from workflow_composer.llm.providers import LocalProvider, CloudProvider
    
    # Local GPU inference
    local = LocalProvider()
    response = await local.complete("Generate workflow for RNA-seq")
    
    # Cloud inference with cost tracking
    cloud = CloudProvider()
    response = await cloud.complete("Complex analysis", model="gpt-4o")
    print(f"Cost: ${response.cost:.4f}")
"""

from .base import (
    # Protocols & Types
    ProviderProtocol,
    ProviderType,
    ModelCapability,
    
    # Data Classes
    ModelInfo,
    ProviderResponse,
    ProviderHealth,
    
    # Exceptions
    ProviderError,
    ProviderUnavailableError,
    ModelNotFoundError,
    
    # Base Class
    BaseProvider,
)

from .local import (
    LocalProvider,
    VLLMBackend,
    OllamaBackend,
)

from .cloud import (
    CloudProvider,
    LightningBackend,
    OpenAIBackend,
    AnthropicBackend,
    CloudModel,
    CLOUD_MODELS,
    DEFAULT_MODEL,
)


__all__ = [
    # Types
    "ProviderProtocol",
    "ProviderType",
    "ModelCapability",
    
    # Data Classes
    "ModelInfo",
    "ProviderResponse",
    "ProviderHealth",
    
    # Exceptions
    "ProviderError",
    "ProviderUnavailableError",
    "ModelNotFoundError",
    
    # Base
    "BaseProvider",
    
    # Providers
    "LocalProvider",
    "CloudProvider",
    
    # Backends
    "VLLMBackend",
    "OllamaBackend",
    "LightningBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    
    # Cloud Model Registry
    "CloudModel",
    "CLOUD_MODELS",
    "DEFAULT_MODEL",
]
