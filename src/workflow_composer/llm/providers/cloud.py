"""
Cloud Provider
==============

Unified interface to cloud LLM services.

Primary: Lightning.ai (unified API to multiple cloud models)
Available: OpenAI, Anthropic (via Lightning.ai or direct)

Usage:
    from workflow_composer.llm.providers import CloudProvider
    
    provider = CloudProvider()
    response = await provider.complete("Explain methylation")
    
    # Use specific model
    response = await provider.complete("...", model="gpt-4")
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Any

from .base import (
    BaseProvider,
    ModelCapability,
    ModelInfo,
    ProviderHealth,
    ProviderResponse,
    ProviderType,
    ProviderUnavailableError,
    ModelNotFoundError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Cloud Model Registry
# =============================================================================

@dataclass
class CloudModel:
    """Cloud model metadata."""
    id: str
    name: str
    provider: str  # openai, anthropic, etc.
    capabilities: List[ModelCapability]
    context_length: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    
    @property
    def avg_cost_per_1k(self) -> float:
        """Average cost per 1k tokens."""
        return (self.cost_per_1k_input + self.cost_per_1k_output) / 2


# Model registry with pricing (as of 2024)
CLOUD_MODELS = {
    # OpenAI models
    "gpt-4o": CloudModel(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        capabilities=[ModelCapability.CODING, ModelCapability.GENERAL, ModelCapability.CHAT],
        context_length=128000,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
    ),
    "gpt-4o-mini": CloudModel(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        capabilities=[ModelCapability.CODING, ModelCapability.GENERAL, ModelCapability.CHAT],
        context_length=128000,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    "gpt-4-turbo": CloudModel(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider="openai",
        capabilities=[ModelCapability.CODING, ModelCapability.GENERAL],
        context_length=128000,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
    ),
    
    # Anthropic models
    "claude-3-5-sonnet": CloudModel(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        capabilities=[ModelCapability.CODING, ModelCapability.GENERAL, ModelCapability.CHAT],
        context_length=200000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "claude-3-opus": CloudModel(
        id="claude-3-opus-20240229",
        name="Claude 3 Opus",
        provider="anthropic",
        capabilities=[ModelCapability.CODING, ModelCapability.GENERAL],
        context_length=200000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
    "claude-3-haiku": CloudModel(
        id="claude-3-haiku-20240307",
        name="Claude 3 Haiku",
        provider="anthropic",
        capabilities=[ModelCapability.GENERAL, ModelCapability.CHAT],
        context_length=200000,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
    ),
    
    # Google models (via Lightning.ai)
    "gemini-1.5-pro": CloudModel(
        id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        provider="google",
        capabilities=[ModelCapability.CODING, ModelCapability.GENERAL],
        context_length=1000000,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
    ),
}

DEFAULT_MODEL = "gpt-4o-mini"


# =============================================================================
# Lightning.ai Backend
# =============================================================================

class LightningBackend:
    """
    Lightning.ai unified API backend.
    
    Provides access to multiple cloud models through a single interface.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("LIGHTNING_API_KEY")
        self._client = None
        self._available = None
    
    def _get_client(self):
        """Lazy load Lightning client."""
        if self._client is None and self._api_key:
            try:
                # Try to import Lightning AI client
                # This is a placeholder - actual import depends on Lightning.ai SDK
                from ..lightning_adapter import LightningAdapter
                self._client = LightningAdapter(api_key=self._api_key)
            except ImportError:
                logger.debug("Lightning adapter not available, using fallback")
                self._client = False
            except Exception as e:
                logger.debug(f"Failed to initialize Lightning client: {e}")
                self._client = False
        return self._client if self._client else None
    
    def is_available(self) -> bool:
        """Check if Lightning.ai is available."""
        if self._available is not None:
            return self._available
        
        if not self._api_key:
            self._available = False
            return False
        
        client = self._get_client()
        if client:
            try:
                self._available = client.is_available()
            except Exception:
                self._available = False
        else:
            self._available = False
        
        return self._available
    
    async def complete(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ProviderResponse:
        """Complete using Lightning.ai."""
        client = self._get_client()
        if not client:
            raise ProviderUnavailableError("lightning", "Client not initialized")
        
        # Get model metadata for cost tracking
        model_meta = CLOUD_MODELS.get(model)
        
        start = time.time()
        
        # Run in thread pool if sync
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.complete(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        )
        
        latency = (time.time() - start) * 1000
        tokens = getattr(response, 'tokens_used', 0)
        
        # Calculate cost
        cost = 0.0
        if model_meta and tokens:
            cost = (tokens / 1000) * model_meta.avg_cost_per_1k
        
        return ProviderResponse(
            content=response.content if hasattr(response, 'content') else str(response),
            model=model,
            provider="lightning",
            provider_type=ProviderType.CLOUD,
            tokens_used=tokens,
            latency_ms=latency,
            cost=cost,
        )


# =============================================================================
# OpenAI Direct Backend
# =============================================================================

class OpenAIBackend:
    """Direct OpenAI API backend."""
    
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None and self._api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                logger.debug("openai package not installed")
                self._client = False
            except Exception as e:
                logger.debug(f"Failed to initialize OpenAI: {e}")
                self._client = False
        return self._client if self._client else None
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self._api_key is not None and self._get_client() is not None
    
    async def complete(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ProviderResponse:
        """Complete using OpenAI directly."""
        client = self._get_client()
        if not client:
            raise ProviderUnavailableError("openai", "Client not available")
        
        model_meta = CLOUD_MODELS.get(model)
        start = time.time()
        
        # Run sync API in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        
        latency = (time.time() - start) * 1000
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        
        cost = 0.0
        if model_meta:
            input_cost = (response.usage.prompt_tokens / 1000) * model_meta.cost_per_1k_input
            output_cost = (response.usage.completion_tokens / 1000) * model_meta.cost_per_1k_output
            cost = input_cost + output_cost
        
        return ProviderResponse(
            content=content,
            model=model,
            provider="openai",
            provider_type=ProviderType.CLOUD,
            tokens_used=tokens,
            latency_ms=latency,
            cost=cost,
        )


# =============================================================================
# Anthropic Direct Backend
# =============================================================================

class AnthropicBackend:
    """Direct Anthropic API backend."""
    
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
    
    def _get_client(self):
        """Lazy load Anthropic client."""
        if self._client is None and self._api_key:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key)
            except ImportError:
                logger.debug("anthropic package not installed")
                self._client = False
            except Exception as e:
                logger.debug(f"Failed to initialize Anthropic: {e}")
                self._client = False
        return self._client if self._client else None
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return self._api_key is not None and self._get_client() is not None
    
    async def complete(
        self,
        prompt: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ProviderResponse:
        """Complete using Anthropic directly."""
        client = self._get_client()
        if not client:
            raise ProviderUnavailableError("anthropic", "Client not available")
        
        # Get actual model ID if short name used
        model_meta = CLOUD_MODELS.get(model.replace("-20241022", "").replace("-20240229", "").replace("-20240307", ""))
        actual_model = model_meta.id if model_meta else model
        
        start = time.time()
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=actual_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        
        latency = (time.time() - start) * 1000
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        cost = 0.0
        if model_meta:
            input_cost = (response.usage.input_tokens / 1000) * model_meta.cost_per_1k_input
            output_cost = (response.usage.output_tokens / 1000) * model_meta.cost_per_1k_output
            cost = input_cost + output_cost
        
        return ProviderResponse(
            content=content,
            model=actual_model,
            provider="anthropic",
            provider_type=ProviderType.CLOUD,
            tokens_used=tokens,
            latency_ms=latency,
            cost=cost,
        )


# =============================================================================
# Cloud Provider
# =============================================================================

class CloudProvider(BaseProvider):
    """
    Unified interface to cloud LLM services.
    
    Manages multiple cloud backends with smart routing:
    - Lightning.ai: Primary unified interface
    - OpenAI: Direct API access
    - Anthropic: Direct API access
    
    Example:
        provider = CloudProvider()
        
        # Auto-select based on model
        response = await provider.complete("...", model="gpt-4o")
        
        # Force specific backend
        response = await provider.complete("...", backend="anthropic")
    """
    
    def __init__(
        self,
        lightning_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        prefer_lightning: bool = True,
    ):
        """
        Initialize cloud provider.
        
        Args:
            lightning_api_key: Lightning.ai API key
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            prefer_lightning: Use Lightning.ai as primary when possible
        """
        super().__init__(name="cloud", provider_type=ProviderType.CLOUD)
        
        self.lightning = LightningBackend(lightning_api_key)
        self.openai = OpenAIBackend(openai_api_key)
        self.anthropic = AnthropicBackend(anthropic_api_key)
        self.prefer_lightning = prefer_lightning
        
        # Model to backend mapping
        self._model_backends = {
            "gpt-": "openai",
            "claude-": "anthropic",
            "gemini-": "lightning",  # Google models via Lightning
        }
    
    def _get_backend_for_model(self, model: str) -> tuple:
        """Get appropriate backend for a model."""
        for prefix, backend_name in self._model_backends.items():
            if model.startswith(prefix):
                if backend_name == "openai":
                    return self.openai, "openai"
                elif backend_name == "anthropic":
                    return self.anthropic, "anthropic"
                elif backend_name == "lightning":
                    return self.lightning, "lightning"
        
        # Default to lightning if available, else first available
        if self.prefer_lightning and self.lightning.is_available():
            return self.lightning, "lightning"
        if self.openai.is_available():
            return self.openai, "openai"
        if self.anthropic.is_available():
            return self.anthropic, "anthropic"
        if self.lightning.is_available():
            return self.lightning, "lightning"
        
        return None, None
    
    def is_available(self) -> bool:
        """Check if any cloud backend is available."""
        return (
            self.lightning.is_available() or
            self.openai.is_available() or
            self.anthropic.is_available()
        )
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        backends = []
        if self.lightning.is_available():
            backends.append("lightning")
        if self.openai.is_available():
            backends.append("openai")
        if self.anthropic.is_available():
            backends.append("anthropic")
        return backends
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        backend: Optional[str] = None,
        **kwargs,
    ) -> ProviderResponse:
        """
        Complete using cloud models.
        
        Args:
            prompt: The prompt to complete
            model: Model to use (e.g., "gpt-4o", "claude-3-5-sonnet")
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            backend: Force specific backend
            **kwargs: Additional arguments
            
        Returns:
            ProviderResponse with completion
        """
        model = model or DEFAULT_MODEL
        
        # Force specific backend
        if backend:
            if backend == "lightning":
                if not self.lightning.is_available():
                    raise ProviderUnavailableError("lightning", "Lightning.ai not available")
                return await self.lightning.complete(prompt, model, temperature, max_tokens, **kwargs)
            elif backend == "openai":
                if not self.openai.is_available():
                    raise ProviderUnavailableError("openai", "OpenAI not available")
                return await self.openai.complete(prompt, model, temperature, max_tokens, **kwargs)
            elif backend == "anthropic":
                if not self.anthropic.is_available():
                    raise ProviderUnavailableError("anthropic", "Anthropic not available")
                return await self.anthropic.complete(prompt, model, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Unknown backend: {backend}")
        
        # Auto-select backend
        selected_backend, backend_name = self._get_backend_for_model(model)
        if selected_backend is None:
            raise ProviderUnavailableError("cloud", "No cloud backend available")
        
        logger.debug(f"Using cloud backend {backend_name} for model {model}")
        return await selected_backend.complete(prompt, model, temperature, max_tokens, **kwargs)
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion from cloud backend."""
        # TODO: Implement true streaming
        response = await self.complete(prompt, model, **kwargs)
        yield response.content
    
    def list_models(self) -> List[ModelInfo]:
        """List all available cloud models."""
        models = []
        available_backends = set(self.get_available_backends())
        
        for model_id, meta in CLOUD_MODELS.items():
            # Check if backend for this model is available
            backend_available = (
                meta.provider in available_backends or
                "lightning" in available_backends  # Lightning can access all
            )
            
            if backend_available:
                models.append(ModelInfo(
                    id=meta.id,
                    name=meta.name,
                    provider=meta.provider,
                    provider_type=ProviderType.CLOUD,
                    capabilities=meta.capabilities,
                    context_length=meta.context_length,
                    cost_per_1k_tokens=meta.avg_cost_per_1k,
                    is_available=True,
                ))
        
        return models
    
    async def health_check(self) -> ProviderHealth:
        """Check health of cloud backends."""
        backends = self.get_available_backends()
        
        return ProviderHealth(
            provider="cloud",
            is_healthy=len(backends) > 0,
            available_models=len(self.list_models()),
            metadata={
                "lightning_available": self.lightning.is_available(),
                "openai_available": self.openai.is_available(),
                "anthropic_available": self.anthropic.is_available(),
            }
        )
    
    def get_model_cost(self, model: str) -> Optional[CloudModel]:
        """Get cost information for a model."""
        return CLOUD_MODELS.get(model)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CloudProvider",
    "LightningBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "CloudModel",
    "CLOUD_MODELS",
    "DEFAULT_MODEL",
]
