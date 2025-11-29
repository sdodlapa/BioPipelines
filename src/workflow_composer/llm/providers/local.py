"""
Local Provider
==============

Unified interface to local LLM models (vLLM, Ollama).

This provider manages local GPU/CPU inference with automatic fallback:
- Primary: vLLM (high-performance GPU inference)
- Fallback: Ollama (lightweight, CPU-friendly)

Usage:
    from workflow_composer.llm.providers import LocalProvider
    
    provider = LocalProvider()
    response = await provider.complete("Explain RNA-seq")
    
    # Force specific backend
    response = await provider.complete("...", backend="ollama")
"""

from __future__ import annotations

import asyncio
import logging
import time
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
# Backend Wrappers
# =============================================================================

class VLLMBackend:
    """Wrapper for vLLM adapter."""
    
    def __init__(self):
        self._adapter = None
        self._available = None
    
    def _get_adapter(self):
        """Lazy load the adapter."""
        if self._adapter is None:
            try:
                from ..vllm_adapter import VLLMAdapter
                self._adapter = VLLMAdapter()
            except Exception as e:
                logger.debug(f"Failed to initialize vLLM adapter: {e}")
                self._adapter = False  # Mark as failed
        return self._adapter if self._adapter else None
    
    def is_available(self) -> bool:
        """Check if vLLM is available."""
        if self._available is not None:
            return self._available
        
        adapter = self._get_adapter()
        if adapter:
            try:
                self._available = adapter.is_available()
            except Exception:
                self._available = False
        else:
            self._available = False
        
        return self._available
    
    def reset_availability(self):
        """Reset availability cache for re-checking."""
        self._available = None
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ProviderResponse:
        """Complete using vLLM."""
        adapter = self._get_adapter()
        if not adapter:
            raise ProviderUnavailableError("vllm", "Adapter not initialized")
        
        start = time.time()
        
        # Use adapter's complete method
        if hasattr(adapter, 'complete_async'):
            response = await adapter.complete_async(prompt, **kwargs)
        else:
            # Fallback to sync in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: adapter.complete(prompt, **kwargs)
            )
        
        latency = (time.time() - start) * 1000
        
        return ProviderResponse(
            content=response.content if hasattr(response, 'content') else str(response),
            model=model or adapter.model,
            provider="vllm",
            provider_type=ProviderType.LOCAL,
            tokens_used=getattr(response, 'tokens_used', 0),
            latency_ms=latency,
            cost=0.0,  # Local is free
        )
    
    def list_models(self) -> List[ModelInfo]:
        """List available vLLM models."""
        adapter = self._get_adapter()
        if not adapter or not self.is_available():
            return []
        
        # vLLM typically serves one model at a time
        model_name = getattr(adapter, 'model', 'unknown')
        return [
            ModelInfo(
                id=model_name,
                name=model_name,
                provider="vllm",
                provider_type=ProviderType.LOCAL,
                capabilities=[ModelCapability.CODING, ModelCapability.GENERAL],
                context_length=32768,
                cost_per_1k_tokens=0.0,
                is_available=True,
            )
        ]


class OllamaBackend:
    """Wrapper for Ollama adapter."""
    
    def __init__(self):
        self._adapter = None
        self._available = None
    
    def _get_adapter(self):
        """Lazy load the adapter."""
        if self._adapter is None:
            try:
                from ..ollama_adapter import OllamaAdapter
                self._adapter = OllamaAdapter()
            except Exception as e:
                logger.debug(f"Failed to initialize Ollama adapter: {e}")
                self._adapter = False
        return self._adapter if self._adapter else None
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available
        
        adapter = self._get_adapter()
        if adapter:
            try:
                self._available = adapter.is_available()
            except Exception:
                self._available = False
        else:
            self._available = False
        
        return self._available
    
    def reset_availability(self):
        """Reset availability cache."""
        self._available = None
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ProviderResponse:
        """Complete using Ollama."""
        adapter = self._get_adapter()
        if not adapter:
            raise ProviderUnavailableError("ollama", "Adapter not initialized")
        
        start = time.time()
        
        # Use adapter's complete method
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: adapter.complete(prompt, **kwargs)
        )
        
        latency = (time.time() - start) * 1000
        
        return ProviderResponse(
            content=response.content if hasattr(response, 'content') else str(response),
            model=model or adapter.model,
            provider="ollama",
            provider_type=ProviderType.LOCAL,
            tokens_used=getattr(response, 'tokens_used', 0),
            latency_ms=latency,
            cost=0.0,
        )
    
    def list_models(self) -> List[ModelInfo]:
        """List available Ollama models."""
        if not self.is_available():
            return []
        
        # Common Ollama models
        return [
            ModelInfo(
                id="llama3:8b",
                name="Llama 3 8B",
                provider="ollama",
                provider_type=ProviderType.LOCAL,
                capabilities=[ModelCapability.GENERAL],
                context_length=8192,
                cost_per_1k_tokens=0.0,
            ),
            ModelInfo(
                id="codellama:13b",
                name="Code Llama 13B",
                provider="ollama",
                provider_type=ProviderType.LOCAL,
                capabilities=[ModelCapability.CODING],
                context_length=16384,
                cost_per_1k_tokens=0.0,
            ),
        ]


# =============================================================================
# Local Provider
# =============================================================================

class LocalProvider(BaseProvider):
    """
    Unified interface to local LLM models.
    
    Manages multiple local backends with automatic fallback:
    - vLLM: Primary, high-performance GPU inference
    - Ollama: Fallback, lightweight CPU inference
    
    Example:
        provider = LocalProvider()
        
        # Auto-select best available backend
        response = await provider.complete("Generate a workflow")
        
        # Force specific backend
        response = await provider.complete("...", backend="ollama")
    """
    
    def __init__(
        self,
        prefer_vllm: bool = True,
        enable_fallback: bool = True,
    ):
        """
        Initialize local provider.
        
        Args:
            prefer_vllm: Prefer vLLM over Ollama when both available
            enable_fallback: Enable fallback to secondary backend
        """
        super().__init__(name="local", provider_type=ProviderType.LOCAL)
        
        self.vllm = VLLMBackend()
        self.ollama = OllamaBackend()
        self.prefer_vllm = prefer_vllm
        self.enable_fallback = enable_fallback
        
        # Cache for quick availability check
        self._backends_checked = False
    
    def _get_primary_backend(self) -> tuple:
        """Get primary and fallback backends based on preference."""
        if self.prefer_vllm:
            return (self.vllm, "vllm"), (self.ollama, "ollama")
        return (self.ollama, "ollama"), (self.vllm, "vllm")
    
    def is_available(self) -> bool:
        """Check if any local backend is available."""
        return self.vllm.is_available() or self.ollama.is_available()
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        backends = []
        if self.vllm.is_available():
            backends.append("vllm")
        if self.ollama.is_available():
            backends.append("ollama")
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
        Complete using local models.
        
        Args:
            prompt: The prompt to complete
            model: Specific model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            backend: Force specific backend ("vllm" or "ollama")
            **kwargs: Additional arguments
            
        Returns:
            ProviderResponse with completion
            
        Raises:
            ProviderUnavailableError: If no backend is available
        """
        # If specific backend requested
        if backend:
            if backend == "vllm":
                if not self.vllm.is_available():
                    raise ProviderUnavailableError("vllm", "vLLM not available")
                return await self.vllm.complete(prompt, model, temperature, max_tokens, **kwargs)
            elif backend == "ollama":
                if not self.ollama.is_available():
                    raise ProviderUnavailableError("ollama", "Ollama not available")
                return await self.ollama.complete(prompt, model, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Unknown backend: {backend}")
        
        # Auto-select with fallback
        (primary, primary_name), (fallback, fallback_name) = self._get_primary_backend()
        
        # Try primary
        if primary.is_available():
            try:
                logger.debug(f"Using primary backend: {primary_name}")
                return await primary.complete(prompt, model, temperature, max_tokens, **kwargs)
            except Exception as e:
                logger.warning(f"Primary backend {primary_name} failed: {e}")
                if not self.enable_fallback:
                    raise
        
        # Try fallback
        if self.enable_fallback and fallback.is_available():
            try:
                logger.debug(f"Using fallback backend: {fallback_name}")
                return await fallback.complete(prompt, model, temperature, max_tokens, **kwargs)
            except Exception as e:
                logger.error(f"Fallback backend {fallback_name} also failed: {e}")
                raise ProviderUnavailableError("local", f"All backends failed: {e}")
        
        raise ProviderUnavailableError("local", "No local backend available")
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion from local backend."""
        # For now, use non-streaming fallback
        # TODO: Implement true streaming for vLLM
        response = await self.complete(prompt, model, **kwargs)
        yield response.content
    
    def list_models(self) -> List[ModelInfo]:
        """List all available local models."""
        models = []
        
        if self.vllm.is_available():
            models.extend(self.vllm.list_models())
        
        if self.ollama.is_available():
            models.extend(self.ollama.list_models())
        
        return models
    
    async def health_check(self) -> ProviderHealth:
        """Check health of local backends."""
        vllm_ok = self.vllm.is_available()
        ollama_ok = self.ollama.is_available()
        
        models = self.list_models()
        
        return ProviderHealth(
            provider="local",
            is_healthy=vllm_ok or ollama_ok,
            available_models=len(models),
            metadata={
                "vllm_available": vllm_ok,
                "ollama_available": ollama_ok,
            }
        )
    
    def reset_cache(self):
        """Reset availability caches for re-checking."""
        self.vllm.reset_availability()
        self.ollama.reset_availability()
        self._backends_checked = False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LocalProvider",
    "VLLMBackend",
    "OllamaBackend",
]
