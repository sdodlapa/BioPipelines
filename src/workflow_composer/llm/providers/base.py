"""
Provider Base Protocol
======================

Defines the protocol for unified LLM providers (local and cloud).

This module provides:
- ProviderProtocol: Interface for all providers
- ModelInfo: Model metadata
- ProviderResponse: Unified response type
- ProviderError: Base exception

Usage:
    class MyProvider(ProviderProtocol):
        async def complete(self, prompt: str, **kwargs) -> LLMResponse:
            ...
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ProviderType(Enum):
    """Type of provider."""
    LOCAL = "local"
    CLOUD = "cloud"


class ModelCapability(Enum):
    """Model capabilities for routing."""
    CODING = "coding"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    GENERAL = "general"
    FAST = "fast"
    MULTIMODAL = "multimodal"
    CHAT = "chat"  # Conversational/chat-optimized


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: str
    provider: str
    provider_type: ProviderType
    capabilities: List[ModelCapability] = field(default_factory=list)
    context_length: int = 4096
    cost_per_1k_tokens: float = 0.0  # 0 for local
    is_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_local(self) -> bool:
        return self.provider_type == ProviderType.LOCAL
    
    @property
    def is_cloud(self) -> bool:
        return self.provider_type == ProviderType.CLOUD
    
    @property
    def is_free(self) -> bool:
        return self.cost_per_1k_tokens == 0.0


@dataclass
class ProviderResponse:
    """Unified response from a provider."""
    content: str
    model: str
    provider: str
    provider_type: ProviderType
    tokens_used: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def text(self) -> str:
        """Alias for content."""
        return self.content


@dataclass
class ProviderHealth:
    """Health status of a provider."""
    provider: str
    is_healthy: bool
    latency_ms: float = 0.0
    available_models: int = 0
    error: Optional[str] = None
    last_checked: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Exceptions
# =============================================================================

class ProviderError(Exception):
    """Base exception for provider errors."""
    
    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.recoverable = recoverable


class ProviderUnavailableError(ProviderError):
    """Provider is not available."""
    
    def __init__(self, provider: str, reason: str = ""):
        message = f"Provider '{provider}' is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(message, provider=provider, recoverable=True)


class ModelNotFoundError(ProviderError):
    """Requested model not found."""
    
    def __init__(self, model: str, provider: str):
        super().__init__(
            f"Model '{model}' not found in provider '{provider}'",
            provider=provider,
            recoverable=True,
        )


class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    
    def __init__(self, provider: str, retry_after: Optional[float] = None):
        message = f"Rate limit exceeded for provider '{provider}'"
        if retry_after:
            message += f", retry after {retry_after}s"
        super().__init__(message, provider=provider, recoverable=True)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Authentication failed."""
    
    def __init__(self, provider: str):
        super().__init__(
            f"Authentication failed for provider '{provider}'",
            provider=provider,
            recoverable=False,
        )


# =============================================================================
# Protocol Definition
# =============================================================================

@runtime_checkable
class ProviderProtocol(Protocol):
    """
    Protocol for unified LLM providers.
    
    All providers (local and cloud) must implement this interface.
    This enables seamless switching and orchestration between providers.
    """
    
    @property
    def name(self) -> str:
        """Provider name (e.g., 'local', 'cloud', 'vllm', 'lightning')."""
        ...
    
    @property
    def provider_type(self) -> ProviderType:
        """Whether this is a local or cloud provider."""
        ...
    
    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        ...
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ProviderResponse:
        """
        Generate a completion.
        
        Args:
            prompt: The prompt to complete
            model: Specific model to use (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific arguments
            
        Returns:
            ProviderResponse with the completion
            
        Raises:
            ProviderUnavailableError: If provider is not available
            ModelNotFoundError: If model is not found
            ProviderError: For other errors
        """
        ...
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream a completion.
        
        Args:
            prompt: The prompt to complete
            model: Specific model to use (optional)
            **kwargs: Additional arguments
            
        Yields:
            Tokens as they are generated
        """
        ...
    
    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        ...
    
    async def health_check(self) -> ProviderHealth:
        """Check provider health."""
        ...


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaseProvider(ABC):
    """
    Abstract base class for providers.
    
    Provides common functionality and enforces the protocol.
    """
    
    def __init__(self, name: str, provider_type: ProviderType):
        self._name = name
        self._provider_type = provider_type
        self._models_cache: Optional[List[ModelInfo]] = None
        self._last_health_check: Optional[ProviderHealth] = None
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def provider_type(self) -> ProviderType:
        return self._provider_type
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ProviderResponse:
        """Generate completion."""
        pass
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Default streaming implementation (non-streaming fallback).
        
        Override for true streaming support.
        """
        response = await self.complete(prompt, model, **kwargs)
        yield response.content
    
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        pass
    
    async def health_check(self) -> ProviderHealth:
        """Check provider health."""
        import time
        
        start = time.time()
        try:
            is_available = self.is_available()
            models = self.list_models() if is_available else []
            latency = (time.time() - start) * 1000
            
            health = ProviderHealth(
                provider=self.name,
                is_healthy=is_available,
                latency_ms=latency,
                available_models=len(models),
            )
        except Exception as e:
            health = ProviderHealth(
                provider=self.name,
                is_healthy=False,
                error=str(e),
            )
        
        self._last_health_check = health
        return health
    
    def complete_sync(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ProviderResponse:
        """Synchronous wrapper for complete()."""
        return asyncio.run(self.complete(prompt, model, **kwargs))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.provider_type.value})"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ProviderType",
    "ModelCapability",
    # Data classes
    "ModelInfo",
    "ProviderResponse",
    "ProviderHealth",
    # Exceptions
    "ProviderError",
    "ProviderUnavailableError",
    "ModelNotFoundError",
    "RateLimitError",
    "AuthenticationError",
    # Protocol and base
    "ProviderProtocol",
    "BaseProvider",
]
