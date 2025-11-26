"""
Provider Router
===============

Smart routing with automatic fallback between providers.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from .base import BaseProvider, Message, ProviderResponse, ProviderError
from .registry import get_registry, ProviderRegistry, ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Result of a routed request."""
    response: ProviderResponse
    provider_used: str
    fallback_used: bool = False
    attempts: int = 1
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ProviderRouter:
    """
    Routes requests to the best available provider with fallback.
    
    Features:
        - Automatic provider selection based on priority
        - Fallback to next provider on failure
        - Failure tracking to deprioritize flaky providers
        - Configurable retry behavior
    
    Example:
        router = ProviderRouter()
        
        # Simple completion with auto-fallback
        response = router.complete("Explain RNA-seq")
        
        # Force specific provider
        response = router.complete("...", preferred_provider="gemini")
        
        # Disable fallback
        response = router.complete("...", fallback=False)
    """
    
    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        skip_providers: Optional[List[str]] = None,
        max_retries: int = 2,
    ):
        """
        Initialize the router.
        
        Args:
            registry: Provider registry (uses global if not provided)
            skip_providers: Provider IDs to skip
            max_retries: Max retries per provider before moving on
        """
        self.registry = registry or get_registry()
        self.skip_providers = set(skip_providers or [])
        self.max_retries = max_retries
        
        # Track failures to deprioritize flaky providers
        self._failure_counts: Dict[str, int] = {}
        self._provider_cache: Dict[str, BaseProvider] = {}
    
    def _get_ordered_providers(self) -> List[ProviderConfig]:
        """Get providers in priority order, accounting for failures."""
        providers = self.registry.list_providers(configured_only=True)
        
        # Filter out skipped providers
        providers = [p for p in providers if p.id not in self.skip_providers]
        
        # Sort by (failure_count, priority)
        def sort_key(p: ProviderConfig) -> tuple:
            failures = self._failure_counts.get(p.id, 0)
            return (failures > 3, p.priority)  # Heavily deprioritize >3 failures
        
        return sorted(providers, key=sort_key)
    
    def _get_provider_instance(self, config: ProviderConfig) -> BaseProvider:
        """Get or create provider instance."""
        if config.id in self._provider_cache:
            return self._provider_cache[config.id]
        
        # Import and instantiate
        provider = self._create_provider(config)
        self._provider_cache[config.id] = provider
        return provider
    
    def _create_provider(self, config: ProviderConfig) -> BaseProvider:
        """Create a provider instance."""
        # Lazy imports to avoid circular dependencies
        if config.id == "lightning":
            from .lightning import LightningProvider
            return LightningProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "gemini":
            from .gemini import GeminiProvider
            return GeminiProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "openai":
            from .openai import OpenAIProvider
            return OpenAIProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "anthropic":
            from .anthropic import AnthropicProvider
            return AnthropicProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "ollama":
            from .ollama import OllamaProvider
            return OllamaProvider(
                model=config.default_model,
                base_url=config.base_url,
            )
        elif config.id == "vllm":
            from .vllm import VLLMProvider
            return VLLMProvider(
                model=config.default_model,
                base_url=config.base_url,
            )
        else:
            raise ValueError(f"Unknown provider: {config.id}")
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        fallback: bool = True,
        **kwargs
    ) -> ProviderResponse:
        """
        Generate a completion using the best available provider.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            preferred_provider: Prefer this provider if available
            preferred_model: Prefer this model
            fallback: Whether to try other providers on failure
            **kwargs: Additional parameters (temperature, max_tokens)
            
        Returns:
            ProviderResponse from the successful provider
            
        Raises:
            ProviderError: If all providers fail
        """
        providers = self._get_ordered_providers()
        
        # Move preferred provider to front if specified
        if preferred_provider:
            providers = sorted(
                providers,
                key=lambda p: 0 if p.id == preferred_provider else 1
            )
        
        errors = []
        attempts = 0
        
        for config in providers:
            for retry in range(self.max_retries):
                attempts += 1
                
                try:
                    provider = self._get_provider_instance(config)
                    
                    # Override model if preferred
                    if preferred_model:
                        provider.model = preferred_model
                    
                    start = time.time()
                    response = provider.complete(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        **kwargs
                    )
                    response.latency_ms = (time.time() - start) * 1000
                    
                    # Clear failure count on success
                    self._failure_counts[config.id] = 0
                    
                    logger.info(
                        f"Completion successful with {config.id} "
                        f"({response.latency_ms:.0f}ms)"
                    )
                    
                    return response
                    
                except Exception as e:
                    error_msg = f"{config.id}: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(f"Provider failed: {error_msg}")
                    
                    # Track failure
                    self._failure_counts[config.id] = \
                        self._failure_counts.get(config.id, 0) + 1
                    
                    # Non-retriable errors
                    if isinstance(e, ProviderError) and not e.retriable:
                        break
            
            if not fallback:
                break
        
        # All failed
        raise ProviderError(
            provider="router",
            message=f"All providers failed after {attempts} attempts. "
                    f"Errors: {'; '.join(errors[-3:])}",  # Last 3 errors
            retriable=False,
        )
    
    def chat(
        self,
        messages: List[Union[Message, Dict]],
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        fallback: bool = True,
        **kwargs
    ) -> ProviderResponse:
        """
        Generate a chat response using the best available provider.
        
        Args:
            messages: List of Message objects or dicts
            preferred_provider: Prefer this provider if available
            preferred_model: Prefer this model
            fallback: Whether to try other providers on failure
            **kwargs: Additional parameters
            
        Returns:
            ProviderResponse from the successful provider
        """
        providers = self._get_ordered_providers()
        
        if preferred_provider:
            providers = sorted(
                providers,
                key=lambda p: 0 if p.id == preferred_provider else 1
            )
        
        # Normalize messages
        normalized = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(Message.from_dict(msg))
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
        
        errors = []
        attempts = 0
        
        for config in providers:
            for retry in range(self.max_retries):
                attempts += 1
                
                try:
                    provider = self._get_provider_instance(config)
                    
                    if preferred_model:
                        provider.model = preferred_model
                    
                    start = time.time()
                    response = provider.chat(normalized, **kwargs)
                    response.latency_ms = (time.time() - start) * 1000
                    
                    self._failure_counts[config.id] = 0
                    
                    return response
                    
                except Exception as e:
                    errors.append(f"{config.id}: {e}")
                    self._failure_counts[config.id] = \
                        self._failure_counts.get(config.id, 0) + 1
                    
                    if isinstance(e, ProviderError) and not e.retriable:
                        break
            
            if not fallback:
                break
        
        raise ProviderError(
            provider="router",
            message=f"All providers failed. Errors: {'; '.join(errors[-3:])}",
            retriable=False,
        )
    
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Async version of complete()."""
        import asyncio
        return await asyncio.to_thread(
            self.complete, prompt, system_prompt, **kwargs
        )
    
    async def chat_async(
        self,
        messages: List[Union[Message, Dict]],
        **kwargs
    ) -> ProviderResponse:
        """Async version of chat()."""
        import asyncio
        return await asyncio.to_thread(self.chat, messages, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current router status."""
        providers = self._get_ordered_providers()
        
        status = []
        for config in providers:
            try:
                provider = self._get_provider_instance(config)
                available = provider.is_available()
            except Exception:
                available = False
            
            status.append({
                "id": config.id,
                "name": config.name,
                "priority": config.priority,
                "available": available,
                "failures": self._failure_counts.get(config.id, 0),
                "free_tier": config.free_tier,
            })
        
        return {
            "providers": status,
            "active_providers": sum(1 for s in status if s["available"]),
        }


# Global router instance
_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    """Get the global provider router."""
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router


def get_best_provider() -> Optional[str]:
    """Get the ID of the best available provider."""
    router = get_router()
    providers = router._get_ordered_providers()
    
    for config in providers:
        try:
            provider = router._get_provider_instance(config)
            if provider.is_available():
                return config.id
        except Exception:
            continue
    
    return None
