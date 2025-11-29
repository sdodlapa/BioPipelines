"""
Model Orchestrator
==================

Central orchestrator for LLM model selection and execution.

The ModelOrchestrator provides:
- Unified interface to local and cloud providers
- Strategy-based routing (local-first, ensemble, cascade)
- Automatic fallback and retry
- Cost tracking and optimization

Usage:
    from workflow_composer.llm import get_orchestrator, Strategy
    
    # Get orchestrator with strategy
    orch = get_orchestrator(strategy=Strategy.LOCAL_FIRST)
    
    # Simple completion
    response = await orch.complete("Generate a workflow")
    
    # With specific model
    response = await orch.complete("...", model="gpt-4o")
    
    # Ensemble for critical decisions
    response = await orch.ensemble("Is this workflow correct?")
    
    # Check cost
    print(f"Total cost: ${orch.total_cost:.4f}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from .providers import (
    LocalProvider,
    CloudProvider,
    ProviderResponse,
    ProviderType,
    ProviderUnavailableError,
)
from .strategies import (
    Strategy,
    StrategyConfig,
    EnsembleMode,
    get_preset,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Response Types
# =============================================================================

@dataclass
class OrchestratorResponse:
    """
    Response from the orchestrator.
    
    Extends ProviderResponse with orchestration metadata.
    """
    content: str
    model: str
    provider: str
    provider_type: ProviderType
    strategy_used: Strategy
    tokens_used: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    
    # Orchestration metadata
    attempts: int = 1
    fallback_used: bool = False
    cached: bool = False
    
    # Ensemble metadata (if applicable)
    ensemble_responses: Optional[List[ProviderResponse]] = None
    
    def __str__(self) -> str:
        return self.content


@dataclass
class UsageStats:
    """Aggregated usage statistics."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    local_requests: int = 0
    cloud_requests: int = 0
    fallbacks_used: int = 0
    cache_hits: int = 0
    errors: int = 0
    
    def add(self, response: OrchestratorResponse):
        """Update stats with a response."""
        self.total_requests += 1
        self.total_tokens += response.tokens_used
        self.total_cost += response.cost
        self.total_latency_ms += response.latency_ms
        
        if response.provider_type == ProviderType.LOCAL:
            self.local_requests += 1
        else:
            self.cloud_requests += 1
        
        if response.fallback_used:
            self.fallbacks_used += 1
        if response.cached:
            self.cache_hits += 1


# =============================================================================
# Model Orchestrator
# =============================================================================

class ModelOrchestrator:
    """
    Central orchestrator for LLM model selection and execution.
    
    Features:
    - Unified API across local and cloud providers
    - Strategy-based routing
    - Automatic fallback on failure
    - Cost tracking and optimization
    - Ensemble patterns for critical decisions
    
    Example:
        orch = ModelOrchestrator(strategy=Strategy.LOCAL_FIRST)
        
        # Basic completion
        response = await orch.complete("Generate RNA-seq workflow")
        print(response.content)
        print(f"Used: {response.provider} ({response.model})")
        print(f"Cost: ${response.cost:.4f}")
        
        # Force specific provider
        response = await orch.complete("...", provider_type=ProviderType.CLOUD)
        
        # Ensemble for important decisions
        response = await orch.ensemble("Validate this workflow...")
    """
    
    def __init__(
        self,
        strategy: Strategy = Strategy.AUTO,
        config: Optional[StrategyConfig] = None,
        local_provider: Optional[LocalProvider] = None,
        cloud_provider: Optional[CloudProvider] = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            strategy: Primary routing strategy
            config: Detailed configuration (overrides strategy if provided)
            local_provider: Custom local provider
            cloud_provider: Custom cloud provider
        """
        self.config = config or StrategyConfig(strategy=strategy)
        self.strategy = self.config.strategy
        
        # Initialize providers
        self.local = local_provider or LocalProvider()
        self.cloud = cloud_provider or CloudProvider()
        
        # Usage tracking
        self.stats = UsageStats()
        
        # Response cache (simple in-memory)
        self._cache: Dict[str, OrchestratorResponse] = {}
        self._cache_max_size = 100
        
        logger.info(f"ModelOrchestrator initialized with strategy: {self.strategy.value}")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def total_cost(self) -> float:
        """Get total cost of all requests."""
        return self.stats.total_cost
    
    @property
    def is_local_available(self) -> bool:
        """Check if local provider is available."""
        return self.local.is_available()
    
    @property
    def is_cloud_available(self) -> bool:
        """Check if cloud provider is available."""
        return self.cloud.is_available()
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        strategy: Optional[Strategy] = None,
        provider_type: Optional[ProviderType] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> OrchestratorResponse:
        """
        Complete a prompt using the configured strategy.
        
        Args:
            prompt: The prompt to complete
            model: Specific model to use (optional)
            strategy: Override default strategy for this request
            provider_type: Force specific provider type
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional provider arguments
            
        Returns:
            OrchestratorResponse with completion and metadata
            
        Raises:
            ProviderUnavailableError: If no provider available for strategy
        """
        strategy = strategy or self.strategy
        
        # Check cache
        if self.config.cache_responses:
            cache_key = self._make_cache_key(prompt, model, strategy)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cached.cached = True
                self.stats.cache_hits += 1
                return cached
        
        # Route based on strategy
        if provider_type:
            response = await self._complete_with_provider(
                provider_type, prompt, model, temperature, max_tokens, **kwargs
            )
        elif strategy == Strategy.LOCAL_ONLY:
            response = await self._complete_local_only(prompt, model, temperature, max_tokens, **kwargs)
        elif strategy == Strategy.CLOUD_ONLY:
            response = await self._complete_cloud_only(prompt, model, temperature, max_tokens, **kwargs)
        elif strategy == Strategy.LOCAL_FIRST:
            response = await self._complete_local_first(prompt, model, temperature, max_tokens, **kwargs)
        elif strategy == Strategy.CASCADE:
            response = await self._complete_cascade(prompt, model, temperature, max_tokens, **kwargs)
        elif strategy == Strategy.PARALLEL:
            response = await self._complete_parallel(prompt, model, temperature, max_tokens, **kwargs)
        elif strategy == Strategy.ENSEMBLE:
            response = await self.ensemble(prompt, model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        else:  # AUTO
            response = await self._complete_auto(prompt, model, temperature, max_tokens, **kwargs)
        
        # Update stats
        self.stats.add(response)
        
        # Cache if enabled
        if self.config.cache_responses:
            self._add_to_cache(cache_key, response)
        
        return response
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens.
        
        Currently uses non-streaming fallback.
        """
        response = await self.complete(prompt, model, **kwargs)
        yield response.content
    
    async def ensemble(
        self,
        prompt: str,
        model: Optional[str] = None,
        mode: Optional[EnsembleMode] = None,
        models: Optional[List[str]] = None,
        **kwargs,
    ) -> OrchestratorResponse:
        """
        Use multiple models and combine results.
        
        Args:
            prompt: The prompt to complete
            model: Default model (can be overridden by models list)
            mode: How to combine results
            models: Specific models to use (None = auto-select)
            **kwargs: Additional arguments
            
        Returns:
            OrchestratorResponse with combined result
        """
        mode = mode or self.config.ensemble_mode
        models = models or self.config.ensemble_models
        
        # Collect responses from available providers
        responses = []
        tasks = []
        
        # Query local if available
        if self.local.is_available():
            tasks.append(self._safe_complete(self.local, prompt, model, **kwargs))
        
        # Query cloud if available
        if self.cloud.is_available():
            tasks.append(self._safe_complete(self.cloud, prompt, model, **kwargs))
        
        if not tasks:
            raise ProviderUnavailableError("ensemble", "No providers available")
        
        # Run in parallel
        results = await asyncio.gather(*tasks)
        responses = [r for r in results if r is not None]
        
        if len(responses) < self.config.min_ensemble_responses:
            logger.warning(f"Only got {len(responses)} ensemble responses, need {self.config.min_ensemble_responses}")
        
        if not responses:
            raise ProviderUnavailableError("ensemble", "All providers failed")
        
        # Combine based on mode
        combined = self._combine_responses(responses, mode)
        
        return OrchestratorResponse(
            content=combined,
            model="ensemble",
            provider="orchestrator",
            provider_type=ProviderType.LOCAL,  # Mixed
            strategy_used=Strategy.ENSEMBLE,
            tokens_used=sum(r.tokens_used for r in responses),
            latency_ms=max(r.latency_ms for r in responses),
            cost=sum(r.cost for r in responses),
            attempts=len(responses),
            ensemble_responses=responses,
        )
    
    async def delegate(
        self,
        task: str,
        context: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
    ) -> OrchestratorResponse:
        """
        Delegate a task to the most appropriate model.
        
        Analyzes the task and routes to the best provider.
        Future: Will use TaskRouter for intelligent routing.
        
        Args:
            task: Task description
            context: Additional context
            provider_type: Force specific provider
            
        Returns:
            OrchestratorResponse
        """
        prompt = task
        if context:
            prompt = f"Context:\n{context}\n\nTask:\n{task}"
        
        return await self.complete(prompt, provider_type=provider_type)
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.stats = UsageStats()
    
    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models by provider."""
        return {
            "local": [m.id for m in self.local.list_models()],
            "cloud": [m.id for m in self.cloud.list_models()],
        }
    
    # =========================================================================
    # Strategy Implementations
    # =========================================================================
    
    async def _complete_with_provider(
        self,
        provider_type: ProviderType,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> OrchestratorResponse:
        """Complete with specific provider type."""
        if provider_type == ProviderType.LOCAL:
            provider = self.local
        else:
            provider = self.cloud
        
        if not provider.is_available():
            raise ProviderUnavailableError(
                provider_type.value,
                f"{provider_type.value} provider not available"
            )
        
        response = await provider.complete(prompt, model, temperature, max_tokens, **kwargs)
        
        return OrchestratorResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            provider_type=response.provider_type,
            strategy_used=self.strategy,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
            cost=response.cost,
        )
    
    async def _complete_local_only(
        self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, **kwargs
    ) -> OrchestratorResponse:
        """Use only local provider, fail if unavailable."""
        if not self.local.is_available():
            raise ProviderUnavailableError("local", "Local provider not available")
        
        response = await self.local.complete(prompt, model, temperature, max_tokens, **kwargs)
        
        return OrchestratorResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            provider_type=ProviderType.LOCAL,
            strategy_used=Strategy.LOCAL_ONLY,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
            cost=0.0,
        )
    
    async def _complete_cloud_only(
        self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, **kwargs
    ) -> OrchestratorResponse:
        """Use only cloud provider."""
        if not self.cloud.is_available():
            raise ProviderUnavailableError("cloud", "Cloud provider not available")
        
        response = await self.cloud.complete(prompt, model, temperature, max_tokens, **kwargs)
        
        return OrchestratorResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            provider_type=ProviderType.CLOUD,
            strategy_used=Strategy.CLOUD_ONLY,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
            cost=response.cost,
        )
    
    async def _complete_local_first(
        self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, **kwargs
    ) -> OrchestratorResponse:
        """Try local first, fallback to cloud."""
        fallback_used = False
        
        # Try local
        if self.local.is_available():
            try:
                response = await self.local.complete(prompt, model, temperature, max_tokens, **kwargs)
                return OrchestratorResponse(
                    content=response.content,
                    model=response.model,
                    provider=response.provider,
                    provider_type=ProviderType.LOCAL,
                    strategy_used=Strategy.LOCAL_FIRST,
                    tokens_used=response.tokens_used,
                    latency_ms=response.latency_ms,
                    cost=0.0,
                )
            except Exception as e:
                logger.warning(f"Local provider failed: {e}, falling back to cloud")
                fallback_used = True
        else:
            fallback_used = True
        
        # Fallback to cloud
        if self.config.fallback_enabled and self.cloud.is_available():
            response = await self.cloud.complete(prompt, model, temperature, max_tokens, **kwargs)
            return OrchestratorResponse(
                content=response.content,
                model=response.model,
                provider=response.provider,
                provider_type=ProviderType.CLOUD,
                strategy_used=Strategy.LOCAL_FIRST,
                tokens_used=response.tokens_used,
                latency_ms=response.latency_ms,
                cost=response.cost,
                fallback_used=fallback_used,
            )
        
        raise ProviderUnavailableError("local_first", "No provider available")
    
    async def _complete_auto(
        self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, **kwargs
    ) -> OrchestratorResponse:
        """
        Auto-select best provider based on availability and preferences.
        
        Current logic:
        1. If local available and cost preference, use local
        2. If specific cloud model requested, use cloud
        3. Otherwise use local-first strategy
        """
        # If specific cloud model requested
        if model and model.startswith(("gpt-", "claude-", "gemini-")):
            if self.cloud.is_available():
                return await self._complete_cloud_only(prompt, model, temperature, max_tokens, **kwargs)
        
        # If prefer cheaper and local available
        if self.config.prefer_cheaper and self.local.is_available():
            return await self._complete_local_first(prompt, model, temperature, max_tokens, **kwargs)
        
        # Default to local-first
        return await self._complete_local_first(prompt, model, temperature, max_tokens, **kwargs)
    
    async def _complete_cascade(
        self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, **kwargs
    ) -> OrchestratorResponse:
        """Try providers in sequence until success."""
        providers = []
        
        if self.local.is_available():
            providers.append(("local", self.local))
        if self.cloud.is_available():
            providers.append(("cloud", self.cloud))
        
        if not providers:
            raise ProviderUnavailableError("cascade", "No providers available")
        
        last_error = None
        attempts = 0
        
        for name, provider in providers:
            attempts += 1
            try:
                response = await provider.complete(prompt, model, temperature, max_tokens, **kwargs)
                return OrchestratorResponse(
                    content=response.content,
                    model=response.model,
                    provider=response.provider,
                    provider_type=response.provider_type,
                    strategy_used=Strategy.CASCADE,
                    tokens_used=response.tokens_used,
                    latency_ms=response.latency_ms,
                    cost=response.cost,
                    attempts=attempts,
                    fallback_used=attempts > 1,
                )
            except Exception as e:
                logger.warning(f"Cascade: {name} failed: {e}")
                last_error = e
        
        raise ProviderUnavailableError("cascade", f"All providers failed: {last_error}")
    
    async def _complete_parallel(
        self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, **kwargs
    ) -> OrchestratorResponse:
        """Race providers in parallel, use fastest."""
        tasks = []
        
        if self.local.is_available():
            tasks.append(self._safe_complete(self.local, prompt, model, **kwargs))
        if self.cloud.is_available():
            tasks.append(self._safe_complete(self.cloud, prompt, model, **kwargs))
        
        if not tasks:
            raise ProviderUnavailableError("parallel", "No providers available")
        
        # Use asyncio.wait to get first completed
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Get result from first completed
        for task in done:
            result = task.result()
            if result:
                return OrchestratorResponse(
                    content=result.content,
                    model=result.model,
                    provider=result.provider,
                    provider_type=result.provider_type,
                    strategy_used=Strategy.PARALLEL,
                    tokens_used=result.tokens_used,
                    latency_ms=result.latency_ms,
                    cost=result.cost,
                )
        
        raise ProviderUnavailableError("parallel", "All providers failed")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    async def _safe_complete(
        self,
        provider,
        prompt: str,
        model: Optional[str],
        **kwargs,
    ) -> Optional[ProviderResponse]:
        """Complete with error handling, return None on failure."""
        try:
            return await provider.complete(prompt, model, **kwargs)
        except Exception as e:
            logger.debug(f"Provider {provider.name} failed: {e}")
            self.stats.errors += 1
            return None
    
    def _combine_responses(
        self,
        responses: List[ProviderResponse],
        mode: EnsembleMode,
    ) -> str:
        """Combine multiple responses based on mode."""
        if not responses:
            return ""
        
        if len(responses) == 1:
            return responses[0].content
        
        if mode == EnsembleMode.BEST:
            # Return longest response (heuristic for "most complete")
            return max(responses, key=lambda r: len(r.content)).content
        
        elif mode == EnsembleMode.VOTE:
            # For simple voting, return most common
            # (simplified: return longest as proxy for "most confident")
            return max(responses, key=lambda r: len(r.content)).content
        
        elif mode == EnsembleMode.MERGE:
            # Merge all responses
            return "\n\n---\n\n".join(r.content for r in responses)
        
        elif mode == EnsembleMode.CONSENSUS:
            # Find common elements (simplified: return longest)
            return max(responses, key=lambda r: len(r.content)).content
        
        return responses[0].content
    
    def _make_cache_key(self, prompt: str, model: Optional[str], strategy: Strategy) -> str:
        """Create cache key from request parameters."""
        import hashlib
        key = f"{strategy.value}:{model or 'default'}:{prompt[:100]}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, response: OrchestratorResponse):
        """Add response to cache with LRU eviction."""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = response


# =============================================================================
# Factory Function
# =============================================================================

_default_orchestrator: Optional[ModelOrchestrator] = None


def get_orchestrator(
    strategy: Strategy = Strategy.AUTO,
    preset: Optional[str] = None,
    **kwargs,
) -> ModelOrchestrator:
    """
    Get or create the model orchestrator.
    
    Args:
        strategy: Orchestration strategy
        preset: Use a preset configuration (development, production, critical, etc.)
        **kwargs: Additional arguments for ModelOrchestrator
        
    Returns:
        ModelOrchestrator instance
    """
    global _default_orchestrator
    
    if preset:
        config = get_preset(preset)
        return ModelOrchestrator(config=config, **kwargs)
    
    if _default_orchestrator is None or _default_orchestrator.strategy != strategy:
        _default_orchestrator = ModelOrchestrator(strategy=strategy, **kwargs)
    
    return _default_orchestrator


def reset_orchestrator():
    """Reset the default orchestrator."""
    global _default_orchestrator
    _default_orchestrator = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ModelOrchestrator",
    "OrchestratorResponse",
    "UsageStats",
    "get_orchestrator",
    "reset_orchestrator",
]
