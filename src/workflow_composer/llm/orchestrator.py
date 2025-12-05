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
    load_profile,
)
from .resource_detector import ResourceDetector, ResourceStatus
from .metrics import RoutingMetrics, RoutingDecision, MetricsContext, get_metrics

# Task-based router for T4 vLLM servers (optional - may not be installed in all envs)
try:
    from workflow_composer.providers.t4_router import T4ModelRouter, TaskCategory
    T4_ROUTER_AVAILABLE = True
except ImportError:
    T4ModelRouter = None
    TaskCategory = None
    T4_ROUTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Debug routing logger (separate from main logger for filtering)
routing_logger = logging.getLogger("workflow_composer.routing")


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
        profile: Optional[str] = None,
        local_provider: Optional[LocalProvider] = None,
        cloud_provider: Optional[CloudProvider] = None,
        auto_detect: bool = False,
    ):
        """
        Initialize orchestrator.
        
        Args:
            strategy: Primary routing strategy
            config: Detailed configuration (overrides strategy if provided)
            profile: Load config from named profile (e.g., 't4_hybrid')
            local_provider: Custom local provider
            cloud_provider: Custom cloud provider
            auto_detect: Auto-detect best strategy from resources
        """
        # Load configuration
        if profile:
            self.config = load_profile(profile)
            logger.info(f"Loaded profile: {profile}")
        elif config:
            self.config = config
        elif auto_detect:
            self.config = self._auto_detect_config()
        else:
            self.config = StrategyConfig(strategy=strategy)
        
        self.strategy = self.config.strategy
        
        # Initialize providers
        self.local = local_provider or LocalProvider()
        self.cloud = cloud_provider or CloudProvider()
        
        # Usage tracking
        self.stats = UsageStats()
        
        # Metrics collection (v2.1)
        self.metrics = get_metrics()
        
        # Response cache (simple in-memory)
        self._cache: Dict[str, OrchestratorResponse] = {}
        self._cache_max_size = 100
        
        # Resource detector for health checks
        self._resource_detector = ResourceDetector(
            vllm_endpoints=self.config.vllm_endpoints or {}
        )
        
        # T4 task-based router (for routing to specialized vLLM servers)
        self._t4_router: Optional[T4ModelRouter] = None
        if T4_ROUTER_AVAILABLE and self.config.vllm_endpoints:
            try:
                self._t4_router = T4ModelRouter()
                logger.debug("T4ModelRouter initialized for task-based routing")
            except Exception as e:
                logger.warning(f"Failed to initialize T4ModelRouter: {e}")
        
        profile_name = self.config.profile_name or "default"
        logger.info(f"ModelOrchestrator initialized: strategy={self.strategy.value}, profile={profile_name}")
    
    def _auto_detect_config(self) -> StrategyConfig:
        """
        Auto-detect best configuration from available resources.
        
        Returns config based on:
        1. Available vLLM endpoints
        2. Configured cloud API keys
        3. SLURM availability
        """
        detector = ResourceDetector()
        status = detector.detect()
        
        profile_name = detector.get_best_strategy()
        logger.info(f"Auto-detected strategy profile: {profile_name} (mode: {status.deployment_mode})")
        
        return load_profile(profile_name)
    
    # =========================================================================
    # Strategy Switching (v2.1)
    # =========================================================================
    
    def switch_strategy(self, profile_or_strategy: str | Strategy | StrategyConfig) -> None:
        """
        Switch to a different strategy at runtime.
        
        Args:
            profile_or_strategy: One of:
                - Profile name (str): e.g., 't4_hybrid', 'cloud_only'
                - Strategy enum: Strategy.LOCAL_FIRST
                - StrategyConfig: Full configuration object
        
        Examples:
            orch.switch_strategy("t4_local_only")  # PHI mode
            orch.switch_strategy(Strategy.CLOUD_ONLY)
            orch.switch_strategy(my_custom_config)
        """
        if isinstance(profile_or_strategy, StrategyConfig):
            new_config = profile_or_strategy
        elif isinstance(profile_or_strategy, Strategy):
            new_config = StrategyConfig(strategy=profile_or_strategy)
        else:
            new_config = load_profile(profile_or_strategy)
        
        old_profile = self.config.profile_name or "default"
        new_profile = new_config.profile_name or "default"
        
        self.config = new_config
        self.strategy = new_config.strategy
        
        # Update resource detector with new endpoints
        if new_config.vllm_endpoints:
            self._resource_detector = ResourceDetector(
                vllm_endpoints=new_config.vllm_endpoints
            )
        
        logger.info(f"Switched strategy: {old_profile} -> {new_profile} (strategy={self.strategy.value})")
    
    def get_current_profile(self) -> str:
        """Get the name of the current strategy profile."""
        return self.config.profile_name or "default"
    
    def get_resource_status(self) -> ResourceStatus:
        """
        Get current resource availability status.
        
        Useful for debugging and monitoring.
        """
        return self._resource_detector.detect()
    
    def can_use_cloud(self, task_type: Optional[str] = None) -> bool:
        """
        Check if cloud APIs can be used (respects data governance).
        
        Args:
            task_type: Optional task type for per-task governance
        
        Returns:
            True if cloud is allowed for this request
        """
        if task_type:
            return self.config.can_use_cloud_for_task(task_type)
        return self.config.allow_cloud
    
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
        
        # Debug routing log
        if self.config.debug_routing:
            self._log_routing_decision(
                prompt=prompt,
                response=response,
                task_type=kwargs.get("task_type", "general"),
            )
        
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
    
    def _log_routing_decision(
        self,
        prompt: str,
        response: OrchestratorResponse,
        task_type: str = "general",
        fallback_chain: Optional[List[str]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Log detailed routing decision for debugging.
        
        Only logs when debug_routing is enabled in config.
        Writes to both logger and metrics system.
        """
        decision = RoutingDecision(
            task_type=task_type,
            query_length=len(prompt),
            strategy_profile=self.config.profile_name or "default",
            model_key=response.model.split("/")[-1] if "/" in response.model else response.model,
            model_id=response.model,
            provider=response.provider,
            fallback_depth=response.attempts - 1,
            fallback_reason="primary_failed" if response.fallback_used else None,
            fallback_chain=fallback_chain or [],
            success=error is None,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            latency_ms=response.latency_ms,
            tokens_generated=response.tokens_used,
            estimated_cost=response.cost,
            debug_context={
                "strategy_used": response.strategy_used.value,
                "provider_type": response.provider_type.value,
                "cached": response.cached,
            }
        )
        
        # Log to dedicated routing logger
        routing_logger.info(
            f"ROUTING: {task_type} -> {response.provider}/{response.model} "
            f"[{response.latency_ms:.0f}ms, fallback={response.fallback_used}]"
        )
        routing_logger.debug(f"ROUTING_DETAIL: {decision.to_json()}")
        
        # Also record in metrics
        if self.metrics:
            self.metrics.log(decision)
    
    # =========================================================================
    # Task-Based Routing (T4 Fleet)
    # =========================================================================
    
    def has_task_router(self) -> bool:
        """Check if task-based T4 routing is available."""
        return self._t4_router is not None
    
    async def complete_with_task(
        self,
        task: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> OrchestratorResponse:
        """
        Complete using task-based routing to specialized models.
        
        This routes the request to the appropriate vLLM server based on
        the task type (code, math, general, embeddings).
        
        Args:
            task: Task category - one of:
                  'code', 'math', 'general', 'embeddings', 'reasoning'
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments
        
        Returns:
            OrchestratorResponse with completion
        
        Raises:
            RuntimeError: If task router not available
            
        Example:
            response = await orch.complete_with_task(
                task="code",
                prompt="Write a Python function to parse FASTQ files"
            )
        """
        if not self._t4_router:
            if not self.config.allow_cloud:
                raise RuntimeError(
                    "T4 task router not available and cloud is disabled. "
                    "Deploy vLLM servers with scripts/llm/deploy_core_models.sh"
                )
            # Fallback to generic completion
            logger.warning("T4 router not available, falling back to generic completion")
            return await self.complete(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        start_time = time.time()
        
        try:
            result = await self._t4_router.complete(
                task=task,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine provider type based on result
            is_local = result.get("is_local", True)
            provider_type = ProviderType.LOCAL if is_local else ProviderType.CLOUD
            
            response = OrchestratorResponse(
                content=result["content"],
                model=result.get("model", "unknown"),
                provider=result.get("model_name", "t4-vllm"),
                provider_type=provider_type,
                strategy_used=self.strategy,
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
                latency_ms=latency_ms,
                cost=0.0 if is_local else self._estimate_cloud_cost(result),
                fallback_used=not is_local,
            )
            
            self.stats.add(response)
            
            # Debug routing log
            if self.config.debug_routing:
                self._log_routing_decision(
                    prompt=prompt,
                    response=response,
                    task_type=task,
                )
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Log error in debug mode
            if self.config.debug_routing:
                error_decision = RoutingDecision(
                    task_type=task,
                    query_length=len(prompt),
                    strategy_profile=self.config.profile_name or "default",
                    model_key="unknown",
                    model_id="unknown",
                    provider="t4-router",
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    latency_ms=latency_ms,
                )
                routing_logger.error(f"ROUTING_ERROR: {task} failed - {e}")
                routing_logger.debug(f"ROUTING_ERROR_DETAIL: {error_decision.to_json()}")
                
                if self.metrics:
                    self.metrics.log(error_decision)
            
            raise
    
    def _estimate_cloud_cost(self, result: Dict[str, Any]) -> float:
        """Estimate cost for cloud API call based on usage."""
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        model = result.get("model", "")
        
        # Rough cost estimates per 1K tokens
        if "gpt-4" in model:
            return (input_tokens * 0.01 + output_tokens * 0.03) / 1000
        elif "gpt-3.5" in model:
            return (input_tokens * 0.0005 + output_tokens * 0.0015) / 1000
        elif "claude" in model:
            return (input_tokens * 0.008 + output_tokens * 0.024) / 1000
        elif "deepseek" in model:
            return (input_tokens * 0.0001 + output_tokens * 0.0002) / 1000
        return 0.0
    
    async def embed_with_task(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Get embeddings using the T4 embedding model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If embedding model not available
        """
        if not self._t4_router:
            raise RuntimeError("T4 task router not available for embeddings")
        
        return await self._t4_router.embed(texts)


# =============================================================================
# Factory Function
# =============================================================================

_default_orchestrator: Optional[ModelOrchestrator] = None


def get_orchestrator(
    strategy: Strategy = Strategy.AUTO,
    preset: Optional[str] = None,
    profile: Optional[str] = None,
    auto_detect: bool = False,
    **kwargs,
) -> ModelOrchestrator:
    """
    Get or create the model orchestrator.
    
    Args:
        strategy: Orchestration strategy
        preset: Use a preset configuration (development, production, critical, etc.)
        profile: Load from YAML profile (t4_hybrid, t4_local_only, cloud_only, etc.)
        auto_detect: Auto-detect best strategy from available resources
        **kwargs: Additional arguments for ModelOrchestrator
        
    Returns:
        ModelOrchestrator instance
        
    Examples:
        # Basic usage
        orch = get_orchestrator()
        
        # With profile
        orch = get_orchestrator(profile="t4_hybrid")
        
        # Auto-detect from environment
        orch = get_orchestrator(auto_detect=True)
        
        # Legacy preset
        orch = get_orchestrator(preset="production")
    """
    global _default_orchestrator
    
    # Profile takes precedence (v2.1)
    if profile:
        return ModelOrchestrator(profile=profile, **kwargs)
    
    # Auto-detect (v2.1)
    if auto_detect:
        return ModelOrchestrator(auto_detect=True, **kwargs)
    
    # Legacy preset support
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
