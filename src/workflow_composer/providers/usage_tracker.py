"""
LLM Usage Tracker and Quota Monitor
====================================

Comprehensive tracking system for cloud LLM providers:
- Real-time quota monitoring (requests, tokens per minute/day)
- Budget tracking and alerts
- Model selection recommendations based on remaining quotas
- Usage history and analytics

Usage:
    from workflow_composer.providers.usage_tracker import UsageTracker, get_tracker
    
    tracker = get_tracker()
    status = tracker.get_all_status()
    tracker.print_status()
    
    # Get best model for a task
    model = tracker.recommend_model(task="code_generation")
"""

from __future__ import annotations

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Provider Configuration with Current Rate Limits (Dec 2025)
# =============================================================================

@dataclass
class ModelSpec:
    """Specification for a single model."""
    id: str
    name: str
    rpm: int = 0              # Requests per minute
    rpd: int = 0              # Requests per day
    tpm: int = 0              # Tokens per minute
    tpd: int = 0              # Tokens per day
    input_price: float = 0.0  # $ per 1M input tokens
    output_price: float = 0.0 # $ per 1M output tokens
    context_length: int = 8192
    best_for: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ProviderQuota:
    """Quota configuration for a provider."""
    provider_id: str
    name: str
    free_tier: bool = True
    models: Dict[str, ModelSpec] = field(default_factory=dict)
    dashboard_url: str = ""
    api_usage_url: str = ""
    notes: str = ""


# Comprehensive provider specifications based on Dec 2025 research
PROVIDER_SPECS: Dict[str, ProviderQuota] = {
    # =========================================================================
    # GEMINI (Google AI Studio) - Best Free Tier
    # =========================================================================
    "gemini": ProviderQuota(
        provider_id="gemini",
        name="Google AI Studio (Gemini)",
        free_tier=True,
        dashboard_url="https://aistudio.google.com/usage",
        api_usage_url="https://aistudio.google.com/usage?timeRange=last-28-days&tab=rate-limit",
        notes="Free tier resets at midnight Pacific time. 1M+ tokens/day available.",
        models={
            "gemini-2.0-flash": ModelSpec(
                id="gemini-2.0-flash",
                name="Gemini 2.0 Flash",
                rpm=15, rpd=200, tpm=1_000_000,
                input_price=0.0, output_price=0.0,  # Free tier
                context_length=1_000_000,
                best_for=["general", "fast_response", "multimodal"],
                notes="Fast, 1M context. Best default choice.",
            ),
            "gemini-2.5-flash": ModelSpec(
                id="gemini-2.5-flash",
                name="Gemini 2.5 Flash",
                rpm=10, rpd=250, tpm=250_000,
                input_price=0.0, output_price=0.0,
                context_length=1_000_000,
                best_for=["thinking", "agentic", "complex_reasoning"],
                notes="Latest flash with thinking capabilities.",
            ),
            "gemini-2.5-pro": ModelSpec(
                id="gemini-2.5-pro",
                name="Gemini 2.5 Pro",
                rpm=2, rpd=50, tpm=125_000,
                input_price=0.0, output_price=0.0,
                context_length=1_000_000,
                best_for=["complex_reasoning", "long_context", "scientific"],
                notes="Most capable. Use for complex tasks only (low quota).",
            ),
            "gemini-2.5-flash-lite": ModelSpec(
                id="gemini-2.5-flash-lite",
                name="Gemini 2.5 Flash-Lite",
                rpm=15, rpd=1000, tpm=250_000,
                input_price=0.0, output_price=0.0,
                context_length=1_000_000,
                best_for=["high_volume", "simple_tasks", "classification"],
                notes="Highest quota model. Best for bulk processing.",
            ),
        },
    ),
    
    # =========================================================================
    # GROQ - Fastest Inference
    # =========================================================================
    "groq": ProviderQuota(
        provider_id="groq",
        name="Groq Cloud",
        free_tier=True,
        dashboard_url="https://console.groq.com/settings/limits",
        api_usage_url="https://console.groq.com/settings/usage",
        notes="Fastest inference (280-500 tok/s). Rate limit info in response headers.",
        models={
            "llama-3.1-8b-instant": ModelSpec(
                id="llama-3.1-8b-instant",
                name="Llama 3.1 8B Instant",
                rpm=30, rpd=14400, tpm=6000, tpd=500_000,
                input_price=0.05, output_price=0.08,  # per 1M tokens
                context_length=131_072,
                best_for=["fast_response", "classification", "simple_tasks"],
                notes="560 tok/s. Highest daily quota. Best for quick tasks.",
            ),
            "llama-3.3-70b-versatile": ModelSpec(
                id="llama-3.3-70b-versatile",
                name="Llama 3.3 70B Versatile",
                rpm=30, rpd=1000, tpm=12000, tpd=100_000,
                input_price=0.59, output_price=0.79,
                context_length=131_072,
                best_for=["general", "reasoning", "code"],
                notes="280 tok/s. Good balance of speed and capability.",
            ),
            "openai/gpt-oss-120b": ModelSpec(
                id="openai/gpt-oss-120b",
                name="GPT-OSS 120B",
                rpm=30, rpd=1000, tpm=8000, tpd=200_000,
                input_price=0.15, output_price=0.60,
                context_length=131_072,
                best_for=["reasoning", "code", "analysis"],
                notes="OpenAI's open-weight model. 500 tok/s.",
            ),
            "openai/gpt-oss-20b": ModelSpec(
                id="openai/gpt-oss-20b",
                name="GPT-OSS 20B",
                rpm=30, rpd=1000, tpm=8000, tpd=200_000,
                input_price=0.075, output_price=0.30,
                context_length=131_072,
                best_for=["simple_tasks", "fast_response", "high_volume"],
                notes="Smaller, faster, cheaper. 1000 tok/s.",
            ),
            "groq/compound": ModelSpec(
                id="groq/compound",
                name="Groq Compound (Agentic)",
                rpm=30, rpd=250, tpm=70_000,
                input_price=0.0, output_price=0.0,  # Included free
                context_length=131_072,
                best_for=["agentic", "tool_use", "web_search"],
                notes="AI system with built-in tools (web search, code exec).",
            ),
        },
    ),
    
    # =========================================================================
    # CEREBRAS - Highest Free Quota
    # =========================================================================
    "cerebras": ProviderQuota(
        provider_id="cerebras",
        name="Cerebras Cloud",
        free_tier=True,
        dashboard_url="https://cloud.cerebras.ai/",
        api_usage_url="https://cloud.cerebras.ai/usage",
        notes="14,400 req/day, 1M tokens/day FREE. Very generous limits.",
        models={
            "llama-3.3-70b": ModelSpec(
                id="llama-3.3-70b",
                name="Llama 3.3 70B",
                rpm=60, rpd=14400, tpm=60_000, tpd=1_000_000,
                input_price=0.0, output_price=0.0,
                context_length=8192,
                best_for=["general", "reasoning", "code"],
                notes="Most popular. Extremely fast on Cerebras hardware.",
            ),
            "qwen3-235b-a22b": ModelSpec(
                id="qwen3-235b-a22b",
                name="Qwen3 235B",
                rpm=30, rpd=14400, tpm=30_000, tpd=500_000,
                input_price=0.0, output_price=0.0,
                context_length=8192,
                best_for=["complex_reasoning", "multilingual", "analysis"],
                notes="235B params FREE! Excellent for complex tasks.",
            ),
            "qwen3-coder-480b": ModelSpec(
                id="qwen3-coder-480b",
                name="Qwen3 Coder 480B",
                rpm=10, rpd=100, tpm=10_000, tpd=100_000,
                input_price=0.0, output_price=0.0,
                context_length=8192,
                best_for=["code_generation", "code_review", "debugging"],
                notes="Best coding model. Low quota - use wisely.",
            ),
            "gpt-oss-120b": ModelSpec(
                id="gpt-oss-120b",
                name="GPT-OSS 120B",
                rpm=60, rpd=14400, tpm=60_000, tpd=1_000_000,
                input_price=0.0, output_price=0.0,
                context_length=8192,
                best_for=["reasoning", "general", "analysis"],
                notes="OpenAI's open model on Cerebras.",
            ),
        },
    ),
    
    # =========================================================================
    # OPENROUTER - Gateway to Free Models
    # =========================================================================
    "openrouter": ProviderQuota(
        provider_id="openrouter",
        name="OpenRouter",
        free_tier=True,
        dashboard_url="https://openrouter.ai/settings/keys",
        api_usage_url="https://openrouter.ai/activity",
        notes="Gateway to 400+ models. Free models marked with :free suffix.",
        models={
            "meta-llama/llama-3.3-70b-instruct:free": ModelSpec(
                id="meta-llama/llama-3.3-70b-instruct:free",
                name="Llama 3.3 70B (Free)",
                rpm=20, rpd=50,
                input_price=0.0, output_price=0.0,
                context_length=131_072,
                best_for=["general", "reasoning"],
                notes="Free tier. Add $10 once for higher limits.",
            ),
            "qwen/qwen3-235b-a22b:free": ModelSpec(
                id="qwen/qwen3-235b-a22b:free",
                name="Qwen3 235B (Free)",
                rpm=20, rpd=50,
                input_price=0.0, output_price=0.0,
                context_length=32_768,
                best_for=["complex_reasoning", "multilingual"],
            ),
            "deepseek/deepseek-r1t-chimera:free": ModelSpec(
                id="deepseek/deepseek-r1t-chimera:free",
                name="DeepSeek R1 Chimera (Free)",
                rpm=20, rpd=50,
                input_price=0.0, output_price=0.0,
                context_length=65_536,
                best_for=["reasoning", "math", "code"],
            ),
            "google/gemma-3-27b-it:free": ModelSpec(
                id="google/gemma-3-27b-it:free",
                name="Gemma 3 27B (Free)",
                rpm=20, rpd=50,
                input_price=0.0, output_price=0.0,
                context_length=8192,
                best_for=["general", "fast_response"],
            ),
        },
    ),
    
    # =========================================================================
    # GITHUB MODELS - Free with Copilot
    # =========================================================================
    "github_models": ProviderQuota(
        provider_id="github_models",
        name="GitHub Models",
        free_tier=True,
        dashboard_url="https://github.com/settings/copilot",
        notes="Free with GitHub Copilot subscription. Quota depends on tier.",
        models={
            "gpt-4o-mini": ModelSpec(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                rpm=15, rpd=150,
                input_price=0.0, output_price=0.0,  # Included with Copilot
                context_length=128_000,
                best_for=["general", "fast_response", "classification"],
            ),
            "gpt-4o": ModelSpec(
                id="gpt-4o",
                name="GPT-4o",
                rpm=10, rpd=50,
                input_price=0.0, output_price=0.0,
                context_length=128_000,
                best_for=["complex_reasoning", "multimodal"],
                notes="Premium model. Limited quota.",
            ),
            "DeepSeek-R1": ModelSpec(
                id="DeepSeek-R1",
                name="DeepSeek R1",
                rpm=15, rpd=150,
                input_price=0.0, output_price=0.0,
                context_length=65_536,
                best_for=["reasoning", "math", "code"],
            ),
        },
    ),
    
    # =========================================================================
    # LIGHTNING.AI - WORKING (via litai SDK)
    # =========================================================================
    "lightning": ProviderQuota(
        provider_id="lightning",
        name="Lightning.ai",
        free_tier=True,
        dashboard_url="https://lightning.ai/account",
        notes="✓ WORKING via litai SDK. Format: provider/model_name. 30M free tokens/month.",
        models={
            "lightning-ai/gpt-oss-20b": ModelSpec(
                id="lightning-ai/gpt-oss-20b",
                name="Lightning GPT-OSS 20B (FREE)",
                rpm=100,
                input_price=0.05, output_price=0.10,
                context_length=32_000,
                best_for=["chat", "intent_parsing", "general"],
                notes="Lightning's own model - fast and free",
            ),
            "openai/gpt-4o": ModelSpec(
                id="openai/gpt-4o",
                name="GPT-4o via Lightning",
                rpm=50,
                input_price=2.50, output_price=10.00,
                context_length=128_000,
                best_for=["reasoning", "code", "scientific"],
            ),
            "openai/gpt-3.5-turbo": ModelSpec(
                id="openai/gpt-3.5-turbo",
                name="GPT-3.5 Turbo via Lightning",
                rpm=100,
                input_price=0.50, output_price=1.50,
                context_length=16_000,
                best_for=["quick_response", "chat"],
                notes="Cheapest option for simple tasks",
            ),
        },
    ),
    
    # =========================================================================
    # OPENAI - Paid Fallback
    # =========================================================================
    "openai": ProviderQuota(
        provider_id="openai",
        name="OpenAI",
        free_tier=False,
        dashboard_url="https://platform.openai.com/usage",
        api_usage_url="https://platform.openai.com/organization/usage",
        notes="PAID. Use only when all free tiers exhausted.",
        models={
            "gpt-4o-mini": ModelSpec(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                rpm=500, rpd=10000,
                input_price=0.15, output_price=0.60,
                context_length=128_000,
                best_for=["general", "fast_response"],
                notes="Cheapest OpenAI model.",
            ),
            "gpt-4o": ModelSpec(
                id="gpt-4o",
                name="GPT-4o",
                rpm=500, rpd=10000,
                input_price=2.50, output_price=10.00,
                context_length=128_000,
                best_for=["complex_reasoning", "multimodal"],
                notes="Premium. Use sparingly.",
            ),
        },
    ),
}


# =============================================================================
# Usage Tracking Data Structures
# =============================================================================

@dataclass
class UsageRecord:
    """Single usage record."""
    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error: Optional[str] = None
    cost_usd: float = 0.0


@dataclass
class ProviderUsage:
    """Aggregated usage for a provider."""
    provider_id: str
    requests_today: int = 0
    tokens_today: int = 0
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    total_cost_usd: float = 0.0
    last_request: Optional[float] = None
    rate_limited: bool = False
    rate_limit_reset: Optional[float] = None
    errors_today: int = 0
    
    def can_request(self, quota: ProviderQuota, model_id: str) -> Tuple[bool, str]:
        """Check if we can make a request to this provider."""
        if self.rate_limited:
            if self.rate_limit_reset and time.time() < self.rate_limit_reset:
                return False, f"Rate limited until {datetime.fromtimestamp(self.rate_limit_reset).strftime('%H:%M:%S')}"
            self.rate_limited = False
        
        model = quota.models.get(model_id)
        if not model:
            return True, "Model not in quota tracking"
        
        if model.rpd and self.requests_today >= model.rpd:
            return False, f"Daily request limit reached ({model.rpd}/day)"
        
        if model.rpm and self.requests_this_minute >= model.rpm:
            return False, f"Minute request limit reached ({model.rpm}/min)"
        
        if model.tpd and self.tokens_today >= model.tpd:
            return False, f"Daily token limit reached ({model.tpd}/day)"
        
        return True, "OK"


# =============================================================================
# Usage Tracker
# =============================================================================

class UsageTracker:
    """
    Comprehensive LLM usage tracker with quota monitoring.
    
    Features:
    - Real-time quota tracking per provider/model
    - Automatic rate limit detection from response headers
    - Model selection recommendations based on remaining quotas
    - Usage history and cost tracking
    - Periodic persistence to disk
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize tracker."""
        self.storage_path = storage_path or Path.home() / ".biopipelines" / "llm_usage.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._usage: Dict[str, ProviderUsage] = {}
        self._history: List[UsageRecord] = []
        self._minute_window_start = time.time()
        self._day_start = self._get_day_start()
        
        # Load persisted data
        self._load()
    
    def _get_day_start(self) -> float:
        """Get timestamp for start of current day (Pacific time for Gemini)."""
        now = datetime.now()
        return datetime(now.year, now.month, now.day).timestamp()
    
    def _check_reset(self):
        """Check if minute/day counters should reset."""
        now = time.time()
        
        # Reset minute counters every 60 seconds
        if now - self._minute_window_start >= 60:
            self._minute_window_start = now
            for usage in self._usage.values():
                usage.requests_this_minute = 0
                usage.tokens_this_minute = 0
        
        # Reset daily counters at midnight
        current_day_start = self._get_day_start()
        if current_day_start > self._day_start:
            self._day_start = current_day_start
            for usage in self._usage.values():
                usage.requests_today = 0
                usage.tokens_today = 0
                usage.errors_today = 0
                usage.rate_limited = False
    
    def _get_usage(self, provider_id: str) -> ProviderUsage:
        """Get or create usage record for provider."""
        if provider_id not in self._usage:
            self._usage[provider_id] = ProviderUsage(provider_id=provider_id)
        return self._usage[provider_id]
    
    def record_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        rate_limit_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Record a completed request.
        
        Args:
            provider: Provider ID (e.g., "gemini", "groq")
            model: Model ID
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            error: Error message if failed
            rate_limit_headers: Headers from response (for Groq rate limits)
        """
        with self._lock:
            self._check_reset()
            
            usage = self._get_usage(provider)
            total_tokens = input_tokens + output_tokens
            
            # Update counters
            usage.requests_today += 1
            usage.tokens_today += total_tokens
            usage.requests_this_minute += 1
            usage.tokens_this_minute += total_tokens
            usage.last_request = time.time()
            
            if not success:
                usage.errors_today += 1
            
            # Calculate cost
            cost = self._calculate_cost(provider, model, input_tokens, output_tokens)
            usage.total_cost_usd += cost
            
            # Parse rate limit headers (Groq provides these)
            if rate_limit_headers:
                self._parse_rate_limit_headers(usage, rate_limit_headers)
            
            # Add to history
            record = UsageRecord(
                timestamp=time.time(),
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                success=success,
                error=error,
                cost_usd=cost,
            )
            self._history.append(record)
            
            # Keep only last 1000 records in memory
            if len(self._history) > 1000:
                self._history = self._history[-1000:]
            
            # Persist periodically
            if len(self._history) % 10 == 0:
                self._save()
    
    def record_rate_limit(self, provider: str, reset_seconds: Optional[float] = None):
        """Record that a provider hit rate limit."""
        with self._lock:
            usage = self._get_usage(provider)
            usage.rate_limited = True
            if reset_seconds:
                usage.rate_limit_reset = time.time() + reset_seconds
            else:
                # Default: wait 60 seconds
                usage.rate_limit_reset = time.time() + 60
    
    def _parse_rate_limit_headers(self, usage: ProviderUsage, headers: Dict[str, str]):
        """Parse Groq-style rate limit headers."""
        # Groq headers:
        # x-ratelimit-remaining-requests, x-ratelimit-remaining-tokens
        # x-ratelimit-reset-requests, x-ratelimit-reset-tokens
        if "x-ratelimit-remaining-requests" in headers:
            remaining = int(headers["x-ratelimit-remaining-requests"])
            if remaining <= 0:
                usage.rate_limited = True
        
        if "retry-after" in headers:
            usage.rate_limited = True
            usage.rate_limit_reset = time.time() + int(headers["retry-after"])
    
    def _calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        spec = PROVIDER_SPECS.get(provider)
        if not spec:
            return 0.0
        
        model_spec = spec.models.get(model)
        if not model_spec:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * model_spec.input_price
        output_cost = (output_tokens / 1_000_000) * model_spec.output_price
        return input_cost + output_cost
    
    def can_use_provider(self, provider: str, model: Optional[str] = None) -> Tuple[bool, str]:
        """Check if a provider/model is available (not rate limited, has quota)."""
        with self._lock:
            self._check_reset()
            
            spec = PROVIDER_SPECS.get(provider)
            if not spec:
                return True, "Unknown provider"
            
            usage = self._get_usage(provider)
            model_id = model or next(iter(spec.models.keys()), "default")
            
            return usage.can_request(spec, model_id)
    
    def get_remaining_quota(self, provider: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Get remaining quota for a provider/model."""
        with self._lock:
            self._check_reset()
            
            spec = PROVIDER_SPECS.get(provider)
            if not spec:
                return {"error": "Unknown provider"}
            
            usage = self._get_usage(provider)
            model_id = model or next(iter(spec.models.keys()), None)
            model_spec = spec.models.get(model_id) if model_id else None
            
            result = {
                "provider": provider,
                "model": model_id,
                "requests_today": usage.requests_today,
                "tokens_today": usage.tokens_today,
                "requests_this_minute": usage.requests_this_minute,
                "rate_limited": usage.rate_limited,
            }
            
            if model_spec:
                result.update({
                    "remaining_requests_day": max(0, model_spec.rpd - usage.requests_today) if model_spec.rpd else "unlimited",
                    "remaining_requests_min": max(0, model_spec.rpm - usage.requests_this_minute) if model_spec.rpm else "unlimited",
                    "remaining_tokens_day": max(0, model_spec.tpd - usage.tokens_today) if model_spec.tpd else "unlimited",
                })
            
            return result
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all configured providers."""
        with self._lock:
            self._check_reset()
            
            result = {}
            for provider_id, spec in PROVIDER_SPECS.items():
                usage = self._get_usage(provider_id)
                
                # Check if configured (has API key)
                from .registry import get_registry
                registry = get_registry()
                config = registry.get_provider_config(provider_id)
                is_configured = config.is_configured() if config else False
                is_enabled = config.enabled if config else True
                
                default_model = next(iter(spec.models.keys()), None)
                can_use, reason = usage.can_request(spec, default_model) if default_model else (True, "OK")
                
                result[provider_id] = {
                    "name": spec.name,
                    "free_tier": spec.free_tier,
                    "configured": is_configured,
                    "enabled": is_enabled,
                    "available": can_use and is_configured and is_enabled,
                    "status_reason": reason if not can_use else ("Not configured" if not is_configured else ("Disabled" if not is_enabled else "OK")),
                    "requests_today": usage.requests_today,
                    "tokens_today": usage.tokens_today,
                    "errors_today": usage.errors_today,
                    "cost_today_usd": usage.total_cost_usd,
                    "rate_limited": usage.rate_limited,
                    "dashboard_url": spec.dashboard_url,
                    "models": list(spec.models.keys()),
                }
            
            return result
    
    def recommend_model(
        self,
        task: str = "general",
        prefer_free: bool = True,
        min_context: int = 4096,
    ) -> Optional[Tuple[str, str]]:
        """
        Recommend best available model for a task.
        
        Args:
            task: Task type (general, code, reasoning, fast_response, etc.)
            prefer_free: Prefer free tier providers
            min_context: Minimum context length required
            
        Returns:
            Tuple of (provider_id, model_id) or None if nothing available
        """
        candidates = []
        
        for provider_id, spec in PROVIDER_SPECS.items():
            if not spec.free_tier and prefer_free:
                continue
            
            # Check if configured and enabled
            from .registry import get_registry
            registry = get_registry()
            config = registry.get_provider_config(provider_id)
            if not config or not config.is_configured() or not config.enabled:
                continue
            
            for model_id, model_spec in spec.models.items():
                # Check context length
                if model_spec.context_length < min_context:
                    continue
                
                # Check if available
                can_use, _ = self.can_use_provider(provider_id, model_id)
                if not can_use:
                    continue
                
                # Score based on task match and remaining quota
                score = 0
                if task in model_spec.best_for:
                    score += 100
                if "general" in model_spec.best_for:
                    score += 50
                
                # Prefer models with more remaining quota
                usage = self._get_usage(provider_id)
                if model_spec.rpd:
                    remaining_pct = (model_spec.rpd - usage.requests_today) / model_spec.rpd
                    score += remaining_pct * 50
                
                candidates.append((score, provider_id, model_id))
        
        if not candidates:
            return None
        
        # Return highest scoring candidate
        candidates.sort(reverse=True)
        _, provider, model = candidates[0]
        return (provider, model)
    
    def print_status(self):
        """Print formatted status to console."""
        status = self.get_all_status()
        
        print("\n" + "=" * 70)
        print("LLM PROVIDER STATUS")
        print("=" * 70)
        
        # Group by availability
        available = []
        unavailable = []
        
        for provider_id, info in status.items():
            if info["available"]:
                available.append((provider_id, info))
            else:
                unavailable.append((provider_id, info))
        
        print(f"\n{'AVAILABLE':^70}")
        print("-" * 70)
        for provider_id, info in available:
            tier = "FREE" if info["free_tier"] else "PAID"
            print(f"  ✅ {info['name']:<25} [{tier}]")
            print(f"     Requests: {info['requests_today']}/day  Tokens: {info['tokens_today']:,}/day")
            if info['cost_today_usd'] > 0:
                print(f"     Cost today: ${info['cost_today_usd']:.4f}")
            print(f"     Models: {', '.join(info['models'][:3])}")
            print()
        
        if unavailable:
            print(f"\n{'UNAVAILABLE':^70}")
            print("-" * 70)
            for provider_id, info in unavailable:
                reason = info["status_reason"]
                print(f"  ❌ {info['name']:<25} - {reason}")
        
        print("\n" + "=" * 70)
        
        # Recommend a model
        rec = self.recommend_model()
        if rec:
            print(f"💡 Recommended: {rec[0]}/{rec[1]}")
        print()
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary across all providers."""
        with self._lock:
            total_cost = sum(u.total_cost_usd for u in self._usage.values())
            total_tokens = sum(u.tokens_today for u in self._usage.values())
            total_requests = sum(u.requests_today for u in self._usage.values())
            
            by_provider = {
                pid: {
                    "cost_usd": u.total_cost_usd,
                    "tokens": u.tokens_today,
                    "requests": u.requests_today,
                }
                for pid, u in self._usage.items()
                if u.requests_today > 0
            }
            
            return {
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "total_requests": total_requests,
                "by_provider": by_provider,
                "period": "today",
            }
    
    def _save(self):
        """Persist usage data to disk."""
        try:
            data = {
                "usage": {pid: asdict(u) for pid, u in self._usage.items()},
                "day_start": self._day_start,
                "saved_at": time.time(),
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save usage data: {e}")
    
    def _load(self):
        """Load persisted usage data."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            saved_day = data.get("day_start", 0)
            current_day = self._get_day_start()
            
            # Only restore if same day
            if saved_day >= current_day:
                for pid, usage_data in data.get("usage", {}).items():
                    self._usage[pid] = ProviderUsage(**usage_data)
                    
        except Exception as e:
            logger.warning(f"Failed to load usage data: {e}")


# =============================================================================
# Global Instance
# =============================================================================

_tracker: Optional[UsageTracker] = None


def get_tracker() -> UsageTracker:
    """Get the global usage tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker


# =============================================================================
# CLI Commands
# =============================================================================

def main():
    """CLI entry point."""
    import sys
    
    tracker = get_tracker()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "status":
            tracker.print_status()
        elif cmd == "cost":
            summary = tracker.get_cost_summary()
            print(json.dumps(summary, indent=2))
        elif cmd == "recommend":
            task = sys.argv[2] if len(sys.argv) > 2 else "general"
            rec = tracker.recommend_model(task=task)
            if rec:
                print(f"Recommended: {rec[0]}/{rec[1]}")
            else:
                print("No model available")
        elif cmd == "quota":
            provider = sys.argv[2] if len(sys.argv) > 2 else "gemini"
            quota = tracker.get_remaining_quota(provider)
            print(json.dumps(quota, indent=2))
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python -m workflow_composer.providers.usage_tracker [status|cost|recommend|quota]")
    else:
        tracker.print_status()


if __name__ == "__main__":
    main()
