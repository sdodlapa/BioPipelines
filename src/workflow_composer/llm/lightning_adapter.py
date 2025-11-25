"""
Lightning.ai LLM Adapter
========================

Unified API access to multiple LLM providers through Lightning.ai.

Benefits:
- One API key for all models (OpenAI, Anthropic, open-source)
- 30 million FREE tokens/month
- Pay-as-you-go after free tier
- Up to 70% cheaper than direct API access

Supported Models:
- DeepSeek-V3 (cheapest, excellent for reasoning/code)
- Llama 3.3 70B (open weights, good general purpose)
- Claude 3.5 Sonnet (best for scientific text)
- GPT-4o (highest quality)
- Mistral Large (good balance)

Usage:
    from workflow_composer.llm.lightning_adapter import LightningAdapter
    
    llm = LightningAdapter(model="deepseek/deepseek-v3")
    response = llm.chat([Message.user("Hello!")])
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import LLMAdapter, Message, LLMResponse, Role

logger = logging.getLogger(__name__)


# Model pricing per million tokens (as of Nov 2025)
MODEL_PRICING = {
    "deepseek-ai/DeepSeek-V3": {"input": 0.14, "output": 0.28},
    "deepseek-ai/DeepSeek-R1": {"input": 0.14, "output": 0.28},
    "deepseek-ai/deepseek-chat": {"input": 0.07, "output": 0.14},
    "meta-llama/Llama-3.3-70B-Instruct": {"input": 0.80, "output": 0.80},
    "meta-llama/Llama-3.1-70B-Instruct": {"input": 0.80, "output": 0.80},
    "meta-llama/Llama-3.1-8B-Instruct": {"input": 0.10, "output": 0.10},
    "mistralai/Mistral-Large-Instruct-2411": {"input": 2.00, "output": 6.00},
    "mistralai/Mistral-7B-Instruct-v0.3": {"input": 0.25, "output": 0.25},
    "Qwen/Qwen2.5-72B-Instruct": {"input": 0.80, "output": 0.80},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# Model recommendations by task
TASK_MODELS = {
    "intent_parsing": "deepseek-ai/DeepSeek-V3",
    "workflow_generation": "deepseek-ai/DeepSeek-V3",
    "module_creation": "deepseek-ai/DeepSeek-V3",
    "code_generation": "deepseek-ai/DeepSeek-V3",
    "scientific_analysis": "Qwen/Qwen2.5-72B-Instruct",
    "chat": "meta-llama/Llama-3.3-70B-Instruct",
    "quick_response": "meta-llama/Llama-3.1-8B-Instruct",
    "high_quality": "deepseek-ai/DeepSeek-V3",
}


@dataclass
class UsageTracker:
    """Track token usage for cost estimation."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    requests: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
    
    def add(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.requests += 1
    
    def estimate_cost(self, model: str) -> float:
        """Estimate cost in USD."""
        pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 1.0})
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def is_within_free_tier(self) -> bool:
        """Check if within 30M free token limit."""
        return self.total_tokens < 30_000_000


class LightningAdapter(LLMAdapter):
    """
    Lightning.ai unified LLM API adapter.
    
    Provides access to multiple models through a single API:
    - One API key for all models
    - 30M free tokens/month
    - Automatic fallback support
    - Usage tracking and cost estimation
    """
    
    FREE_TIER_LIMIT = 30_000_000  # 30M tokens/month
    
    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-V3",
        api_key: Optional[str] = None,
        fallback_model: Optional[str] = "meta-llama/Llama-3.1-8B-Instruct",
        task: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize Lightning.ai adapter.
        
        Args:
            model: Model to use (e.g., "deepseek/deepseek-v3")
            api_key: Lightning.ai API key (or set LIGHTNING_API_KEY env var)
            fallback_model: Model to use if primary fails
            task: Task type for automatic model selection
            temperature: Default sampling temperature
            max_tokens: Default max tokens for generation
        """
        # Auto-select model based on task
        if task and task in TASK_MODELS:
            model = TASK_MODELS[task]
            logger.info(f"Auto-selected model '{model}' for task '{task}'")
        
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        self.api_key = api_key or os.environ.get("LIGHTNING_API_KEY")
        self.fallback_model = fallback_model
        self.usage = UsageTracker()
        
        # Initialize client
        self._client = None
        self._use_openai_compat = False
        self._init_client()
        
        logger.info(f"LightningAdapter initialized with model: {self.model}")
    
    def _init_client(self):
        """Initialize the Lightning.ai client."""
        if not self.api_key:
            logger.warning(
                "LIGHTNING_API_KEY not set. Get your free API key at: "
                "https://lightning.ai/models"
            )
            return
        
        # Use OpenAI-compatible API (more reliable than litai)
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.lightning.ai/v1",
                api_key=self.api_key,
            )
            self._use_openai_compat = True
            logger.debug("Using OpenAI-compatible client for Lightning.ai")
        except ImportError:
            logger.error(
                "openai package not installed. "
                "Install with: pip install openai"
            )
            self._client = None
    
    @property
    def provider_name(self) -> str:
        return "lightning"
    
    @property
    def name(self) -> str:
        return f"lightning:{self.model}"
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion for a single prompt."""
        messages = [Message.user(prompt)]
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat completion request.
        
        Args:
            messages: List of Message objects
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content
        """
        if not self._client:
            raise RuntimeError(
                "Lightning.ai client not initialized. "
                "Set LIGHTNING_API_KEY environment variable."
            )
        
        # Convert messages to dict format
        msg_dicts = [
            {"role": msg.role.value if hasattr(msg.role, 'value') else msg.role, 
             "content": msg.content}
            for msg in messages
        ]
        
        try:
            return self._call_api(msg_dicts, temperature, max_tokens, **kwargs)
        except Exception as e:
            if self.fallback_model:
                logger.warning(f"Primary model failed: {e}, trying fallback")
                return self._call_with_fallback(msg_dicts, temperature, max_tokens, **kwargs)
            raise
    
    def _call_api(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Call the Lightning.ai API."""
        # Use OpenAI-compatible endpoint
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        content = response.choices[0].message.content
        
        # Track usage
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
            completion_tokens = getattr(response.usage, 'completion_tokens', 0)
            self.usage.add(prompt_tokens, completion_tokens)
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            tokens_used=prompt_tokens + completion_tokens,
        )
    
    def _call_with_fallback(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Call API with fallback model."""
        original_model = self.model
        self.model = self.fallback_model
        
        try:
            return self._call_api(messages, temperature, max_tokens, **kwargs)
        finally:
            self.model = original_model
    
    def get_model_for_task(self, task: str) -> str:
        """Get recommended model for a specific task."""
        return TASK_MODELS.get(task, self.model)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self.usage.total_tokens,
            "input_tokens": self.usage.total_input_tokens,
            "output_tokens": self.usage.total_output_tokens,
            "requests": self.usage.requests,
            "estimated_cost_usd": self.usage.estimate_cost(self.model),
            "within_free_tier": self.usage.is_within_free_tier(),
            "free_tier_remaining": max(0, self.FREE_TIER_LIMIT - self.usage.total_tokens),
        }
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a given number of tokens."""
        pricing = MODEL_PRICING.get(self.model, {"input": 1.0, "output": 1.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    @staticmethod
    def list_models() -> Dict[str, Dict[str, Any]]:
        """List available models with pricing."""
        return {
            model: {
                "input_cost_per_1m": pricing["input"],
                "output_cost_per_1m": pricing["output"],
                "recommended_for": [
                    task for task, m in TASK_MODELS.items() if m == model
                ]
            }
            for model, pricing in MODEL_PRICING.items()
        }


class LightningModelRouter:
    """
    Intelligent model router for Lightning.ai.
    
    Automatically selects the best model based on:
    - Task type
    - Budget constraints
    - Quality requirements
    - Token usage tracking
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("LIGHTNING_API_KEY")
        self._adapters: Dict[str, LightningAdapter] = {}
        self.total_usage = UsageTracker()
    
    def get_adapter(
        self,
        task: str = "chat",
        prefer_cheap: bool = True,
        prefer_quality: bool = False,
    ) -> LightningAdapter:
        """
        Get adapter for a specific task.
        
        Args:
            task: Task type (intent_parsing, workflow_generation, etc.)
            prefer_cheap: Prefer cheaper models
            prefer_quality: Prefer higher quality models
            
        Returns:
            Configured LightningAdapter
        """
        # Determine model based on preferences
        if prefer_quality:
            model = "anthropic/claude-3.5-sonnet"
        elif prefer_cheap:
            model = "deepseek/deepseek-v3"
        else:
            model = TASK_MODELS.get(task, "deepseek/deepseek-v3")
        
        # Cache adapters
        if model not in self._adapters:
            self._adapters[model] = LightningAdapter(
                model=model,
                api_key=self.api_key,
            )
        
        return self._adapters[model]
    
    def chat(
        self,
        messages: List[Message],
        task: str = "chat",
        **kwargs
    ) -> LLMResponse:
        """
        Route chat request to appropriate model.
        
        Args:
            messages: Chat messages
            task: Task type for model selection
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse from selected model
        """
        adapter = self.get_adapter(task)
        response = adapter.chat(messages, **kwargs)
        
        # Aggregate usage
        if adapter.usage.requests > 0:
            self.total_usage.add(
                adapter.usage.total_input_tokens,
                adapter.usage.total_output_tokens
            )
        
        return response
    
    def get_total_usage(self) -> Dict[str, Any]:
        """Get aggregated usage across all models."""
        total_cost = sum(
            adapter.usage.estimate_cost(model)
            for model, adapter in self._adapters.items()
        )
        
        return {
            "total_tokens": self.total_usage.total_tokens,
            "total_cost_usd": total_cost,
            "within_free_tier": self.total_usage.is_within_free_tier(),
            "models_used": list(self._adapters.keys()),
        }


# Convenience function
def create_lightning_llm(
    task: Optional[str] = None,
    model: Optional[str] = None,
) -> LightningAdapter:
    """
    Create a Lightning.ai LLM adapter.
    
    Args:
        task: Task type for auto model selection
        model: Specific model to use
        
    Returns:
        Configured LightningAdapter
    """
    if model:
        return LightningAdapter(model=model)
    elif task:
        return LightningAdapter(task=task)
    else:
        return LightningAdapter()  # Default to DeepSeek-V3
