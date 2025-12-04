"""
Provider Model Registry - Central catalog of all LLM models.

This module provides unified access to all available models across providers.
It loads the configuration from provider_models.yaml and provides methods
to query models by various criteria.

Usage:
    from workflow_composer.providers.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    
    # Get all free models
    free_models = registry.get_free_models()
    
    # Get recommended model for task
    model = registry.get_recommended_model("code_generation")
    
    # Get all models from a provider
    groq_models = registry.get_provider_models("groq")
"""

from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model types."""
    TEXT = "text"
    AUDIO = "audio"
    TTS = "tts"
    IMAGE = "image"
    EMBEDDING = "embedding"
    REALTIME = "realtime"


class TaskType(Enum):
    """Task types for model recommendations."""
    QUICK_RESPONSE = "quick_response"
    INTENT_PARSING = "intent_parsing"
    CODE_GENERATION = "code_generation"
    COMPLEX_REASONING = "complex_reasoning"
    SCIENTIFIC_ANALYSIS = "scientific_analysis"
    WORKFLOW_GENERATION = "workflow_generation"
    LONG_CONTEXT = "long_context"
    CONTENT_MODERATION = "content_moderation"
    TRANSCRIPTION = "transcription"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    HIGHEST_QUALITY = "highest_quality"


@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    provider: str
    display_name: str
    context_length: int = 8192
    input_price: float = 0.0  # per 1M tokens
    output_price: float = 0.0  # per 1M tokens
    speed: Optional[int] = None  # tokens/sec
    free_tier: bool = False
    preview: bool = False
    experimental: bool = False
    model_type: ModelType = ModelType.TEXT
    best_for: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    rate_limits: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_id(self) -> str:
        """Return full model ID with provider prefix."""
        return f"{self.provider}:{self.id}"
    
    @property
    def is_free(self) -> bool:
        """Check if model is free to use."""
        return self.free_tier or self.id.endswith(":free")
    
    def __str__(self) -> str:
        free_marker = " (FREE)" if self.is_free else ""
        return f"{self.display_name}{free_marker} [{self.provider}]"


@dataclass
class ProviderInfo:
    """Information about a provider."""
    name: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    sdk: Optional[str] = None
    priority: int = 50
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    
    @property
    def model_count(self) -> int:
        return len(self.models)
    
    @property
    def free_model_count(self) -> int:
        return sum(1 for m in self.models.values() if m.is_free)


class ModelRegistry:
    """
    Central registry for all LLM models across providers.
    
    This class loads model configurations from provider_models.yaml
    and provides methods to query and select models based on various criteria.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the model registry.
        
        Args:
            config_path: Path to provider_models.yaml. If None, uses default location.
        """
        self._providers: Dict[str, ProviderInfo] = {}
        self._models: Dict[str, ModelInfo] = {}  # full_id -> ModelInfo
        self._task_recommendations: Dict[str, Dict[str, Any]] = {}
        
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "provider_models.yaml"
        
        self._load_config(config_path)
    
    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        if not config_path.exists():
            logger.warning(f"Model catalog config not found: {config_path}")
            return
        
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load model catalog: {e}")
            return
        
        # Load task recommendations
        if "task_recommendations" in config:
            self._task_recommendations = config.pop("task_recommendations")
        
        # Load providers and models
        for provider_name, provider_config in config.items():
            if not isinstance(provider_config, dict):
                continue
            
            provider = ProviderInfo(
                name=provider_name,
                base_url=provider_config.get("base_url"),
                api_key_env=provider_config.get("api_key_env"),
                sdk=provider_config.get("sdk"),
                priority=provider_config.get("priority", 50),
            )
            
            models_config = provider_config.get("models", {})
            for model_id, model_config in models_config.items():
                if not isinstance(model_config, dict):
                    continue
                
                # Determine model type
                model_type_str = model_config.get("type", "text")
                try:
                    model_type = ModelType(model_type_str)
                except ValueError:
                    model_type = ModelType.TEXT
                
                model = ModelInfo(
                    id=model_id,
                    provider=provider_name,
                    display_name=model_config.get("display_name", model_id),
                    context_length=model_config.get("context", model_config.get("context_length", 8192)),
                    input_price=model_config.get("input_price", 0.0),
                    output_price=model_config.get("output_price", 0.0),
                    speed=model_config.get("speed"),
                    free_tier=model_config.get("free_tier", model_config.get("free", model_config.get("free_only", False))),
                    preview=model_config.get("preview", False),
                    experimental=model_config.get("experimental", False),
                    model_type=model_type,
                    best_for=model_config.get("best_for", []),
                    capabilities=model_config.get("capabilities", []),
                    rate_limits={
                        "tpm": model_config.get("tpm"),
                        "rpm": model_config.get("rpm"),
                        "rpd": model_config.get("rpd", model_config.get("free_rpd")),
                    },
                    extra={k: v for k, v in model_config.items() if k not in {
                        "display_name", "context", "context_length", "input_price", "output_price",
                        "speed", "free_tier", "free", "free_only", "preview", "experimental",
                        "type", "best_for", "capabilities", "tpm", "rpm", "rpd", "free_rpd"
                    }}
                )
                
                provider.models[model_id] = model
                self._models[model.full_id] = model
            
            self._providers[provider_name] = provider
        
        logger.info(
            f"Loaded {len(self._models)} models from {len(self._providers)} providers"
        )
    
    # =========================================================================
    # Provider queries
    # =========================================================================
    
    def get_provider(self, name: str) -> Optional[ProviderInfo]:
        """Get provider info by name."""
        return self._providers.get(name)
    
    def get_providers(self) -> List[ProviderInfo]:
        """Get all providers."""
        return list(self._providers.values())
    
    def get_provider_names(self) -> List[str]:
        """Get list of provider names."""
        return list(self._providers.keys())
    
    # =========================================================================
    # Model queries
    # =========================================================================
    
    def get_model(self, full_id: str) -> Optional[ModelInfo]:
        """
        Get model by full ID (provider:model_id).
        
        Args:
            full_id: Full model ID like "groq:llama-3.1-8b-instant"
        """
        return self._models.get(full_id)
    
    def get_provider_models(self, provider: str) -> List[ModelInfo]:
        """Get all models from a specific provider."""
        provider_info = self._providers.get(provider)
        if not provider_info:
            return []
        return list(provider_info.models.values())
    
    def get_all_models(self) -> List[ModelInfo]:
        """Get all models across all providers."""
        return list(self._models.values())
    
    def get_free_models(self) -> List[ModelInfo]:
        """Get all free models."""
        return [m for m in self._models.values() if m.is_free]
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """Get models by type (text, audio, embedding, etc.)."""
        return [m for m in self._models.values() if m.model_type == model_type]
    
    def get_models_with_capability(self, capability: str) -> List[ModelInfo]:
        """Get models with a specific capability."""
        return [m for m in self._models.values() if capability in m.capabilities]
    
    def get_models_for_task(self, task: str) -> List[ModelInfo]:
        """Get models best suited for a specific task."""
        return [m for m in self._models.values() if task in m.best_for]
    
    def get_fastest_models(self, min_speed: int = 500) -> List[ModelInfo]:
        """Get models with speed above threshold."""
        return sorted(
            [m for m in self._models.values() if m.speed and m.speed >= min_speed],
            key=lambda m: m.speed or 0,
            reverse=True
        )
    
    def get_cheapest_models(self, max_price: float = 1.0) -> List[ModelInfo]:
        """Get models with input price below threshold (per 1M tokens)."""
        return sorted(
            [m for m in self._models.values() if m.input_price <= max_price],
            key=lambda m: m.input_price
        )
    
    def get_long_context_models(self, min_context: int = 100000) -> List[ModelInfo]:
        """Get models with context length above threshold."""
        return sorted(
            [m for m in self._models.values() if m.context_length >= min_context],
            key=lambda m: m.context_length,
            reverse=True
        )
    
    # =========================================================================
    # Task-based recommendations
    # =========================================================================
    
    def get_recommended_model(
        self, 
        task: str | TaskType,
        prefer_free: bool = True,
        available_providers: Optional[List[str]] = None
    ) -> Optional[ModelInfo]:
        """
        Get recommended model for a task.
        
        Args:
            task: Task type (e.g., "code_generation", "intent_parsing")
            prefer_free: If True, prefer free models when available
            available_providers: List of available provider names (ones with valid API keys)
        """
        if isinstance(task, TaskType):
            task = task.value
        
        recommendations = self._task_recommendations.get(task, {})
        if not recommendations:
            logger.warning(f"No recommendations for task: {task}")
            return None
        
        # Try primary first, then fallbacks
        candidates = [recommendations.get("primary")] + recommendations.get("fallback", [])
        
        for model_id in candidates:
            if not model_id:
                continue
            
            model = self.get_model(model_id)
            if not model:
                continue
            
            # Check if provider is available
            if available_providers and model.provider not in available_providers:
                continue
            
            # Check free preference
            if prefer_free and not model.is_free:
                # Keep looking for free options
                continue
            
            return model
        
        # If prefer_free didn't find anything, return first available
        for model_id in candidates:
            if not model_id:
                continue
            model = self.get_model(model_id)
            if model:
                if available_providers and model.provider not in available_providers:
                    continue
                return model
        
        return None
    
    def get_fallback_chain(
        self, 
        task: str | TaskType
    ) -> List[ModelInfo]:
        """
        Get ordered list of models for a task (primary + fallbacks).
        
        Useful for implementing retry logic with fallback providers.
        """
        if isinstance(task, TaskType):
            task = task.value
        
        recommendations = self._task_recommendations.get(task, {})
        if not recommendations:
            return []
        
        candidates = [recommendations.get("primary")] + recommendations.get("fallback", [])
        
        models = []
        for model_id in candidates:
            if model_id:
                model = self.get_model(model_id)
                if model:
                    models.append(model)
        
        return models
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the model catalog."""
        all_models = list(self._models.values())
        free_models = [m for m in all_models if m.is_free]
        
        provider_stats = {}
        for provider in self._providers.values():
            provider_stats[provider.name] = {
                "total": provider.model_count,
                "free": provider.free_model_count,
            }
        
        return {
            "total_providers": len(self._providers),
            "total_models": len(all_models),
            "free_models": len(free_models),
            "paid_models": len(all_models) - len(free_models),
            "providers": provider_stats,
            "by_type": {
                t.value: len([m for m in all_models if m.model_type == t])
                for t in ModelType
            },
            "avg_context_length": sum(m.context_length for m in all_models) // len(all_models) if all_models else 0,
            "max_context_length": max(m.context_length for m in all_models) if all_models else 0,
            "fastest_speed": max(m.speed for m in all_models if m.speed) if any(m.speed for m in all_models) else 0,
        }
    
    def print_summary(self) -> None:
        """Print a summary of the model catalog."""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("BioPipelines Model Catalog Summary")
        print("=" * 60)
        print(f"\nTotal Providers: {stats['total_providers']}")
        print(f"Total Models: {stats['total_models']}")
        print(f"  - Free: {stats['free_models']}")
        print(f"  - Paid: {stats['paid_models']}")
        print(f"\nMax Context: {stats['max_context_length']:,} tokens")
        print(f"Fastest Speed: {stats['fastest_speed']:,} tok/s")
        
        print("\nBy Provider:")
        for name, info in stats['providers'].items():
            print(f"  {name}: {info['total']} models ({info['free']} free)")
        
        print("\nBy Type:")
        for type_name, count in stats['by_type'].items():
            if count > 0:
                print(f"  {type_name}: {count}")
        
        print("=" * 60 + "\n")


# Singleton instance for easy access
_registry: Optional[ModelRegistry] = None

def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


# Convenience functions
def get_model(full_id: str) -> Optional[ModelInfo]:
    """Get model by full ID (provider:model_id)."""
    return get_registry().get_model(full_id)


def get_free_models() -> List[ModelInfo]:
    """Get all free models."""
    return get_registry().get_free_models()


def get_recommended_model(task: str, **kwargs) -> Optional[ModelInfo]:
    """Get recommended model for a task."""
    return get_registry().get_recommended_model(task, **kwargs)


if __name__ == "__main__":
    # Test the registry
    registry = ModelRegistry()
    registry.print_summary()
    
    print("\nFree models:")
    for model in registry.get_free_models()[:10]:
        print(f"  - {model}")
    
    print("\nFastest models (>500 tok/s):")
    for model in registry.get_fastest_models(500)[:5]:
        print(f"  - {model}: {model.speed} tok/s")
    
    print("\nCode generation recommendations:")
    chain = registry.get_fallback_chain("code_generation")
    for i, model in enumerate(chain):
        print(f"  {i+1}. {model}")
