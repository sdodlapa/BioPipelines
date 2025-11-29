"""
Orchestration Strategies
========================

Defines strategies for model selection and orchestration.

Strategies control how the orchestrator routes requests:
- AUTO: Smart selection based on task and availability
- LOCAL_FIRST: Prefer local, fallback to cloud
- LOCAL_ONLY: Only use local models (fail if unavailable)
- CLOUD_ONLY: Only use cloud models
- ENSEMBLE: Use multiple models and combine results
- PARALLEL: Race multiple models, use fastest
- CASCADE: Chain models in sequence (refinement)
- CHAIN: Use output of one model as input to another

Usage:
    from workflow_composer.llm.strategies import Strategy, EnsembleMode
    
    orchestrator = ModelOrchestrator(strategy=Strategy.LOCAL_FIRST)
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class Strategy(Enum):
    """Primary orchestration strategy."""
    
    AUTO = "auto"
    """Smart selection based on task type, cost, and availability."""
    
    LOCAL_FIRST = "local_first"
    """Prefer local GPU models, fallback to cloud if unavailable."""
    
    LOCAL_ONLY = "local_only"
    """Only use local models. Fail if unavailable."""
    
    CLOUD_ONLY = "cloud_only"
    """Only use cloud models."""
    
    ENSEMBLE = "ensemble"
    """Use multiple models and combine results."""
    
    PARALLEL = "parallel"
    """Race multiple models in parallel, use fastest response."""
    
    CASCADE = "cascade"
    """Try models in sequence until success (fallback chain)."""
    
    CHAIN = "chain"
    """Chain models: output of one feeds into next (refinement)."""


class EnsembleMode(Enum):
    """How to combine results in ensemble mode."""
    
    VOTE = "vote"
    """Majority voting (for classification/decisions)."""
    
    BEST = "best"
    """Select best result based on confidence/score."""
    
    MERGE = "merge"
    """Merge all results (for aggregation)."""
    
    CONSENSUS = "consensus"
    """Find common elements across responses."""


class ChainRole(Enum):
    """Role of a model in a chain."""
    
    DRAFT = "draft"
    """Initial fast draft generation."""
    
    REFINE = "refine"
    """Refine and improve the draft."""
    
    VERIFY = "verify"
    """Verify and validate the result."""
    
    SUMMARIZE = "summarize"
    """Summarize or extract key points."""


@dataclass
class StrategyConfig:
    """Configuration for orchestration strategy."""
    
    strategy: Strategy = Strategy.AUTO
    """Primary strategy to use."""
    
    fallback_enabled: bool = True
    """Enable fallback to other providers on failure."""
    
    max_retries: int = 2
    """Maximum retries per provider."""
    
    timeout_seconds: float = 60.0
    """Timeout for each request."""
    
    # Ensemble settings
    ensemble_mode: EnsembleMode = EnsembleMode.BEST
    """How to combine ensemble results."""
    
    ensemble_models: Optional[List[str]] = None
    """Specific models to use in ensemble (None = auto-select)."""
    
    min_ensemble_responses: int = 2
    """Minimum responses required for ensemble."""
    
    # Cost settings
    max_cost_per_request: Optional[float] = None
    """Maximum cost per request in dollars."""
    
    prefer_cheaper: bool = True
    """Prefer cheaper models when quality is similar."""
    
    # Performance settings
    prefer_faster: bool = False
    """Prefer faster models even if slightly lower quality."""
    
    cache_responses: bool = True
    """Cache responses for identical prompts."""


# Preset configurations for common scenarios
PRESETS = {
    "development": StrategyConfig(
        strategy=Strategy.LOCAL_FIRST,
        fallback_enabled=True,
        max_cost_per_request=0.01,
        prefer_cheaper=True,
    ),
    "production": StrategyConfig(
        strategy=Strategy.AUTO,
        fallback_enabled=True,
        max_retries=3,
        cache_responses=True,
    ),
    "critical": StrategyConfig(
        strategy=Strategy.ENSEMBLE,
        ensemble_mode=EnsembleMode.CONSENSUS,
        min_ensemble_responses=3,
        fallback_enabled=True,
    ),
    "cost_optimized": StrategyConfig(
        strategy=Strategy.LOCAL_ONLY,
        fallback_enabled=False,
        prefer_cheaper=True,
    ),
    "quality_first": StrategyConfig(
        strategy=Strategy.CLOUD_ONLY,
        prefer_cheaper=False,
        prefer_faster=False,
    ),
    "speed_first": StrategyConfig(
        strategy=Strategy.PARALLEL,
        prefer_faster=True,
        timeout_seconds=30.0,
    ),
}


def get_preset(name: str) -> StrategyConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESETS[name]


__all__ = [
    "Strategy",
    "EnsembleMode",
    "ChainRole",
    "StrategyConfig",
    "PRESETS",
    "get_preset",
]
