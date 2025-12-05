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

Strategy Profiles (v2.0):
- Profiles are YAML-based configurations loaded at runtime
- Support for dynamic strategy switching via switch_strategy()
- Data governance via allow_cloud flag

Usage:
    from workflow_composer.llm.strategies import Strategy, StrategyConfig
    
    # Basic usage
    orchestrator = ModelOrchestrator(strategy=Strategy.LOCAL_FIRST)
    
    # Profile-based usage (v2.0)
    config = StrategyConfig.from_yaml("config/strategies/t4_hybrid.yaml")
    orchestrator = ModelOrchestrator(config=config)
"""

from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import yaml for profile loading
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


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
    """
    Configuration for orchestration strategy.
    
    Extended in v2.0 with profile support, data governance, and vLLM endpoints.
    """
    
    # === Core Strategy Settings ===
    strategy: Strategy = Strategy.AUTO
    """Primary strategy to use."""
    
    fallback_enabled: bool = True
    """Enable fallback to other providers on failure."""
    
    max_retries: int = 2
    """Maximum retries per provider."""
    
    timeout_seconds: float = 60.0
    """Timeout for each request."""
    
    # === Profile Settings (v2.0) ===
    profile_name: Optional[str] = None
    """Name of the loaded profile (e.g., 't4_hybrid', 'cloud_only')."""
    
    profile_description: Optional[str] = None
    """Human-readable description of the profile."""
    
    # === Data Governance (v2.0) ===
    allow_cloud: bool = True
    """Allow requests to cloud APIs. Set False for PHI/sensitive data."""
    
    allow_cloud_for_tasks: Optional[List[str]] = None
    """Override allow_cloud per task type. If set, only these tasks can use cloud."""
    
    # === Debugging (v2.0) ===
    debug_routing: bool = False
    """Enable verbose routing decision logging."""
    
    # === vLLM Endpoints (v2.0) ===
    vllm_endpoints: Dict[str, str] = field(default_factory=dict)
    """Mapping of model_key -> URL for local vLLM servers."""
    
    task_routing: Dict[str, str] = field(default_factory=dict)
    """Mapping of task_type -> model_key for routing decisions."""
    
    # === Fallback Configuration ===
    fallback_chain: List[str] = field(default_factory=lambda: ["deepseek-v3", "claude-3.5-sonnet"])
    """Ordered list of fallback models/providers."""
    
    # === Ensemble Settings ===
    ensemble_mode: EnsembleMode = EnsembleMode.BEST
    """How to combine ensemble results."""
    
    ensemble_models: Optional[List[str]] = None
    """Specific models to use in ensemble (None = auto-select)."""
    
    min_ensemble_responses: int = 2
    """Minimum responses required for ensemble."""
    
    # === Cost Settings ===
    max_cost_per_request: Optional[float] = None
    """Maximum cost per request in dollars."""
    
    prefer_cheaper: bool = True
    """Prefer cheaper models when quality is similar."""
    
    # === Performance Settings ===
    prefer_faster: bool = False
    """Prefer faster models even if slightly lower quality."""
    
    cache_responses: bool = True
    """Cache responses for identical prompts."""
    
    # === Methods ===
    
    def can_use_cloud_for_task(self, task_type: str) -> bool:
        """
        Check if cloud APIs can be used for a specific task.
        
        Respects both global allow_cloud and per-task overrides.
        """
        if not self.allow_cloud:
            return False
        
        if self.allow_cloud_for_tasks is None:
            return True
        
        return task_type in self.allow_cloud_for_tasks
    
    def get_model_for_task(self, task_type: str) -> Optional[str]:
        """
        Get the configured model key for a task type.
        
        Returns None if no specific routing is configured.
        """
        return self.task_routing.get(task_type)
    
    def get_vllm_url(self, model_key: str) -> Optional[str]:
        """Get vLLM URL for a model key."""
        return self.vllm_endpoints.get(model_key)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "StrategyConfig":
        """
        Load strategy configuration from YAML file.
        
        Args:
            path: Path to YAML file (absolute or relative to config/strategies/)
        
        Returns:
            StrategyConfig instance
        
        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        if not HAS_YAML:
            raise ImportError("PyYAML required for profile loading. Install: pip install pyyaml")
        
        path = Path(path)
        
        # If not absolute, try config/strategies/
        if not path.is_absolute():
            # Try relative to current dir first
            if not path.exists():
                # Try config/strategies/
                config_path = Path("config/strategies") / path
                if config_path.exists():
                    path = config_path
                else:
                    # Try with .yaml extension
                    path = Path("config/strategies") / f"{path}.yaml"
        
        if not path.exists():
            raise FileNotFoundError(f"Strategy profile not found: {path}")
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Invalid profile format: expected dict, got {type(data)}")
        
        # Map YAML fields to dataclass fields
        config_data = {}
        
        # Simple string/bool/number fields
        simple_fields = [
            "profile_name", "profile_description", "allow_cloud", 
            "debug_routing", "fallback_enabled", "max_retries",
            "timeout_seconds", "max_cost_per_request", "prefer_cheaper",
            "prefer_faster", "cache_responses", "min_ensemble_responses",
        ]
        for field_name in simple_fields:
            if field_name in data:
                config_data[field_name] = data[field_name]
        
        # Strategy enum
        if "strategy" in data:
            config_data["strategy"] = Strategy(data["strategy"])
        
        # Ensemble mode enum
        if "ensemble_mode" in data:
            config_data["ensemble_mode"] = EnsembleMode(data["ensemble_mode"])
        
        # Dict fields
        if "vllm_endpoints" in data:
            config_data["vllm_endpoints"] = data["vllm_endpoints"]
        if "task_routing" in data:
            config_data["task_routing"] = data["task_routing"]
        
        # List fields
        if "fallback_chain" in data:
            config_data["fallback_chain"] = data["fallback_chain"]
        if "ensemble_models" in data:
            config_data["ensemble_models"] = data["ensemble_models"]
        if "allow_cloud_for_tasks" in data:
            config_data["allow_cloud_for_tasks"] = data["allow_cloud_for_tasks"]
        
        return cls(**config_data)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required. Install: pip install pyyaml")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "profile_name": self.profile_name,
            "profile_description": self.profile_description,
            "strategy": self.strategy.value,
            "allow_cloud": self.allow_cloud,
            "debug_routing": self.debug_routing,
            "fallback_enabled": self.fallback_enabled,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "vllm_endpoints": self.vllm_endpoints,
            "task_routing": self.task_routing,
            "fallback_chain": self.fallback_chain,
        }
        
        # Only include optional fields if set
        if self.max_cost_per_request is not None:
            data["max_cost_per_request"] = self.max_cost_per_request
        if self.allow_cloud_for_tasks is not None:
            data["allow_cloud_for_tasks"] = self.allow_cloud_for_tasks
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved strategy profile to {path}")


# Preset configurations for common scenarios
PRESETS = {
    "development": StrategyConfig(
        profile_name="development",
        profile_description="Development mode - fast iteration, minimal cost",
        strategy=Strategy.LOCAL_FIRST,
        fallback_enabled=True,
        max_cost_per_request=0.01,
        prefer_cheaper=True,
        debug_routing=True,
    ),
    "production": StrategyConfig(
        profile_name="production",
        profile_description="Production mode - reliable, auto-fallback",
        strategy=Strategy.AUTO,
        fallback_enabled=True,
        max_retries=3,
        cache_responses=True,
    ),
    "critical": StrategyConfig(
        profile_name="critical",
        profile_description="Critical tasks - ensemble voting for reliability",
        strategy=Strategy.ENSEMBLE,
        ensemble_mode=EnsembleMode.CONSENSUS,
        min_ensemble_responses=3,
        fallback_enabled=True,
    ),
    "cost_optimized": StrategyConfig(
        profile_name="cost_optimized",
        profile_description="Cost optimized - local only, no cloud spending",
        strategy=Strategy.LOCAL_ONLY,
        fallback_enabled=False,
        prefer_cheaper=True,
        allow_cloud=False,
    ),
    "quality_first": StrategyConfig(
        profile_name="quality_first",
        profile_description="Quality first - best models regardless of cost",
        strategy=Strategy.CLOUD_ONLY,
        prefer_cheaper=False,
        prefer_faster=False,
    ),
    "speed_first": StrategyConfig(
        profile_name="speed_first",
        profile_description="Speed first - parallel execution, fastest wins",
        strategy=Strategy.PARALLEL,
        prefer_faster=True,
        timeout_seconds=30.0,
    ),
    # v2.0 Profile-based presets
    "t4_hybrid": StrategyConfig(
        profile_name="t4_hybrid",
        profile_description="T4 vLLM fleet with cloud fallback",
        strategy=Strategy.LOCAL_FIRST,
        fallback_enabled=True,
        allow_cloud=True,
        vllm_endpoints={
            "generalist": "http://localhost:8001",
            "coder": "http://localhost:8002",
            "math": "http://localhost:8003",
            "embeddings": "http://localhost:8004",
        },
        task_routing={
            "intent_parsing": "generalist",
            "code_generation": "coder",
            "code_validation": "coder",
            "data_analysis": "generalist",
            "math_statistics": "math",
            "documentation": "generalist",
            "biomedical": "generalist",
            "safety": "generalist",
            "embeddings": "embeddings",
        },
        fallback_chain=["deepseek-v3", "claude-3.5-sonnet"],
    ),
    "t4_local_only": StrategyConfig(
        profile_name="t4_local_only",
        profile_description="T4 vLLM fleet only - no cloud (PHI safe)",
        strategy=Strategy.LOCAL_ONLY,
        fallback_enabled=False,
        allow_cloud=False,
        vllm_endpoints={
            "generalist": "http://localhost:8001",
            "coder": "http://localhost:8002",
            "math": "http://localhost:8003",
            "embeddings": "http://localhost:8004",
        },
        task_routing={
            "intent_parsing": "generalist",
            "code_generation": "coder",
            "code_validation": "coder",
            "data_analysis": "generalist",
            "math_statistics": "math",
            "documentation": "generalist",
            "biomedical": "generalist",
            "safety": "generalist",
            "embeddings": "embeddings",
        },
    ),
    "cloud_only": StrategyConfig(
        profile_name="cloud_only",
        profile_description="Cloud APIs only - no local GPUs",
        strategy=Strategy.CLOUD_ONLY,
        fallback_enabled=True,
        allow_cloud=True,
        vllm_endpoints={},
        fallback_chain=["deepseek-v3", "gpt-4o", "claude-3.5-sonnet"],
    ),
}


def get_preset(name: str) -> StrategyConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESETS[name]


def load_profile(name_or_path: str) -> StrategyConfig:
    """
    Load a strategy profile by name or path.
    
    Tries in order:
    1. Built-in PRESETS (e.g., "t4_hybrid", "development")
    2. YAML file in config/strategies/ (e.g., "custom.yaml")
    3. Absolute path to YAML file
    
    Args:
        name_or_path: Preset name or path to YAML file
    
    Returns:
        StrategyConfig instance
    """
    # Try preset first
    if name_or_path in PRESETS:
        logger.debug(f"Loading preset: {name_or_path}")
        return PRESETS[name_or_path]
    
    # Try as YAML file
    logger.debug(f"Loading profile from file: {name_or_path}")
    return StrategyConfig.from_yaml(name_or_path)


__all__ = [
    "Strategy",
    "EnsembleMode",
    "ChainRole",
    "StrategyConfig",
    "PRESETS",
    "get_preset",
    "load_profile",
]
