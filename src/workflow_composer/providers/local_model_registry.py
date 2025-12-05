"""
Local Model Registry for BioPipelines
=====================================

Provides access to the local model catalog for specialized tasks.
Maps task categories to optimal <10B models that can run on L4 GPUs.

Usage:
    from workflow_composer.providers.local_model_registry import (
        get_model_for_task,
        get_deployment_config,
        TaskCategory,
    )
    
    # Get recommended model for code generation
    model = get_model_for_task(TaskCategory.CODE_GENERATION)
    print(model["huggingface_id"])  # "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Get multi-model deployment configuration
    config = get_deployment_config("standard_l4")
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TaskCategory(str, Enum):
    """Task categories for model selection."""
    INTENT_PARSING = "intent_parsing"
    CODE_GENERATION = "code_generation"
    CODE_VALIDATION = "code_validation"
    DATA_ANALYSIS = "data_analysis"
    ORCHESTRATION = "orchestration"
    DOCUMENTATION = "documentation"
    MATH_STATISTICS = "math_statistics"
    BIO_MEDICAL = "bio_medical"
    EMBEDDINGS = "embeddings"
    SAFETY = "safety"


@dataclass
class LocalModel:
    """A local model specification."""
    name: str
    huggingface_id: str
    parameters: str
    vram_fp16: str
    vram_int8: Optional[str] = None
    context_length: Optional[str] = None
    license: Optional[str] = None
    ollama_cmd: Optional[str] = None
    vllm_cmd: Optional[str] = None
    download_cmd: Optional[str] = None
    strengths: List[str] = None
    
    def __post_init__(self):
        if self.strengths is None:
            self.strengths = []
    
    def get_vram(self, precision: str = "fp16") -> str:
        """Get VRAM requirement for given precision."""
        if precision == "int8" and self.vram_int8:
            return self.vram_int8
        return self.vram_fp16
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "huggingface_id": self.huggingface_id,
            "parameters": self.parameters,
            "vram_fp16": self.vram_fp16,
            "vram_int8": self.vram_int8,
            "context_length": self.context_length,
            "license": self.license,
            "ollama_cmd": self.ollama_cmd,
            "strengths": self.strengths,
        }


class LocalModelRegistry:
    """
    Registry of local models for different task categories.
    
    Loads model configurations from config/local_model_catalog.yaml
    and provides methods to select optimal models for each task.
    """
    
    def __init__(self, catalog_path: Optional[Path] = None):
        """Initialize registry from catalog file."""
        if catalog_path is None:
            # Default to config directory
            catalog_path = Path(__file__).parent.parent.parent.parent / "config" / "local_model_catalog.yaml"
        
        self.catalog_path = catalog_path
        self._catalog: Dict[str, Any] = {}
        self._models_cache: Dict[str, LocalModel] = {}
        
        self._load_catalog()
    
    def _load_catalog(self) -> None:
        """Load the model catalog from YAML."""
        if not self.catalog_path.exists():
            logger.warning(f"Local model catalog not found: {self.catalog_path}")
            return
        
        try:
            with open(self.catalog_path, 'r') as f:
                self._catalog = yaml.safe_load(f) or {}
            logger.info(f"Loaded local model catalog with {len(self._catalog)} categories")
        except Exception as e:
            logger.error(f"Failed to load local model catalog: {e}")
    
    def get_model_for_task(
        self, 
        task: TaskCategory,
        fallback: bool = False,
        compact: bool = False,
    ) -> Optional[LocalModel]:
        """
        Get the recommended model for a task category.
        
        Args:
            task: The task category
            fallback: If True, return fallback model instead of primary
            compact: If True, return compact model (if available)
            
        Returns:
            LocalModel or None if not found
        """
        category_data = self._catalog.get(task.value, {})
        
        if compact and "compact" in category_data:
            model_data = category_data["compact"]
        elif fallback and "fallback" in category_data:
            model_data = category_data["fallback"]
        elif "primary" in category_data:
            model_data = category_data["primary"]
        else:
            return None
        
        return self._parse_model(model_data)
    
    def _parse_model(self, data: Dict[str, Any]) -> LocalModel:
        """Parse model data into LocalModel object."""
        return LocalModel(
            name=data.get("name", "Unknown"),
            huggingface_id=data.get("huggingface_id", ""),
            parameters=str(data.get("parameters", "Unknown")),
            vram_fp16=data.get("vram_fp16", "Unknown"),
            vram_int8=data.get("vram_int8"),
            context_length=data.get("context_length"),
            license=data.get("license"),
            ollama_cmd=data.get("ollama_cmd"),
            vllm_cmd=data.get("vllm_cmd"),
            download_cmd=data.get("download_cmd"),
            strengths=data.get("strengths", []),
        )
    
    def get_deployment_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a deployment configuration.
        
        Args:
            config_name: One of "minimal_l4", "standard_l4", "multi_gpu"
            
        Returns:
            Deployment configuration dict
        """
        configs = self._catalog.get("deployment_configs", {})
        return configs.get(config_name)
    
    def get_all_models(self) -> List[LocalModel]:
        """Get all models in the catalog."""
        models = []
        for category in TaskCategory:
            for variant in ["primary", "fallback", "compact"]:
                category_data = self._catalog.get(category.value, {})
                if variant in category_data:
                    models.append(self._parse_model(category_data[variant]))
        return models
    
    def get_models_by_vram(self, max_vram_gb: float) -> List[LocalModel]:
        """Get models that fit within VRAM limit."""
        def parse_vram(vram_str: str) -> float:
            """Parse VRAM string like '~16GB' to float."""
            try:
                # Remove ~ and GB, convert to float
                return float(vram_str.replace("~", "").replace("GB", "").strip())
            except:
                return float('inf')
        
        models = self.get_all_models()
        return [m for m in models if parse_vram(m.vram_fp16) <= max_vram_gb]
    
    def get_quick_reference(self) -> Dict[str, Dict[str, Any]]:
        """Get the quick reference table."""
        return self._catalog.get("quick_reference", {})
    
    def get_vram_summary(self) -> Dict[str, Dict[str, str]]:
        """Get VRAM requirements summary."""
        return self._catalog.get("vram_summary", {})
    
    def get_ollama_models(self) -> List[Dict[str, str]]:
        """Get models available via Ollama."""
        result = []
        for model in self.get_all_models():
            if model.ollama_cmd:
                result.append({
                    "name": model.name,
                    "ollama_cmd": model.ollama_cmd,
                    "huggingface_id": model.huggingface_id,
                })
        return result


# Module-level registry instance
_registry: Optional[LocalModelRegistry] = None


def get_local_registry() -> LocalModelRegistry:
    """Get or create the local model registry singleton."""
    global _registry
    if _registry is None:
        _registry = LocalModelRegistry()
    return _registry


def get_model_for_task(
    task: TaskCategory,
    fallback: bool = False,
    compact: bool = False,
) -> Optional[LocalModel]:
    """
    Convenience function to get model for a task.
    
    Args:
        task: The task category
        fallback: If True, return fallback model
        compact: If True, return compact model
        
    Returns:
        LocalModel or None
    """
    return get_local_registry().get_model_for_task(task, fallback, compact)


def get_deployment_config(config_name: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get deployment config.
    
    Args:
        config_name: One of "minimal_l4", "standard_l4", "multi_gpu"
        
    Returns:
        Configuration dict
    """
    return get_local_registry().get_deployment_config(config_name)


def recommend_models_for_vram(vram_gb: float) -> Dict[TaskCategory, LocalModel]:
    """
    Recommend models for each task given VRAM constraint.
    
    Args:
        vram_gb: Available VRAM in GB
        
    Returns:
        Dict mapping task categories to recommended models
    """
    registry = get_local_registry()
    recommendations = {}
    
    for task in TaskCategory:
        # Try primary first
        model = registry.get_model_for_task(task, fallback=False)
        if model:
            vram = float(model.vram_fp16.replace("~", "").replace("GB", ""))
            if vram <= vram_gb:
                recommendations[task] = model
                continue
        
        # Try fallback
        model = registry.get_model_for_task(task, fallback=True)
        if model:
            vram = float(model.vram_fp16.replace("~", "").replace("GB", ""))
            if vram <= vram_gb:
                recommendations[task] = model
                continue
        
        # Try compact
        model = registry.get_model_for_task(task, compact=True)
        if model:
            recommendations[task] = model
    
    return recommendations


# Convenience exports
__all__ = [
    "TaskCategory",
    "LocalModel",
    "LocalModelRegistry",
    "get_local_registry",
    "get_model_for_task",
    "get_deployment_config",
    "recommend_models_for_vram",
]
