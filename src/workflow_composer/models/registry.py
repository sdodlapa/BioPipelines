"""
Model and Provider Configuration Registry.

Provides centralized management of model definitions and provider configurations.
Supports both API providers (Lightning, Gemini, OpenAI) and local vLLM models.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ProviderType(Enum):
    """Type of model provider."""
    API = "api"
    LOCAL = "local"


class ModelCapability(Enum):
    """Capabilities a model can have."""
    CODE = "code"
    CHAT = "chat"
    FILL_IN_MIDDLE = "fill-in-middle"
    REASONING = "reasoning"
    AGENTIC = "agentic"
    VISION = "vision"


@dataclass
class VLLMArgs:
    """Arguments for vLLM server."""
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    dtype: str = "float16"
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    
    def to_cli_args(self) -> List[str]:
        """Convert to CLI arguments for vLLM."""
        args = [
            f"--tensor-parallel-size={self.tensor_parallel_size}",
            f"--max-model-len={self.max_model_len}",
            f"--dtype={self.dtype}",
            f"--gpu-memory-utilization={self.gpu_memory_utilization}",
        ]
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        return args


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    id: str
    name: str
    hf_id: str  # HuggingFace model ID
    provider_type: ProviderType
    size_gb: float
    gpus_required: int
    context_length: int
    capabilities: List[ModelCapability] = field(default_factory=list)
    vllm_args: Optional[VLLMArgs] = None
    priority: int = 100  # Higher = lower priority
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, id: str, data: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        capabilities = [
            ModelCapability(c) for c in data.get("capabilities", [])
        ]
        
        vllm_args = None
        if "vllm_args" in data:
            vllm_args = VLLMArgs(**data["vllm_args"])
            
        return cls(
            id=id,
            name=data.get("name", id),
            hf_id=data.get("hf_id", ""),
            provider_type=ProviderType(data.get("type", "local")),
            size_gb=data.get("size_gb", 0),
            gpus_required=data.get("gpus_required", 1),
            context_length=data.get("context_length", 8192),
            capabilities=capabilities,
            vllm_args=vllm_args,
            priority=data.get("priority", 100),
            enabled=data.get("enabled", True),
        )


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    id: str
    name: str
    provider_type: ProviderType
    priority: int  # Lower = higher priority
    env_key: Optional[str] = None  # Environment variable for API key
    base_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    free_tier: bool = False
    rate_limit: Optional[str] = None  # e.g., "100/min"
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, id: str, data: Dict[str, Any]) -> "ProviderConfig":
        """Create ProviderConfig from dictionary."""
        return cls(
            id=id,
            name=data.get("name", id),
            provider_type=ProviderType(data.get("type", "api")),
            priority=data.get("priority", 100),
            env_key=data.get("env_key"),
            base_url=data.get("base_url"),
            models=data.get("models", []),
            free_tier=data.get("free_tier", False),
            rate_limit=data.get("rate_limit"),
            enabled=data.get("enabled", True),
        )
    
    def is_available(self) -> bool:
        """Check if provider is available (has required credentials)."""
        if not self.enabled:
            return False
        if self.provider_type == ProviderType.API and self.env_key:
            return bool(os.environ.get(self.env_key))
        return True


class ModelRegistry:
    """
    Central registry for all models and providers.
    
    Loads configuration from YAML files and provides lookup methods.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the registry.
        
        Args:
            config_dir: Directory containing configuration files.
                        Defaults to src/workflow_composer/models/configs/
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        self.config_dir = Path(config_dir)
        
        self._models: Dict[str, ModelConfig] = {}
        self._providers: Dict[str, ProviderConfig] = {}
        
        self._load_default_configs()
        self._load_yaml_configs()
    
    def _load_default_configs(self):
        """Load default hardcoded configurations."""
        # Default providers
        default_providers = {
            "lightning": ProviderConfig(
                id="lightning",
                name="Lightning.ai",
                provider_type=ProviderType.API,
                priority=1,
                env_key="LIGHTNING_API_KEY",
                base_url="https://api.lightning.ai/v1",
                models=["llama-3.1-8b", "mistral-7b"],
                free_tier=True,
                rate_limit="100/min",
            ),
            "gemini": ProviderConfig(
                id="gemini",
                name="Google Gemini",
                provider_type=ProviderType.API,
                priority=2,
                env_key="GOOGLE_API_KEY",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                models=["gemini-2.0-flash", "gemini-2.5-pro"],
                free_tier=True,
                rate_limit="15/min",
            ),
            "github_copilot": ProviderConfig(
                id="github_copilot",
                name="GitHub Copilot",
                provider_type=ProviderType.API,
                priority=3,
                env_key="GITHUB_TOKEN",
                models=["copilot"],
                free_tier=False,
            ),
            "openai": ProviderConfig(
                id="openai",
                name="OpenAI",
                provider_type=ProviderType.API,
                priority=4,
                env_key="OPENAI_API_KEY",
                base_url="https://api.openai.com/v1",
                models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                free_tier=False,
                rate_limit="500/min",
            ),
            "vllm": ProviderConfig(
                id="vllm",
                name="vLLM Local",
                provider_type=ProviderType.LOCAL,
                priority=5,
                base_url="http://localhost:8000/v1",
                models=[
                    "qwen-coder-32b",
                    "deepseek-coder-v2",
                    "llama-3.3-70b",
                    "minimax-m2",
                    "codellama-34b",
                ],
            ),
        }
        self._providers.update(default_providers)
        
        # Default local models
        default_models = {
            "qwen-coder-32b": ModelConfig(
                id="qwen-coder-32b",
                name="Qwen2.5-Coder-32B-Instruct",
                hf_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                provider_type=ProviderType.LOCAL,
                size_gb=65,
                gpus_required=1,
                context_length=32768,
                capabilities=[
                    ModelCapability.CODE,
                    ModelCapability.CHAT,
                    ModelCapability.FILL_IN_MIDDLE,
                ],
                vllm_args=VLLMArgs(
                    tensor_parallel_size=1,
                    max_model_len=32768,
                    dtype="float16",
                ),
                priority=5,
            ),
            "deepseek-coder-v2": ModelConfig(
                id="deepseek-coder-v2",
                name="DeepSeek-Coder-V2-Instruct",
                hf_id="deepseek-ai/DeepSeek-Coder-V2-Instruct",
                provider_type=ProviderType.LOCAL,
                size_gb=120,
                gpus_required=2,
                context_length=128000,
                capabilities=[
                    ModelCapability.CODE,
                    ModelCapability.CHAT,
                    ModelCapability.FILL_IN_MIDDLE,
                    ModelCapability.REASONING,
                ],
                vllm_args=VLLMArgs(
                    tensor_parallel_size=2,
                    max_model_len=65536,
                    dtype="float16",
                ),
                priority=6,
            ),
            "llama-3.3-70b": ModelConfig(
                id="llama-3.3-70b",
                name="Llama-3.3-70B-Instruct",
                hf_id="meta-llama/Llama-3.3-70B-Instruct",
                provider_type=ProviderType.LOCAL,
                size_gb=140,
                gpus_required=2,
                context_length=128000,
                capabilities=[
                    ModelCapability.CODE,
                    ModelCapability.CHAT,
                    ModelCapability.REASONING,
                ],
                vllm_args=VLLMArgs(
                    tensor_parallel_size=2,
                    max_model_len=32768,
                    dtype="float16",
                ),
                priority=7,
            ),
            "minimax-m2": ModelConfig(
                id="minimax-m2",
                name="MiniMax-M2",
                hf_id="MiniMaxAI/MiniMax-M2",
                provider_type=ProviderType.LOCAL,
                size_gb=230,
                gpus_required=4,
                context_length=128000,
                capabilities=[
                    ModelCapability.CODE,
                    ModelCapability.CHAT,
                    ModelCapability.REASONING,
                    ModelCapability.AGENTIC,
                ],
                vllm_args=VLLMArgs(
                    tensor_parallel_size=4,
                    max_model_len=32768,
                    dtype="float8",
                    gpu_memory_utilization=0.95,
                ),
                priority=8,
            ),
            "codellama-34b": ModelConfig(
                id="codellama-34b",
                name="CodeLlama-34B-Instruct",
                hf_id="codellama/CodeLlama-34b-Instruct-hf",
                provider_type=ProviderType.LOCAL,
                size_gb=70,
                gpus_required=1,
                context_length=16384,
                capabilities=[
                    ModelCapability.CODE,
                    ModelCapability.FILL_IN_MIDDLE,
                ],
                vllm_args=VLLMArgs(
                    tensor_parallel_size=1,
                    max_model_len=16384,
                    dtype="float16",
                ),
                priority=9,
            ),
            # H100-optimized models (80GB VRAM each)
            "deepseek-coder-v2-lite": ModelConfig(
                id="deepseek-coder-v2-lite",
                name="DeepSeek-Coder-V2-Lite-Instruct",
                hf_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                provider_type=ProviderType.LOCAL,
                size_gb=32,
                gpus_required=1,
                context_length=128000,
                capabilities=[
                    ModelCapability.CODE,
                    ModelCapability.CHAT,
                    ModelCapability.FILL_IN_MIDDLE,
                    ModelCapability.REASONING,
                ],
                vllm_args=VLLMArgs(
                    tensor_parallel_size=1,
                    max_model_len=65536,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.90,
                ),
                priority=3,  # High priority for H100
            ),
            "qwen2.5-coder-32b-h100": ModelConfig(
                id="qwen2.5-coder-32b-h100",
                name="Qwen2.5-Coder-32B-H100",
                hf_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                provider_type=ProviderType.LOCAL,
                size_gb=65,
                gpus_required=1,
                context_length=131072,
                capabilities=[
                    ModelCapability.CODE,
                    ModelCapability.CHAT,
                    ModelCapability.FILL_IN_MIDDLE,
                    ModelCapability.AGENTIC,
                ],
                vllm_args=VLLMArgs(
                    tensor_parallel_size=1,
                    max_model_len=65536,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.90,
                ),
                priority=4,  # Great for H100 single GPU
            ),
            "llama-3.3-70b-h100": ModelConfig(
                id="llama-3.3-70b-h100",
                name="Llama-3.3-70B-H100-Optimized",
                hf_id="meta-llama/Llama-3.3-70B-Instruct",
                provider_type=ProviderType.LOCAL,
                size_gb=140,
                gpus_required=2,
                context_length=128000,
                capabilities=[
                    ModelCapability.CODE,
                    ModelCapability.CHAT,
                    ModelCapability.REASONING,
                    ModelCapability.AGENTIC,
                ],
                vllm_args=VLLMArgs(
                    tensor_parallel_size=2,
                    max_model_len=65536,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.90,
                ),
                priority=2,  # Primary model for H100 2-GPU setup
            ),
        }
        self._models.update(default_models)
    
    def _load_yaml_configs(self):
        """Load configurations from YAML files if they exist."""
        models_file = self.config_dir / "models.yaml"
        providers_file = self.config_dir / "providers.yaml"
        
        if models_file.exists():
            with open(models_file) as f:
                data = yaml.safe_load(f) or {}
                for model_id, model_data in data.get("models", {}).items():
                    self._models[model_id] = ModelConfig.from_dict(
                        model_id, model_data
                    )
        
        if providers_file.exists():
            with open(providers_file) as f:
                data = yaml.safe_load(f) or {}
                for provider_id, provider_data in data.get("providers", {}).items():
                    self._providers[provider_id] = ProviderConfig.from_dict(
                        provider_id, provider_data
                    )
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        return self._models.get(model_id)
    
    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get provider configuration by ID."""
        return self._providers.get(provider_id)
    
    def list_models(
        self,
        capability: Optional[ModelCapability] = None,
        enabled_only: bool = True,
    ) -> List[ModelConfig]:
        """
        List available models.
        
        Args:
            capability: Filter by capability
            enabled_only: Only return enabled models
            
        Returns:
            List of matching model configurations
        """
        models = list(self._models.values())
        
        if enabled_only:
            models = [m for m in models if m.enabled]
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        return sorted(models, key=lambda m: m.priority)
    
    def list_providers(
        self,
        provider_type: Optional[ProviderType] = None,
        available_only: bool = True,
    ) -> List[ProviderConfig]:
        """
        List available providers.
        
        Args:
            provider_type: Filter by type (API or LOCAL)
            available_only: Only return available providers
            
        Returns:
            List of matching provider configurations
        """
        providers = list(self._providers.values())
        
        if available_only:
            providers = [p for p in providers if p.is_available()]
        
        if provider_type:
            providers = [p for p in providers if p.provider_type == provider_type]
        
        return sorted(providers, key=lambda p: p.priority)
    
    def register_model(self, config: ModelConfig):
        """Register a new model."""
        self._models[config.id] = config
    
    def register_provider(self, config: ProviderConfig):
        """Register a new provider."""
        self._providers[config.id] = config


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
