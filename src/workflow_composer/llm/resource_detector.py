"""
Resource Detector
=================

Detects available LLM resources (vLLM endpoints, cloud APIs, SLURM).
Uses simple health checks rather than GPU introspection for reliability.

Design Philosophy:
- Health-check based: "Is the endpoint alive?" vs "What GPU is available?"
- Fast: All checks should complete in <5 seconds
- Non-blocking: Failures don't block startup

Usage:
    from workflow_composer.llm.resource_detector import ResourceDetector, ResourceStatus
    
    detector = ResourceDetector()
    status = detector.detect()
    
    if status.has_local_models:
        print(f"Local models available: {status.available_models}")
    if status.has_cloud_apis:
        print(f"Cloud APIs configured: {status.available_cloud_apis}")
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import logging

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


# Default vLLM endpoints (configured in deploy_core_models.sh)
DEFAULT_VLLM_ENDPOINTS = {
    "generalist": "http://localhost:8001",
    "coder": "http://localhost:8002", 
    "math": "http://localhost:8003",
    "embeddings": "http://localhost:8004",
}

# Cloud API environment variable names
CLOUD_API_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}

# Connection info directory (created by deploy_core_models.sh)
CONNECTION_DIR = Path.home() / "BioPipelines" / ".model_connections"


@dataclass
class ResourceStatus:
    """Status of available LLM resources."""
    
    # vLLM endpoints
    vllm_endpoints: Dict[str, bool] = field(default_factory=dict)
    """Mapping of model_key -> is_healthy for each vLLM endpoint."""
    
    vllm_urls: Dict[str, str] = field(default_factory=dict)
    """Mapping of model_key -> actual URL (from connection files or defaults)."""
    
    # Cloud APIs
    cloud_apis: Dict[str, bool] = field(default_factory=dict)
    """Mapping of api_name -> has_key for each cloud provider."""
    
    # SLURM
    slurm_available: bool = False
    """Whether SLURM commands are available."""
    
    slurm_partitions: List[str] = field(default_factory=list)
    """Available SLURM partitions."""
    
    # Derived properties
    @property
    def has_local_models(self) -> bool:
        """True if any local vLLM endpoint is healthy."""
        return any(self.vllm_endpoints.values())
    
    @property
    def has_cloud_apis(self) -> bool:
        """True if any cloud API key is configured."""
        return any(self.cloud_apis.values())
    
    @property
    def available_models(self) -> List[str]:
        """List of healthy local model keys."""
        return [k for k, v in self.vllm_endpoints.items() if v]
    
    @property
    def available_cloud_apis(self) -> List[str]:
        """List of configured cloud API names."""
        return [k for k, v in self.cloud_apis.items() if v]
    
    @property
    def deployment_mode(self) -> str:
        """Inferred deployment mode: 'hybrid', 'local_only', 'cloud_only', 'none'."""
        has_local = self.has_local_models
        has_cloud = self.has_cloud_apis
        
        if has_local and has_cloud:
            return "hybrid"
        elif has_local:
            return "local_only"
        elif has_cloud:
            return "cloud_only"
        else:
            return "none"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "vllm_endpoints": self.vllm_endpoints,
            "vllm_urls": self.vllm_urls,
            "cloud_apis": self.cloud_apis,
            "slurm_available": self.slurm_available,
            "slurm_partitions": self.slurm_partitions,
            "has_local_models": self.has_local_models,
            "has_cloud_apis": self.has_cloud_apis,
            "deployment_mode": self.deployment_mode,
        }


class ResourceDetector:
    """
    Detects available LLM resources.
    
    Uses health checks for vLLM endpoints and environment variable checks
    for cloud APIs. Designed for speed and reliability over precision.
    """
    
    def __init__(
        self,
        vllm_endpoints: Optional[Dict[str, str]] = None,
        connection_dir: Optional[Path] = None,
        health_timeout: float = 2.0,
    ):
        """
        Initialize resource detector.
        
        Args:
            vllm_endpoints: Override default endpoints (model_key -> URL)
            connection_dir: Directory with .env files from deploy script
            health_timeout: Timeout in seconds for health checks
        """
        self.vllm_endpoints = vllm_endpoints or DEFAULT_VLLM_ENDPOINTS.copy()
        self.connection_dir = connection_dir or CONNECTION_DIR
        self.health_timeout = health_timeout
    
    def detect(self) -> ResourceStatus:
        """
        Run all detection methods and return status.
        
        This is the main entry point. Catches all exceptions to prevent
        detection failures from blocking application startup.
        """
        # Load actual endpoints from connection files (if deployed)
        actual_endpoints = self._load_connection_info()
        
        return ResourceStatus(
            vllm_endpoints=self._check_vllm_health(actual_endpoints),
            vllm_urls=actual_endpoints,
            cloud_apis=self._check_cloud_keys(),
            slurm_available=self._check_slurm(),
            slurm_partitions=self._get_slurm_partitions(),
        )
    
    def _load_connection_info(self) -> Dict[str, str]:
        """Load vLLM URLs from connection files created by deploy script."""
        endpoints = self.vllm_endpoints.copy()
        
        if not self.connection_dir.exists():
            logger.debug(f"Connection dir not found: {self.connection_dir}")
            return endpoints
        
        for model_key in self.vllm_endpoints:
            env_file = self.connection_dir / f"{model_key}.env"
            if env_file.exists():
                try:
                    with open(env_file) as f:
                        for line in f:
                            if line.startswith("URL="):
                                url = line.strip().split("=", 1)[1]
                                endpoints[model_key] = url
                                logger.debug(f"Loaded {model_key} URL: {url}")
                                break
                except Exception as e:
                    logger.warning(f"Failed to read {env_file}: {e}")
        
        return endpoints
    
    def _check_vllm_health(self, endpoints: Dict[str, str]) -> Dict[str, bool]:
        """Check health of vLLM endpoints."""
        if not HAS_REQUESTS:
            logger.warning("requests library not available, skipping vLLM health checks")
            return {k: False for k in endpoints}
        
        results = {}
        for model_key, url in endpoints.items():
            results[model_key] = self._is_endpoint_healthy(url)
        
        return results
    
    def _is_endpoint_healthy(self, url: str) -> bool:
        """Check if a single endpoint is healthy."""
        try:
            # Try /health endpoint first (vLLM standard)
            response = requests.get(
                f"{url}/health",
                timeout=self.health_timeout,
            )
            if response.ok:
                return True
            
            # Fallback: try /v1/models (OpenAI-compatible)
            response = requests.get(
                f"{url}/v1/models",
                timeout=self.health_timeout,
            )
            return response.ok
            
        except requests.exceptions.RequestException:
            return False
        except Exception as e:
            logger.debug(f"Health check failed for {url}: {e}")
            return False
    
    def _check_cloud_keys(self) -> Dict[str, bool]:
        """Check which cloud API keys are configured."""
        results = {}
        
        for api_name, env_vars in CLOUD_API_KEYS.items():
            if isinstance(env_vars, str):
                env_vars = [env_vars]
            
            # Check if any of the environment variables are set
            has_key = any(
                bool(os.getenv(var))
                for var in env_vars
            )
            results[api_name] = has_key
        
        return results
    
    def _check_slurm(self) -> bool:
        """Check if SLURM is available."""
        # Check if we're in a SLURM job
        if os.getenv("SLURM_JOB_ID"):
            return True
        
        # Check if squeue command exists
        return shutil.which("squeue") is not None
    
    def _get_slurm_partitions(self) -> List[str]:
        """Get available SLURM partitions."""
        if not self._check_slurm():
            return []
        
        try:
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%P"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                partitions = [
                    p.strip().rstrip("*")
                    for p in result.stdout.strip().split("\n")
                    if p.strip()
                ]
                return partitions
        except Exception as e:
            logger.debug(f"Failed to get SLURM partitions: {e}")
        
        return []
    
    def get_best_strategy(self) -> str:
        """
        Suggest the best strategy profile based on detected resources.
        
        Returns:
            Profile name: 't4_hybrid', 't4_local_only', 'cloud_only', or 'development'
        """
        status = self.detect()
        
        if status.has_local_models and status.has_cloud_apis:
            return "t4_hybrid"
        elif status.has_local_models:
            return "t4_local_only"
        elif status.has_cloud_apis:
            return "cloud_only"
        else:
            return "development"


# Convenience function for quick checks
def detect_resources() -> ResourceStatus:
    """Quick resource detection with default settings."""
    return ResourceDetector().detect()


def is_vllm_available(model_key: str = "generalist") -> bool:
    """Check if a specific vLLM model is available."""
    status = detect_resources()
    return status.vllm_endpoints.get(model_key, False)


def get_available_cloud_apis() -> List[str]:
    """Get list of configured cloud APIs."""
    status = detect_resources()
    return status.available_cloud_apis


__all__ = [
    "ResourceDetector",
    "ResourceStatus",
    "detect_resources",
    "is_vllm_available",
    "get_available_cloud_apis",
]
