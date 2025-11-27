"""
Multi-Model Deployment Configuration
=====================================

Configure and manage multiple vLLM instances for different agent roles.

Architecture:
- Main Model (GPU 0-1): MiniMax-M2 for coding and complex reasoning
- Router Model (GPU 2): Smaller model for fast intent routing
- Embeddings (GPU 3): BGE model for memory/RAG

This enables:
- Fast routing (<100ms) with small model
- High-quality code generation with large model
- Efficient memory retrieval with dedicated embeddings
"""

import os
import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ModelRole(Enum):
    """Role of a model in the system."""
    MAIN = "main"           # Primary model for generation
    ROUTER = "router"       # Fast routing/classification
    CODING = "coding"       # Code generation/debugging
    EMBEDDINGS = "embeddings"  # Vector embeddings


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    role: ModelRole
    hf_model: str
    gpu_ids: List[int]
    port: int
    
    # vLLM parameters
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    dtype: str = "auto"
    quantization: Optional[str] = None
    
    # Resource limits
    gpu_memory_utilization: float = 0.9
    
    # Extras
    extra_args: List[str] = field(default_factory=list)
    
    @property
    def url(self) -> str:
        """Get the API URL for this model."""
        return f"http://localhost:{self.port}/v1"
    
    def to_vllm_args(self) -> List[str]:
        """Convert to vLLM command line arguments."""
        args = [
            "--model", self.hf_model,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        
        if self.quantization:
            args.extend(["--quantization", self.quantization])
        
        args.extend(self.extra_args)
        
        return args


# =============================================================================
# Predefined Configurations
# =============================================================================

# Configuration for 4x H100 80GB setup
QUAD_H100_CONFIG = {
    "main": ModelConfig(
        name="main",
        role=ModelRole.MAIN,
        hf_model="MiniMaxAI/MiniMax-M2-Lite",
        gpu_ids=[0, 1, 2, 3],  # Use all 4 GPUs for 230B MoE
        port=8000,
        tensor_parallel_size=4,
        max_model_len=32768,
        dtype="auto",
        gpu_memory_utilization=0.9,
    ),
}

# Configuration for 2x H100 80GB setup (simpler - one main model)
DUAL_H100_CONFIG = {
    "main": ModelConfig(
        name="main",
        role=ModelRole.MAIN,
        hf_model="meta-llama/Llama-3.3-70B-Instruct",
        gpu_ids=[0, 1],
        port=8000,
        tensor_parallel_size=2,
        max_model_len=16384,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    ),
}

# Configuration for 1x T4 16GB (local dev)
SINGLE_T4_CONFIG = {
    "main": ModelConfig(
        name="main",
        role=ModelRole.MAIN,
        hf_model="microsoft/Phi-3-mini-4k-instruct",
        gpu_ids=[0],
        port=8000,
        tensor_parallel_size=1,
        max_model_len=4096,
        dtype="float16",
        gpu_memory_utilization=0.85,
    ),
}

# Multi-model configuration for advanced setups (experimental)
# Uses specialized models for different tasks
MULTI_MODEL_CONFIG = {
    "main": ModelConfig(
        name="main",
        role=ModelRole.MAIN,
        hf_model="MiniMaxAI/MiniMax-M2-Lite",
        gpu_ids=[0, 1],  # GPUs 0-1 for main
        port=8000,
        tensor_parallel_size=2,
        max_model_len=16384,
        dtype="auto",
        gpu_memory_utilization=0.85,
    ),
    "router": ModelConfig(
        name="router",
        role=ModelRole.ROUTER,
        hf_model="microsoft/Phi-3-mini-4k-instruct",
        gpu_ids=[2],  # GPU 2 for fast routing
        port=8001,
        tensor_parallel_size=1,
        max_model_len=4096,
        dtype="float16",
        gpu_memory_utilization=0.5,  # Small footprint
    ),
    "embeddings": ModelConfig(
        name="embeddings",
        role=ModelRole.EMBEDDINGS,
        hf_model="BAAI/bge-large-en-v1.5",
        gpu_ids=[3],  # GPU 3 for embeddings
        port=8002,
        tensor_parallel_size=1,
        max_model_len=512,
        dtype="float16",
        gpu_memory_utilization=0.3,
    ),
}


# =============================================================================
# Deployment Manager
# =============================================================================

class MultiModelDeployment:
    """
    Manages deployment of multiple vLLM instances.
    
    Example:
        deployment = MultiModelDeployment.auto_detect()
        
        # Start all models
        deployment.start_all()
        
        # Get URLs
        main_url = deployment.get_url("main")
        router_url = deployment.get_url("router")
    """
    
    def __init__(self, configs: Dict[str, ModelConfig]):
        """
        Initialize with model configurations.
        
        Args:
            configs: Dictionary of name -> ModelConfig
        """
        self.configs = configs
        self._processes: Dict[str, subprocess.Popen] = {}
        self._hf_token = self._get_hf_token()
    
    @classmethod
    def auto_detect(cls) -> "MultiModelDeployment":
        """
        Auto-detect GPU configuration and return appropriate deployment.
        """
        gpu_count = cls._detect_gpus()
        logger.info(f"Detected {gpu_count} GPUs")
        
        if gpu_count >= 4:
            logger.info("Using 4x H100 configuration (MiniMax-M2)")
            return cls(QUAD_H100_CONFIG)
        elif gpu_count >= 2:
            logger.info("Using 2x H100 configuration (Llama-70B)")
            return cls(DUAL_H100_CONFIG)
        elif gpu_count >= 1:
            logger.info("Using 1x T4 configuration (Phi-3)")
            return cls(SINGLE_T4_CONFIG)
        else:
            logger.warning("No GPUs detected - running in CPU mode")
            return cls({})
    
    @staticmethod
    def _detect_gpus() -> int:
        """Detect number of available GPUs."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return len([l for l in result.stdout.strip().split('\n') if l.strip()])
        except Exception:
            pass
        
        # Try CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            return len(cuda_devices.split(","))
        
        return 0
    
    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token."""
        # Check environment
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if token:
            return token
        
        # Check file
        token_file = Path.home() / ".secrets" / "hf_token"
        if token_file.exists():
            return token_file.read_text().strip()
        
        # Check workspace
        workspace_token = Path(".secrets/hf_token")
        if workspace_token.exists():
            return workspace_token.read_text().strip()
        
        return None
    
    def get_config(self, name: str) -> Optional[ModelConfig]:
        """Get configuration for a model."""
        return self.configs.get(name)
    
    def get_url(self, name: str) -> Optional[str]:
        """Get API URL for a model."""
        config = self.get_config(name)
        return config.url if config else None
    
    def start(self, name: str) -> bool:
        """
        Start a single model.
        
        Args:
            name: Model name (e.g., "main", "router")
            
        Returns:
            True if started successfully
        """
        config = self.get_config(name)
        if not config:
            logger.error(f"Unknown model: {name}")
            return False
        
        if name in self._processes:
            logger.warning(f"Model {name} already running")
            return True
        
        logger.info(f"Starting {name} model: {config.hf_model}")
        
        # Build command
        cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"]
        cmd.extend(config.to_vllm_args())
        
        # Set environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
        if self._hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = self._hf_token
        
        try:
            # Start process
            log_file = Path(f"logs/vllm_{name}.log")
            log_file.parent.mkdir(exist_ok=True)
            
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            
            self._processes[name] = process
            logger.info(f"Started {name} (PID: {process.pid}) on port {config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return False
    
    def start_all(self) -> Dict[str, bool]:
        """Start all configured models."""
        results = {}
        for name in self.configs:
            results[name] = self.start(name)
        return results
    
    def stop(self, name: str) -> bool:
        """Stop a model."""
        if name not in self._processes:
            return True
        
        process = self._processes[name]
        logger.info(f"Stopping {name} (PID: {process.pid})")
        
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        
        del self._processes[name]
        return True
    
    def stop_all(self):
        """Stop all models."""
        for name in list(self._processes.keys()):
            self.stop(name)
    
    def status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models."""
        status = {}
        for name, config in self.configs.items():
            process = self._processes.get(name)
            status[name] = {
                "model": config.hf_model,
                "port": config.port,
                "gpus": config.gpu_ids,
                "running": process is not None and process.poll() is None,
                "pid": process.pid if process else None,
            }
        return status
    
    def generate_sbatch_script(
        self,
        output_path: str = "scripts/start_multi_model.sh",
        partition: str = "h100quadflex",
    ) -> str:
        """
        Generate SBATCH script for deploying all models.
        
        Args:
            output_path: Path to write script
            partition: SLURM partition
            
        Returns:
            Path to generated script
        """
        # Determine resources based on config
        total_gpus = max(
            max(c.gpu_ids) + 1 for c in self.configs.values()
        ) if self.configs else 1
        
        script = f'''#!/bin/bash
#SBATCH --job-name=multi-model-vllm
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{total_gpus}
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/multi_model_%j.out
#SBATCH --error=logs/multi_model_%j.err

# =============================================================================
# Multi-Model vLLM Deployment
# =============================================================================
# Generated by BioPipelines Multi-Model Deployment System
# Models: {", ".join(self.configs.keys())}
# Total GPUs: {total_gpus}

set -e

# Setup environment
module load anaconda3 cuda/12.4
source activate biopipelines

# Set HuggingFace token
export HUGGING_FACE_HUB_TOKEN="${{HUGGING_FACE_HUB_TOKEN:-$(cat ~/.secrets/hf_token 2>/dev/null || cat .secrets/hf_token 2>/dev/null || echo "")}}"

# Log GPU info
echo "=== GPU Configuration ==="
nvidia-smi -L

mkdir -p logs

'''
        
        # Add startup for each model
        for name, config in self.configs.items():
            gpu_str = ",".join(map(str, config.gpu_ids))
            args_str = " ".join(config.to_vllm_args())
            
            script += f'''
# Start {name} model
echo "Starting {name}: {config.hf_model}"
CUDA_VISIBLE_DEVICES={gpu_str} python -m vllm.entrypoints.openai.api_server \\
    {args_str} \\
    > logs/vllm_{name}.log 2>&1 &
echo "{name.upper()}_PID=$!"

'''
        
        # Add health check and wait
        script += '''
# Wait for models to load
echo "Waiting for models to load..."
sleep 30

# Health check
'''
        
        for name, config in self.configs.items():
            script += f'''
if curl -s http://localhost:{config.port}/health > /dev/null; then
    echo "✓ {name} model ready on port {config.port}"
else
    echo "✗ {name} model not responding on port {config.port}"
fi
'''
        
        script += '''
# Print access info
echo ""
echo "=========================================="
echo "Multi-Model Deployment Ready"
echo "=========================================="
'''
        
        for name, config in self.configs.items():
            script += f'echo "  {name:12}: http://$SLURM_NODELIST:{config.port}/v1"\n'
        
        script += '''
echo ""
echo "SSH Tunnel:"
'''
        
        ports = [c.port for c in self.configs.values()]
        tunnel_args = " ".join([f"-L {p}:$SLURM_NODELIST:{p}" for p in ports])
        script += f'echo "  ssh {tunnel_args} ${{USER}}@$(hostname)"\n'
        
        script += '''
echo "=========================================="

# Wait for all processes
wait
'''
        
        # Write script
        output = Path(output_path)
        output.parent.mkdir(exist_ok=True)
        output.write_text(script)
        output.chmod(0o755)
        
        logger.info(f"Generated multi-model script: {output_path}")
        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

_deployment_instance: Optional[MultiModelDeployment] = None


def get_deployment() -> MultiModelDeployment:
    """Get the global deployment instance."""
    global _deployment_instance
    if _deployment_instance is None:
        _deployment_instance = MultiModelDeployment.auto_detect()
    return _deployment_instance


def get_model_url(role: str = "main") -> str:
    """Get URL for a model by role."""
    deployment = get_deployment()
    url = deployment.get_url(role)
    if url:
        return url
    # Fall back to environment
    return os.environ.get("VLLM_URL", "http://localhost:8000/v1")


def configure_from_env() -> Dict[str, str]:
    """
    Configure model URLs from environment variables.
    
    Environment variables:
    - VLLM_URL: Main model URL (default)
    - VLLM_ROUTER_URL: Router model URL
    - VLLM_EMBEDDINGS_URL: Embeddings URL
    
    Returns:
        Dictionary of role -> URL
    """
    return {
        "main": os.environ.get("VLLM_URL", "http://localhost:8000/v1"),
        "router": os.environ.get("VLLM_ROUTER_URL", os.environ.get("VLLM_URL", "http://localhost:8000/v1")),
        "embeddings": os.environ.get("VLLM_EMBEDDINGS_URL", None),
    }
