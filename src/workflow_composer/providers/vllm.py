"""
vLLM Provider
=============

High-performance local GPU inference.
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from .base import BaseProvider, Message, ProviderResponse, ProviderError

logger = logging.getLogger(__name__)


class VLLMProvider(BaseProvider):
    """
    Provider for local vLLM server.
    
    vLLM provides high-performance GPU inference with an OpenAI-compatible API.
    Use for running open-source models on your own hardware.
    
    Supported models (configured for 8x H100):
        - Qwen/Qwen2.5-Coder-32B-Instruct (1 GPU)
        - deepseek-ai/DeepSeek-Coder-V2-Instruct (2 GPUs)
        - meta-llama/Llama-3.3-70B-Instruct (2 GPUs)
        - MiniMaxAI/MiniMax-M2 (4 GPUs)
        - codellama/CodeLlama-34b-Instruct-hf (1 GPU)
    
    Setup:
        1. Start vLLM server: scripts/llm/start_local_model.sh qwen-coder-32b
        2. Or submit SLURM job: sbatch scripts/llm/slurm_vllm_qwen.sh
    """
    
    name = "vllm"
    default_model = "Qwen/Qwen2.5-Coder-32B-Instruct"
    supports_streaming = True
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        
        # Support multiple environment variables
        host = os.environ.get("VLLM_HOST", "localhost")
        port = os.environ.get("VLLM_PORT", "8000")
        
        self.base_url = base_url or os.environ.get(
            "VLLM_API_BASE",
            f"http://{host}:{port}/v1"
        )
    
    def is_available(self) -> bool:
        """Check if vLLM server is running."""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Generate completion using vLLM."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using vLLM."""
        import requests
        
        messages = self._normalize_messages(messages)
        
        # Check what model is loaded
        model = kwargs.get("model", self.model)
        loaded_model = self._get_loaded_model()
        if loaded_model:
            model = loaded_model
        
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        start = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=300,  # Long timeout for GPU inference
            )
            
            if response.status_code != 200:
                raise ProviderError(
                    self.name,
                    f"API error: {response.status_code} - {response.text[:200]}",
                    retriable=response.status_code >= 500,
                    status_code=response.status_code,
                )
            
            data = response.json()
            
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            
            return ProviderResponse(
                content=content,
                provider=self.name,
                model=model,
                tokens_used=usage.get("total_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_ms=(time.time() - start) * 1000,
                finish_reason=data["choices"][0].get("finish_reason", "stop"),
                raw_response=data,
            )
            
        except requests.exceptions.ConnectionError:
            raise ProviderError(
                self.name,
                "vLLM server not running. Start with: scripts/llm/start_local_model.sh",
                retriable=False,
            )
        except requests.exceptions.RequestException as e:
            raise ProviderError(
                self.name,
                f"Request failed: {e}",
                retriable=True,
            )
    
    def _get_loaded_model(self) -> Optional[str]:
        """Get the currently loaded model from vLLM server."""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    return models[0].get("id")
        except Exception:
            pass
        return None
    
    def list_models(self) -> List[str]:
        """List models loaded in vLLM server."""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return []
    
    def health_check(self) -> Dict[str, Any]:
        """Detailed health check for vLLM."""
        base_check = super().health_check()
        
        if base_check.get("available"):
            models = self.list_models()
            base_check["models_loaded"] = models
            base_check["model"] = models[0] if models else None
        
        return base_check
