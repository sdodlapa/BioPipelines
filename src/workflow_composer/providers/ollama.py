"""
Ollama Provider
===============

Local inference with Ollama (free, private).
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from .base import BaseProvider, Message, ProviderResponse, ProviderError

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """
    Provider for local Ollama server.
    
    Ollama provides free local inference with easy model management.
    Excellent for development and privacy-sensitive workloads.
    
    Supported models (examples):
        - llama3:8b (default)
        - mistral:7b
        - codellama:13b
        - deepseek-coder:6.7b
    
    Setup:
        1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
        2. Pull a model: ollama pull llama3:8b
        3. Start server: ollama serve (or it runs automatically)
    """
    
    name = "ollama"
    default_model = "llama3:8b"
    supports_streaming = True
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.base_url = base_url or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        )
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Generate completion using Ollama."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using Ollama."""
        import requests
        
        messages = self._normalize_messages(messages)
        
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        
        start = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,  # Longer timeout for local inference
            )
            
            if response.status_code != 200:
                raise ProviderError(
                    self.name,
                    f"API error: {response.status_code} - {response.text[:200]}",
                    retriable=response.status_code >= 500,
                    status_code=response.status_code,
                )
            
            data = response.json()
            
            content = data.get("message", {}).get("content", "")
            
            # Ollama provides token counts
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            
            return ProviderResponse(
                content=content,
                provider=self.name,
                model=self.model,
                tokens_used=prompt_tokens + completion_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=(time.time() - start) * 1000,
                finish_reason=data.get("done_reason", "stop"),
                raw_response=data,
            )
            
        except requests.exceptions.ConnectionError:
            raise ProviderError(
                self.name,
                "Ollama server not running. Start with: ollama serve",
                retriable=False,
            )
        except requests.exceptions.RequestException as e:
            raise ProviderError(
                self.name,
                f"Request failed: {e}",
                retriable=True,
            )
    
    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        import requests
        
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        
        return []
    
    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama library."""
        import requests
        
        try:
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=600,  # Model downloads can be slow
            )
            return resp.status_code == 200
        except Exception:
            return False
