"""
GitHub Models Provider
======================

FREE tier with GitHub Pro+ account.
Uses OpenAI-compatible API via Azure.
"""

import os
import time
import logging
from typing import Optional, List

from .base import BaseProvider, Message, ProviderResponse, ProviderError

logger = logging.getLogger(__name__)


class GitHubModelsProvider(BaseProvider):
    """
    Provider for GitHub Models API.
    
    Free tier available with GitHub Pro+ account.
    Uses GitHub token with `models:read` scope.
    
    Endpoint: https://models.inference.ai.azure.com
    
    Supported models:
        - gpt-4o-mini (default, free)
        - gpt-4o
        - DeepSeek-R1
        - Llama-3.1-405B-Instruct
        - Phi-4
        - Mistral-large-2411
    """
    
    name = "github_models"
    default_model = "gpt-4o-mini"
    supports_streaming = True
    
    # Azure-hosted GitHub Models endpoint
    BASE_URL = "https://models.inference.ai.azure.com"
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.environ.get("GITHUB_TOKEN")
        self.base_url = self.BASE_URL
    
    def is_available(self) -> bool:
        """Check if GitHub Models is configured and reachable."""
        if not self.api_key:
            return False
        
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # Try to list models (or just verify auth works)
            resp = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"GitHub Models check failed: {e}")
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Generate completion using GitHub Models."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using GitHub Models (OpenAI-compatible)."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "GITHUB_TOKEN not set",
                retriable=False,
            )
        
        import requests
        
        messages = self._normalize_messages(messages)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        model = kwargs.get("model", self.model)
        
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
                headers=headers,
                json=payload,
                timeout=120,
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
            
        except requests.exceptions.RequestException as e:
            raise ProviderError(
                self.name,
                f"Request failed: {e}",
                retriable=True,
            )
