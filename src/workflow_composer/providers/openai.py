"""
OpenAI Provider
===============

Paid API with high reliability and quality.
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from .base import BaseProvider, Message, ProviderResponse, ProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI API.
    
    Paid service with high reliability and quality.
    Used as fallback when free tiers are exhausted.
    
    Supported models:
        - gpt-4o (recommended)
        - gpt-4o-mini (default, cost-effective)
        - gpt-4-turbo
        - gpt-3.5-turbo
    """
    
    name = "openai"
    default_model = "gpt-4o-mini"
    supports_streaming = True
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.organization = organization or os.environ.get("OPENAI_ORG_ID")
    
    def is_available(self) -> bool:
        """Check if OpenAI is configured and reachable."""
        if not self.api_key:
            return False
        
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"}
            if self.organization:
                headers["OpenAI-Organization"] = self.organization
            
            resp = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Generate completion using OpenAI."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using OpenAI."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "OPENAI_API_KEY not set",
                retriable=False,
            )
        
        import requests
        
        messages = self._normalize_messages(messages)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        
        payload = {
            "model": kwargs.get("model", self.model),
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
                model=self.model,
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
