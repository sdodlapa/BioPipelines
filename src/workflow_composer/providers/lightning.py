"""
Lightning.ai Provider
=====================

FREE tier with 30M tokens/month.
Supports multiple models through unified API.

Uses OpenAI-compatible endpoint for reliability.
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from .base import BaseProvider, Message, ProviderResponse, ProviderError

logger = logging.getLogger(__name__)


class LightningProvider(BaseProvider):
    """
    Provider for Lightning.ai API.
    
    Lightning.ai offers 30M FREE tokens/month with access to many models.
    This is the recommended default provider for development.
    
    Uses OpenAI-compatible API endpoint for reliability.
    
    Supported models (via Lightning.ai):
        - lightning-ai/DeepSeek-V3.1 (best value, excellent for code)
        - lightning-ai/llama-3.3-70b (general purpose)
        - openai/gpt-4o (high quality)
        - anthropic/claude-3-5-sonnet-20240620 (scientific)
        - google/gemini-2.5-flash-lite-preview-06-17
    
    Note: Requires credits in your Lightning.ai account.
    Free tier gives 30M tokens/month.
    """
    
    name = "lightning"
    default_model = "lightning-ai/DeepSeek-V3.1"
    supports_streaming = True
    
    # Correct base URL for Lightning.ai Models API
    DEFAULT_BASE_URL = "https://lightning.ai/api/v1"
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.environ.get("LIGHTNING_API_KEY")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI-compatible client."""
        if not self.api_key:
            return
        
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            logger.debug("Lightning.ai OpenAI-compatible client initialized")
        except ImportError:
            logger.warning("openai package not installed, using requests fallback")
            self._client = None
    
    def is_available(self) -> bool:
        """Check if Lightning.ai is configured and has credits."""
        if not self.api_key:
            return False
        
        # Check if the models endpoint is reachable
        try:
            import requests
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
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
        """Generate completion using Lightning.ai."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using Lightning.ai."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "LIGHTNING_API_KEY not set. Get free key at https://lightning.ai/models",
                retriable=False,
            )
        
        messages = self._normalize_messages(messages)
        start = time.time()
        
        # Use OpenAI-compatible client if available
        if self._client:
            return self._chat_with_openai_client(messages, start, **kwargs)
        else:
            return self._chat_with_requests(messages, start, **kwargs)
    
    def _chat_with_openai_client(
        self,
        messages: List[Message],
        start: float,
        **kwargs
    ) -> ProviderResponse:
        """Use OpenAI-compatible client for chat."""
        try:
            response = self._client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[m.to_dict() for m in messages],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            return ProviderResponse(
                content=content,
                provider=self.name,
                model=self.model,
                tokens_used=usage.total_tokens if usage else 0,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=(time.time() - start) * 1000,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
            
        except Exception as e:
            raise ProviderError(
                self.name,
                f"API error: {e}",
                retriable=True,
            )
    
    def _chat_with_requests(
        self,
        messages: List[Message],
        start: float,
        **kwargs
    ) -> ProviderResponse:
        """Fallback to requests library."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [m.to_dict() for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
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
