"""
Anthropic Provider
==================

Paid API with excellent quality for complex reasoning.
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from .base import BaseProvider, Message, ProviderResponse, ProviderError, Role

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic Claude API.
    
    Paid service with excellent quality for complex reasoning.
    
    Supported models:
        - claude-3-5-sonnet-20241022 (recommended)
        - claude-3-opus-20240229 (most capable)
        - claude-3-haiku-20240307 (fastest)
    """
    
    name = "anthropic"
    default_model = "claude-3-5-sonnet-20241022"
    supports_streaming = True
    
    # API version
    API_VERSION = "2023-06-01"
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1"
    
    def is_available(self) -> bool:
        """Check if Anthropic is configured."""
        return bool(self.api_key)
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Generate completion using Claude."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using Claude."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "ANTHROPIC_API_KEY not set",
                retriable=False,
            )
        
        import requests
        
        messages = self._normalize_messages(messages)
        
        # Extract system message (Claude handles it separately)
        system_content = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content
            else:
                chat_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": self.API_VERSION,
        }
        
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if system_content:
            payload["system"] = system_content
        
        # Claude uses different parameter names
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        elif self.temperature != 0.7:  # Only if not default
            payload["temperature"] = self.temperature
        
        start = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/messages",
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
            
            # Extract content from Claude's format
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")
            
            usage = data.get("usage", {})
            
            return ProviderResponse(
                content=content,
                provider=self.name,
                model=self.model,
                tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                latency_ms=(time.time() - start) * 1000,
                finish_reason=data.get("stop_reason", "end_turn"),
                raw_response=data,
            )
            
        except requests.exceptions.RequestException as e:
            raise ProviderError(
                self.name,
                f"Request failed: {e}",
                retriable=True,
            )
