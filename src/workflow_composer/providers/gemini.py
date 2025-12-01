"""
Google Gemini Provider
======================

FREE tier with rate limits.
Excellent for coding and reasoning tasks.
"""

import os
import time
import logging
import asyncio
from typing import Optional, List, Dict, Any, Iterator, AsyncIterator

from .base import BaseProvider, Message, ProviderResponse, ProviderError, Role

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """
    Provider for Google Gemini API.
    
    Gemini offers a free tier with rate limits.
    Excellent quality for coding and reasoning tasks.
    
    Supported models:
        - gemini-2.0-flash (default, fast)
        - gemini-2.5-pro-preview-06-05 (latest, powerful)
        - gemini-2.5-flash-preview-05-20 (fast, capable)
    """
    
    name = "gemini"
    default_model = "gemini-2.0-flash"
    supports_streaming = True
    
    # Model name to API path mapping
    MODEL_PATHS = {
        "gemini-2.0-flash": "models/gemini-2.0-flash",
        "gemini-2.5-pro": "models/gemini-2.5-pro-preview-06-05",
        "gemini-2.5-pro-preview-06-05": "models/gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash": "models/gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-preview-05-20": "models/gemini-2.5-flash-preview-05-20",
    }
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def is_available(self) -> bool:
        """Check if Gemini is configured and reachable."""
        if not self.api_key:
            return False
        
        try:
            import requests
            resp = requests.get(
                f"{self.base_url}/models",
                params={"key": self.api_key},
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
        """Generate completion using Gemini."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using Gemini."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "GOOGLE_API_KEY not set",
                retriable=False,
            )
        
        import requests
        
        messages = self._normalize_messages(messages)
        
        # Get model path
        model = kwargs.get("model", self.model)
        model_path = self.MODEL_PATHS.get(model, f"models/{model}")
        
        # Convert messages to Gemini format
        contents = self._convert_messages(messages)
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            },
        }
        
        url = f"{self.base_url}/{model_path}:generateContent"
        
        start = time.time()
        
        try:
            response = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                headers={"Content-Type": "application/json"},
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
            
            # Extract content
            try:
                content = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                raise ProviderError(
                    self.name,
                    f"Invalid response format: {data}",
                    retriable=False,
                )
            
            # Token usage
            usage = data.get("usageMetadata", {})
            
            return ProviderResponse(
                content=content,
                provider=self.name,
                model=model,
                tokens_used=usage.get("totalTokenCount", 0),
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
                latency_ms=(time.time() - start) * 1000,
                raw_response=data,
            )
            
        except requests.exceptions.RequestException as e:
            raise ProviderError(
                self.name,
                f"Request failed: {e}",
                retriable=True,
            )
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict]:
        """Convert messages to Gemini format."""
        contents = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Gemini doesn't have system role, prepend to first user message
                # or add as user+model exchange
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"System instruction: {msg.content}"}],
                })
                contents.append({
                    "role": "model",
                    "parts": [{"text": "Understood. I will follow these instructions."}],
                })
            elif msg.role == Role.USER:
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg.content}],
                })
            elif msg.role == Role.ASSISTANT:
                contents.append({
                    "role": "model",
                    "parts": [{"text": msg.content}],
                })
        
        return contents
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """Stream completion token by token using Gemini's streaming API."""
        messages = self._build_messages(prompt, system_prompt)
        yield from self.chat_stream(messages, **kwargs)
    
    def chat_stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> Iterator[str]:
        """Stream chat response token by token using Gemini's streaming API."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "GOOGLE_API_KEY not set",
                retriable=False,
            )
        
        import requests
        
        messages = self._normalize_messages(messages)
        
        model = kwargs.get("model", self.model)
        model_path = self.MODEL_PATHS.get(model, f"models/{model}")
        
        contents = self._convert_messages(messages)
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            },
        }
        
        # Use streamGenerateContent endpoint
        url = f"{self.base_url}/{model_path}:streamGenerateContent"
        
        try:
            response = requests.post(
                url,
                params={"key": self.api_key, "alt": "sse"},
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
                stream=True,
            )
            
            if response.status_code != 200:
                raise ProviderError(
                    self.name,
                    f"API error: {response.status_code} - {response.text[:200]}",
                    retriable=response.status_code >= 500,
                    status_code=response.status_code,
                )
            
            # Parse SSE stream
            import json
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if 'candidates' in data:
                                for candidate in data['candidates']:
                                    if 'content' in candidate:
                                        for part in candidate['content'].get('parts', []):
                                            if 'text' in part:
                                                yield part['text']
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.RequestException as e:
            raise ProviderError(
                self.name,
                f"Stream request failed: {e}",
                retriable=True,
            )
    
    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Async stream completion using Gemini's streaming API."""
        messages = self._build_messages(prompt, system_prompt)
        async for chunk in self.chat_stream_async(messages, **kwargs):
            yield chunk
    
    async def chat_stream_async(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """Async stream chat response using Gemini's streaming API."""
        # Use sync streaming in thread for now
        # TODO: Use aiohttp for true async
        loop = asyncio.get_event_loop()
        
        def get_chunks():
            return list(self.chat_stream(messages, **kwargs))
        
        chunks = await loop.run_in_executor(None, get_chunks)
        for chunk in chunks:
            yield chunk
