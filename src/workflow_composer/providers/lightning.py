"""
Lightning.ai Provider
=====================

FREE tier with 30M tokens/month.
Supports multiple models through the litai SDK.

STATUS (Dec 2025): ✅ WORKING - Uses litai SDK (not OpenAI-compatible API)

This provider wraps the working LightningAdapter from llm/lightning_adapter.py
which uses the litai SDK for proper authentication.

Working Models:
    - lightning-ai/gpt-oss-20b (FREE - Lightning's own model)
    - lightning-ai/gpt-oss-120b (FREE - larger variant)
    - openai/gpt-4o (via Lightning)
    - openai/gpt-4-turbo
    - openai/gpt-4
    - openai/gpt-3.5-turbo
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from .base import BaseProvider, Message, ProviderResponse, ProviderError

logger = logging.getLogger(__name__)


# Import the working adapter
try:
    from ..llm.lightning_adapter import LightningAdapter, WORKING_MODELS
    HAS_LIGHTNING_ADAPTER = True
except ImportError:
    HAS_LIGHTNING_ADAPTER = False
    WORKING_MODELS = []


class LightningProvider(BaseProvider):
    """
    Provider for Lightning.ai API.
    
    ✅ WORKING (Dec 2025): Uses litai SDK for proper authentication.
    
    Lightning.ai offers 30M FREE tokens/month with access to many models.
    
    Supported models (confirmed working):
        - lightning-ai/gpt-oss-20b (FREE)
        - lightning-ai/gpt-oss-120b (FREE)
        - openai/gpt-4o
        - openai/gpt-4-turbo
        - openai/gpt-4
        - openai/gpt-3.5-turbo
    """
    
    name = "lightning"
    default_model = "lightning-ai/gpt-oss-20b"
    supports_streaming = False  # litai SDK doesn't support streaming yet
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model or self.default_model, **kwargs)
        self.api_key = api_key or self._load_api_key()
        self._adapter = None
        self._init_adapter()
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment or secrets file."""
        # Check environment first
        if os.environ.get("LIGHTNING_API_KEY"):
            return os.environ["LIGHTNING_API_KEY"]
        
        # Check secrets file
        secrets_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", ".secrets", "lightning_key"
        )
        if os.path.exists(secrets_path):
            with open(secrets_path) as f:
                return f.read().strip()
        
        return None
    
    def _init_adapter(self):
        """Initialize the Lightning adapter."""
        if not HAS_LIGHTNING_ADAPTER:
            logger.warning("LightningAdapter not available")
            return
        
        if not self.api_key:
            logger.warning("LIGHTNING_API_KEY not set")
            return
        
        try:
            self._adapter = LightningAdapter(
                model=self.model,
                api_key=self.api_key,
            )
            logger.debug(f"Lightning adapter initialized with model: {self.model}")
        except Exception as e:
            logger.warning(f"Failed to initialize Lightning adapter: {e}")
            self._adapter = None
    
    def is_available(self) -> bool:
        """Check if Lightning.ai is configured and working."""
        if not self.api_key:
            return False
        if not HAS_LIGHTNING_ADAPTER:
            return False
        return self._adapter is not None
    
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
        """Generate chat response using Lightning.ai via litai SDK."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "LIGHTNING_API_KEY not set. Get free key at https://lightning.ai",
                retriable=False,
            )
        
        if not self._adapter:
            raise ProviderError(
                self.name,
                "Lightning adapter not initialized",
                retriable=False,
            )
        
        messages = self._normalize_messages(messages)
        start = time.time()
        
        try:
            # Convert to adapter message format
            from ..llm.base import Message as LLMMessage
            
            llm_messages = []
            for m in messages:
                if m.role == "system":
                    llm_messages.append(LLMMessage.system(m.content))
                elif m.role == "user":
                    llm_messages.append(LLMMessage.user(m.content))
                elif m.role == "assistant":
                    llm_messages.append(LLMMessage.assistant(m.content))
            
            # Call the adapter
            response = self._adapter.chat(llm_messages)
            
            return ProviderResponse(
                content=response.content,
                provider=self.name,
                model=self.model,
                tokens_used=response.tokens_used,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=(time.time() - start) * 1000,
                finish_reason="stop",
                raw_response=response.raw_response,
            )
            
        except Exception as e:
            raise ProviderError(
                self.name,
                f"API error: {e}",
                retriable=True,
            )
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models."""
        return WORKING_MODELS if WORKING_MODELS else [cls.default_model]
