"""
Base Provider Interface
=======================

Abstract base class that all providers must implement.
Combines the best features from llm/base.py and models/providers/base.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Union
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class Role(Enum):
    """Message roles for chat completions."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    A single message in a conversation.
    
    Examples:
        msg = Message.system("You are a helpful assistant.")
        msg = Message.user("Explain RNA-seq.")
        msg = Message(Role.ASSISTANT, "RNA-seq is...")
    """
    role: Role
    content: str
    name: Optional[str] = None  # For tool messages
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API calls."""
        d = {
            "role": self.role.value,
            "content": self.content
        }
        if self.name:
            d["name"] = self.name
        return d
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(Role.SYSTEM, content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(Role.USER, content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(Role.ASSISTANT, content)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        role = Role(d.get("role", "user"))
        return cls(
            role=role,
            content=d.get("content", ""),
            name=d.get("name"),
        )


@dataclass
class ProviderResponse:
    """
    Response from a provider.
    
    Unified response format for all providers.
    """
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0
    finish_reason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = None
    cached: bool = False
    
    @property
    def text(self) -> str:
        """Alias for content."""
        return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
        }


class ProviderError(Exception):
    """
    Error from a provider.
    
    Attributes:
        provider: Name of the provider that failed
        message: Error message
        retriable: Whether the request can be retried
        status_code: HTTP status code if applicable
    """
    def __init__(
        self,
        provider: str,
        message: str,
        retriable: bool = True,
        status_code: Optional[int] = None,
    ):
        self.provider = provider
        self.retriable = retriable
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    All providers (OpenAI, Gemini, Lightning, etc.) must implement this interface.
    Provides both sync and async methods for flexibility.
    
    Attributes:
        name: Provider identifier (e.g., "openai", "gemini")
        default_model: Default model to use
        temperature: Sampling temperature (0.0 - 2.0)
        max_tokens: Maximum tokens in response
    """
    
    # Class attributes to be overridden by subclasses
    name: str = "base"
    default_model: str = ""
    supports_streaming: bool = False
    supports_async: bool = True
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the provider.
        
        Args:
            model: Model to use (defaults to provider's default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: API key (or uses environment variable)
            base_url: Custom API base URL
            **kwargs: Additional provider-specific options
        """
        self.model = model or self.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self._kwargs = kwargs
        self._initialized = False
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """
        Generate a completion for a single prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            ProviderResponse with the generated content
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """
        Generate a response in a multi-turn conversation.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            ProviderResponse with the assistant's response
        """
        pass
    
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """
        Async version of complete().
        
        Default implementation wraps sync method.
        Override for true async support.
        """
        import asyncio
        return await asyncio.to_thread(
            self.complete, prompt, system_prompt, **kwargs
        )
    
    async def chat_async(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """
        Async version of chat().
        
        Default implementation wraps sync method.
        Override for true async support.
        """
        import asyncio
        return await asyncio.to_thread(self.chat, messages, **kwargs)
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream a completion token by token.
        
        Default implementation yields the full response.
        Override for true streaming support.
        """
        response = self.complete(prompt, system_prompt, **kwargs)
        yield response.content
    
    def chat_stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> Iterator[str]:
        """
        Stream a chat response token by token.
        
        Default implementation yields the full response.
        Override for true streaming support.
        """
        response = self.chat(messages, **kwargs)
        yield response.content
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available and configured.
        
        Should verify:
        - API key is set (for API providers)
        - Server is reachable (for local providers)
        
        Returns:
            True if provider can be used
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.
        
        Returns:
            Dictionary with:
                - available: bool
                - latency_ms: float (optional)
                - error: str (if not available)
                - models: List[str] (optional)
        """
        try:
            start = time.time()
            available = self.is_available()
            latency = (time.time() - start) * 1000
            
            return {
                "available": available,
                "latency_ms": latency,
                "provider": self.name,
                "model": self.model,
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "provider": self.name,
            }
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Message]:
        """Helper to build message list from prompt."""
        messages = []
        if system_prompt:
            messages.append(Message.system(system_prompt))
        messages.append(Message.user(prompt))
        return messages
    
    def _normalize_messages(
        self,
        messages: List[Union[Message, Dict[str, Any]]]
    ) -> List[Message]:
        """Normalize messages to Message objects."""
        normalized = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(Message.from_dict(msg))
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
        return normalized
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


class MockProvider(BaseProvider):
    """
    Mock provider for testing.
    
    Returns predefined responses or echoes input.
    """
    
    name = "mock"
    default_model = "mock-model"
    
    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "Mock response",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.responses = responses or {}
        self.default_response = default_response
        self.call_history: List[Dict[str, Any]] = []
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        self.call_history.append({
            "type": "complete",
            "prompt": prompt,
            "system_prompt": system_prompt,
        })
        
        # Check for keyword matches
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt.lower():
                return ProviderResponse(
                    content=response,
                    provider=self.name,
                    model=self.model,
                )
        
        return ProviderResponse(
            content=self.default_response,
            provider=self.name,
            model=self.model,
        )
    
    def chat(self, messages: List[Message], **kwargs) -> ProviderResponse:
        messages = self._normalize_messages(messages)
        last_user = next(
            (m.content for m in reversed(messages) if m.role == Role.USER),
            ""
        )
        return self.complete(last_user, **kwargs)
    
    def is_available(self) -> bool:
        return True
