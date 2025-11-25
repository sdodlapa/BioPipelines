"""
HuggingFace LLM Adapter
=======================

Adapter for HuggingFace models via the Inference API, local transformers, or vLLM.

Supports:
- HuggingFace Inference API (cloud)
- Local transformers (requires torch, transformers packages)
- vLLM server (high-performance GPU inference)

Usage:
    from workflow_composer.llm import HuggingFaceAdapter
    
    # Using HuggingFace Inference API
    llm = HuggingFaceAdapter(model="meta-llama/Llama-3-8b-chat-hf", backend="api")
    
    # Using local transformers
    llm = HuggingFaceAdapter(model="meta-llama/Llama-3-8b-chat-hf", backend="transformers")
    
    # Using vLLM server (recommended for GPU inference)
    llm = HuggingFaceAdapter(
        model="meta-llama/Llama-3.1-8B-Instruct",
        backend="vllm",
        vllm_url="http://localhost:8000"
    )
"""

import json
import os
import logging
from typing import List, Optional, Dict, Any
import urllib.request
import urllib.error

from .base import LLMAdapter, LLMResponse, Message

logger = logging.getLogger(__name__)


class HuggingFaceAdapter(LLMAdapter):
    """
    Adapter for HuggingFace models.
    
    Can use either:
    1. HuggingFace Inference API (cloud, requires HF_TOKEN)
    2. Local transformers library (requires transformers, torch)
    3. vLLM server (high-performance GPU inference)
    
    Recommended models:
    - meta-llama/Llama-3.1-8B-Instruct (general purpose)
    - meta-llama/Llama-3.1-70B-Instruct (high quality)
    - mistralai/Mistral-7B-Instruct-v0.3 (fast, good quality)
    - Qwen/Qwen2.5-7B-Instruct (multilingual)
    - codellama/CodeLlama-34b-Instruct-hf (code generation)
    - deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (code focused)
    """
    
    API_BASE = "https://api-inference.huggingface.co/models"
    
    # Backend options
    BACKEND_API = "api"
    BACKEND_TRANSFORMERS = "transformers"
    BACKEND_VLLM = "vllm"
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        token: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        backend: str = "api",
        use_api: Optional[bool] = None,  # Deprecated, use backend instead
        device: str = "auto",
        vllm_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize HuggingFace adapter.
        
        Args:
            model: HuggingFace model ID
            token: HuggingFace API token (default: from HF_TOKEN env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            backend: Backend to use - "api", "transformers", or "vllm"
            use_api: Deprecated - use backend="api" or backend="transformers"
            device: Device for local/transformers inference ("auto", "cuda", "cpu")
            vllm_url: vLLM server URL (default: http://localhost:8000)
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.device = device
        self.vllm_url = vllm_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        
        # Handle deprecated use_api parameter
        if use_api is not None:
            logger.warning("use_api is deprecated. Use backend='api' or backend='transformers' instead.")
            backend = self.BACKEND_API if use_api else self.BACKEND_TRANSFORMERS
        
        self.backend = backend.lower()
        
        self._pipeline = None  # Lazy load for transformers
        self._vllm_adapter = None  # Lazy load for vLLM
        
        if self.backend == self.BACKEND_API and not self.token:
            logger.debug("No HuggingFace token provided. Set HF_TOKEN environment variable.")
    
    @property
    def use_api(self) -> bool:
        """Deprecated: Check if using API backend."""
        return self.backend == self.BACKEND_API
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    def _get_vllm_adapter(self):
        """Lazy load vLLM adapter."""
        if self._vllm_adapter is None:
            from .vllm_adapter import VLLMAdapter
            self._vllm_adapter = VLLMAdapter(
                model=self.model,
                base_url=self.vllm_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        return self._vllm_adapter
    
    def _get_pipeline(self):
        """Lazy load transformers pipeline for local inference."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                import torch
                
                device_map = self.device
                if device_map == "auto":
                    device_map = "cuda" if torch.cuda.is_available() else "cpu"
                
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    device_map=device_map,
                    torch_dtype=torch.float16 if device_map == "cuda" else torch.float32,
                    token=self.token
                )
            except ImportError:
                raise ImportError(
                    "Local HuggingFace inference requires transformers and torch. "
                    "Install with: pip install transformers torch"
                )
        return self._pipeline
    
    def _api_request(self, prompt: str, **kwargs) -> str:
        """Make request to HuggingFace Inference API."""
        url = f"{self.API_BASE}/{self.model}"
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        data = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "return_full_text": False,
            }
        }
        
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return str(result)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"HuggingFace API error: {e.code} - {error_body}")
            raise
    
    def _local_generate(self, prompt: str, **kwargs) -> str:
        """Generate using local transformers pipeline."""
        pipe = self._get_pipeline()
        
        result = pipe(
            prompt,
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            do_sample=True,
            return_full_text=False
        )
        
        if result and len(result) > 0:
            return result[0].get("generated_text", "")
        return ""
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a completion.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated text
        """
        if self.backend == self.BACKEND_VLLM:
            return self._get_vllm_adapter().complete(prompt, **kwargs)
        elif self.backend == self.BACKEND_API:
            content = self._api_request(prompt, **kwargs)
        else:  # transformers
            content = self._local_generate(prompt, **kwargs)
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            tokens_used=0,  # Not easily available
            finish_reason="stop"
        )
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate a chat response.
        
        Formats messages into a prompt and uses completion.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with assistant's response
        """
        if self.backend == self.BACKEND_VLLM:
            return self._get_vllm_adapter().chat(messages, **kwargs)
        
        # Format messages into a prompt for API/transformers backends
        # Use chat template if available, otherwise simple format
        prompt_parts = []
        for msg in messages:
            if msg.role.value == "system":
                prompt_parts.append(f"<|system|>\n{msg.content}</s>")
            elif msg.role.value == "user":
                prompt_parts.append(f"<|user|>\n{msg.content}</s>")
            elif msg.role.value == "assistant":
                prompt_parts.append(f"<|assistant|>\n{msg.content}</s>")
        
        prompt_parts.append("<|assistant|>\n")
        prompt = "\n".join(prompt_parts)
        
        return self.complete(prompt, **kwargs)
    
    def is_available(self) -> bool:
        """Check if HuggingFace backend is accessible."""
        if self.backend == self.BACKEND_VLLM:
            return self._get_vllm_adapter().is_available()
        elif self.backend == self.BACKEND_API:
            if not self.token:
                return False
            try:
                # Try a minimal request
                self._api_request("Hi", max_tokens=5)
                return True
            except Exception as e:
                logger.warning(f"HuggingFace API not available: {e}")
                return False
        else:  # transformers
            try:
                import transformers
                import torch
                return True
            except ImportError:
                return False
