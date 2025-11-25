"""
vLLM LLM Adapter
================

Adapter for running open-source HuggingFace models via vLLM server.

vLLM provides high-throughput, low-latency inference for LLMs with:
- PagedAttention for efficient memory management
- Continuous batching for high throughput
- OpenAI-compatible API server

Requirements:
    - vLLM server running: python -m vllm.entrypoints.openai.api_server --model <model>
    - Or via Docker: docker run --gpus all -p 8000:8000 vllm/vllm-openai --model <model>

Usage:
    from workflow_composer.llm import VLLMAdapter
    
    # Connect to local vLLM server
    llm = VLLMAdapter(
        model="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000"
    )
    
    response = llm.complete("Explain RNA-seq analysis")

Supported Models (examples):
    - meta-llama/Llama-3.1-8B-Instruct
    - meta-llama/Llama-3.1-70B-Instruct
    - mistralai/Mistral-7B-Instruct-v0.3
    - mistralai/Mixtral-8x7B-Instruct-v0.1
    - Qwen/Qwen2.5-7B-Instruct
    - deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
    - microsoft/Phi-3.5-mini-instruct
    - codellama/CodeLlama-34b-Instruct-hf
"""

import json
import os
import logging
from typing import List, Iterator, Optional, Dict, Any
import urllib.request
import urllib.error

from .base import LLMAdapter, LLMResponse, Message

logger = logging.getLogger(__name__)


class VLLMAdapter(LLMAdapter):
    """
    Adapter for vLLM OpenAI-compatible API server.
    
    vLLM serves HuggingFace models with high performance using:
    - PagedAttention for efficient GPU memory usage
    - Continuous batching for high throughput
    - OpenAI-compatible REST API
    
    Attributes:
        model: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        base_url: vLLM server URL (default: http://localhost:8000)
        temperature: Sampling temperature (0.0 - 1.0)
        max_tokens: Maximum tokens in response
    """
    
    DEFAULT_BASE_URL = "http://localhost:8000"
    
    # Recommended models for bioinformatics
    RECOMMENDED_MODELS = {
        # General purpose
        "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        
        # Code-focused (good for workflow generation)
        "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
        "deepseek-coder": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        
        # Smaller/faster models
        "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
        "gemma-2b": "google/gemma-2-2b-it",
    }
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120,
        **kwargs
    ):
        """
        Initialize vLLM adapter.
        
        Args:
            model: HuggingFace model ID or alias from RECOMMENDED_MODELS
            base_url: vLLM server URL (default: from VLLM_BASE_URL env or localhost:8000)
            api_key: Optional API key (for authenticated deployments)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        # Resolve model aliases
        if model in self.RECOMMENDED_MODELS:
            model = self.RECOMMENDED_MODELS[model]
        
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        self.base_url = (
            base_url or 
            os.environ.get("VLLM_BASE_URL") or 
            self.DEFAULT_BASE_URL
        ).rstrip("/")
        
        self.api_key = api_key or os.environ.get("VLLM_API_KEY")
        self.timeout = timeout
    
    @property
    def provider_name(self) -> str:
        return "vllm"
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to vLLM server (OpenAI-compatible API)."""
        url = f"{self.base_url}/v1{endpoint}"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"vLLM API error: {e.code} - {error_body}")
            raise RuntimeError(f"vLLM request failed: {e.code} - {error_body}")
        except urllib.error.URLError as e:
            logger.error(f"vLLM connection error: {e}")
            raise ConnectionError(
                f"Failed to connect to vLLM server at {self.base_url}. "
                f"Make sure vLLM is running: python -m vllm.entrypoints.openai.api_server --model {self.model}"
            ) from e
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a completion.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional vLLM/OpenAI parameters
            
        Returns:
            LLMResponse with generated text
        """
        # Use chat endpoint for instruction-tuned models
        messages = [Message.user(prompt)]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate a chat response.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse with assistant's response
        """
        data = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        # Add optional parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop", "seed", 
                    "top_k", "repetition_penalty"]:
            if key in kwargs:
                data[key] = kwargs[key]
        
        response = self._make_request("/chat/completions", data)
        
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})
        
        return LLMResponse(
            content=message.get("content", ""),
            model=response.get("model", self.model),
            provider=self.provider_name,
            tokens_used=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=response
        )
    
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream a completion token by token.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Yields:
            String chunks of the response
        """
        messages = [Message.user(prompt)]
        yield from self.chat_stream(messages, **kwargs)
    
    def chat_stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        """
        Stream a chat response token by token.
        
        Args:
            messages: Conversation messages
            **kwargs: Additional parameters
            
        Yields:
            String chunks of the response
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }
        
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        json_str = line[6:]
                        if json_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"vLLM streaming error: {e}")
            raise
    
    def get_models(self) -> List[str]:
        """
        Get list of models available on the vLLM server.
        
        Returns:
            List of model names
        """
        url = f"{self.base_url}/v1/models"
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        request = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                return [m.get("id", "") for m in result.get("data", [])]
        except Exception as e:
            logger.warning(f"Failed to get vLLM models: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if vLLM server is accessible."""
        try:
            models = self.get_models()
            return len(models) > 0
        except Exception as e:
            logger.warning(f"vLLM not available: {e}")
            return False
    
    @classmethod
    def get_recommended_models(cls) -> Dict[str, str]:
        """Get dictionary of recommended model aliases and their full names."""
        return cls.RECOMMENDED_MODELS.copy()
    
    @staticmethod
    def get_launch_command(
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        port: int = 8000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None
    ) -> str:
        """
        Generate command to launch vLLM server.
        
        Args:
            model: HuggingFace model ID
            port: Port to serve on
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            quantization: Quantization method (awq, gptq, squeezellm)
            
        Returns:
            Shell command to start vLLM server
        """
        cmd = [
            "python -m vllm.entrypoints.openai.api_server",
            f"--model {model}",
            f"--port {port}",
            f"--tensor-parallel-size {tensor_parallel_size}",
            f"--gpu-memory-utilization {gpu_memory_utilization}",
        ]
        
        if max_model_len:
            cmd.append(f"--max-model-len {max_model_len}")
        
        if quantization:
            cmd.append(f"--quantization {quantization}")
        
        return " \\\n    ".join(cmd)
