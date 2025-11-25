"""
LLM Adapter Layer
=================

Provides a unified interface to multiple LLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Anthropic (Claude 3.5, Claude 3)
- Ollama (local models: Llama3, Mistral, CodeLlama)
- HuggingFace (any HF model via API, transformers, or vLLM)
- vLLM (high-performance GPU inference for HF models)
- Custom endpoints

Usage:
    from workflow_composer.llm import get_llm, VLLMAdapter
    
    # Using factory
    llm = get_llm("vllm", model="meta-llama/Llama-3.1-8B-Instruct")
    
    # OpenAI
    llm = get_llm("openai", model="gpt-4o")
    
    # Direct instantiation
    llm = VLLMAdapter(model="codellama/CodeLlama-34b-Instruct-hf")
    
    # Generate completion
    response = llm.complete("Explain RNA-seq analysis")
    
    # List available providers
    providers = list_providers()
    
    # Check which are available
    available = check_providers()
"""

from .base import LLMAdapter, Message, LLMResponse
from .ollama_adapter import OllamaAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .vllm_adapter import VLLMAdapter
from .factory import get_llm, LLMFactory, list_providers, check_providers, register_provider

__all__ = [
    "LLMAdapter",
    "Message", 
    "LLMResponse",
    "OllamaAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "VLLMAdapter",
    "get_llm",
    "LLMFactory",
    "list_providers",
    "check_providers",
    "register_provider"
]
