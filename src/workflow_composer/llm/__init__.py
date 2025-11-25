"""
LLM Adapter Layer
=================

Provides a unified interface to multiple LLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Anthropic (Claude 3.5, Claude 3)
- Ollama (local models: Llama3, Mistral, CodeLlama)
- HuggingFace (any HF model via API, transformers, or vLLM)
- vLLM (high-performance GPU inference for HF models)
- Lightning.ai (30M FREE tokens/month, unified API for all models)
- Custom endpoints

Usage:
    from workflow_composer.llm import get_llm, VLLMAdapter, LightningAdapter
    
    # Using factory
    llm = get_llm("vllm", model="meta-llama/Llama-3.1-8B-Instruct")
    
    # Lightning.ai (recommended - 30M free tokens!)
    llm = get_llm("lightning", model="deepseek/deepseek-v3")
    
    # OpenAI
    llm = get_llm("openai", model="gpt-4o")
    
    # Direct instantiation
    llm = VLLMAdapter(model="codellama/CodeLlama-34b-Instruct-hf")
    llm = LightningAdapter(task="workflow_generation")
    
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
from .lightning_adapter import LightningAdapter, LightningModelRouter, create_lightning_llm
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
    "LightningAdapter",
    "LightningModelRouter",
    "create_lightning_llm",
    "get_llm",
    "LLMFactory",
    "list_providers",
    "check_providers",
    "register_provider"
]
