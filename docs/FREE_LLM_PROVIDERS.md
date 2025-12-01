# Free & Low-Cost LLM API Providers for BioPipelines

**Version**: 1.0.0  
**Date**: December 1, 2025  
**Status**: Research Complete

---

## Executive Summary

This document consolidates research from ChatGPT analysis and additional web exploration to identify free/cheap cloud LLM APIs that can be integrated into BioPipelines' agentic system. The goal is to **chain multiple free tiers** to create a resilient, cost-effective multi-provider architecture.

### Key Insights

1. **~80% of queries** can be handled by free tiers when using smart routing
2. **Provider cascade** with fallback prevents rate limit failures
3. **Most free tiers** have restrictive limits (20-50 req/day) but work well in aggregate
4. **OpenAI-compatible endpoints** make integration trivial
5. **Data privacy**: Some free tiers use prompts for training (check per-provider)

---

## Table of Contents

1. [Provider Comparison Matrix](#provider-comparison-matrix)
2. [Tier 1: Gateway APIs (Multi-Model)](#tier-1-gateway-apis-multi-model)
3. [Tier 2: Free Providers (Unlimited)](#tier-2-free-providers-unlimited)
4. [Tier 3: Trial Credit Providers](#tier-3-trial-credit-providers)
5. [Integration Strategy for BioPipelines](#integration-strategy-for-biopipelines)
6. [Implementation Plan](#implementation-plan)
7. [Privacy & Compliance Notes](#privacy--compliance-notes)

---

## Provider Comparison Matrix

### Best Free Providers (Ranked by Value)

| Provider | Free Tier Limits | Best Models | OpenAI Compatible | Priority |
|----------|------------------|-------------|-------------------|----------|
| **Google AI Studio** | 500 req/day, 1M tokens/day | Gemini 2.5 Pro/Flash | ✅ | 1 |
| **Groq** | 1K-14K req/day by model | Llama 3.3 70B, GPT-OSS 120B | ✅ | 2 |
| **Cerebras** | 14,400 req/day, 1M tokens/day | Llama 3.3 70B, Qwen3 235B | ✅ | 3 |
| **OpenRouter** | 50 req/day (free models) | 20+ free models | ✅ | 4 |
| **Mistral** | 1M tokens/month | Mistral Small, Codestral | ✅ | 5 |
| **GitHub Models** | Copilot tier dependent | GPT-4o, DeepSeek-R1, o3 | ✅ | 6 |
| **Cloudflare Workers AI** | 10K neurons/day | Llama, Mistral, Qwen | ✅ | 7 |
| **Cohere** | 1K req/month | Command-R, Aya | ✅ | 8 |

### Trial Credit Providers

| Provider | Credits | Duration | Best For |
|----------|---------|----------|----------|
| **Baseten** | $30 | Unlimited | Custom deployments |
| **SambaNova** | $5 | 3 months | DeepSeek, Llama |
| **Hyperbolic** | $1-$26 | Varies | DeepSeek V3, Qwen |
| **Scaleway** | 1M tokens | Unlimited | EU-based processing |
| **AI21** | $10 | 3 months | Jamba models |
| **Fireworks** | $1 | Unlimited | Fast inference |

---

## Tier 1: Gateway APIs (Multi-Model)

### 1.1 OpenRouter 🌟
**Priority: HIGH** - Single API to 400+ models

**Free Tier Details:**
```yaml
rate_limits:
  requests_per_minute: 20
  requests_per_day: 50  # Base free tier
  requests_per_day_with_topup: 1000  # After $10 lifetime topup

free_models:  # All marked :free
  - google/gemma-3-27b-it:free
  - meta-llama/llama-3.3-70b-instruct:free
  - mistralai/mistral-small-3.1-24b-instruct:free
  - qwen/qwen3-235b-a22b:free
  - qwen/qwen3-coder:free
  - deepseek/deepseek-r1t-chimera:free
  - x-ai/grok-4.1-fast:free
  - openai/gpt-oss-20b:free
  - moonshotai/kimi-k2:free

byok_free_tier:
  description: "Bring Your Own Key - 1M free requests/month"
  after_limit: "5% fee"
```

**Integration:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "https://biopipelines.app",
        "X-Title": "BioPipelines"
    }
)

# Use any free model
response = client.chat.completions.create(
    model="meta-llama/llama-3.3-70b-instruct:free",
    messages=[{"role": "user", "content": query}]
)
```

---

### 1.2 Together AI
**Priority: MEDIUM** - Strong open models, free endpoints

**Free Tier Details:**
```yaml
free_endpoints:  # Reduced rate limits
  - meta-llama/Llama-3.3-70B-Instruct-Turbo-Free  # 0.6 req/min (36/hour)
  - meta-llama/Llama-Vision-Free
  - deepseek-ai/DeepSeek-R1-Distill-Llama-70B-Free

signup_credit: $1  # ~50K tokens

paid_models:  # Very cheap
  deepseek_v3: "$0.14/M input, $0.28/M output"
  llama_3_3_70b: "$0.54/M input"
```

**Integration:**
```python
client = OpenAI(
    base_url="https://api.together.ai/v1",
    api_key=os.environ["TOGETHER_API_KEY"]
)
```

---

## Tier 2: Free Providers (Unlimited)

### 2.1 Google AI Studio 🌟🌟
**Priority: HIGHEST** - Best free tier overall

**Free Tier Details:**
```yaml
gemini_2.5_pro:
  tokens_per_day: 3,000,000
  requests_per_day: 50
  requests_per_minute: 2
  
gemini_2.5_flash:
  tokens_per_minute: 250,000
  requests_per_day: 250
  requests_per_minute: 10
  
gemini_2.5_flash_lite:
  tokens_per_minute: 250,000
  requests_per_day: 1000
  requests_per_minute: 15

gemini_2.0_flash:
  tokens_per_minute: 1,000,000
  requests_per_day: 200
  requests_per_minute: 15

gemma_3_27b:
  tokens_per_minute: 15,000
  requests_per_day: 14,400
  requests_per_minute: 30

tools:
  google_search: "500 req/day free"
  code_execution: "Free"
  
privacy_note: "Data used for training in free tier (except UK/CH/EEA/EU)"
```

**Integration:**
```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# OpenAI-compatible endpoint also available
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"]
)
```

---

### 2.2 Groq 🌟
**Priority: HIGH** - Fastest inference, generous free tier

**Free Tier Details:**
```yaml
models:
  llama-3.3-70b-versatile:
    requests_per_day: 1000
    tokens_per_minute: 12,000
    speed: "280 tok/s"
    
  llama-3.1-8b-instant:
    requests_per_day: 14,400
    tokens_per_minute: 6,000
    speed: "560 tok/s"
    
  openai/gpt-oss-120b:
    requests_per_day: 1000
    tokens_per_minute: 8,000
    speed: "500 tok/s"
    
  groq/compound:  # Agentic with web search + code execution
    requests_per_day: 250
    tokens_per_minute: 70,000
    features: ["web_search", "code_execution", "browser"]

  whisper-large-v3:  # Speech-to-text
    audio_seconds_per_minute: 7,200
    requests_per_day: 2,000
```

**Integration:**
```python
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Or OpenAI-compatible
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)

# Use Compound for agentic tasks with built-in tools
response = client.chat.completions.create(
    model="groq/compound",
    messages=[{"role": "user", "content": "Search for RNA-seq best practices"}]
)
```

---

### 2.3 Cerebras 🌟
**Priority: HIGH** - Very generous, fast inference

**Free Tier Details:**
```yaml
models:
  gpt-oss-120b:
    requests_per_day: 14,400
    tokens_per_day: 1,000,000
    tokens_per_minute: 60,000
    
  qwen3-235b-a22b:  # Massive model, free!
    requests_per_day: 14,400
    tokens_per_day: 1,000,000
    
  qwen3-coder-480b:  # Best coding model
    requests_per_day: 100
    tokens_per_day: 1,000,000
    
  llama-3.3-70b:
    requests_per_day: 14,400
    tokens_per_day: 1,000,000
    
  llama-4-scout:
    requests_per_day: 14,400
    tokens_per_day: 1,000,000
```

**Integration:**
```python
client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ["CEREBRAS_API_KEY"]
)
```

---

### 2.4 Mistral La Plateforme
**Priority: MEDIUM** - Official Mistral models

**Free Tier Details:**
```yaml
experiment_plan:  # Requires phone verification
  requests_per_second: 1
  tokens_per_minute: 500,000
  tokens_per_month: 1,000,000,000  # 1B tokens!
  
codestral:  # Separate free tier
  requests_per_minute: 30
  requests_per_day: 2000
  
privacy_note: "Experiment plan requires opting into data training"
```

---

### 2.5 GitHub Models 🌟
**Priority: HIGH** (if you have Copilot)

**Models Available (Free with GitHub account):**
```yaml
models:
  - OpenAI GPT-4o
  - OpenAI GPT-4o-mini
  - OpenAI o1, o3, o3-mini
  - DeepSeek-R1
  - DeepSeek-V3
  - Llama-3.3-70B
  - Llama-4-Scout
  - Llama-4-Maverick
  - Grok 3, Grok 3 Mini
  - Mistral Medium 3

limits:  # Depends on subscription tier
  free_tier: "Low - ~50 req/day per model"
  copilot_pro: "Higher limits"
  copilot_pro_plus: "Much higher limits"
  
note: "Restrictive input/output token limits"
```

**Integration:**
```python
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"]
)
```

---

### 2.6 Cloudflare Workers AI
**Priority: LOW** - Good backup option

**Free Tier Details:**
```yaml
limits:
  neurons_per_day: 10,000  # Varies by model size
  
models:  # Subset of available
  - @cf/meta/llama-3.3-70b-instruct-fp8
  - @cf/mistral/mistral-small-3.1-24b-instruct
  - @cf/qwen/qwen-2.5-coder-32b-instruct
  - @cf/openai/gpt-oss-120b
  - @cf/deepseek/deepseek-r1-distill-qwen-32b
```

---

### 2.7 NVIDIA NIM
**Priority: MEDIUM** - GPU-optimized inference

**Free Tier Details:**
```yaml
limits:
  requests_per_minute: 40
  
requirements:
  - Phone number verification
  - Context windows may be limited
  
models:
  - Many open models with NVIDIA optimizations
```

---

### 2.8 Cohere
**Priority: LOW** - Good for multilingual

**Free Tier Details:**
```yaml
limits:
  requests_per_minute: 20
  requests_per_month: 1000
  
models:
  - command-r-plus-08-2024
  - c4ai-aya-expanse-32b  # Multilingual
```

---

## Tier 3: Trial Credit Providers

### 3.1 SambaNova Cloud
```yaml
credits: $5
duration: 3 months
models:
  - DeepSeek-R1-0528
  - DeepSeek-V3.1
  - Llama-3.3-70B
  - Whisper-Large-v3
speed: "Very fast - custom hardware"
```

### 3.2 Hyperbolic
```yaml
credits: "$1 signup, $25 with survey"
models:
  - DeepSeek-V3
  - Qwen3-235B
  - gpt-oss-120b
```

### 3.3 Scaleway
```yaml
credits: "1M free tokens"
location: "EU (GDPR compliant)"
models:
  - DeepSeek-R1
  - Llama-3.3-70B
  - Gemma-3-27B
```

### 3.4 Fireworks AI
```yaml
credits: "$1"
features: "Very fast inference, good for production"
models:
  - DeepSeek-V3
  - Llama-4-Scout
  - Many vision/audio models
```

---

## Integration Strategy for BioPipelines

### 5.1 Provider Priority System

```python
# providers/cascade.py
PROVIDER_PRIORITY = {
    # Tier 1: Highest daily limits, most reliable
    "gemini": 1,           # 250+ req/day, 1M tokens/day
    "cerebras": 2,         # 14,400 req/day per model
    "groq": 3,             # 1,000+ req/day, very fast
    
    # Tier 2: Good limits, diverse models
    "openrouter_free": 4,  # 50 req/day, many models
    "mistral": 5,          # 1B tokens/month
    "github_models": 6,    # If Copilot subscription
    
    # Tier 3: Lower limits, good backup
    "cloudflare": 7,       # 10K neurons/day
    "together_free": 8,    # 36 req/hour
    "cohere": 9,           # 1K req/month
    
    # Tier 4: Trial credits (use sparingly)
    "sambanova": 10,
    "hyperbolic": 11,
    "fireworks": 12,
    
    # Tier 5: Local fallback
    "ollama": 20,
    "vllm": 21,
    
    # Tier 6: Paid fallback (last resort)
    "lightning": 50,       # DeepSeek, cheap
    "openai": 99,          # Expensive, always available
}
```

### 5.2 Smart Routing Logic

```python
class MultiProviderRouter:
    """
    Routes requests to optimal provider based on:
    1. Task type (reasoning, coding, simple)
    2. Provider availability (rate limits, health)
    3. Cost optimization
    4. Response quality requirements
    """
    
    TASK_ROUTING = {
        # For complex reasoning
        "reasoning": ["cerebras:qwen3-235b", "gemini:pro", "groq:compound"],
        
        # For code generation
        "coding": ["cerebras:qwen3-coder", "mistral:codestral", "groq:gpt-oss-120b"],
        
        # For simple classification/parsing
        "simple": ["gemini:flash-lite", "groq:llama-8b", "cloudflare:llama-8b"],
        
        # For intent parsing (arbiter)
        "intent_parsing": ["gemini:flash", "groq:llama-70b", "cerebras:llama-70b"],
    }
    
    def route(self, task_type: str, query: str) -> ProviderResponse:
        """Route to best available provider."""
        candidates = self.TASK_ROUTING.get(task_type, self.TASK_ROUTING["simple"])
        
        for provider_model in candidates:
            provider, model = provider_model.split(":")
            
            if self._is_available(provider):
                try:
                    return self._call_provider(provider, model, query)
                except RateLimitError:
                    self._mark_cooldown(provider)
                    continue
        
        # Fall back to local or paid
        return self._fallback(query)
```

### 5.3 Rate Limit Tracking

```python
@dataclass
class ProviderStatus:
    requests_today: int = 0
    tokens_today: int = 0
    last_reset: datetime = None
    cooldown_until: datetime = None
    
class RateLimitTracker:
    """Track usage across all providers."""
    
    DAILY_LIMITS = {
        "gemini": {"requests": 250, "tokens": 1_000_000},
        "cerebras": {"requests": 14_400, "tokens": 1_000_000},
        "groq": {"requests": 1_000, "tokens": 100_000},
        "openrouter": {"requests": 50, "tokens": 500_000},
        "mistral": {"requests": 86_400, "tokens": 33_000_000},  # 1B/month
    }
    
    def can_use(self, provider: str) -> bool:
        status = self.status[provider]
        limits = self.DAILY_LIMITS[provider]
        
        # Check cooldown
        if status.cooldown_until and datetime.now() < status.cooldown_until:
            return False
        
        # Check daily limits
        if status.requests_today >= limits["requests"]:
            return False
        
        return True
    
    def record_usage(self, provider: str, tokens: int):
        status = self.status[provider]
        status.requests_today += 1
        status.tokens_today += tokens
```

### 5.4 Integration with Existing CascadingProviderRouter

Update `src/workflow_composer/providers/router.py`:

```python
# Add new providers to existing cascade
PROVIDER_PRIORITY = {
    # Cloud providers with free tiers
    "gemini": 1,              # NEW: Google AI Studio
    "cerebras": 2,            # NEW: Cerebras Cloud
    "groq": 3,                # NEW: Groq Cloud
    "openrouter_free": 4,     # NEW: OpenRouter free models
    
    # Existing providers
    "lightning": 5,           # Lightning.ai (DeepSeek)
    "github_models": 6,       # GitHub Models
    "mistral": 7,             # Mistral La Plateforme
    
    # Local providers
    "ollama": 15,
    "vllm": 16,
    
    # Expensive fallback
    "openai": 99,
}

# Model mappings per provider
PROVIDER_MODELS = {
    "gemini": "gemini-2.5-flash",
    "cerebras": "llama-3.3-70b",
    "groq": "llama-3.3-70b-versatile",
    "openrouter_free": "meta-llama/llama-3.3-70b-instruct:free",
}
```

---

## Implementation Plan

### Phase 1: Add New Providers (Week 1)

```python
# src/workflow_composer/providers/gemini.py
class GeminiProvider(BaseProvider):
    """Google AI Studio / Gemini API provider."""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEFAULT_MODEL = "gemini-2.5-flash"
    
    def __init__(self):
        self.client = OpenAI(
            base_url=self.BASE_URL,
            api_key=os.environ.get("GEMINI_API_KEY")
        )

# src/workflow_composer/providers/cerebras.py
class CerebrasProvider(BaseProvider):
    """Cerebras Cloud provider."""
    
    BASE_URL = "https://api.cerebras.ai/v1"
    DEFAULT_MODEL = "llama-3.3-70b"

# src/workflow_composer/providers/groq.py  
class GroqProvider(BaseProvider):
    """Groq Cloud provider."""
    
    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
```

### Phase 2: Update Router (Week 1)

```python
# Update router.py to include new providers
# Add rate limit tracking per provider
# Add automatic fallback on 429 errors
```

### Phase 3: Environment Configuration (Week 1)

```bash
# .env additions
GEMINI_API_KEY=your_key_here
CEREBRAS_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
```

### Phase 4: Testing & Monitoring (Week 2)

```python
# Add metrics for provider usage
# Track cost savings vs single-provider
# Monitor rate limit hits and fallback frequency
```

---

## Privacy & Compliance Notes

### Data Training Opt-Out

| Provider | Free Tier Training | Opt-Out Available |
|----------|-------------------|-------------------|
| Google AI Studio | Yes (outside EU) | Use paid tier |
| Groq | No | N/A |
| Cerebras | No | N/A |
| OpenRouter | Varies by model | Check model docs |
| Mistral (Experiment) | Yes | Use paid tier |
| GitHub Models | No | N/A |

### Recommendations for Sensitive Data

1. **Use Groq or Cerebras** for sensitive queries (no training)
2. **Use local models** (Ollama, vLLM) for PHI/PII
3. **Route to Google only** for non-sensitive queries
4. **Implement query classification** to route appropriately

---

## Cost Estimation

### Free Tier Capacity (Per Day)

```
Gemini Flash:        250 req × 4K tokens = 1M tokens
Cerebras Llama 70B:  14,400 req × 4K tokens = 57M tokens  
Groq Llama 70B:      1,000 req × 4K tokens = 4M tokens
OpenRouter Free:     50 req × 4K tokens = 200K tokens
-----------------------------------------------------------
Total Free:          ~62M tokens/day (if all used)
```

### BioPipelines Usage Estimate

```
Intent Parsing:      ~100 queries/day × 500 tokens = 50K tokens
Workflow Generation: ~20 queries/day × 2K tokens = 40K tokens
Diagnostics:         ~10 queries/day × 1K tokens = 10K tokens
-----------------------------------------------------------
Total Daily Usage:   ~100K tokens/day

Free Tier Coverage:  ~620x our needs!
```

### When Free Tiers Exhaust

| Monthly Volume | Recommended Approach | Est. Cost |
|----------------|---------------------|-----------|
| < 1M tokens | Free tiers only | $0 |
| 1M-10M tokens | Free + Lightning.ai | $0-5/mo |
| 10M-100M tokens | Free + Together AI | $5-50/mo |
| > 100M tokens | Dedicated vLLM cluster | $100+/mo |

---

## Quick Start

### 1. Get API Keys

```bash
# Required (choose 2-3 minimum)
# Google AI Studio: https://aistudio.google.com/apikey
# Groq: https://console.groq.com/keys
# Cerebras: https://cloud.cerebras.ai/
# OpenRouter: https://openrouter.ai/settings/keys

# Optional
# Mistral: https://console.mistral.ai/
# Together: https://api.together.ai/settings/api-keys
```

### 2. Configure Environment

```bash
# Add to .env
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
CEREBRAS_API_KEY=csk-...
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Test Providers

```python
from workflow_composer.providers import get_cascading_router

router = get_cascading_router()
status = router.get_status()
print(f"Available providers: {status['available']}")

# Test a query
response = router.complete("What is RNA-seq?")
print(f"Provider used: {response.provider}")
```

---

## References

- [GitHub: cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) - Maintained list
- [Google AI Pricing](https://ai.google.dev/pricing)
- [Groq Documentation](https://console.groq.com/docs)
- [Cerebras Cloud](https://cloud.cerebras.ai/)
- [OpenRouter Models](https://openrouter.ai/models)
- [Together AI Models](https://docs.together.ai/docs/inference-models)

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-01 | Initial research document created |
