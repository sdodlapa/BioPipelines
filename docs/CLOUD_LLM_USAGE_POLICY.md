# Cloud LLM Usage Policy for BioPipelines

**Version**: 3.2.0  
**Date**: December 4, 2025  
**Status**: Active

---

## Executive Summary

This document defines the **Cloud LLM Usage Policy** for BioPipelines, ensuring optimal cost efficiency while maintaining service quality. The policy establishes a **tiered cascade system** that prioritizes free-tier providers, then subscribed/budgeted providers, and finally pay-as-you-go options.

### Key Principles

1. **Free First**: Always try free-tier providers before paid options
2. **Cheapest Within Budget**: Use the most cost-effective model for each task
3. **Quality Where It Matters**: Use higher-quality models only for complex reasoning
4. **Fallback Resilience**: Multiple provider cascade prevents outages
5. **Cost Tracking**: All usage is tracked for budget management

### Quick Status Check

```bash
# Check all provider status and quotas
python -c "from src.workflow_composer.providers.usage_tracker import get_tracker; get_tracker().print_status()"

# Get cost summary
python -c "from src.workflow_composer.providers.usage_tracker import get_tracker; import json; print(json.dumps(get_tracker().get_cost_summary(), indent=2))"

# Get model recommendation for a task
python -c "from src.workflow_composer.providers.usage_tracker import get_tracker; print(get_tracker().recommend_model(task='code_generation'))"

# Browse complete model catalog (158 models!)
python -c "from src.workflow_composer.providers.model_registry import ModelRegistry; ModelRegistry().print_summary()"

# Get all free models
python -c "from src.workflow_composer.providers.model_registry import get_free_models; print([str(m) for m in get_free_models()])"
```

### New: Complete Model Catalog

As of v3.2.0, BioPipelines includes a **complete model registry** with 158 models across 7 providers:

- **Configuration**: `config/provider_models.yaml` - Complete model definitions
- **Python API**: `src/workflow_composer/providers/model_registry.py` - Programmatic access

```python
from workflow_composer.providers import (
    get_model_registry,
    get_free_models,
    get_recommended_model,
)

# Get recommended model for code generation
model = get_recommended_model("code_generation")
print(f"Using: {model}")  # DeepSeek Chat V3 [openrouter]

# Get all free models (43 available!)
for model in get_free_models():
    print(f"  - {model}")
```

---

## Table of Contents

1. [Current Provider Configuration](#1-current-provider-configuration)
2. [Comprehensive Model Guide](#2-comprehensive-model-guide)
3. [Model Selection by Task](#3-model-selection-by-task)
4. [Usage Tracking & Quotas](#4-usage-tracking--quotas)
5. [Lightning.ai Status](#5-lightningai-status)
6. [Cost Comparison](#6-cost-comparison)
7. [Implementation Status](#7-implementation-status)
8. [Recommendations](#8-recommendations)
9. [Environment Variables](#9-environment-variables)

---

## 1. Current Provider Configuration

### Active Providers (as of Dec 2025)

| Provider | Priority | Status | API Key Location | Primary Use |
|----------|----------|--------|------------------|-------------|
| **Gemini** | 1 | ✅ Active | `.secrets/google_api_key` | Default (best free tier) |
| **Cerebras** | 2 | ✅ Active | `.secrets/cerebras_key` | Large models (235B free!) |
| **Groq** | 3 | ✅ Active | `.secrets/groq_key` | Fast inference |
| **OpenRouter** | 4 | ✅ Active | `.secrets/openrouter_key` | Multi-model gateway |
| **Lightning.ai** | 5 | ✅ Active | `.secrets/lightning_key` | 30M tokens/mo free |
| **GitHub Models** | 6 | ✅ Active | `.secrets/github_token` | Copilot Pro+ integration |
| **OpenAI** | 99 | ✅ Active | `.secrets/openai_key` | Last resort (paid) |
| **Ollama** | 15 | ⚠️ Local | — | Local development |
| **vLLM** | 16 | ⚠️ Local | — | HPC batch processing |

### Current Cascade (Working!)

```python
# The router automatically cascades through providers in priority order
# Registry: src/workflow_composer/providers/registry.py

# Verified working cascade (tested Dec 4, 2025):
# 1. Gemini (gemini-2.0-flash) → ✅ 518ms response
# 2. Cerebras → fallback if Gemini rate-limited
# 3. Groq → fallback if Cerebras rate-limited
# 4. OpenRouter → fallback if Groq rate-limited
# 5. Lightning.ai (lightning-ai/gpt-oss-20b) → ✅ FIXED! 30M tokens/mo
# 6. GitHub Models → Copilot Pro+ quota
# ... continues to OpenAI (priority 99) as last resort
```

### Key Files

- **Secret Keys**: `.secrets/` directory (loaded by `start_server.sh`)
- **Provider Registry**: `src/workflow_composer/providers/registry.py`
- **Router Logic**: `src/workflow_composer/providers/router.py`
- **Individual Providers**: `src/workflow_composer/providers/{gemini,groq,cerebras}.py`

---

## 2. Tiered Provider Cascade

### Priority Order (RECOMMENDED)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TIER 1: FREE UNLIMITED                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Gemini    │  │   Groq      │  │  Cerebras   │  │ OpenRouter  │ │
│  │ Flash-Lite  │  │ Llama-8B    │  │ gpt-oss-120b│  │   :free     │ │
│  │ 1000 req/d  │  │14400 req/d  │  │14400 req/d  │  │   50 req/d  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                    TIER 2: GITHUB COPILOT PRO+                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    GitHub Models API                             │ │
│  │  GPT-4o, DeepSeek-R1, o3, Llama-3.3-70B (within Pro+ limits)   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                    TIER 3: LIGHTNING.AI BUDGET                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ GPT-OSS-20B │  │DeepSeek V3.1│  │ Llama 3.3   │  │ gpt-oss-120b│ │
│  │ $0.0125/1M  │  │ $0.14/1M    │  │  $0.80/1M   │  │  $0.05/1M   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                    TIER 4: LOCAL (NO COST)                           │
│  ┌─────────────┐  ┌─────────────┐                                   │
│  │   Ollama    │  │    vLLM     │                                   │
│  │  (offline)  │  │  (HPC GPU)  │                                   │
│  └─────────────┘  └─────────────┘                                   │
├─────────────────────────────────────────────────────────────────────┤
│                    TIER 5: PAY AS YOU GO                             │
│  ┌─────────────┐  ┌─────────────┐                                   │
│  │   OpenAI    │  │  Anthropic  │                                   │
│  │ ($$$)       │  │   ($$$)     │                                   │
│  └─────────────┘  └─────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Cascade Logic

```python
# Recommended cascade order in router
PROVIDER_PRIORITY = {
    # Tier 1: Free unlimited (daily resets)
    "groq_free": 1,              # 14,400 req/day for Llama-8B
    "cerebras_free": 2,          # 14,400 req/day, 1M tokens
    "gemini_free": 3,            # 1,000 req/day for Flash-Lite
    "openrouter_free": 4,        # 50 req/day (limited but diverse models)
    
    # Tier 2: Subscription (GitHub Copilot Pro+)
    "github_models": 5,          # Within Pro+ quota
    
    # Tier 3: Lightning.ai (cheap pay-as-go)
    "lightning_gpt_oss_20b": 10, # $0.0125/M - CHEAPEST
    "lightning_gpt_oss_120b": 11,# $0.05/M - Very cheap
    "lightning_deepseek": 12,    # $0.14/M - Good for reasoning
    "lightning_llama": 13,       # $0.80/M - General purpose
    
    # Tier 4: Local (when available)
    "ollama": 20,
    "vllm": 21,
    
    # Tier 5: Direct APIs (expensive, last resort)
    "openai": 99,
    "anthropic": 100,
}
```

---

## 3. Model Selection by Task

### Task-to-Model Mapping

| Task Type | Recommended Model | Provider | Cost/1M tokens | Notes |
|-----------|------------------|----------|----------------|-------|
| **Simple Classification** | Llama-3.1-8B | Groq (free) | $0 | Fast, 14K req/day |
| **Intent Parsing** | GPT-OSS-20B | Lightning | $0.0125 | Cheapest paid |
| **Entity Extraction** | Llama-3.3-70B | Cerebras (free) | $0 | 14K req/day |
| **Workflow Generation** | DeepSeek-V3.1 | Lightning | $0.14 | Best reasoning/$  |
| **Code Generation** | DeepSeek-V3.1 | Lightning | $0.14 | Excellent for code |
| **Complex Reasoning** | gpt-oss-120b | Cerebras/Lightning | $0-0.05 | Large model |
| **Scientific QA** | Gemini 2.5 Pro | Google (free) | $0 | 50 req/day |
| **Fallback** | GPT-4o-mini | OpenAI | $0.15 | Reliable |

### Current Implementation

```python
# From lightning_adapter.py
TASK_MODELS = {
    "intent_parsing": "deepseek-ai/DeepSeek-V3",      # Could use cheaper GPT-OSS-20B
    "workflow_generation": "deepseek-ai/DeepSeek-V3",
    "module_creation": "deepseek-ai/DeepSeek-V3",
    "code_generation": "deepseek-ai/DeepSeek-V3",
    "scientific_analysis": "Qwen/Qwen2.5-72B-Instruct",
    "chat": "meta-llama/Llama-3.3-70B-Instruct",
    "quick_response": "meta-llama/Llama-3.1-8B-Instruct",
    "high_quality": "deepseek-ai/DeepSeek-V3",
}
```

---

## 4. Cost Comparison

### Complete Model Catalog by Provider

---

### 🟢 Google AI Studio (Gemini) - BEST FREE TIER

**Dashboard**: https://aistudio.google.com/usage  
**API Key**: https://aistudio.google.com/apikey

| Model | Model ID | Context | Free RPM | Free RPD | Paid Input $/M | Paid Output $/M | Best For |
|-------|----------|---------|----------|----------|----------------|-----------------|----------|
| **Gemini 3 Pro** | `gemini-3-pro-preview` | 200K+ | — | — | $2.00-4.00 | $12.00-18.00 | Best multimodal, agentic |
| **Gemini 2.5 Pro** | `gemini-2.5-pro` | 1M | 2 | 50 | $1.25-2.50 | $10.00-15.00 | Complex reasoning, code |
| **Gemini 2.5 Flash** ⭐ | `gemini-2.5-flash` | 1M | 10 | 250 | $0.30 | $2.50 | Large scale, agentic |
| **Gemini 2.5 Flash-Lite** ⭐ | `gemini-2.5-flash-lite` | 1M | 15 | 1,000 | $0.10 | $0.40 | Cheapest, high volume |
| **Gemini 2.0 Flash** | `gemini-2.0-flash` | 1M | 15 | 200 | $0.10 | $0.40 | Balanced multimodal |
| **Gemini 2.0 Flash-Lite** | `gemini-2.0-flash-lite` | 1M | 30 | 200 | $0.075 | $0.30 | Cost effective |
| **Gemma 3 & 3n** | `gemma-3-*` | — | 30 | 14,400 | FREE | FREE | Local deployment |
| **Gemini Embedding** | `gemini-embedding-001` | — | 100 | 1,000 | $0.15 | — | Vector embeddings |

**Free Tier Limits**: 1M+ tokens/day, resets at midnight Pacific time.

---

### 🟢 Groq - FASTEST INFERENCE

**Dashboard**: https://console.groq.com/usage  
**API Key**: https://console.groq.com/keys

| Model | Model ID | Speed | Context | TPM | RPM | Input $/M | Output $/M | Best For |
|-------|----------|-------|---------|-----|-----|-----------|------------|----------|
| **OpenAI GPT-OSS 120B** ⭐ | `openai/gpt-oss-120b` | 500 t/s | 131K | 250K | 1K | $0.15 | $0.60 | General, reasoning |
| **OpenAI GPT-OSS 20B** | `openai/gpt-oss-20b` | 1000 t/s | 131K | 250K | 1K | $0.075 | $0.30 | Fast tasks |
| **Llama 3.3 70B** ⭐ | `llama-3.3-70b-versatile` | 280 t/s | 131K | 300K | 1K | $0.59 | $0.79 | General purpose |
| **Llama 3.1 8B** | `llama-3.1-8b-instant` | 560 t/s | 131K | 250K | 1K | $0.05 | $0.08 | Quick responses |
| **Llama 4 Maverick 17B** | `meta-llama/llama-4-maverick-17b-128e-instruct` | 600 t/s | 131K | 300K | 1K | $0.20 | $0.60 | Preview |
| **Llama 4 Scout 17B** | `meta-llama/llama-4-scout-17b-16e-instruct` | 750 t/s | 131K | 300K | 1K | $0.11 | $0.34 | Preview |
| **Qwen 3 32B** | `qwen/qwen3-32b` | 400 t/s | 131K | 300K | 1K | $0.29 | $0.59 | Code, reasoning |
| **Kimi K2** | `moonshotai/kimi-k2-instruct-0905` | 200 t/s | 262K | 250K | 1K | $1.00 | $3.00 | Long context |
| **Llama Guard 4 12B** | `meta-llama/llama-guard-4-12b` | 1200 t/s | 131K | 30K | 100 | $0.20 | $0.20 | Content moderation |
| **Groq Compound** | `groq/compound` | 450 t/s | 131K | 200K | 200 | — | — | Web search + code |
| **Whisper Large V3** | `whisper-large-v3` | — | — | — | 300 | $0.111/hr | — | Speech-to-text |

**Free Tier**: Very generous - 14,400 req/day for Llama 8B, 1,000 req/day for 70B models.

---

### 🟢 Cerebras - ULTRA FAST (2000+ tok/s)

**Dashboard**: https://cloud.cerebras.ai/  
**Pricing**: https://www.cerebras.ai/pricing

| Model | Model ID | Speed | Params | Free Tier | Input $/M | Output $/M | Best For |
|-------|----------|-------|--------|-----------|-----------|------------|----------|
| **OpenAI GPT-OSS 120B** ⭐ | `gpt-oss-120b` | 3000 t/s | 120B | ✅ | $0.35 | $0.75 | General, reasoning |
| **Llama 3.3 70B** | `llama-3.3-70b` | 2100 t/s | 70B | ✅ | $0.85 | $1.20 | General purpose |
| **Llama 3.1 8B** | `llama3.1-8b` | 2200 t/s | 8B | ✅ | $0.10 | $0.10 | Quick responses |
| **Qwen 3 32B** | `qwen-3-32b` | 2600 t/s | 32B | ✅ | $0.40 | $0.80 | Code, math |
| **Qwen 3 235B** ⭐ | `qwen-3-235b-a22b-instruct-2507` | 1400 t/s | 235B | ✅ Preview | $0.60 | $1.20 | Complex reasoning |
| **Z.ai GLM 4.6** | `zai-glm-4.6` | 1000 t/s | 357B | ✅ Preview | $2.25 | $2.75 | Massive model |

**Free Tier**: All models available free! 14,400 req/day, 1M tokens/day. Developer tier starts at $10.

---

### 🟢 Lightning.ai - WORKING (via litai SDK)

**Dashboard**: https://lightning.ai/account  
**Format**: `provider/model_name`

| Model | Model ID | Context | Free Tier | Input $/M | Output $/M | Notes |
|-------|----------|---------|-----------|-----------|------------|-------|
| **Lightning GPT-OSS 20B** ⭐ | `lightning-ai/gpt-oss-20b` | 32K | ✅ 30M tokens/mo | $0.05 | $0.10 | Lightning's own model - fast |
| **OpenAI GPT-4o** | `openai/gpt-4o` | 128K | Credits | $2.50 | $10.00 | Highest quality |
| **OpenAI GPT-4 Turbo** | `openai/gpt-4-turbo` | 128K | Credits | $10.00 | $30.00 | Fast GPT-4 |
| **OpenAI GPT-4** | `openai/gpt-4` | 128K | Credits | $30.00 | $60.00 | Classic GPT-4 |
| **OpenAI GPT-3.5 Turbo** | `openai/gpt-3.5-turbo` | 16K | Credits | $0.50 | $1.50 | Cheap fallback |

**Note**: Lightning.ai provides access to OpenAI models via their platform. Free tier: 30M tokens/month + 15 credits.

---

### 🟢 OpenRouter - MULTI-MODEL GATEWAY

**Dashboard**: https://openrouter.ai/activity  
**Models**: https://openrouter.ai/models

| Model | Model ID | Context | Input $/M | Output $/M | Notes |
|-------|----------|---------|-----------|------------|-------|
| **Meta Llama 3.3 70B :free** ⭐ | `meta-llama/llama-3.3-70b-instruct:free` | 131K | FREE | FREE | 50 req/day |
| **Google Gemma 3 27B :free** | `google/gemma-3-27b-it:free` | 96K | FREE | FREE | 50 req/day |
| **Qwen 3 235B :free** | `qwen/qwen3-235b-a22b:free` | 40K | FREE | FREE | 50 req/day |
| **OpenAI GPT-OSS 20B :free** | `openai/gpt-oss-20b:free` | 131K | FREE | FREE | 50 req/day |
| **DeepSeek Chat V3** | `deepseek/deepseek-chat-v3-0324` | 64K | $0.14 | $0.28 | Best value reasoning |
| **Anthropic Claude 3.5 Sonnet** | `anthropic/claude-3.5-sonnet` | 200K | $3.00 | $15.00 | Scientific writing |
| **OpenAI GPT-4o** | `openai/gpt-4o-2024-11-20` | 128K | $2.50 | $10.00 | Latest GPT-4o |

**Free Tier**: 50 req/day for `:free` models. After $10 lifetime top-up: 1,000 req/day.

---

### 🟢 GitHub Models - COPILOT PRO+ ACCESS

**Playground**: https://github.com/marketplace/models  
**Docs**: https://docs.github.com/en/github-models

**Rate Limits by Copilot Tier:**

| Model Tier | Copilot Free | Copilot Pro | Copilot Business | Copilot Enterprise |
|------------|--------------|-------------|------------------|-------------------|
| **Low (Llama, Phi, etc.)** | 15 RPM, 150 RPD | 15 RPM, 150 RPD | 15 RPM, 300 RPD | 20 RPM, 450 RPD |
| **High (GPT-4o, Claude)** | 10 RPM, 50 RPD | 10 RPM, 50 RPD | 10 RPM, 100 RPD | 15 RPM, 150 RPD |
| **DeepSeek-R1** | 1 RPM, 8 RPD | 1 RPM, 8 RPD | 2 RPM, 10 RPD | 2 RPM, 12 RPD |
| **xAI Grok-3** | 1 RPM, 15 RPD | 1 RPM, 15 RPD | 2 RPM, 20 RPD | 2 RPM, 30 RPD |
| **OpenAI o1/o3/gpt-5** | — | 1 RPM, 8 RPD | 2 RPM, 10 RPD | 2 RPM, 12 RPD |

**Available Models**: GPT-4o, GPT-4o-mini, DeepSeek-R1, Llama 3.3 70B, Phi-4, Mistral Large, xAI Grok-3, Claude 3.5 Sonnet, and more.

---

### 🟡 OpenAI - DIRECT API (PAID)

**Dashboard**: https://platform.openai.com/usage  
**Pricing**: https://openai.com/api/pricing/

| Model | Model ID | Context | Input $/M | Output $/M | Notes |
|-------|----------|---------|-----------|------------|-------|
| **GPT-4o** | `gpt-4o` | 128K | $2.50 | $10.00 | Best quality |
| **GPT-4o mini** ⭐ | `gpt-4o-mini` | 128K | $0.15 | $0.60 | Best value |
| **GPT-4 Turbo** | `gpt-4-turbo` | 128K | $10.00 | $30.00 | Fast GPT-4 |
| **o1** | `o1` | 200K | $15.00 | $60.00 | Reasoning |
| **o1-mini** | `o1-mini` | 128K | $1.10 | $4.40 | Fast reasoning |
| **o3-mini** | `o3-mini` | 200K | $1.10 | $4.40 | Latest reasoning |
| **GPT-3.5 Turbo** | `gpt-3.5-turbo` | 16K | $0.50 | $1.50 | Legacy cheap |

**Recommendation**: Use OpenAI as last-resort fallback (priority 99) since free alternatives exist.

---

### 📊 Cost Comparison Summary

#### Estimated Monthly Cost (13.5M tokens/month usage)

```
┌────────────────────────────────────────────────────────────┐
│ Strategy                 │ Monthly Cost │ Notes            │
├────────────────────────────────────────────────────────────┤
│ FREE TIER FIRST ⭐       │ ~$0.00       │ Gemini+Groq+Cereb│
│ Lightning GPT-OSS-20B    │ $0.17        │ If free tier full│
│ DeepSeek via OpenRouter  │ $1.89        │ Good for reasoning│
│ GPT-4o via OpenAI        │ $33.75       │ Premium quality   │
└────────────────────────────────────────────────────────────┘

RECOMMENDED: Use free tier cascade (80% of queries) + Lightning.ai GPT-OSS-20B (20%)
Estimated cost: ~$0.03/month
```

---

## 5. Lightning.ai Configuration

### ✅ NOW WORKING (Fixed Dec 4, 2025)

Lightning.ai was broken due to using OpenAI-compatible API format. Fixed by switching to `litai` SDK.

**Working Models (tested):**
- `lightning-ai/gpt-oss-20b` ⭐ (FREE 30M tokens/month)
- `openai/gpt-4o`
- `openai/gpt-4-turbo`
- `openai/gpt-4`
- `openai/gpt-3.5-turbo`

```python
# Test Lightning.ai is working
from src.workflow_composer.llm.lightning_adapter import LightningAdapter
adapter = LightningAdapter(model="lightning-ai/gpt-oss-20b")
response = adapter.chat([Message.user("Hello!")])
```

---

## 6. Implementation Status

### Current State ✅ WORKING

| Component | Status | Notes |
|-----------|--------|-------|
| Provider Registry | ✅ Complete | All 8 providers configured in `registry.py` |
| Provider Router | ✅ Complete | Cascade with rate-limit detection |
| Groq Adapter | ✅ Complete | `providers/groq.py` |
| Cerebras Adapter | ✅ Complete | `providers/cerebras.py` |
| Gemini Adapter | ✅ Complete | `providers/gemini.py` |
| OpenRouter Adapter | ✅ Complete | `providers/openrouter.py` |
| API Keys | ✅ Configured | All in `.secrets/` directory |
| Cascade Working | ✅ Verified | Gemini used first (518ms) |

### What Was Already Implemented

✅ **Free Tier Cascade**: Gemini → Cerebras → Groq → OpenRouter → GitHub Models  
✅ **Automatic Fallback**: Rate-limited providers skipped for session  
✅ **API Key Management**: Secrets stored in `.secrets/`, loaded by `start_server.sh`  
✅ **Priority-Based Routing**: Lower priority number = tried first  

### Issue Found: Lightning.ai Disabled

Lightning.ai is marked **disabled** in `registry.py` due to API returning empty responses:

```python
# From registry.py line 120
"lightning": ProviderConfig(
    ...
    # DISABLED: API returns empty responses (status 200, 0 bytes) as of Dec 2025
    # Re-enable when Lightning.ai fixes their API or account is verified
    enabled=False,
),
```

**Action**: Lightning can be re-enabled once their API issues are resolved.

---

## 8. Recommendations

### Current Status: Good, But Can Improve ✅

The infrastructure is working well:
- Free tier cascade is active (Gemini → Cerebras → Groq → OpenRouter)
- Rate limiting detection and automatic fallback working
- All API keys configured and tested

### Recommended Improvements

1. **Re-enable Lightning.ai** (when their API is fixed)
   - Monitor https://status.lightning.ai/
   - Test with: `curl -X POST https://lightning.ai/api/v1/chat/completions ...`

2. **Integrate the new ProviderRouter with the Intent Arbiter**
   - Currently the arbiter may still use the old Lightning adapter
   - Update `arbiter.py` to use `providers.router.get_router()`

3. **Add Cost Tracking Integration**
   - Connect `CostTracker` to `ProviderRouter` for real-time cost monitoring
   - Track per-provider usage in session

4. **Monitoring Dashboard** (future)
   - Add endpoint to show current cascade status
   - Track rate limit hits per provider

### Usage Policy Summary (CURRENT)

```
╔══════════════════════════════════════════════════════════════════╗
║                    LLM USAGE POLICY (ACTIVE)                     ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Cascade uses FREE tier first (Gemini → Cerebras → Groq)     ║
║  2. Rate-limited providers auto-skipped for session              ║
║  3. OpenAI (priority 99) only used if all free tiers fail       ║
║  4. Local models (Ollama/vLLM) used when available              ║
║  5. API keys loaded from .secrets/ by start_server.sh           ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 9. Environment Variables

### How Keys Are Loaded

The `start_server.sh` script automatically loads all keys from `.secrets/`:

```bash
# From start_server.sh (lines 436-444):
[ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key)
[ -f ".secrets/cerebras_key" ] && export CEREBRAS_API_KEY=$(cat .secrets/cerebras_key)
[ -f ".secrets/groq_key" ] && export GROQ_API_KEY=$(cat .secrets/groq_key)
[ -f ".secrets/openrouter_key" ] && export OPENROUTER_API_KEY=$(cat .secrets/openrouter_key)
[ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=$(cat .secrets/lightning_key)
[ -f ".secrets/github_token" ] && export GITHUB_TOKEN=$(cat .secrets/github_token)
[ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key)
```

### Secret Files (All Configured ✅)

| File | Environment Variable | Provider |
|------|---------------------|----------|
| `.secrets/google_api_key` | `GOOGLE_API_KEY` | Gemini |
| `.secrets/cerebras_key` | `CEREBRAS_API_KEY` | Cerebras |
| `.secrets/groq_key` | `GROQ_API_KEY` | Groq |
| `.secrets/openrouter_key` | `OPENROUTER_API_KEY` | OpenRouter |
| `.secrets/lightning_key` | `LIGHTNING_API_KEY` | Lightning.ai |
| `.secrets/github_token` | `GITHUB_TOKEN` | GitHub Models |
| `.secrets/openai_key` | `OPENAI_API_KEY` | OpenAI |

### Manual Loading (for development/testing)

```bash
# Quick load all keys
source <(cat <<'EOF'
export GOOGLE_API_KEY=$(cat .secrets/google_api_key)
export CEREBRAS_API_KEY=$(cat .secrets/cerebras_key)
export GROQ_API_KEY=$(cat .secrets/groq_key)
export OPENROUTER_API_KEY=$(cat .secrets/openrouter_key)
export LIGHTNING_API_KEY=$(cat .secrets/lightning_key)
export GITHUB_TOKEN=$(cat .secrets/github_token)
export OPENAI_API_KEY=$(cat .secrets/openai_key)
EOF
)

# Test cascade
python -c "from src.workflow_composer.providers.router import get_router; get_router().print_status()"
```

---

## Quick Start Checklist

- [ ] Get Groq API key (free): https://console.groq.com/keys
- [ ] Get Cerebras API key (free): https://cloud.cerebras.ai/
- [ ] Get Gemini API key (free): https://aistudio.google.com/apikey
- [ ] Update `.env` with new keys
- [ ] Change `DEFAULT_MODEL` to `gpt-oss-20b` in `cloud.py`
- [ ] Update `TASK_MODELS` to use cheaper models
- [ ] Set budget alert to $5/month in CostTracker

---

## References

- [FREE_LLM_PROVIDERS.md](./FREE_LLM_PROVIDERS.md) - Detailed provider research
- [LLM_ORCHESTRATION_PLAN.md](./LLM_ORCHESTRATION_PLAN.md) - Architecture plan
- [Lightning.ai Pricing](https://lightning.ai/pricing/)
- [Groq Console](https://console.groq.com/)
- [Cerebras Cloud](https://cloud.cerebras.ai/)
- [Google AI Studio](https://aistudio.google.com/)

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2025-12-04 | 2.0.0 | Complete rewrite with usage policy, tier cascade |
| 2025-12-01 | 1.0.0 | Initial FREE_LLM_PROVIDERS research |
