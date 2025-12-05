# BioPipelines Local Model Strategy
## Comprehensive Guide to Sub-10B Models for Agentic Workflows

**Last Updated:** December 5, 2025  
**Target Hardware:** 10× T4 GPUs (16GB each) across multiple nodes  
**Strategy:** Single-T4 models locally + Cloud APIs for complex tasks

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Revised Strategy: T4-Only + Cloud Fallback](#revised-strategy-t4-only--cloud-fallback)
3. [Task Category Taxonomy](#task-category-taxonomy)
4. [Model Recommendations by Category](#model-recommendations-by-category)
5. [GPU Memory Planning](#gpu-memory-planning)
6. [Deployment Configurations](#deployment-configurations)
7. [Critical Evaluation](#critical-evaluation)

---

## Executive Summary

### The Multi-Model Strategy

Instead of relying on one large model (70B+) or expensive cloud APIs, we deploy **specialized sub-10B models** for different tasks. This approach offers:

| Advantage | Description |
|-----------|-------------|
| **Cost** | Zero API costs after initial GPU investment |
| **Latency** | Sub-second responses with proper batching |
| **Privacy** | All data stays on-premise |
| **Specialization** | Each model optimized for its task |
| **Redundancy** | Multiple fallback options per category |

### Quick Reference: Primary Model Per Category

| Category | Primary Model | Params | VRAM (FP16) |
|----------|--------------|--------|-------------|
| Intent Parsing | Llama-3.2-3B-Instruct | 3.2B | ~7GB |
| Orchestration | Nemotron-8B-Orchestrator | 8B | ~17GB |
| Code Generation | Qwen2.5-Coder-7B-Instruct | 7.6B | ~16GB |
| Code Validation | Qwen2.5-Coder-7B-Instruct | 7.6B | ~16GB |
| Data Analysis | Phi-3.5-mini-instruct | 3.8B | ~8GB |
| Math/Statistics | Qwen2.5-Math-7B-Instruct | 7B | ~15GB |
| Bio/Medical | BioMistral-7B | 7B | ~15GB |
| Documentation | Gemma-2-9B-IT | 9B | ~19GB |
| Embeddings | BGE-M3 | 0.6B | ~2GB |
| Safety/Guardrails | Nemotron-Safety-8B | 8B | ~17GB |

---

## Revised Strategy: T4-Only + Cloud Fallback

### Why This Change?

After analyzing our hardware (10× T4 GPUs with 16GB each across SLURM nodes), we've adopted a **simplified approach**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    T4-ONLY + CLOUD HYBRID ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   LOCAL (T4 GPUs - 16GB each)              CLOUD (API Fallback)             │
│   ┌─────────────────────────┐              ┌─────────────────────────┐      │
│   │  ✅ Models ≤7GB FP16    │              │  DeepSeek-V3 ($0.27/M)  │      │
│   │  ✅ Models ≤14GB INT8   │              │  Claude-3.5 Sonnet      │      │
│   │                         │              │  GPT-4o                 │      │
│   │  Fast, private, free    │─────────────▶│  Complex tasks, fallback│      │
│   └─────────────────────────┘              └─────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Constraints:**
- Each T4 has only **16GB VRAM**
- Multi-GPU tensor parallelism across nodes is impractical (network bottleneck)
- Each node runs **independent vLLM server**
- A **router/load balancer** distributes requests across nodes

### T4 VRAM Compatibility Matrix

| Model | Params | FP16 VRAM | INT8 VRAM | INT4 VRAM | T4 Status |
|-------|--------|-----------|-----------|-----------|-----------|
| **Qwen2.5-Coder-1.5B** | 1.5B | ~3GB | ~2GB | ~1GB | ✅ FP16 fits |
| **Gemma-2-2B-IT** | 2.6B | ~6GB | ~3GB | ~2GB | ✅ FP16 fits |
| **Llama-3.2-3B-Instruct** | 3.2B | ~7GB | ~4GB | ~2GB | ✅ FP16 fits |
| **Phi-3.5-mini-instruct** | 3.8B | ~8GB | ~4GB | ~2.5GB | ✅ FP16 fits |
| **Qwen2.5-Coder-7B** | 7.6B | ~16GB | ~8GB | ~4GB | ⚠️ Tight FP16, ✅ INT8 |
| **Llama-3.1-8B-Instruct** | 8B | ~17GB | ~8.5GB | ~4.5GB | ❌ FP16, ✅ INT8 |
| **BioMistral-7B** | 7B | ~15GB | ~7.5GB | ~4GB | ⚠️ Tight FP16, ✅ INT8 |
| **Qwen2.5-Math-7B** | 7B | ~15GB | ~7.5GB | ~4GB | ⚠️ Tight FP16, ✅ INT8 |
| **Gemma-2-9B-IT** | 9B | ~19GB | ~9.5GB | ~5GB | ❌ FP16, ✅ INT8 |
| **BGE-M3** | 0.6B | ~1.5GB | ~1GB | N/A | ✅ FP16 fits |

**Legend:**
- ✅ = Comfortable fit with room for KV cache
- ⚠️ = Fits but limited KV cache / batch size
- ❌ = Does not fit

### Recommended T4-Optimized Model Selection

Based on what fits comfortably on a single T4:

| Category | **T4 Local Model** | Quantization | VRAM | Cloud Fallback |
|----------|-------------------|--------------|------|----------------|
| Intent Parsing | **Llama-3.2-3B-Instruct** | FP16 | ~7GB | DeepSeek-V3 |
| Orchestration | ❌ Use Cloud | - | - | **DeepSeek-V3** or Claude-3.5 |
| Code Generation | **Qwen2.5-Coder-7B** | INT8 | ~8GB | DeepSeek-V3 |
| Code Validation | **Qwen2.5-Coder-1.5B** | FP16 | ~3GB | DeepSeek-V3 |
| Data Analysis | **Phi-3.5-mini-instruct** | FP16 | ~8GB | DeepSeek-V3 |
| Math/Statistics | **Qwen2.5-Math-7B** | INT8 | ~7.5GB | DeepSeek-V3 |
| Bio/Medical | **BioMistral-7B** | INT8 | ~7.5GB | Claude-3.5 (Claude knows biology) |
| Documentation | **Gemma-2-9B-IT** | INT8 | ~9.5GB | Claude-3.5 |
| Embeddings | **BGE-M3** | FP16 | ~1.5GB | OpenAI Embeddings |
| Safety/Guardrails | **Llama-Guard-3-1B** | FP16 | ~2.5GB | Claude-3.5 |

### T4 Fleet Deployment Strategy

With 10 T4s available, here's the optimal distribution:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         10× T4 FLEET ALLOCATION                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  HIGH-FREQUENCY TASKS (Need multiple replicas for throughput)              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ T4-1: Llama-3.2-3B (Intent)      │ T4-2: Llama-3.2-3B (Intent)     │   │
│  │       FP16, ~7GB                 │       FP16, ~7GB                │   │
│  │       + Llama-Guard-3 ~2.5GB     │       + BGE-M3 ~1.5GB           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  CODE TASKS (Core developer workflow)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ T4-3: Qwen2.5-Coder-7B (Gen)     │ T4-4: Qwen2.5-Coder-7B (Gen)    │   │
│  │       INT8, ~8GB                 │       INT8, ~8GB                │   │
│  │       + Qwen2.5-Coder-1.5B       │       + Qwen2.5-Coder-1.5B      │   │
│  │         (Validation) ~3GB        │         (Validation) ~3GB       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  SPECIALIZED MODELS (Lower frequency, dedicated GPUs)                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ T4-5: Phi-3.5-mini              │ T4-6: Qwen2.5-Math-7B            │   │
│  │       (Data Analysis)           │       (Statistics)               │   │
│  │       FP16, ~8GB                │       INT8, ~7.5GB               │   │
│  ├─────────────────────────────────┼────────────────────────────────────┤   │
│  │ T4-7: BioMistral-7B             │ T4-8: Gemma-2-9B-IT              │   │
│  │       (Bio/Medical)             │       (Documentation)            │   │
│  │       INT8, ~7.5GB              │       INT8, ~9.5GB               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  EMBEDDINGS & RESERVE                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ T4-9: BGE-M3 (Embeddings)       │ T4-10: RESERVE / Hot Spare       │   │
│  │       FP16, ~1.5GB              │        For scaling high-demand   │   │
│  │       Can add more models       │        models dynamically        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Cloud API Pricing Comparison

| Provider | Model | Input ($/1M) | Output ($/1M) | Best For |
|----------|-------|--------------|---------------|----------|
| **DeepSeek** | DeepSeek-V3 | **$0.27** | **$1.10** | Most tasks (cheapest) |
| OpenAI | GPT-4o | $2.50 | $10.00 | Broad knowledge |
| Anthropic | Claude-3.5 Sonnet | $3.00 | $15.00 | Complex reasoning, biology |
| Google | Gemini 1.5 Pro | $1.25 | $5.00 | Long context |

**Recommendation:** Use **DeepSeek-V3** as primary cloud fallback (10x cheaper than GPT-4).

### When to Use Cloud vs Local

```python
# Decision logic in the orchestrator
def choose_model(task_type: str, complexity: str, context_length: int):
    """Route requests to local or cloud models."""
    
    # Always local (fast, free)
    LOCAL_PREFERRED = {
        "intent_parsing",      # Llama-3.2-3B
        "embeddings",          # BGE-M3
        "safety_check",        # Llama-Guard-3
        "quick_validation",    # Qwen2.5-Coder-1.5B
    }
    
    # Always cloud (too complex for small models)
    CLOUD_REQUIRED = {
        "orchestration",       # Needs Orchestrator-8B or better
        "complex_reasoning",   # Multi-step planning
        "long_context",        # >32K tokens
    }
    
    # Hybrid: try local first, fallback to cloud
    HYBRID = {
        "code_generation",     # Local Qwen2.5-7B, fallback DeepSeek
        "data_analysis",       # Local Phi-3.5, fallback Claude
        "bio_medical",         # Local BioMistral, fallback Claude
    }
    
    if task_type in LOCAL_PREFERRED:
        return LocalModel(task_type)
    elif task_type in CLOUD_REQUIRED or complexity == "high":
        return CloudModel("deepseek-v3")
    else:
        return HybridModel(local_first=True, fallback="deepseek-v3")
```

### Cost Comparison: Local vs Cloud

**Scenario:** 100,000 queries/month, average 1000 tokens input + 500 tokens output

| Strategy | Monthly Cost | Notes |
|----------|-------------|-------|
| **100% Cloud (GPT-4o)** | ~$375 | ($2.50 × 100 + $10.00 × 50) |
| **100% Cloud (DeepSeek)** | ~$82 | ($0.27 × 100 + $1.10 × 50) |
| **T4 Local (Power only)** | ~$50 | 10 × T4 @ ~25W each = 250W, 24/7 |
| **Hybrid (80% local, 20% DeepSeek)** | ~$66 | $50 local + $16 cloud |

**Winner:** Hybrid approach with DeepSeek fallback is most cost-effective.

---

## Task Category Taxonomy

Based on BioPipelines codebase analysis, here are the **10 distinct task categories**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BioPipelines Task Categories                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ 1. INTENT        │    │ 2. ORCHESTRATION │    │ 3. CODE          │       │
│  │    PARSING       │───▶│    & PLANNING    │───▶│    GENERATION    │       │
│  │                  │    │                  │    │                  │       │
│  │ agents/intent/   │    │ llm/             │    │ generators/      │       │
│  │ classification.py│    │ orchestrator_8b  │    │ specialists/     │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│           │                      │                       │                   │
│           ▼                      ▼                       ▼                   │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ 4. CODE          │    │ 5. DATA          │    │ 6. MATH &        │       │
│  │    VALIDATION    │    │    ANALYSIS      │    │    STATISTICS    │       │
│  │                  │    │                  │    │                  │       │
│  │ agents/          │    │ analysis/        │    │ statistical      │       │
│  │ coding_agent.py  │    │ visualization    │    │ reasoning        │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│           │                      │                       │                   │
│           ▼                      ▼                       ▼                   │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ 7. BIO/MEDICAL   │    │ 8. DOCUMENTATION │    │ 9. EMBEDDINGS    │       │
│  │    REASONING     │    │    & WRITING     │    │    & RETRIEVAL   │       │
│  │                  │    │                  │    │                  │       │
│  │ domain-specific  │    │ specialists/     │    │ agents/rag/      │       │
│  │ explanations     │    │ docs.py          │    │ knowledge_base   │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│                                  │                                           │
│                                  ▼                                           │
│                         ┌──────────────────┐                                │
│                         │ 10. SAFETY &     │                                │
│                         │     GUARDRAILS   │                                │
│                         │                  │                                │
│                         │ input/output     │                                │
│                         │ filtering        │                                │
│                         └──────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Recommendations by Category

### Category 1: Intent Parsing & Query Understanding

**Purpose:** Convert natural language queries into structured intents, extract entities, classify task types, and identify tool calls.

**Used by:**
- `agents/intent/parser.py` - Pattern and semantic intent parsing
- `agents/classification.py` - Task type classification
- `agents/intent/semantic.py` - Semantic similarity matching

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **Llama-3.2-3B-Instruct** | 3.2B | ~7GB | Purpose-built for agentic tasks. Meta trained this specifically for tool-use, function-calling, and instruction following. 128K context handles long conversation history. Fast inference enables real-time parsing. |
| **P2** | Qwen2.5-7B-Instruct | 7B | ~15GB | Stronger reasoning than Llama-3.2-3B but larger footprint. Excellent multilingual support. Better for complex, ambiguous queries. |
| **P3** | Gemma-2-2B-IT | 2.6B | ~6GB | Smallest option for ultra-constrained scenarios. Good reasoning for size (MMLU 51.3%). Fast enough for T4 GPUs. |
| **P4** | Llama-3.1-8B-Instruct | 8B | ~17GB | Most capable but heavy for just parsing. Use if combining parsing with other tasks. |

#### Recommendations
```yaml
primary: Llama-3.2-3B-Instruct  # Best balance of speed + capability
fallback: Gemma-2-2B-IT          # When VRAM is tight
heavy_duty: Qwen2.5-7B-Instruct  # For complex multi-intent queries
```

#### BioPipelines Integration Points
```python
# In agents/intent/parser.py - IntentParser._classify_with_llm()
# In agents/classification.py - classify_task()
# In agents/intent/unified_parser.py - UnifiedIntentParser.parse()
```

---

### Category 2: Orchestration & Multi-Step Planning

**Purpose:** Decide which tools/models to use, coordinate sub-agents, plan multi-step workflows, handle retries and fallbacks.

**Used by:**
- `llm/orchestrator_8b.py` - Main orchestration layer
- `agents/orchestrator.py` - Agent coordination
- `agents/specialists/supervisor.py` - Multi-agent supervision

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **Nemotron-8B-Orchestrator** | 8B | ~17GB | **Purpose-built for orchestration.** Trained on 15M+ agentic trajectories (ToolOrchestra). Understands cost/latency tradeoffs. Best at deciding "which tool for this task?" |
| **P2** | Granite-3.3-8B-Instruct | 8B | ~17GB | IBM's Apache-2.0 licensed alternative. Strong tool-calling and RAG support. Better for enterprises needing permissive licensing. |
| **P3** | Qwen2.5-7B-Instruct | 7B | ~15GB | Excellent generalist that can orchestrate well. Slightly worse at pure orchestration but more versatile. |
| **P4** | Yi-1.5-9B-Chat | 9B | ~19GB | Strong reasoning and planning. Apache-2.0 license. Bilingual (EN/CN) if needed. |
| **P5** | Llama-3.1-8B-Instruct | 8B | ~17GB | Reliable fallback with native tool-use support. Large community, well-tested. |

#### Recommendations
```yaml
primary: Nemotron-8B-Orchestrator    # Best for pure orchestration
alternative: Granite-3.3-8B-Instruct  # If Apache-2.0 license needed
generalist: Qwen2.5-7B-Instruct       # If also using for other tasks
```

#### Key Decision: Dedicated vs Shared Orchestrator
- **Dedicated (Recommended):** Run Nemotron-8B solely for orchestration. It excels at this specific task.
- **Shared:** Use Qwen2.5-7B for both orchestration and other tasks (saves VRAM but slightly worse orchestration).

---

### Category 3: Code Generation & Workflow Creation

**Purpose:** Generate Snakemake/Nextflow workflows, Python scripts, R code, SQL queries, shell commands.

**Used by:**
- `generators/` - Workflow generation
- `agents/specialists/codegen.py` - Code generation agent
- `agents/tools/` - Tool implementation

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **Qwen2.5-Coder-7B-Instruct** | 7.6B | ~16GB | **State-of-the-art open-source coding.** 62.8% HumanEval, 69.6% MBPP. Supports 338 languages including Snakemake, Nextflow DSL. 128K context for large codebases. |
| **P2** | DeepSeek-Coder-V2-Lite-Instruct | 16B (2.4B active) | ~12GB | MoE architecture = fast inference despite size. Matches GPT-4 Turbo on code benchmarks. Great for complex multi-file generation. |
| **P3** | StarCoder2-7B | 7B | ~15GB | BigCode's latest. Strong on completion and repo-level tasks. Permissive license (BigCode OpenRAIL-M). |
| **P4** | CodeGemma-7B-IT | 7B | ~15GB | Google's code specialist. Strong Python/JS/TS. Good integration with Gemma ecosystem. |
| **P5** | Qwen2.5-Coder-1.5B-Instruct | 1.5B | ~3GB | Tiny but capable. For quick completions when main model is busy. Can run on T4 alongside others. |
| **P6** | Magicoder-CL-7B | 7B | ~15GB | RL-tuned for competitive coding. Good as "second opinion" model for alternative implementations. |

#### Recommendations
```yaml
primary: Qwen2.5-Coder-7B-Instruct     # Best overall
fast_alternative: DeepSeek-Coder-V2-Lite  # MoE = fast inference
lightweight: Qwen2.5-Coder-1.5B        # For constrained scenarios
second_opinion: Magicoder-CL-7B        # Alternative implementations
```

#### Language Coverage for BioPipelines
| Language | Primary Model Support |
|----------|----------------------|
| Python | ✅ Excellent (all models) |
| Snakemake | ✅ Qwen2.5-Coder (trained on it) |
| Nextflow DSL | ✅ Qwen2.5-Coder, DeepSeek-Coder |
| R | ✅ Good (all models) |
| Bash/Shell | ✅ Excellent (all models) |
| SQL | ✅ Good (all models) |

---

### Category 4: Code Validation & Error Resolution

**Purpose:** Validate workflow syntax, diagnose job failures, read stack traces, suggest fixes.

**Used by:**
- `agents/coding_agent.py` - Error diagnosis
- `agents/specialists/validator.py` - Syntax validation
- `agents/self_healing.py` - Auto-fix attempts

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **Qwen2.5-Coder-7B-Instruct** | 7.6B | ~16GB | **Same model as code generation (share instance).** Excellent at understanding errors, reading traces, suggesting minimal patches. |
| **P2** | Phi-3.5-mini-instruct | 3.8B | ~8GB | Strong reasoning helps with complex error chains. 128K context for large log files. Lighter than Qwen2.5-Coder. |
| **P3** | Yi-1.5-9B-Chat | 9B | ~19GB | Can read very long traces. Combined reasoning + code understanding. Good for "explain this error" tasks. |
| **P4** | Llama-3.1-8B-Instruct | 8B | ~17GB | Robust natural-language explanations. Good at generating human-readable error summaries. |

#### Recommendations
```yaml
primary: Qwen2.5-Coder-7B-Instruct  # Share with code-gen (same weights)
lightweight: Phi-3.5-mini-instruct   # When dedicated validation needed
long_context: Yi-1.5-9B-Chat         # For very large log files
```

#### Practical Tip
Run **one instance** of Qwen2.5-Coder-7B for both code generation and validation. Create two "agents" (different system prompts) sharing the same model weights.

---

### Category 5: Data Analysis & Visualization

**Purpose:** Interpret analysis results, generate pandas/matplotlib/seaborn code, create statistical summaries.

**Used by:**
- Analysis result interpretation
- Visualization code generation
- Report figure creation

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **Phi-3.5-mini-instruct** | 3.8B | ~8GB | **Best reasoning-per-parameter for data tasks.** Strong math (MATH 48.5%), good at table reasoning, generates clean plotting code. Fast inference. |
| **P2** | Qwen2.5-7B-Instruct | 7B | ~15GB | Stronger overall but heavier. Better for complex multi-step analyses. |
| **P3** | Llama-3.2-3B-Instruct | 3.2B | ~7GB | Fast alternative for simple analyses. Good enough for basic pandas operations. |

#### Recommendations
```yaml
primary: Phi-3.5-mini-instruct   # Best for data/stats tasks
heavy_duty: Qwen2.5-7B-Instruct  # Complex analyses
lightweight: Llama-3.2-3B        # Quick simple tasks
```

---

### Category 6: Math & Statistical Reasoning

**Purpose:** Statistical tests, uncertainty quantification, mathematical derivations, significance calculations.

**Used by:**
- Advanced statistical analysis
- Method comparison
- Power calculations

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **Qwen2.5-Math-7B-Instruct** | 7B | ~15GB | **Dedicated math model.** Outperforms even larger models on MATH benchmark (83.6-85.3% with CoT). Contest-level problem solving. |
| **P2** | DeepSeek-Math-7B-Instruct | 7B | ~15GB | Strong symbolic manipulation and proofs. Widely used and tested. Good alternative to Qwen. |
| **P3** | Mathstral-7B-v0.1 | 7B | ~15GB | Mistral-7B fine-tuned for math/science. Good general math but less specialized than above. |
| **P4** | Phi-3.5-mini-instruct | 3.8B | ~8GB | Lightweight option. Good for basic stats but struggles with advanced topics. |

#### Recommendations
```yaml
primary: Qwen2.5-Math-7B-Instruct    # Best for serious math
alternative: DeepSeek-Math-7B        # Good backup
lightweight: Phi-3.5-mini            # Basic stats only
```

#### Use Case Split
| Task | Model |
|------|-------|
| Basic descriptive stats | Phi-3.5-mini |
| Hypothesis testing | Qwen2.5-Math-7B |
| Power analysis | Qwen2.5-Math-7B |
| Bayesian inference | Qwen2.5-Math-7B |
| Simple p-value interpretation | Phi-3.5-mini |

---

### Category 7: Bio/Medical Domain Reasoning

**Purpose:** Explain biological mechanisms, compare methods, summarize papers, describe datasets.

**Used by:**
- Domain-specific explanations
- Method recommendations
- Literature-style outputs

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **BioMistral-7B** | 7B | ~15GB | **Best bio/clinical performance at this size.** Significantly improves over Mistral-7B on MedQA, MedMCQA, PubMedQA. Understands biomedical terminology. |
| **P2** | Llama3-Med42-8B | 8B | ~17GB | Llama-3-8B adapted to clinical QA. Strong Elo ratings on medical benchmarks. Better for clinical-style questions. |
| **P3** | Meditron-7B | 7B | ~15GB | Domain-pretrained on PubMed/full-text. Good base for fine-tuning on custom corpora. Less instruction-tuned. |
| **P4** | BioGPT-Large | 1.5B | ~3GB | Tiny MIT-licensed option. Good for entity-rich biomedical text generation. Can run alongside everything else. |

#### Recommendations
```yaml
primary: BioMistral-7B           # Best overall bio reasoning
clinical: Llama3-Med42-8B        # If clinical focus needed
lightweight: BioGPT-Large        # Tiny, always available
fine_tuning_base: Meditron-7B    # For custom domain adaptation
```

#### ⚠️ Important Caveat
These models are **NOT** approved for clinical decision-making. Always include disclaimers:
> "This is for research purposes only. Not medical advice."

---

### Category 8: Documentation & Narrative Writing

**Purpose:** Generate reports, READMEs, explain workflows, write user-friendly documentation.

**Used by:**
- `agents/specialists/docs.py` - Documentation agent
- Report generation
- Workflow explanations

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **Gemma-2-9B-IT** | 9B | ~19GB | **Best writing quality at this size.** Strong reasoning, good long-form coherence. Produces professional-quality documentation. |
| **P2** | Yi-1.5-9B-Chat | 9B | ~19GB | Strong reading comprehension and reasoning. Good multi-purpose writer. Apache-2.0 license. |
| **P3** | Llama-3.2-3B-Instruct | 3.2B | ~7GB | Very fast for short docs. Good for comments, commit messages, quick explanations. |
| **P4** | Qwen2.5-7B-Instruct | 7B | ~15GB | Good writer that can also handle other tasks. Use if already loaded for other purposes. |

#### Recommendations
```yaml
primary: Gemma-2-9B-IT           # Best writing quality
fast: Llama-3.2-3B-Instruct      # Quick docs
versatile: Yi-1.5-9B-Chat        # Good all-around
```

#### Practical Tip
Documentation doesn't need a dedicated model. **Reuse your orchestrator** (Gemma-2-9B or Yi-1.5-9B) for writing tasks.

---

### Category 9: Embeddings & Retrieval (RAG)

**Purpose:** Semantic search over papers, code, logs, configs, metadata.

**Used by:**
- `agents/rag/` - RAG system
- `agents/rag/knowledge_base.py` - Document indexing
- Similarity search

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **BGE-M3** | 0.6B | ~2GB | **Best hybrid retrieval.** Supports dense + sparse + ColBERT. Multilingual. Up to 8K token documents. |
| **P2** | BGE-base-en-v1.5 | 0.1B | ~0.5GB | Smaller, English-only. Top MTEB performance. Very fast. |
| **P3** | Llama-Embed-Nemotron-8B | 8B | ~17GB | Higher quality but heavy. Universal embedding with instruction awareness. Use for critical retrieval. |
| **P4** | E5-mistral-7b-instruct | 7B | ~15GB | Strong instruction-following embeddings. Good for nuanced queries. |

#### Recommendations
```yaml
primary: BGE-M3                      # Best balance
lightweight: BGE-base-en-v1.5        # Fast English-only
high_quality: Llama-Embed-Nemotron-8B  # When quality critical
```

#### Note
Embedding models are **different from chat models**. They're encoders (no text generation). Much smaller VRAM footprint—run BGE-M3 alongside all your generative models.

---

### Category 10: Safety & Guardrails

**Purpose:** Filter unsafe inputs/outputs, classify content, enforce policies.

**Used by:**
- Input validation
- Output filtering
- Content safety

#### Priority-Ordered Models

| Priority | Model | Params | VRAM | Why This Priority |
|----------|-------|--------|------|-------------------|
| **P1** | **Nemotron-Safety-8B-V3** | 8B | ~17GB | **Purpose-built safety classifier.** NVIDIA's content-safety model, integrates with NeMo Guardrails. Trained for open-source LLM outputs. |
| **P2** | Llama-Guard-3-8B | 8B | ~17GB | Meta's safety classifier. Good at content categorization. Works well with Llama models. |
| **P3** | Rule-based filters | N/A | N/A | For simple cases, regex + keyword filters are faster and cheaper. Use ML models for nuanced cases. |

#### Recommendations
```yaml
primary: Nemotron-Safety-8B-V3  # Best ML-based safety
alternative: Llama-Guard-3-8B   # If using Llama stack
fallback: Rule-based regex      # Always have as backup
```

#### Implementation Strategy
1. **Fast path:** Rule-based regex for obvious violations (faster)
2. **ML path:** Nemotron-Safety for ambiguous content
3. **Always on:** Run safety check on ALL outputs before returning to user

---

## Complete Model Catalog

### All Recommended Models (Sorted by Category)

| Category | Model | HuggingFace ID | Params | VRAM (FP16) | VRAM (INT8) | License |
|----------|-------|----------------|--------|-------------|-------------|---------|
| **Intent** | Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | ~7GB | ~4GB | Llama 3.2 |
| **Intent** | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7B | ~15GB | ~8GB | Apache-2.0 |
| **Intent** | Gemma-2-2B-IT | `google/gemma-2-2b-it` | 2.6B | ~6GB | ~3GB | Gemma |
| **Orchestration** | Nemotron-8B-Orchestrator | `nvidia/Llama-3.1-Nemotron-8B-Orchestrator` | 8B | ~17GB | ~9GB | Llama 3.1 |
| **Orchestration** | Granite-3.3-8B-Instruct | `ibm-granite/granite-3.3-8b-instruct` | 8B | ~17GB | ~9GB | Apache-2.0 |
| **Orchestration** | Yi-1.5-9B-Chat | `01-ai/Yi-1.5-9B-Chat` | 9B | ~19GB | ~10GB | Apache-2.0 |
| **Code** | Qwen2.5-Coder-7B-Instruct | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7.6B | ~16GB | ~8GB | Apache-2.0 |
| **Code** | DeepSeek-Coder-V2-Lite | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B (2.4B active) | ~12GB | ~6GB | DeepSeek |
| **Code** | StarCoder2-7B | `bigcode/starcoder2-7b` | 7B | ~15GB | ~8GB | BigCode OpenRAIL-M |
| **Code** | Qwen2.5-Coder-1.5B | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | ~3GB | ~2GB | Apache-2.0 |
| **Analysis** | Phi-3.5-mini-instruct | `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~8GB | ~4GB | MIT |
| **Math** | Qwen2.5-Math-7B-Instruct | `Qwen/Qwen2.5-Math-7B-Instruct` | 7B | ~15GB | ~8GB | Apache-2.0 |
| **Math** | DeepSeek-Math-7B | `deepseek-ai/deepseek-math-7b-instruct` | 7B | ~15GB | ~8GB | DeepSeek |
| **Bio** | BioMistral-7B | `BioMistral/BioMistral-7B` | 7B | ~15GB | ~8GB | Apache-2.0 |
| **Bio** | Med42-v2-8B | `m42-health/Llama3-Med42-8B` | 8B | ~17GB | ~9GB | Llama 3 |
| **Bio** | BioGPT-Large | `microsoft/BioGPT-Large` | 1.5B | ~3GB | ~2GB | MIT |
| **Docs** | Gemma-2-9B-IT | `google/gemma-2-9b-it` | 9B | ~19GB | ~10GB | Gemma |
| **Embedding** | BGE-M3 | `BAAI/bge-m3` | 0.6B | ~2GB | N/A | MIT |
| **Embedding** | BGE-base-en-v1.5 | `BAAI/bge-base-en-v1.5` | 0.1B | ~0.5GB | N/A | MIT |
| **Safety** | Nemotron-Safety-8B | `nvidia/Llama-3.1-Nemotron-Safety-8B-V3` | 8B | ~17GB | ~9GB | Llama 3.1 |

---

## GPU Memory Planning

### Available GPU Configurations

| GPU | VRAM | Price (GCP) | Best For |
|-----|------|-------------|----------|
| T4 | 16GB | ~$0.35/hr | Small models (≤7B INT8) |
| L4 | 24GB | ~$0.50/hr | Medium models (≤8B FP16) |
| A10G | 24GB | ~$1.00/hr | Same as L4, faster |
| A100-40GB | 40GB | ~$3.00/hr | Multiple 7B models |

### VRAM Requirements by Precision

| Precision | Rule of Thumb | Example (7B model) |
|-----------|---------------|-------------------|
| FP32 | 4 bytes × params | ~28GB |
| FP16/BF16 | 2 bytes × params | ~14GB |
| INT8 | 1 byte × params | ~7GB |
| INT4 (GPTQ/AWQ) | 0.5 bytes × params | ~4GB |

### What Fits on Each GPU

#### Single T4 (16GB)
```
Option A: One 7B model (INT8)
  - Qwen2.5-Coder-7B (INT8): ~8GB
  - Remaining: ~8GB for KV cache + other models
  
Option B: Multiple small models
  - Llama-3.2-3B-Instruct: ~7GB
  - BGE-M3: ~2GB
  - Total: ~9GB
```

#### Single L4 (24GB)
```
Option A: One 8B model (FP16)
  - Nemotron-8B-Orchestrator: ~17GB
  - BGE-M3: ~2GB
  - Remaining: ~5GB for KV cache

Option B: Two medium models (INT8)
  - Qwen2.5-Coder-7B (INT8): ~8GB
  - Phi-3.5-mini (INT8): ~4GB
  - BGE-M3: ~2GB
  - Total: ~14GB
```

#### 2× L4 (48GB total)
```
GPU 0 (24GB):
  - Nemotron-8B-Orchestrator (FP16): ~17GB
  - Llama-3.2-3B-Instruct (FP16): ~7GB
  
GPU 1 (24GB):
  - Qwen2.5-Coder-7B (FP16): ~16GB
  - BGE-M3: ~2GB
  - Phi-3.5-mini (INT8): ~4GB
```

#### 1× L4 + 1× T4 (40GB total)
```
L4 (24GB) - Heavy models:
  - Qwen2.5-Coder-7B (FP16): ~16GB
  - Remaining: ~8GB for KV cache

T4 (16GB) - Light models:
  - Llama-3.2-3B-Instruct (FP16): ~7GB
  - BGE-M3: ~2GB
  - Phi-3.5-mini (INT8): ~4GB
  - Total: ~13GB
```

---

## Deployment Configurations

### Configuration 1: Minimal (1× L4)

**Best for:** Testing, low-traffic deployments

```yaml
name: minimal_l4
gpus: 1× L4 (24GB)
strategy: swap_on_demand

models:
  always_loaded:
    - name: Llama-3.2-3B-Instruct
      role: [intent_parsing, documentation]
      vram: ~7GB
    - name: BGE-M3
      role: [embeddings]
      vram: ~2GB
  
  swap_in:
    - name: Qwen2.5-Coder-7B-Instruct
      role: [code_generation, code_validation]
      vram: ~16GB
      swap_out: [Llama-3.2-3B]
    - name: Phi-3.5-mini-instruct
      role: [data_analysis]
      vram: ~8GB

total_fixed: ~9GB
swap_budget: ~15GB
```

### Configuration 2: Standard (2× L4)

**Best for:** Production, moderate traffic

```yaml
name: standard_2xl4
gpus: 2× L4 (48GB total)
strategy: dedicated_specialists

gpu_0:  # Orchestration + NLU
  models:
    - name: Nemotron-8B-Orchestrator
      role: [orchestration]
      vram: ~17GB
    - name: Llama-3.2-3B-Instruct
      role: [intent_parsing, documentation]
      vram: ~7GB
  total: ~24GB

gpu_1:  # Code + Analysis
  models:
    - name: Qwen2.5-Coder-7B-Instruct
      role: [code_generation, code_validation]
      vram: ~16GB
    - name: Phi-3.5-mini-instruct (INT8)
      role: [data_analysis]
      vram: ~4GB
    - name: BGE-M3
      role: [embeddings]
      vram: ~2GB
  total: ~22GB
```

### Configuration 3: Full Zoo (3× L4 or equivalent)

**Best for:** High traffic, full specialization

```yaml
name: full_zoo_3xl4
gpus: 3× L4 (72GB total)
strategy: full_specialization

gpu_0:  # Orchestration + Safety
  models:
    - name: Nemotron-8B-Orchestrator
      role: [orchestration]
      vram: ~17GB
    - name: Nemotron-Safety-8B (optional)
      role: [safety]
      vram: ~17GB (or skip if rule-based is enough)
  
gpu_1:  # Code Specialists
  models:
    - name: Qwen2.5-Coder-7B-Instruct
      role: [code_generation, code_validation]
      vram: ~16GB
    - name: Llama-3.2-3B-Instruct
      role: [intent_parsing]
      vram: ~7GB

gpu_2:  # Analysis + Bio + Docs
  models:
    - name: Phi-3.5-mini-instruct
      role: [data_analysis]
      vram: ~8GB
    - name: BioMistral-7B
      role: [bio_reasoning]
      vram: ~15GB
    - name: BGE-M3
      role: [embeddings]
      vram: ~2GB
```

### Configuration 4: Mixed L4 + T4

**Best for:** Cost optimization

```yaml
name: mixed_l4_t4
gpus: 1× L4 (24GB) + 2× T4 (32GB)
strategy: split_by_model_size

l4:  # Heavy models
  models:
    - name: Qwen2.5-Coder-7B-Instruct
      role: [code_generation, code_validation]
      vram: ~16GB
    - name: Phi-3.5-mini-instruct
      role: [data_analysis]
      vram: ~8GB

t4_0:  # Orchestration (INT8)
  models:
    - name: Nemotron-8B-Orchestrator (INT8)
      role: [orchestration]
      vram: ~9GB
    - name: BGE-M3
      role: [embeddings]
      vram: ~2GB

t4_1:  # Light models
  models:
    - name: Llama-3.2-3B-Instruct
      role: [intent_parsing, documentation]
      vram: ~7GB
    - name: BioGPT-Large
      role: [bio_reasoning_light]
      vram: ~3GB
```

---

## Critical Evaluation

### Part 1: Is the Multi-Model Strategy Actually Worth It?

Before committing to running 5-10 specialized models locally, let's honestly evaluate whether this approach makes sense for BioPipelines.

#### The Core Question

> **Should we run multiple <10B models on L4/T4 GPUs, or would a simpler approach work better?**

#### Alternative Approaches to Consider

| Approach | Description | Cost | Complexity |
|----------|-------------|------|------------|
| **A: Multi-Model Zoo** | 5-10 specialized models on 2-3 GPUs | ~$1,000/mo hardware | High |
| **B: Single Strong Model** | One 70B model on 1-2 A100s | ~$3,000/mo hardware | Low |
| **C: Cloud API Only** | DeepSeek/GPT-4o via API | ~$50-500/mo usage | Very Low |
| **D: Hybrid (Recommended)** | 2-3 local models + cloud fallback | ~$500/mo + API | Medium |

---

### Part 2: Honest Assessment of Multi-Model Challenges

#### Challenge 1: Orchestration Overhead

```
User Query
    │
    ▼
┌─────────────────┐
│ Intent Parser   │ ← Model 1 inference (~100-300ms)
│ (Llama-3.2-3B)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Orchestrator    │ ← Model 2 inference (~200-500ms)
│ (Nemotron-8B)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Code Generator  │ ← Model 3 inference (~500-2000ms)
│ (Qwen-Coder-7B) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validator       │ ← Model 4 inference (~300-800ms)
│ (Qwen-Coder-7B) │
└─────────────────┘

Total latency: 1.1 - 3.6 seconds for multi-step workflow
vs. Single model: 0.5 - 1.5 seconds
```

**Reality Check:** Each model hop adds latency. For simple queries, this creates unnecessary overhead.

#### Challenge 2: Quality vs. Specialization Trade-off

| Task | Specialized Model | vs. General 70B Model |
|------|-------------------|----------------------|
| Intent Parsing | Llama-3.2-3B: Good | Llama-70B: Excellent |
| Code Generation | Qwen-Coder-7B: Good | GPT-4: Excellent |
| Bio Reasoning | BioMistral-7B: Adequate | Claude-3: Better context |
| Complex Queries | Multiple hops needed | Single inference |

**Reality Check:** A single strong model often outperforms a zoo of weak specialists on complex tasks.

#### Challenge 3: Error Propagation

```
Query: "Analyze my ChIP-seq data and create a heatmap of binding sites"

Multi-Model Chain:
  Intent Parser → misclassifies as "visualization" only
      ↓
  Orchestrator → routes to visualization agent (skips ChIP-seq analysis)
      ↓
  Wrong result

Single Strong Model:
  Understands full context → generates complete pipeline
```

**Reality Check:** Errors compound across model hops. Misclassification in step 1 ruins everything.

#### Challenge 4: Memory Fragmentation

With 2× L4 GPUs (48GB total):

```
Scenario A: Run 5 models simultaneously
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0 (24GB)                                                    │
│ ┌──────────────┐ ┌──────────────┐ ┌────────┐                   │
│ │ Nemotron-8B  │ │ Llama-3.2-3B │ │ BGE-M3 │ ← Fragmented     │
│ │    17GB      │ │     7GB      │ │  2GB   │   (26GB needed)  │
│ └──────────────┘ └──────────────┘ └────────┘   DOESN'T FIT!   │
└─────────────────────────────────────────────────────────────────┘

Scenario B: Model swapping
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0 (24GB)                                                    │
│ ┌──────────────────────────────┐                               │
│ │ Active Model (16-17GB)       │ ← Only one "big" model        │
│ │ + KV Cache (4-6GB)           │   at a time                   │
│ └──────────────────────────────┘                               │
│ ┌──────────────────────────────┐                               │
│ │ Idle models in RAM (waiting) │ ← Swapping takes 5-15 sec    │
│ └──────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

**Reality Check:** You can't actually run 5 medium models on 2× L4 simultaneously. Swapping kills latency.

---

### Part 3: Realistic GPU Configurations

#### What ACTUALLY Fits on L4 + T4 Combinations

##### Configuration: 1× L4 (24GB) + 1× T4 (16GB) = 40GB Total

**Realistic Deployment (Simultaneous):**

| GPU | Model | VRAM | Role |
|-----|-------|------|------|
| L4 | Qwen2.5-Coder-7B (FP16) | 16GB | Code Gen + Validation |
| L4 | KV Cache | 6GB | For long context |
| T4 | Llama-3.2-3B (FP16) | 7GB | Intent + Docs |
| T4 | BGE-M3 | 2GB | Embeddings |
| T4 | Phi-3.5-mini (INT8) | 4GB | Data Analysis |

**Total: 3 generative models + 1 embedding model running simultaneously**

You **cannot** add:
- ❌ Orchestrator (Nemotron-8B needs 17GB - won't fit on T4)
- ❌ Bio model (BioMistral-7B needs 15GB - won't fit on T4)
- ❌ Math model (Qwen2.5-Math needs 15GB - won't fit on T4)

##### Configuration: 2× L4 (48GB Total)

**Realistic Deployment (Simultaneous):**

| GPU | Model | VRAM | Role |
|-----|-------|------|------|
| L4-0 | Nemotron-8B-Orchestrator | 17GB | Orchestration |
| L4-0 | Llama-3.2-3B | 7GB | Intent Parsing |
| L4-1 | Qwen2.5-Coder-7B | 16GB | Code Gen |
| L4-1 | BGE-M3 | 2GB | Embeddings |
| L4-1 | Phi-3.5-mini (INT8) | 4GB | Analysis |

**Total: 4 generative models + 1 embedding model**

You still **cannot** simultaneously run:
- ❌ Dedicated Math model
- ❌ Dedicated Bio model
- ❌ Safety model
- ❌ Documentation model (separate from intent)

##### Configuration: 3× L4 (72GB Total)

**This is where multi-model starts making sense:**

| GPU | Models | Total VRAM |
|-----|--------|------------|
| L4-0 | Nemotron-8B (17GB) + Safety-8B (17GB) | 34GB ❌ |

Wait - even 3× L4 has limits. Let's be realistic:

| GPU | Models | VRAM Used |
|-----|--------|-----------|
| L4-0 | Nemotron-8B-Orchestrator (17GB) + Llama-3.2-3B (7GB) | 24GB ✅ |
| L4-1 | Qwen2.5-Coder-7B (16GB) + BGE-M3 (2GB) + Phi-3.5-mini-INT8 (4GB) | 22GB ✅ |
| L4-2 | BioMistral-7B (15GB) + Qwen2.5-Math-7B-INT8 (8GB) | 23GB ✅ |

**Now we can run 6 generative + 1 embedding model**

---

### Part 4: The Honest Recommendation

#### For BioPipelines Specifically

Given that BioPipelines is a **bioinformatics workflow generator**, let's prioritize:

| Priority | Capability | Why |
|----------|------------|-----|
| 1 | Code Generation | Core function - generate Snakemake/Nextflow |
| 2 | Code Validation | Catch errors before execution |
| 3 | Intent Parsing | Understand what user wants |
| 4 | Bio Domain Knowledge | Explain methods, suggest tools |
| 5 | Orchestration | Route complex queries |
| 6 | Everything else | Nice to have |

#### Recommended Configuration: Pragmatic Hybrid

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDED ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        LOCAL (2× L4 = 48GB)                            │ │
│  │                                                                        │ │
│  │   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐     │ │
│  │   │ Qwen2.5-Coder   │   │ Llama-3.2-3B    │   │ BGE-M3          │     │ │
│  │   │ 7B              │   │                 │   │                 │     │ │
│  │   │                 │   │                 │   │                 │     │ │
│  │   │ • Code Gen      │   │ • Intent Parse  │   │ • Embeddings    │     │ │
│  │   │ • Code Validate │   │ • Quick Docs    │   │ • RAG           │     │ │
│  │   │ • Debugging     │   │ • Routing       │   │                 │     │ │
│  │   │                 │   │                 │   │                 │     │ │
│  │   │ ~16GB           │   │ ~7GB            │   │ ~2GB            │     │ │
│  │   └─────────────────┘   └─────────────────┘   └─────────────────┘     │ │
│  │                                                                        │ │
│  │   Handles: 80% of requests (workflow generation, quick questions)     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      │ Fallback for complex queries          │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        CLOUD API (On-Demand)                           │ │
│  │                                                                        │ │
│  │   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐     │ │
│  │   │ DeepSeek-V3     │   │ Claude-3.5      │   │ GPT-4o          │     │ │
│  │   │                 │   │                 │   │                 │     │ │
│  │   │ • Complex code  │   │ • Bio reasoning │   │ • Fallback      │     │ │
│  │   │ • Multi-file    │   │ • Paper summary │   │ • Edge cases    │     │ │
│  │   │ • $0.27/M tok   │   │ • Long context  │   │                 │     │ │
│  │   └─────────────────┘   └─────────────────┘   └─────────────────┘     │ │
│  │                                                                        │ │
│  │   Handles: 20% of requests (complex reasoning, bio domain, edge cases)│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why This Configuration?

| Decision | Rationale |
|----------|-----------|
| **Qwen2.5-Coder-7B as primary** | Handles 90% of BioPipelines use cases (code gen/validation). State-of-the-art at 7B. |
| **Llama-3.2-3B for intent/routing** | Fast, good enough for parsing. Can also do light docs/explanations. |
| **BGE-M3 for embeddings** | Tiny (2GB), essential for RAG over docs/papers. |
| **Skip dedicated orchestrator** | Llama-3.2-3B can route to 2-3 models. Nemotron adds 17GB for marginal gain. |
| **Skip dedicated bio model locally** | BioMistral is good but not essential. Cloud APIs better for deep bio reasoning. |
| **Skip dedicated math model locally** | Qwen2.5-Coder handles most stats. DeepSeek-V3 API for complex math. |
| **Cloud for complex queries** | DeepSeek-V3 at $0.27/M tokens is cheaper than running 3rd GPU. |

#### Cost Comparison

| Configuration | Monthly Cost | Capability |
|---------------|-------------|------------|
| 3× L4 (full zoo) | ~$1,080 | 6 local models, no API |
| 2× L4 + cloud | ~$720 + ~$50 API | 3 local models + unlimited cloud fallback |
| 1× L4 + cloud | ~$360 + ~$100 API | 2 local models + more cloud |

**Winner: 2× L4 + cloud** - Best balance of cost, capability, and simplicity.

---

### Part 5: Implementation Roadmap

#### Phase 1: Minimal Viable (Week 1-2)
```bash
# Single L4 GPU deployment
Models:
  1. Qwen2.5-Coder-7B (primary - code gen/validation)
  2. BGE-M3 (embeddings)

Cloud fallback:
  - DeepSeek-V3 API for everything else
```

#### Phase 2: Standard (Week 3-4)
```bash
# Add second L4 GPU
Models:
  1. Qwen2.5-Coder-7B (code tasks)
  2. Llama-3.2-3B (intent/docs)
  3. BGE-M3 (embeddings)
  4. Phi-3.5-mini (data analysis - INT8)

Cloud fallback:
  - DeepSeek-V3 for complex code
  - Claude-3.5 for bio reasoning
```

#### Phase 3: Full (Month 2+)
```bash
# Add third L4 GPU only if:
# - Query volume > 10K/day
# - Latency requirements < 500ms
# - Privacy requirements prohibit any cloud usage

Models:
  + Nemotron-8B-Orchestrator (dedicated routing)
  + BioMistral-7B (domain knowledge)
```

---

### Part 6: Multi-Node GPU Configurations (4× L4 + 4× T4)

> **Your Setup:** 4× L4 GPUs + 4× T4 GPUs across 2 separate nodes

#### Total Resources

| GPU Type | Count | VRAM Each | Total VRAM | Best For |
|----------|-------|-----------|------------|----------|
| L4 | 4 | 24GB | **96GB** | 7-9B models in FP16 |
| T4 | 4 | 16GB | **64GB** | 3-7B models in INT8 |
| **Combined** | **8** | - | **160GB** | Full model zoo |

#### The Challenge: Multi-Node Communication

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NODE 1 (L4 GPUs)                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                           │
│  │ L4-0    │ │ L4-1    │ │ L4-2    │ │ L4-3    │                           │
│  │ 24GB    │ │ 24GB    │ │ 24GB    │ │ 24GB    │                           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                           │
│       │          │          │          │                                    │
│       └──────────┴──────────┴──────────┘                                    │
│                      │ NVLink/PCIe (fast)                                   │
│                      ▼                                                      │
│              ┌──────────────┐                                               │
│              │ vLLM Server  │                                               │
│              │ Port 8000    │                                               │
│              └──────────────┘                                               │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       │ Network (slower - 10-100Gbps)
                       │
┌──────────────────────┴──────────────────────────────────────────────────────┐
│                           NODE 2 (T4 GPUs)                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                           │
│  │ T4-0    │ │ T4-1    │ │ T4-2    │ │ T4-3    │                           │
│  │ 16GB    │ │ 16GB    │ │ 16GB    │ │ 16GB    │                           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                           │
│       │          │          │          │                                    │
│       └──────────┴──────────┴──────────┘                                    │
│                      │                                                      │
│                      ▼                                                      │
│              ┌──────────────┐                                               │
│              │ vLLM Server  │                                               │
│              │ Port 8001    │                                               │
│              └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** You can't do tensor parallelism across nodes (too slow). Instead, run **separate vLLM instances** on each node and route requests via HTTP.

#### Optimal Model Distribution: 4× L4 + 4× T4

##### Node 1: L4 GPUs (96GB total) - Heavy Models

| GPU | Model | VRAM | Role |
|-----|-------|------|------|
| L4-0 | Nemotron-8B-Orchestrator | 17GB | Central orchestration |
| L4-0 | Llama-3.2-3B-Instruct | 7GB | Intent parsing |
| L4-1 | Qwen2.5-Coder-7B-Instruct | 16GB | Code generation |
| L4-1 | Phi-3.5-mini (INT8) | 4GB | Quick validation |
| L4-2 | Qwen2.5-Math-7B-Instruct | 15GB | Math/statistics |
| L4-2 | BGE-M3 | 2GB | Embeddings |
| L4-3 | Gemma-2-9B-IT | 19GB | Documentation |

**L4 Node Total: ~80GB used, 16GB headroom for KV cache**

##### Node 2: T4 GPUs (64GB total) - Lighter Models

| GPU | Model | VRAM | Role |
|-----|-------|------|------|
| T4-0 | BioMistral-7B (INT8) | 8GB | Bio reasoning |
| T4-0 | BioGPT-Large | 3GB | Bio entity extraction |
| T4-1 | DeepSeek-Coder-V2-Lite (INT4) | 8GB | Backup code gen |
| T4-1 | BGE-base-en-v1.5 | 0.5GB | Fast embeddings |
| T4-2 | Llama-3.2-3B (backup) | 7GB | Fallback intent |
| T4-2 | Phi-3.5-mini | 8GB | Backup analysis |
| T4-3 | Nemotron-Safety-8B (INT8) | 9GB | Safety filtering |

**T4 Node Total: ~51GB used, 13GB headroom**

#### Does This Solve GPU Issues? **YES!**

| Issue | With 4×L4 + 4×T4 | Status |
|-------|------------------|--------|
| VRAM for full model zoo | 160GB available | ✅ Solved |
| Running 10 models simultaneously | ~131GB needed | ✅ Solved |
| Dedicated orchestrator | Fits on L4-0 | ✅ Solved |
| Dedicated bio model | Fits on T4-0 | ✅ Solved |
| Dedicated math model | Fits on L4-2 | ✅ Solved |
| Dedicated safety model | Fits on T4-3 | ✅ Solved |
| Redundancy/fallbacks | Multiple backups | ✅ Solved |

#### Architecture for Multi-Node Deployment

```yaml
# docker-compose.yml or SLURM job script concept

# NODE 1: L4 GPUs (primary inference)
services:
  orchestrator:
    model: nvidia/Llama-3.1-Nemotron-8B-Orchestrator
    gpu: L4-0
    port: 8000
    
  coder:
    model: Qwen/Qwen2.5-Coder-7B-Instruct
    gpu: L4-1
    port: 8001
    
  math:
    model: Qwen/Qwen2.5-Math-7B-Instruct
    gpu: L4-2
    port: 8002
    
  docs:
    model: google/gemma-2-9b-it
    gpu: L4-3
    port: 8003

# NODE 2: T4 GPUs (secondary/specialized)
  bio:
    model: BioMistral/BioMistral-7B
    gpu: T4-0
    port: 9000
    quantization: int8
    
  safety:
    model: nvidia/Llama-3.1-Nemotron-Safety-8B-V3
    gpu: T4-3
    port: 9003
    quantization: int8

# Load balancer / API Gateway
  gateway:
    routes:
      /code/*: http://node1:8001
      /orchestrate/*: http://node1:8000
      /bio/*: http://node2:9000
      /safety/*: http://node2:9003
```

#### Latency Considerations

| Route | Latency Added |
|-------|---------------|
| Same GPU | ~0ms |
| Same node, different GPU | ~1-5ms |
| Cross-node (network) | ~10-50ms |

**Mitigation:** Keep hot-path models (orchestrator, coder) on same node.

---

### Part 6b: Alternative - Use Your Existing H100/A100 GPUs

Looking at your cluster, you have access to **much more powerful GPUs**:

| Partition | GPU | VRAM | Available |
|-----------|-----|------|-----------|
| `h100flex` | H100 | 80GB | 15 nodes × 1 GPU |
| `h100dualflex` | H100 | 80GB | 4 nodes × 2 GPUs |
| `h100quadflex` | H100 | 80GB | 4 nodes × 4 GPUs |
| `a100flex` | A100 | 40-80GB | 10 nodes × 1 GPU |
| `t4flex` | T4 | 16GB | 10 nodes × 1 GPU |

#### Option A: Single H100 (80GB) - Simplest

```
┌─────────────────────────────────────────────────────────────────┐
│                    1× H100 (80GB)                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Qwen2.5-Coder-7B (16GB) + Nemotron-8B (17GB) +          │   │
│  │ Llama-3.2-3B (7GB) + Phi-3.5 (8GB) + BioMistral (15GB)  │   │
│  │ + BGE-M3 (2GB) = 65GB                                    │   │
│  │                                                          │   │
│  │ Remaining: 15GB for KV cache                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  5 generative models + 1 embedding = FULL ZOO on ONE GPU!      │
└─────────────────────────────────────────────────────────────────┘
```

**Pros:** Simplest setup, no cross-GPU/node communication, lowest latency  
**Cons:** Single point of failure, may have queue contention

#### Option B: 2× H100 (160GB) - Full Redundancy

| GPU | Models | Purpose |
|-----|--------|---------|
| H100-0 | All primary models (~65GB) | Main inference |
| H100-1 | All models duplicated | Redundancy + load balancing |

#### Option C: 1× H100 + 4× T4 (Hybrid)

| GPU | Models | Purpose |
|-----|--------|---------|
| H100 | All heavy models (65GB) | Primary inference |
| T4-0 | Llama-3.2-3B + BGE-M3 | Intent + embeddings (fast path) |
| T4-1 | Phi-3.5-mini | Quick analysis |
| T4-2 | BioGPT-Large + backup | Lightweight bio |
| T4-3 | Safety-8B (INT8) | Safety filtering |

**This is probably optimal for your cluster!**

---

### Part 7: Revised Recommendations

#### For Your Cluster (H100 + T4 Available)

| Priority | Configuration | Why |
|----------|---------------|-----|
| **1st** | 1× H100 + 2× T4 | Run full zoo on H100, offload intent/embeddings/safety to T4s |
| **2nd** | 2× H100 | Full redundancy, load balancing |
| **3rd** | 4× L4 + 4× T4 (once available) | More distributed, good for scaling |
| **4th** | 4× A100 | Similar to H100 approach |

#### Immediate Action (With Current Resources)

```bash
# Request 1× H100 + 2× T4
salloc --partition=h100flex --gres=gpu:1 --time=8:00:00 &
salloc --partition=t4flex --gres=gpu:1 --nodes=2 --time=8:00:00 &

# On H100 node: Run main models via vLLM
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
    --model-name coder \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.4 &

vllm serve nvidia/Llama-3.1-Nemotron-8B-Orchestrator \
    --model-name orchestrator \
    --port 8001 \
    --gpu-memory-utilization 0.3 &

# On T4 nodes: Run lightweight models
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --model-name intent \
    --port 8002 \
    --gpu-memory-utilization 0.5 &
```

---

### Part 8: When to Use Each Approach

#### Use Local Models When:
- ✅ Simple, well-defined tasks (80% of queries)
- ✅ Code generation for common patterns
- ✅ Quick intent classification
- ✅ Embedding generation for RAG
- ✅ Privacy-sensitive data

#### Use Cloud APIs When:
- ✅ Complex multi-step reasoning
- ✅ Novel or unusual requests
- ✅ Deep bio/medical domain questions
- ✅ Very long context (>32K tokens)
- ✅ When local model confidence is low

#### Decision Tree for Runtime Routing

```
                         ┌─────────────────────────────────────┐
                         │          Incoming Query             │
                         └─────────────────┬───────────────────┘
                                           │
                                           ▼
                         ┌─────────────────────────────────────┐
                         │   Llama-3.2-3B: Classify Intent     │
                         └─────────────────┬───────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
           ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
           │  CODE_TASK    │     │  SIMPLE_QA    │     │  COMPLEX      │
           │               │     │               │     │               │
           │ confidence    │     │ confidence    │     │ confidence    │
           │ > 0.8?        │     │ > 0.9?        │     │ < 0.7 OR      │
           │               │     │               │     │ multi-step?   │
           └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
                   │                     │                     │
                   ▼                     ▼                     ▼
           ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
           │ Qwen2.5-Coder │     │ Llama-3.2-3B  │     │ Cloud API     │
           │ (LOCAL)       │     │ (LOCAL)       │     │ (DeepSeek/    │
           │               │     │               │     │  Claude)      │
           └───────────────┘     └───────────────┘     └───────────────┘
```

---

### Part 9: Final Verdict

#### The Multi-Model Zoo: Verdict by GPU Configuration

| Configuration | VRAM | Models Possible | Verdict |
|---------------|------|-----------------|---------|
| **1× T4 (16GB)** | 16GB | 1-2 small | ❌ Not viable for multi-model |
| **4× T4 (64GB)** | 64GB | 4-5 (INT8) | ⚠️ Limited - all models need quantization |
| **1× L4 (24GB)** | 24GB | 2-3 | ⚠️ Limited - need cloud fallback |
| **2× L4 (48GB)** | 48GB | 3-4 | ✅ Viable with cloud fallback |
| **4× L4 (96GB)** | 96GB | 6-7 | ✅ Full zoo possible |
| **4× L4 + 4× T4 (160GB)** | 160GB | 10+ | ✅✅ **Full zoo with redundancy** |
| **1× H100 (80GB)** | 80GB | 5-6 | ✅✅ **Full zoo on single GPU!** |
| **2× H100 (160GB)** | 160GB | 10+ | 🚀 Overkill - consider 70B model |
| **1× H100 + 2× T4 (112GB)** | 112GB | 7-8 | ✅✅ **Optimal for your cluster** |

#### For Your Specific Cluster: Recommended Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED: 1× H100 + 2× T4                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  H100 (80GB) - Primary Inference Node                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Qwen2.5-Coder-7B (16GB)     - Code generation & validation            │ │
│  │  Nemotron-8B-Orchestrator (17GB) - Central routing                     │ │
│  │  Qwen2.5-Math-7B (15GB)      - Math/statistics                         │ │
│  │  BioMistral-7B (15GB)        - Bio domain                              │ │
│  │  ──────────────────────────────────────────────────────────            │ │
│  │  Total: 63GB used, 17GB for KV cache                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  T4-0 (16GB) - Fast Path                    T4-1 (16GB) - Safety & Backup  │
│  ┌─────────────────────────────────┐        ┌─────────────────────────────┐ │
│  │  Llama-3.2-3B (7GB) - Intent   │        │  Safety-8B INT8 (9GB)       │ │
│  │  BGE-M3 (2GB) - Embeddings     │        │  Phi-3.5 INT8 (4GB)         │ │
│  │  Remaining: 7GB                 │        │  Remaining: 3GB             │ │
│  └─────────────────────────────────┘        └─────────────────────────────┘ │
│                                                                              │
│  Total: 8 specialized models, zero cloud dependency                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Alternative: 4× L4 + 4× T4 (When Available)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FUTURE: 4× L4 + 4× T4 (160GB)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Node 1: 4× L4 (96GB)                    Node 2: 4× T4 (64GB)               │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐ │
│  │ L4-0: Orchestrator (17GB)       │    │ T4-0: BioMistral-7B INT8 (8GB) │ │
│  │       + Llama-3.2-3B (7GB)      │    │       + BioGPT (3GB)           │ │
│  │                                 │    │                                 │ │
│  │ L4-1: Qwen-Coder-7B (16GB)      │    │ T4-1: DeepSeek-Coder INT4 (8GB)│ │
│  │       + Phi-3.5 INT8 (4GB)      │    │       + BGE-base (0.5GB)       │ │
│  │                                 │    │                                 │ │
│  │ L4-2: Qwen-Math-7B (15GB)       │    │ T4-2: Llama-3.2-3B (7GB)       │ │
│  │       + BGE-M3 (2GB)            │    │       + Phi-3.5 (8GB)          │ │
│  │                                 │    │                                 │ │
│  │ L4-3: Gemma-2-9B (19GB)         │    │ T4-3: Safety-8B INT8 (9GB)     │ │
│  └─────────────────────────────────┘    └─────────────────────────────────┘ │
│                                                                              │
│  Total: 10+ models with full redundancy across nodes                        │
│  Advantage: Better fault tolerance, horizontal scaling                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Summary: Your Options Ranked

| Rank | Configuration | Cluster Partition | Recommendation |
|------|---------------|-------------------|----------------|
| 🥇 | **1× H100 + 2× T4** | `h100flex` + `t4flex` | Start here - full capability, simple |
| 🥈 | **4× L4 + 4× T4** | (request L4s) | When L4s available - good scaling |
| 🥉 | **2× H100** | `h100dualflex` | If you need redundancy |
| 4 | **1× A100 + 2× T4** | `a100flex` + `t4flex` | A100 40GB holds 3-4 models |
| 5 | **4× T4 only** | `t4flex` | Budget option, limited capability |

#### The Bottom Line (Updated)

> **With 4× L4 + 4× T4 (160GB total):**
> 
> ✅ **YES, this completely solves all GPU issues!**
> 
> You can run the **complete 10-model zoo** simultaneously:
> - All 6 task-specific models (Code, Math, Bio, Docs, Orchestrator, Safety)
> - All 2 embedding models (BGE-M3, BGE-base)
> - Plus backups/fallbacks for redundancy
> - **Zero cloud API dependency**
> 
> **However, with your current H100 access:**
> 
> **1× H100 (80GB)** can run 5-6 models on a SINGLE GPU, which is simpler to manage.
> 
> **Recommended immediate action:** Start with `h100flex` + 2× `t4flex`:
> - H100: Coder + Orchestrator + Math + Bio = 63GB
> - T4-0: Intent + Embeddings = 9GB
> - T4-1: Safety + Backup = 13GB
> 
> This gives you **8 models with zero cross-node complexity**.

---

## Next Steps

> **📋 For detailed implementation instructions, see: [LOCAL_MODEL_IMPLEMENTATION.md](./LOCAL_MODEL_IMPLEMENTATION.md)**

1. **Request H100 + T4 allocation** from your cluster
   ```bash
   # Immediate: Test with available resources
   salloc -p h100flex --gres=gpu:1 -t 4:00:00
   salloc -p t4flex --gres=gpu:1 -N 2 -t 4:00:00
   ```

2. **Request L4 GPUs** from admin for future scaling

3. **Follow Implementation Guide** - Path A (H100 + T4) for immediate deployment

4. **Set up vLLM** for multi-model serving on H100

5. **Implement routing logic** to direct requests to appropriate models

6. **Benchmark** local models vs cloud APIs for your specific use cases

---

## Appendix: Download Commands

### Via Ollama (Easiest)
```bash
# Core models
ollama pull llama3.2:3b          # Intent parsing
ollama pull qwen2.5-coder:7b     # Code generation
ollama pull phi3.5               # Data analysis
ollama pull gemma2:9b            # Documentation

# Optional
ollama pull llama3.1:8b          # Orchestration fallback
ollama pull deepseek-coder-v2:16b  # Alternative code model
```

### Via HuggingFace CLI
```bash
# Install
pip install huggingface_hub

# Download models
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct
huggingface-cli download microsoft/Phi-3.5-mini-instruct
huggingface-cli download nvidia/Llama-3.1-Nemotron-8B-Orchestrator
huggingface-cli download BAAI/bge-m3
huggingface-cli download BioMistral/BioMistral-7B
```

### Via vLLM Serve
```bash
# Start vLLM server with multiple models
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
    --model-name qwen-coder \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8

# In separate terminal
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --model-name llama-intent \
    --port 8001 \
    --max-model-len 4096
```
