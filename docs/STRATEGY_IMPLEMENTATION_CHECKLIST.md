# Dynamic Strategy Selection — Implementation Checklist

**Purpose:** Actionable implementation guide extracted from `DYNAMIC_STRATEGY_IMPLEMENTATION_PLAN.md`  
**Created:** December 5, 2025  
**Status:** ✅ Phases 1-2 COMPLETE  
**Estimated Effort:** 2 weeks (Phases 1-2), +2 weeks optional (Phases 3-4)

---

## Overview

This document provides step-by-step implementation tasks. For rationale and design 
decisions, see `DYNAMIC_STRATEGY_IMPLEMENTATION_PLAN.md`.

**Core Principle:** Extend existing components, don't create parallel systems.

---

## Phase 1: Foundation (Week 1) ✅ COMPLETE

### 1.1 Deploy Static vLLM Servers ✅

**Goal:** 4 long-running vLLM servers on T4 nodes

#### Tasks

- [x] **1.1.1** Create vLLM server startup script
  - File: `scripts/llm/deploy_core_models.sh`
  - Supports start/stop/status/restart commands
  - Manages all 4 models (generalist, coder, math, embeddings)
  
- [x] **1.1.2** Create SLURM job integration
  - Script uses t4flex partition
  - Connection file management for dynamic ports
  
- [x] **1.1.3-1.1.6** Server configurations defined for all 4 models:
  - Generalist: Qwen/Qwen2.5-7B-Instruct-AWQ (8001)
  - Coder: Qwen/Qwen2.5-Coder-7B-Instruct-AWQ (8002)  
  - Math: Qwen/Qwen2.5-Math-7B-Instruct (8003)
  - Embeddings: BAAI/bge-m3 (8004)

- [x] **1.1.7** Create health monitoring
  - File: `scripts/llm/monitor_vllm_health.py`
  - Comprehensive health checks with alerts

#### Verification
```bash
# Test each server
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health

# Test inference
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-7B-Instruct-AWQ", "prompt": "Hello", "max_tokens": 10}'
```

---

### 1.2 Create ResourceDetector (Minimal) ✅

**Goal:** Simple health-check based detection, not GPU introspection

#### Tasks

- [x] **1.2.1** Create ResourceDetector class
  - File: `src/workflow_composer/llm/resource_detector.py`
  - Includes ResourceStatus dataclass with deployment_mode property
  - Health checks for vLLM endpoints
  - Cloud API key detection
  - SLURM availability check
  
- [x] **1.2.2** Add unit tests
  - Included in `tests/test_strategy_routing.py`

---

### 1.3 Extend StrategyConfig ✅

**Goal:** Add profile support to existing dataclass

#### Tasks

- [x] **1.3.1** Extended StrategyConfig in `src/workflow_composer/llm/strategies.py`
  - Added: profile_name, profile_description, allow_cloud, allow_cloud_for_tasks
  - Added: debug_routing, vllm_endpoints, task_routing, data_governance

- [x] **1.3.2** Added profile loading method
  - `load_profile(name)` function loads from `config/strategies/{name}.yaml`

- [x] **1.3.3** Updated existing code to use new fields

---

### 1.4 Create Strategy Profile YAML Files ✅

**Goal:** 4 profile files covering main deployment scenarios

#### Tasks

- [x] **1.4.1** Created `config/strategies/` directory

- [x] **1.4.2** Created `t4_hybrid.yaml` (primary profile)
  - T4 vLLM fleet with cloud fallback
  - Task routing to specialized models
  - DeepSeek/Claude fallback chain

- [x] **1.4.3** Created `t4_local_only.yaml` (no cloud, PHI-safe)
  - allow_cloud: false
  - Data governance: PHI mode

- [x] **1.4.4** Created `cloud_only.yaml` (development/no GPUs)
  - No local endpoints
  - Cloud-first routing

- [x] **1.4.5** Created `development.yaml` (fast iteration)
  - Debug routing enabled
  - Relaxed settings for testing
    code_generation: coder
    code_validation: coder
    math_statistics: math
    data_analysis: generalist
    documentation: generalist
    embeddings: embeddings
    orchestration: cloud  # Always cloud
    biomedical: generalist
    safety: generalist
  
  fallback:
    primary: deepseek-v3
    secondary: claude-3.5-sonnet
  ```

- [ ] **1.4.3** Create `t4_local_only.yaml` (no cloud, PHI-safe)
  ```yaml
  profile_name: t4_local_only
  description: "T4 only, no cloud fallback (PHI-safe)"
  allow_cloud: false
  # ... same task_routing, no fallback section
  ```

- [ ] **1.4.4** Create `cloud_only.yaml` (development/no GPUs)
  ```yaml
  profile_name: cloud_only
  description: "Cloud APIs only, no local models"
  allow_cloud: true
  vllm_endpoints: {}
  task_routing:
    "*": cloud
  fallback:
    primary: deepseek-v3
  ```

- [ ] **1.4.5** Create `development.yaml` (fast iteration)
  ```yaml
  profile_name: development
  description: "Minimal config for local development"
  allow_cloud: true
  debug_routing: true
  # Use smallest/fastest models or mocks
  ```

---

### 1.5 Basic Metrics Collection ✅

**Goal:** Track baseline quality metrics from day 1

#### Tasks

- [x] **1.5.1** Created metrics schema
  - File: `src/workflow_composer/llm/metrics.py`
  - RoutingDecision dataclass with full context
  - RoutingMetrics class for aggregation

- [x] **1.5.2** Added logging to router
  - Logs to `logs/routing_metrics.jsonl`
  - Thread-safe with buffering

- [x] **1.5.3** Created summary methods
  - `get_summary()` returns statistics
  - `print_summary()` for console output

---

## Phase 2: Integration (Week 2) ✅ COMPLETE

### 2.1 Add switch_strategy() to ModelOrchestrator ✅

**Goal:** Allow runtime strategy changes

#### Tasks

- [x] **2.1.1** Added `switch_strategy()` method
  - Accepts Strategy enum, profile name, or StrategyConfig
  - Updates config and reinitializes routers

- [x] **2.1.2** Added resource-aware initialization
  - `auto_detect=True` parameter
  - `_auto_detect_config()` internal method
  - `get_resource_status()` public method

- [x] **2.1.3** Added integration tests
  - File: `tests/test_strategy_routing.py`
  - 38 passing tests

---

### 2.2 Wire T4ModelRouter into Provider System ✅

**Goal:** Task-based routing integrated with orchestrator

#### Tasks

- [x] **2.2.1** Added T4ModelRouter import to orchestrator
  - Conditional import (graceful fallback if not available)
  - T4_ROUTER_AVAILABLE flag

- [x] **2.2.2** Added `complete_with_task()` method
  - Routes by task type (code, math, general, embeddings)
  - Falls back to generic completion if router unavailable

- [x] **2.2.3** Added `has_task_router()` check
  - Returns True if T4 router is configured

- [x] **2.2.4** Added `embed_with_task()` for embeddings

- [x] **2.2.5** Added cost estimation for cloud fallback
          if self.config.allow_cloud or self._is_local_available(task):
              try:
                  return await self.t4_router.route(request)
              except LocalModelUnavailable:
                  if not self.config.allow_cloud:
                      raise  # PHI mode, can't fallback
          
          # Fallback to cloud
          return await self.cloud_router.route(request)
  ```

- [ ] **2.2.2** Update T4ModelRouter to use config endpoints
  - Currently hardcoded, change to config-driven

- [ ] **2.2.3** Add fallback metrics (track fallback_depth)

---

### 2.3 Add CLI Strategy Flag ✅

**Goal:** `--strategy` flag for all entry points

#### Tasks

- [x] **2.3.1** Added to main CLI entry point
  - `--strategy` / `-s` option on `generate` and `chat` commands
  - Accepts strategy names or profile names

- [x] **2.3.2** Added `strategy` subcommand
  - `--list`: List available strategies and profiles
  - `--check`: Detect resources and recommend strategy
  - `--test`: Test a strategy with optional query

- [x] **2.3.3** Updated CLI help text

---

### 2.4 Integration Tests ✅

**Goal:** Comprehensive test coverage

#### Tasks

- [x] **2.4.1** Created test suite
  - File: `tests/test_strategy_routing.py`
  - 38 tests covering all components
  - Tests for: StrategyConfig, ResourceDetector, Orchestrator, Metrics, CLI, T4 routing

- [x] **2.4.2** All tests passing
  - Run with: `pytest tests/test_strategy_routing.py -v`

---

## Phase 3: Optimization (Week 3-4) — OPTIONAL

*Only implement after Phase 1-2 are stable and metrics show need*

### 3.1 Complexity Routing (RouteLLM)

- [ ] **3.1.1** Install RouteLLM: `pip install routellm`
- [ ] **3.1.2** Create complexity router wrapper
- [ ] **3.1.3** Enable only for `code_generation` and `biomedical` tasks
- [ ] **3.1.4** Tune threshold based on quality metrics

### 3.2 Prefix Caching

- [ ] **3.2.1** Enable in vLLM server config: `--enable-prefix-caching`
- [ ] **3.2.2** Create bioinformatics context templates
- [ ] **3.2.3** Measure TTFT improvement

### 3.3 Debug Routing Logs

- [ ] **3.3.1** Create RoutingDecision dataclass
- [ ] **3.3.2** Log full context when `debug_routing=True`
- [ ] **3.3.3** Create log analysis dashboard

---

## Phase 4: Specialized Models (Month 2) — CONDITIONAL

*Only implement if Phase 1-2 metrics show quality issues*

### Trigger Conditions

| Add Model | Trigger | Metric Threshold |
|-----------|---------|------------------|
| BioMistral-7B | Biomedical accuracy | <85% on test set |
| Llama-Guard-3 | Safety false negatives | >5% missed flags |
| Llama-3.2-3B | Intent parsing | <90% accuracy |

### Tasks (if triggered)

- [ ] **4.1** Deploy additional vLLM server
- [ ] **4.2** Update task_routing in profiles
- [ ] **4.3** A/B test generalist vs specialized

---

## Quick Reference

### File Locations

| Component | Path |
|-----------|------|
| ResourceDetector | `src/workflow_composer/llm/resource_detector.py` |
| StrategyConfig extension | `src/workflow_composer/llm/strategies.py` |
| Profile YAMLs | `config/strategies/*.yaml` |
| vLLM scripts | `scripts/llm/start_vllm_server.sh` |
| Health monitor | `scripts/llm/monitor_vllm_health.py` |
| Metrics | `src/workflow_composer/llm/metrics.py` |
| Integration tests | `tests/integration/test_strategy_*.py` |

### Commands

```bash
# Start vLLM server manually
./scripts/llm/start_vllm_server.sh generalist

# Start via SLURM
sbatch scripts/llm/slurm_vllm_job.sh generalist

# Run with specific strategy
python -m workflow_composer --strategy t4_hybrid

# Run integration tests
pytest tests/integration/ -m integration -v

# Check routing metrics
python scripts/analyze_routing_metrics.py --since 24h
```

### Environment Variables

```bash
BIOPIPELINES_STRATEGY=t4_hybrid          # Default strategy
BIOPIPELINES_DEBUG_ROUTING=1             # Enable verbose routing logs
DEEPSEEK_API_KEY=xxx                     # Cloud fallback
OPENAI_API_KEY=xxx                       # For embeddings fallback
```

---

## Success Criteria

### Phase 1 Complete When: ✅ DONE
- [x] 4 vLLM server deployment script ready
- [x] ResourceDetector correctly reports server status
- [x] StrategyConfig loads from YAML without errors
- [x] Metrics logging to `logs/routing_metrics.jsonl`

### Phase 2 Complete When: ✅ DONE
- [x] `switch_strategy()` works without restart
- [x] T4 → cloud fallback logic implemented
- [x] CLI `--strategy` flag documented and working
- [x] Integration tests pass (38/38)

### Phase 3 Complete When:
- [ ] RouteLLM reduces cloud API calls by >30%
- [ ] Prefix caching reduces TTFT by >20%
- [ ] Debug logs provide actionable troubleshooting info

---

*This checklist is a living document. Update task status as work progresses.*
