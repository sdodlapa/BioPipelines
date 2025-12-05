# Dynamic Strategy Selection — Implementation Checklist

**Purpose:** Actionable implementation guide extracted from `DYNAMIC_STRATEGY_IMPLEMENTATION_PLAN.md`  
**Created:** December 5, 2025  
**Status:** Ready for Implementation  
**Estimated Effort:** 2 weeks (Phases 1-2), +2 weeks optional (Phases 3-4)

---

## Overview

This document provides step-by-step implementation tasks. For rationale and design 
decisions, see `DYNAMIC_STRATEGY_IMPLEMENTATION_PLAN.md`.

**Core Principle:** Extend existing components, don't create parallel systems.

---

## Phase 1: Foundation (Week 1)

### 1.1 Deploy Static vLLM Servers

**Goal:** 4 long-running vLLM servers on T4 nodes

#### Tasks

- [ ] **1.1.1** Create vLLM server startup script
  - File: `scripts/llm/start_vllm_server.sh`
  - Accept model name as parameter
  - Configure port mapping (8001-8004)
  
- [ ] **1.1.2** Create SLURM job script for persistent allocation
  - File: `scripts/llm/slurm_vllm_job.sh`
  - 24-hour allocation on t4flex partition
  - Auto-restart on failure
  
- [ ] **1.1.3** Deploy Generalist server
  ```bash
  Model: Qwen/Qwen2.5-7B-Instruct-AWQ
  Port: 8001
  VRAM: ~8GB
  ```
  
- [ ] **1.1.4** Deploy Coder server
  ```bash
  Model: Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
  Port: 8002
  VRAM: ~8GB
  ```
  
- [ ] **1.1.5** Deploy Math server
  ```bash
  Model: Qwen/Qwen2.5-Math-7B-Instruct (INT8)
  Port: 8003
  VRAM: ~10GB
  ```
  
- [ ] **1.1.6** Deploy Embeddings server
  ```bash
  Model: BAAI/bge-m3
  Port: 8004
  VRAM: ~4GB
  ```

- [ ] **1.1.7** Create health monitoring cron job
  - File: `scripts/llm/monitor_vllm_health.py`
  - Check every 5 minutes
  - Alert on failure, auto-restart

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

### 1.2 Create ResourceDetector (Minimal)

**Goal:** Simple health-check based detection, not GPU introspection

#### Tasks

- [ ] **1.2.1** Create ResourceDetector class
  - File: `src/workflow_composer/llm/resource_detector.py`
  - ~50 lines of code
  
  ```python
  @dataclass
  class ResourceStatus:
      vllm_endpoints: Dict[str, bool]  # endpoint -> is_healthy
      cloud_apis: Dict[str, bool]       # api_name -> has_key
      slurm_available: bool
  
  class ResourceDetector:
      def __init__(self, vllm_endpoints: List[str]):
          self.endpoints = vllm_endpoints
      
      def detect(self) -> ResourceStatus:
          return ResourceStatus(
              vllm_endpoints=self._check_vllm_health(),
              cloud_apis=self._check_cloud_keys(),
              slurm_available=self._check_slurm(),
          )
      
      def _check_vllm_health(self) -> Dict[str, bool]:
          results = {}
          for endpoint in self.endpoints:
              try:
                  r = requests.get(f"{endpoint}/health", timeout=2)
                  results[endpoint] = r.ok
              except:
                  results[endpoint] = False
          return results
  ```

- [ ] **1.2.2** Add unit tests
  - File: `tests/unit/test_resource_detector.py`
  - Mock HTTP responses

#### Verification
```python
detector = ResourceDetector([
    "http://localhost:8001",
    "http://localhost:8002",
])
status = detector.detect()
assert status.vllm_endpoints["http://localhost:8001"] == True
```

---

### 1.3 Extend StrategyConfig

**Goal:** Add profile support to existing dataclass

#### Tasks

- [ ] **1.3.1** Extend StrategyConfig in `src/workflow_composer/llm/strategies.py`
  
  ```python
  # Add these fields to existing StrategyConfig
  profile_name: Optional[str] = None
  allow_cloud: bool = True          # Data governance
  debug_routing: bool = False       # Verbose logging
  vllm_endpoints: Dict[str, str] = field(default_factory=dict)
  ```

- [ ] **1.3.2** Add profile loading method
  
  ```python
  @classmethod
  def from_yaml(cls, path: Path) -> "StrategyConfig":
      with open(path) as f:
          data = yaml.safe_load(f)
      return cls(**data)
  ```

- [ ] **1.3.3** Update existing PRESETS dict with vllm_endpoints

#### Verification
```python
config = StrategyConfig.from_yaml("config/strategies/t4_hybrid.yaml")
assert config.profile_name == "t4_hybrid"
assert config.allow_cloud == True
```

---

### 1.4 Create Strategy Profile YAML Files

**Goal:** 4 profile files covering main deployment scenarios

#### Tasks

- [ ] **1.4.1** Create `config/strategies/` directory

- [ ] **1.4.2** Create `t4_hybrid.yaml` (primary profile)
  ```yaml
  profile_name: t4_hybrid
  description: "T4 vLLM fleet with DeepSeek cloud fallback"
  allow_cloud: true
  debug_routing: false
  
  vllm_endpoints:
    generalist: "http://t4-node-01:8001"
    coder: "http://t4-node-02:8002"
    math: "http://t4-node-03:8003"
    embeddings: "http://t4-node-04:8004"
  
  task_routing:
    intent_parsing: generalist
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

### 1.5 Basic Metrics Collection

**Goal:** Track baseline quality metrics from day 1

#### Tasks

- [ ] **1.5.1** Create metrics schema
  - File: `src/workflow_composer/llm/metrics.py`
  
  ```python
  @dataclass
  class RoutingMetric:
      timestamp: datetime
      task_type: str
      query_length: int
      model_used: str
      fallback_depth: int  # 0=primary, 1+=fallback
      latency_ms: float
      success: bool
      error: Optional[str] = None
  ```

- [ ] **1.5.2** Add logging to existing router
  - Log every routing decision to JSON file
  - `logs/routing_metrics.jsonl`

- [ ] **1.5.3** Create simple analysis script
  - File: `scripts/analyze_routing_metrics.py`
  - Report: fallback rate, latency by task, error rate

---

## Phase 2: Integration (Week 2)

### 2.1 Add switch_strategy() to ModelOrchestrator

**Goal:** Allow runtime strategy changes

#### Tasks

- [ ] **2.1.1** Add method to `src/workflow_composer/llm/orchestrator.py`
  
  ```python
  def switch_strategy(self, profile_name: str) -> None:
      """Switch to a different strategy profile."""
      profile_path = Path(f"config/strategies/{profile_name}.yaml")
      if not profile_path.exists():
          raise ValueError(f"Unknown profile: {profile_name}")
      
      new_config = StrategyConfig.from_yaml(profile_path)
      self._apply_config(new_config)
      logger.info(f"Switched to strategy: {profile_name}")
  
  def _apply_config(self, config: StrategyConfig) -> None:
      """Apply configuration changes."""
      self.strategy_config = config
      self._reinitialize_routers()
  ```

- [ ] **2.1.2** Add resource-aware initialization
  
  ```python
  def __init__(self, strategy: Optional[str] = None):
      if strategy is None:
          strategy = self._auto_detect_strategy()
      self.switch_strategy(strategy)
  
  def _auto_detect_strategy(self) -> str:
      """Detect best strategy based on available resources."""
      detector = ResourceDetector(KNOWN_VLLM_ENDPOINTS)
      status = detector.detect()
      
      if any(status.vllm_endpoints.values()):
          return "t4_hybrid"
      elif any(status.cloud_apis.values()):
          return "cloud_only"
      else:
          raise RuntimeError("No LLM backends available")
  ```

- [ ] **2.1.3** Add integration tests
  - File: `tests/integration/test_strategy_switching.py`

---

### 2.2 Wire T4ModelRouter into CascadingProviderRouter

**Goal:** Unified routing through existing provider cascade

#### Tasks

- [ ] **2.2.1** Modify `src/workflow_composer/providers/router.py`
  
  ```python
  class CascadingProviderRouter:
      def __init__(self, config: StrategyConfig):
          self.config = config
          self.t4_router = T4ModelRouter(config.vllm_endpoints)
          self.cloud_router = ProviderRouter(config.fallback)
      
      async def route(self, request: LLMRequest) -> LLMResponse:
          task = request.task_type
          
          # Check if local routing allowed and available
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

### 2.3 Add CLI Strategy Flag

**Goal:** `--strategy` flag for all entry points

#### Tasks

- [ ] **2.3.1** Add to main CLI entry point
  
  ```python
  @click.option(
      "--strategy", "-s",
      type=click.Choice(["t4_hybrid", "t4_local_only", "cloud_only", "development"]),
      default=None,
      help="Strategy profile to use (auto-detected if not specified)"
  )
  def main(strategy: Optional[str]):
      orchestrator = ModelOrchestrator(strategy=strategy)
  ```

- [ ] **2.3.2** Add environment variable fallback
  ```python
  strategy = strategy or os.getenv("BIOPIPELINES_STRATEGY")
  ```

- [ ] **2.3.3** Update CLI help text and README

---

### 2.4 Integration Tests

**Goal:** End-to-end tests with actual vLLM servers

#### Tasks

- [ ] **2.4.1** Create integration test suite
  - File: `tests/integration/test_strategy_integration.py`
  - Requires: At least one vLLM server running
  
  ```python
  @pytest.mark.integration
  def test_t4_hybrid_routing():
      orchestrator = ModelOrchestrator(strategy="t4_hybrid")
      response = orchestrator.generate(
          prompt="Write a Python function",
          task_type="code_generation"
      )
      assert response.model.startswith("Qwen")  # Used local
  
  @pytest.mark.integration  
  def test_fallback_on_server_down():
      # Stop t4-coder, verify fallback to cloud
      pass
  ```

- [ ] **2.4.2** Create CI workflow for integration tests
  - Only run when vLLM servers are available
  - Skip gracefully otherwise

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

### Phase 1 Complete When:
- [ ] 4 vLLM servers running and healthy for 24+ hours
- [ ] ResourceDetector correctly reports server status
- [ ] StrategyConfig loads from YAML without errors
- [ ] Metrics logging to `logs/routing_metrics.jsonl`

### Phase 2 Complete When:
- [ ] `switch_strategy()` works without restart
- [ ] T4 → cloud fallback works when server down
- [ ] CLI `--strategy` flag documented and working
- [ ] Integration tests pass on CI

### Phase 3 Complete When:
- [ ] RouteLLM reduces cloud API calls by >30%
- [ ] Prefix caching reduces TTFT by >20%
- [ ] Debug logs provide actionable troubleshooting info

---

*This checklist is a living document. Update task status as work progresses.*
