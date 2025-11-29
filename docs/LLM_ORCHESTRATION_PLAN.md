# LLM Orchestration Implementation Plan

**Version**: 1.0.0  
**Date**: November 29, 2025  
**Status**: IN PROGRESS

---

## Executive Summary

This document outlines the implementation plan for a unified LLM orchestration layer that enables:
- Seamless switching between local (GPU) and cloud models
- Intelligent task-based routing
- Ensemble patterns for critical decisions
- Cost optimization through smart fallback strategies

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ModelOrchestrator                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                            Public API                                      │  │
│  │  complete() | stream() | ensemble() | delegate()                          │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                            │
│  ┌──────────────────────────────────┴───────────────────────────────────────┐   │
│  │                          Strategy Layer                                   │   │
│  │  AUTO | LOCAL_FIRST | CLOUD_ONLY | ENSEMBLE | PARALLEL | CASCADE         │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│  ┌──────────────────────────────────┴───────────────────────────────────────┐   │
│  │                          Provider Layer                                   │   │
│  │  ┌─────────────────────────┐        ┌─────────────────────────────────┐  │   │
│  │  │     LocalProvider       │        │       CloudProvider              │  │   │
│  │  │  • VLLMBackend          │        │  • LightningBackend              │  │   │
│  │  │  • OllamaBackend        │        │  • (OpenAI, Anthropic via LIT)   │  │   │
│  │  └─────────────────────────┘        └─────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Provider Layer ✅ COMPLETE
**Goal**: Create unified LocalProvider and CloudProvider abstractions

| Task | File | Status |
|------|------|--------|
| 1.1 Create provider base protocol | `llm/providers/base.py` | ✅ DONE |
| 1.2 Implement LocalProvider | `llm/providers/local.py` | ✅ DONE |
| 1.3 Implement CloudProvider | `llm/providers/cloud.py` | ✅ DONE |
| 1.4 Create backends wrapper | `llm/providers/backends.py` | ⏭️ SKIP (inline) |
| 1.5 Add provider tests | `tests/unit/test_providers.py` | ⬜ TODO |
| 1.6 Update __init__.py exports | `llm/__init__.py` | ✅ DONE |

### Phase 2: ModelOrchestrator Core ⏳ IN PROGRESS
**Goal**: Create the main orchestrator with basic strategies

| Task | File | Status |
|------|------|--------|
| 2.1 Define Strategy enum | `llm/strategies.py` | ✅ DONE |
| 2.2 Implement ModelOrchestrator | `llm/orchestrator.py` | ✅ DONE |
| 2.3 Implement LOCAL_FIRST strategy | `llm/orchestrator.py` | ✅ DONE |
| 2.4 Implement AUTO strategy | `llm/orchestrator.py` | ✅ DONE |
| 2.5 Add orchestrator tests | `tests/unit/test_orchestrator.py` | ⬜ TODO |
| 2.6 Add get_orchestrator() factory | `llm/__init__.py` | ✅ DONE |

### Phase 3: Task-Based Routing ✅ COMPLETE
**Goal**: Intelligent routing based on task type

| Task | File | Status |
|------|------|--------|
| 3.1 Define TaskType enum | `llm/task_router.py` | ✅ DONE |
| 3.2 Implement TaskRouter | `llm/task_router.py` | ✅ DONE |
| 3.3 Add task classification | `llm/task_router.py` | ✅ DONE |
| 3.4 Integrate with orchestrator | `llm/orchestrator.py` | ⬜ TODO |
| 3.5 Add routing tests | `tests/unit/test_task_router.py` | ⬜ TODO |

### Phase 4: Ensemble & Advanced Patterns ✅ MOSTLY COMPLETE
**Goal**: Multi-model ensemble for critical decisions

| Task | File | Status |
|------|------|--------|
| 4.1 Define EnsembleStrategy enum | `llm/strategies.py` | ✅ DONE (EnsembleMode) |
| 4.2 Implement VOTE ensemble | `llm/orchestrator.py` | ✅ DONE |
| 4.3 Implement CHAIN ensemble | `llm/orchestrator.py` | ⬜ TODO |
| 4.4 Implement PARALLEL race | `llm/orchestrator.py` | ✅ DONE |
| 4.5 Add CostTracker | `llm/cost_tracker.py` | ⬜ TODO |
| 4.6 Add ensemble tests | `tests/unit/test_ensemble.py` | ⬜ TODO |

### Phase 5: Integration & Documentation ⏳ IN PROGRESS
**Goal**: Integrate with existing codebase

| Task | File | Status |
|------|------|--------|
| 5.1 Update BioPipelines facade | `facade.py` | ⬜ TODO |
| 5.2 Update UnifiedAgent | `agents/unified_agent.py` | ⬜ TODO |
| 5.3 Update Composer | `composer.py` | ⬜ TODO |
| 5.4 Update architecture docs | `docs/ARCHITECTURE.md` | ⬜ TODO |
| 5.5 Create usage examples | `examples/orchestrator_usage.py` | ✅ DONE |

---

## File Structure

```
src/workflow_composer/llm/
├── __init__.py                 # Updated exports
├── base.py                     # Existing LLMAdapter
├── factory.py                  # Existing get_llm()
│
├── orchestrator.py             # NEW: ModelOrchestrator
├── strategies.py               # NEW: Strategy, EnsembleStrategy
├── task_router.py              # NEW: TaskRouter, TaskType
├── cost_tracker.py             # NEW: CostTracker
│
├── providers/                  # NEW: Unified provider layer
│   ├── __init__.py
│   ├── base.py                 # ProviderProtocol
│   ├── local.py                # LocalProvider
│   ├── cloud.py                # CloudProvider
│   └── backends.py             # Backend wrappers
│
└── (existing adapters remain)
    ├── vllm_adapter.py
    ├── ollama_adapter.py
    ├── lightning_adapter.py
    └── ...
```

---

## Detailed Specifications

### 1. ProviderProtocol

```python
@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol for unified providers."""
    
    @property
    def name(self) -> str: ...
    
    @property
    def provider_type(self) -> Literal["local", "cloud"]: ...
    
    def is_available(self) -> bool: ...
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse: ...
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]: ...
    
    def list_models(self) -> List[ModelInfo]: ...
```

### 2. Strategy Enum

```python
class Strategy(Enum):
    """Orchestration strategies."""
    AUTO = "auto"              # Smart routing based on task
    LOCAL_ONLY = "local"       # Force local models
    CLOUD_ONLY = "cloud"       # Force cloud models
    LOCAL_FIRST = "local_first"  # Try local, fallback to cloud
    CLOUD_FIRST = "cloud_first"  # Try cloud, fallback to local
    ENSEMBLE = "ensemble"      # Multiple models
    PARALLEL = "parallel"      # Race local and cloud
    CASCADE = "cascade"        # Escalate on low confidence
```

### 3. TaskType Enum

```python
class TaskType(Enum):
    """Task types for intelligent routing."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    ERROR_DIAGNOSIS = "error_diagnosis"
    DATA_INTERPRETATION = "data_interpretation"
    WORKFLOW_GENERATION = "workflow_generation"
    GENERAL = "general"
```

### 4. ModelOrchestrator API

```python
class ModelOrchestrator:
    """Intelligent multi-model orchestration."""
    
    async def complete(
        self,
        prompt: str,
        strategy: Strategy = Strategy.AUTO,
        task_type: Optional[TaskType] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> OrchestratedResponse: ...
    
    async def stream(
        self,
        prompt: str,
        strategy: Strategy = Strategy.AUTO,
        **kwargs
    ) -> AsyncIterator[str]: ...
    
    async def ensemble(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
        strategy: EnsembleStrategy = EnsembleStrategy.VOTE,
    ) -> EnsembleResponse: ...
```

---

## Testing Strategy

### Unit Tests
- Provider availability mocking
- Strategy selection logic
- Task classification
- Fallback behavior

### Integration Tests
- Real local model calls (if available)
- Cloud API calls (with test keys)
- End-to-end orchestration

### Performance Tests
- Latency comparison (local vs cloud)
- Parallel execution overhead
- Fallback timing

---

## Success Criteria

| Metric | Target |
|--------|--------|
| All unit tests pass | 100% |
| Local fallback works | Verified |
| Cloud fallback works | Verified |
| Strategy selection | Correct routing |
| Backward compatibility | Existing code unchanged |
| Documentation | Complete |

---

## Progress Log

| Date | Phase | Task | Status | Notes |
|------|-------|------|--------|-------|
| 2025-11-29 | 1 | Plan created | ✅ Done | This document |
| 2025-11-29 | 1 | 1.1 Provider base | ⬜ TODO | Starting now |

---

## Notes

- Existing adapters remain unchanged for backward compatibility
- New orchestrator is opt-in via `get_orchestrator()`
- All async methods have sync wrappers
- Cost tracking is optional but recommended
