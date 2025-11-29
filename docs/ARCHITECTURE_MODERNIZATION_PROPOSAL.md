# BioPipelines Architecture Modernization Proposal

**Date**: November 29, 2025  
**Status**: ✅ IMPLEMENTED (Phase 1 Complete)
**Last Updated**: November 30, 2025
**Effort Estimate**: 2-4 weeks for full implementation

---

## Implementation Status

### ✅ COMPLETED Tasks (Phase 1)
| Task | Description | Status |
|------|-------------|--------|
| **#1** | Dependency Injection Container | ✅ `infrastructure/container.py` |
| **#3** | BioPipelines Facade | ✅ `facade.py` |
| **#6** | Protocol-based Interfaces | ✅ `infrastructure/protocols.py` |
| **#7** | Configuration Consolidation | ✅ `infrastructure/settings.py` |
| **#8** | Error Hierarchy | ✅ `infrastructure/exceptions.py` |
| **#10** | Structured Logging | ✅ `infrastructure/logging.py` |

### ❌ DEFERRED Tasks (After Critical Assessment)
| Task | Description | Reason for Deferral |
|------|-------------|---------------------|
| **#2** | Domain-Driven Structure | Would break 50+ imports across 34K LOC. Current structure is functional. |
| **#4** | CQRS Pattern | Tool system already well-organized. CQRS adds complexity without clear benefit. |
| **#5** | Event-Driven Architecture | Protocol defined for future use. Implementing now is premature. |
| **#9** | Async/Sync Wrappers | Already implemented in `unified_agent.py` and `providers/`. No work needed. |

### Test Results
- **217 tests passed**, 6 skipped
- **24 new infrastructure tests** covering all new modules

---

## Executive Summary

This document proposes **significant architectural improvements** to bring BioPipelines to professional/enterprise standards. The proposals range from "quick wins" to "major refactoring" based on modern software patterns.

### Current State

| Metric | Current | Target |
|--------|---------|--------|
| **Total Lines** | 53,498 | ~40,000 (25% reduction) |
| **Files** | 129 | ~80 (modular consolidation) |
| **Test Coverage** | 22% | 70%+ |
| **Largest File** | 1,088 lines | <400 lines |
| **Duplicate Code** | Moderate | Minimal |
| **API Consistency** | Mixed | Unified |

---

## 🔴 CRITICAL ISSUES (Must Fix)

### 1. No Dependency Injection Container

**Problem**: Components create their own dependencies, making testing hard and coupling tight.

**Current Pattern** (scattered throughout):
```python
class Composer:
    def __init__(self):
        self.llm = get_llm(...)           # Creates dependency internally
        self.tool_selector = ToolSelector(...)  # Creates dependency internally
```

**Proposed Pattern** (dependency injection):
```python
from dataclasses import dataclass
from typing import Protocol

class LLMProtocol(Protocol):
    def complete(self, prompt: str) -> str: ...

@dataclass
class AppContainer:
    """Central dependency container."""
    llm: LLMProtocol
    tool_selector: ToolSelector
    module_mapper: ModuleMapper
    config: Config
    
    @classmethod
    def from_config(cls, config: Config) -> "AppContainer":
        return cls(
            llm=get_llm(config.llm.provider),
            tool_selector=ToolSelector(config.paths.tools),
            module_mapper=ModuleMapper(config.paths.modules),
            config=config,
        )

class Composer:
    def __init__(self, container: AppContainer):
        self.llm = container.llm
        self.tool_selector = container.tool_selector
```

**Benefits**:
- Easy testing with mock dependencies
- Clear dependency graph
- Single configuration point

**Effort**: Medium (1 week)

---

### 2. No Clear Domain Model

**Problem**: Business logic is scattered. No clear separation of:
- Domain entities (Workflow, Tool, Sample, etc.)
- Domain services (WorkflowGenerator, ToolSelector, etc.)
- Infrastructure (LLM adapters, file I/O, etc.)

**Proposed Structure** (Domain-Driven Design):
```
src/workflow_composer/
├── domain/                    # Pure business logic (no I/O)
│   ├── entities/
│   │   ├── workflow.py       # Workflow, WorkflowStep
│   │   ├── tool.py           # Tool, ToolParameter
│   │   ├── sample.py         # Sample, SampleMetadata
│   │   └── job.py            # Job, JobStatus
│   ├── services/
│   │   ├── workflow_builder.py  # Pure workflow construction
│   │   ├── tool_matcher.py      # Pure tool matching logic
│   │   └── job_scheduler.py     # Pure scheduling logic
│   └── value_objects/
│       ├── genome.py         # GenomeAssembly, Organism
│       └── analysis.py       # AnalysisType, DataType
│
├── application/               # Use cases / orchestration
│   ├── commands/             # Write operations
│   │   ├── generate_workflow.py
│   │   ├── submit_job.py
│   │   └── download_data.py
│   ├── queries/              # Read operations
│   │   ├── get_job_status.py
│   │   ├── list_workflows.py
│   │   └── search_tools.py
│   └── services/
│       └── composer_service.py
│
├── infrastructure/            # External I/O
│   ├── llm/                  # LLM adapters
│   ├── storage/              # File, cloud storage
│   ├── slurm/                # SLURM interface
│   └── repositories/         # Data access
│
└── interfaces/                # Entry points
    ├── cli/
    ├── web/
    └── api/
```

**Benefits**:
- Clear separation of concerns
- Domain logic is testable without mocking I/O
- Easy to understand business rules

**Effort**: High (2-3 weeks)

---

### 3. Multiple Entry Points with No Common Base

**Problem**: 5+ different "agent" entry points:
- `UnifiedAgent`
- `AutonomousAgent`
- `AgentOrchestrator`
- `AgentBridge`
- `ReactAgent`

**Current Reality**: Users don't know which to use.

**Proposed Solution** (Facade Pattern):

```python
# One entry point to rule them all
class BioPipelines:
    """
    The single entry point for BioPipelines.
    
    Usage:
        bp = BioPipelines()
        
        # Chat mode
        response = bp.chat("Create an RNA-seq workflow")
        
        # Direct workflow generation
        workflow = bp.generate_workflow("RNA-seq", samples=["s1.fq", "s2.fq"])
        
        # Submit to cluster
        job = bp.submit(workflow, cluster="slurm")
        
        # Monitor
        status = bp.status(job.id)
    """
    
    def __init__(self, config: Optional[Config] = None):
        self._container = AppContainer.from_config(config or Config.load())
        self._agent = UnifiedAgent(self._container)
    
    def chat(self, message: str) -> str:
        """Natural language interface."""
        return self._agent.process_sync(message).message
    
    def generate_workflow(
        self, 
        analysis_type: str,
        samples: List[str],
        **options
    ) -> Workflow:
        """Programmatic workflow generation."""
        pass
    
    def submit(self, workflow: Workflow, cluster: str = "local") -> Job:
        """Submit workflow for execution."""
        pass
    
    def status(self, job_id: str) -> JobStatus:
        """Get job status."""
        pass
```

**Benefits**:
- Single, obvious entry point
- Hides complexity
- Backward compatible (keep old entry points as aliases)

**Effort**: Low (3-5 days)

---

## 🟡 SIGNIFICANT IMPROVEMENTS

### 4. Adopt Command Query Responsibility Segregation (CQRS)

**Problem**: Same code paths for reads and writes creates complexity.

**Proposed Pattern**:
```python
# Commands (writes) - explicit, validated, logged
@dataclass
class GenerateWorkflowCommand:
    query: str
    output_dir: Path
    options: Dict[str, Any]

class GenerateWorkflowHandler:
    def handle(self, cmd: GenerateWorkflowCommand) -> Workflow:
        # Validate
        # Generate
        # Persist
        # Audit
        pass

# Queries (reads) - simple, cacheable
@dataclass  
class GetJobStatusQuery:
    job_id: str

class GetJobStatusHandler:
    def handle(self, query: GetJobStatusQuery) -> JobStatus:
        # Just fetch and return
        pass
```

**Benefits**:
- Clear separation of read/write concerns
- Easier to optimize (cache reads, validate writes)
- Better audit trail

**Effort**: Medium (1 week)

---

### 5. Event-Driven Architecture for Autonomous Operations

**Problem**: Current autonomous agent uses polling and callbacks, which doesn't scale.

**Proposed Pattern** (Event Sourcing):
```python
# Events
@dataclass
class WorkflowGeneratedEvent:
    workflow_id: str
    timestamp: datetime
    user_id: str

@dataclass
class JobSubmittedEvent:
    job_id: str
    workflow_id: str
    cluster: str

@dataclass
class JobCompletedEvent:
    job_id: str
    status: str
    outputs: List[str]

# Event Bus
class EventBus:
    def publish(self, event: Event) -> None: ...
    def subscribe(self, event_type: Type[Event], handler: Callable) -> None: ...

# Handlers react to events
class SlackNotifier:
    def on_job_failed(self, event: JobFailedEvent):
        slack.post(f"Job {event.job_id} failed: {event.error}")

class AutoRecovery:
    def on_job_failed(self, event: JobFailedEvent):
        if event.is_recoverable:
            commands.resubmit_job(event.job_id, increase_memory=True)
```

**Benefits**:
- Decoupled components
- Easy to add new behaviors (just subscribe to events)
- Natural audit trail

**Effort**: High (2 weeks)

---

### 6. Protocol-Based Interfaces (Structural Typing)

**Problem**: Too many ABCs and inheritance hierarchies.

**Proposed Pattern** (Python Protocols):
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMProtocol(Protocol):
    """Any LLM must provide these methods."""
    
    def complete(self, prompt: str) -> str: ...
    def chat(self, messages: List[Message]) -> str: ...
    
    @property
    def model_name(self) -> str: ...

# Now any class that has these methods works
class VLLMAdapter:  # No inheritance needed!
    def complete(self, prompt: str) -> str: ...
    def chat(self, messages: List[Message]) -> str: ...
    
    @property
    def model_name(self) -> str: ...

# Type checking works
def generate_workflow(llm: LLMProtocol) -> Workflow:
    response = llm.complete("...")  # Type safe
```

**Benefits**:
- No coupling through inheritance
- Easier testing (any object with matching methods works)
- More Pythonic

**Effort**: Low (3-5 days)

---

## 🟢 QUICK WINS (Low effort, high impact)

### 7. Consolidate Configuration

**Current**: Config spread across:
- `config/composer.yaml`
- `config/analysis_definitions.yaml`
- `config/tool_mappings.yaml`
- Environment variables
- Hardcoded defaults

**Proposed**: Single `pydantic-settings` configuration:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """All configuration in one place."""
    
    # LLM
    llm_provider: str = "lightning"
    llm_model: str = "deepseek-ai/deepseek-v3"
    llm_api_key: Optional[str] = None
    
    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("generated_workflows")
    
    # SLURM
    slurm_partition: str = "main"
    slurm_default_memory: str = "16G"
    
    class Config:
        env_prefix = "BIOPIPELINES_"
        env_file = ".env"

settings = Settings()  # Auto-loads from env + .env
```

**Effort**: Low (2 days)

---

### 8. Proper Error Hierarchy

**Current**: Mixed use of exceptions, error codes, and ToolResult.success=False.

**Proposed**:
```python
# Base exceptions
class BioPipelinesError(Exception):
    """Base for all BioPipelines errors."""
    pass

class ConfigurationError(BioPipelinesError):
    """Invalid configuration."""
    pass

class ToolNotFoundError(BioPipelinesError):
    """Requested tool doesn't exist."""
    pass

class WorkflowValidationError(BioPipelinesError):
    """Workflow failed validation."""
    def __init__(self, errors: List[ValidationError]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation errors")

class LLMError(BioPipelinesError):
    """LLM call failed."""
    pass

class SLURMError(BioPipelinesError):
    """SLURM operation failed."""
    pass
```

**Effort**: Low (2 days)

---

### 9. Standardize Async/Sync Patterns

**Current**: Inconsistent async usage - some components are async, some sync, some have both with `_sync` suffix.

**Proposed**: Consistent pattern:
```python
class ToolExecutor:
    """Tools are async-first, with sync wrapper."""
    
    async def execute(self, tool_name: str, **params) -> ToolResult:
        """Async execution (primary)."""
        pass
    
    def execute_sync(self, tool_name: str, **params) -> ToolResult:
        """Sync wrapper for non-async contexts."""
        return asyncio.run(self.execute(tool_name, **params))

# Or use anyio for flexible sync/async
import anyio

async def run_tool(name: str) -> Result:
    ...

# Works in both contexts
result = anyio.from_thread.run(run_tool, "scan_data")
```

**Effort**: Medium (1 week)

---

### 10. Add Structured Logging

**Current**: Mixed logging, print statements, and no correlation IDs.

**Proposed**:
```python
import structlog

log = structlog.get_logger()

# Every operation gets a correlation ID
@contextmanager
def operation_context(operation: str, **metadata):
    op_id = str(uuid4())[:8]
    with structlog.contextvars.bound_contextvars(
        operation=operation,
        operation_id=op_id,
        **metadata
    ):
        log.info("operation.started")
        try:
            yield op_id
            log.info("operation.completed")
        except Exception as e:
            log.error("operation.failed", error=str(e))
            raise

# Usage
with operation_context("generate_workflow", query=user_query) as op_id:
    workflow = generate(...)
    
# Output: {"event": "operation.started", "operation": "generate_workflow", 
#          "operation_id": "a1b2c3d4", "query": "RNA-seq...", "timestamp": "..."}
```

**Effort**: Low (2-3 days)

---

## 📊 Proposed Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Implement `BioPipelines` facade (#3)
- [ ] Add Protocol-based interfaces (#6)
- [ ] Consolidate configuration (#7)
- [ ] Add error hierarchy (#8)

### Phase 2: Core Architecture (Week 2-3)
- [ ] Implement dependency injection container (#1)
- [ ] Refactor to Domain-Driven structure (#2)
- [ ] Standardize async/sync patterns (#9)
- [ ] Add structured logging (#10)

### Phase 3: Advanced Patterns (Week 3-4)
- [ ] Implement CQRS for commands/queries (#4)
- [ ] Add event-driven architecture (#5)
- [ ] Increase test coverage to 70%

---

## 🎯 Success Metrics

| Metric | Before | After | How to Measure |
|--------|--------|-------|----------------|
| Test coverage | 22% | 70% | pytest --cov |
| Cyclomatic complexity | Unknown | <10 per function | radon |
| Import time | ~3s | <0.5s | time python -c "import workflow_composer" |
| Lines per file | Max 1088 | Max 400 | wc -l |
| New developer onboarding | Days | Hours | Time to first PR |

---

## 💡 Alternative: Incremental Improvement

If major refactoring is too risky, here's an incremental path:

1. **Keep current architecture** but add facades
2. **Add Protocols** without removing existing ABCs
3. **Add structured logging** alongside existing logging
4. **Add test coverage** without restructuring

This is safer but slower to reach professional standards.

---

## Decision Needed

1. **Full modernization** (2-4 weeks, significant change)
2. **Incremental improvement** (ongoing, lower risk)
3. **Hybrid** (Quick wins now, major refactoring later)

The recommendation is **Option 3 (Hybrid)**: Implement Quick Wins (7-10) immediately, then Phase 1 foundation, and evaluate further phases based on results.

---

*This proposal was created after comprehensive codebase analysis on November 29, 2025.*
