# BioPipelines Core Components

**Version**: 2.0.0  
**Date**: November 29, 2025

---

## Component Overview

This document details the key components of BioPipelines v2.0, their responsibilities, interfaces, and interactions.

---

## 1. BioPipelines Facade

**Location**: `src/workflow_composer/facade.py`

The unified entry point that hides internal complexity and provides a stable API.

### Interface

```python
class BioPipelines:
    """Single entry point for all BioPipelines functionality."""
    
    # Conversation
    def chat(self, query: str) -> AgentResponse:
        """Process natural language query."""
    
    # Workflow Operations
    def generate_workflow(self, workflow_type: str, input_dir: str, **kwargs) -> ToolResult:
        """Generate a workflow pipeline."""
    
    def list_workflows(self) -> ToolResult:
        """List available workflow types."""
    
    # Job Management
    def submit(self, workflow_dir: str, profile: str = "slurm") -> ToolResult:
        """Submit workflow to SLURM."""
    
    def status(self, job_id: str) -> ToolResult:
        """Get job status."""
    
    def cancel(self, job_id: str) -> ToolResult:
        """Cancel running job."""
    
    # Data Operations
    def scan_data(self, path: str = None) -> ToolResult:
        """Scan directory for data files."""
    
    def search_databases(self, query: str) -> ToolResult:
        """Search GEO, ENCODE, SRA, TCGA."""
    
    # Diagnostics
    def diagnose(self, job_id: str = None, log_content: str = None) -> ToolResult:
        """Diagnose errors."""
```

### Usage Example

```python
from workflow_composer import BioPipelines

# Initialize
bp = BioPipelines()

# Natural language interaction
response = bp.chat("Find RNA-seq data for breast cancer")
print(response.message)

# Programmatic API
result = bp.generate_workflow("rnaseq", "/data/fastq")
if result.success:
    job = bp.submit(result.data["workflow_dir"])
    print(f"Job submitted: {job.data['job_id']}")
```

---

## 2. Unified Agent

**Location**: `src/workflow_composer/agents/unified_agent.py`

The AI orchestrator that understands user intent and coordinates tool execution.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UnifiedAgent                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Query Router   │  │  Tool Detector  │  │ Autonomy Ctrl   │  │
│  │                 │  │                 │  │                 │  │
│  │ Classify intent │  │ Pattern match   │  │ Permission chk  │  │
│  │ Route to tool   │  │ Select tool     │  │ Approval flow   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │                     AgentTools                             │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │  │
│  │  │  Data   │ │Workflow │ │Execution│ │Diagnosis│ ...      │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Autonomy Levels

| Level | Description | Example Actions |
|-------|-------------|-----------------|
| `SUPERVISED` | All actions require approval | Any destructive operation |
| `GUIDED` | Suggestions with explanations | Workflow modifications |
| `AUTONOMOUS` | Execute with notification | Data scanning, status checks |
| `FULL_AUTO` | Silent execution | Read-only operations |

### Tool Classification

```python
# Permission mappings
TOOL_PERMISSIONS = {
    ToolName.SCAN_DATA: "read",
    ToolName.SEARCH_DATABASES: "read",
    ToolName.GENERATE_WORKFLOW: "write",
    ToolName.SUBMIT_JOB: "execute",
    ToolName.CANCEL_JOB: "execute",
    ToolName.DIAGNOSE_ERROR: "read",
}
```

---

## 3. Dependency Injection Container

**Location**: `src/workflow_composer/infrastructure/container.py`

Thread-safe dependency injection container supporting multiple lifecycles.

### Scopes

| Scope | Behavior |
|-------|----------|
| `SINGLETON` | One instance per container (default) |
| `TRANSIENT` | New instance on each resolve |
| `SCOPED` | One instance per scope context |

### Interface

```python
class Container:
    def register(
        self,
        name: str,
        factory: Callable,
        scope: Scope = Scope.SINGLETON
    ) -> None:
        """Register a service factory."""
    
    def resolve(self, name: str) -> Any:
        """Resolve a service by name."""
    
    def create_scope(self) -> "Container":
        """Create a child scope."""
    
    @staticmethod
    def inject(*dependencies: str):
        """Decorator for automatic dependency injection."""
```

### Usage

```python
from workflow_composer.infrastructure import Container, Scope

# Create container
container = Container()

# Register services
container.register("config", lambda: load_config())
container.register("llm", lambda: VLLMAdapter(), scope=Scope.SINGLETON)
container.register("logger", lambda: get_logger(), scope=Scope.TRANSIENT)

# Resolve dependencies
llm = container.resolve("llm")

# Use decorator
@Container.inject("llm", "config")
def generate_workflow(llm, config, query: str):
    return llm.complete(query)
```

---

## 4. Protocol Definitions

**Location**: `src/workflow_composer/infrastructure/protocols.py`

Python Protocols for structural typing and loose coupling.

### Core Protocols

```python
@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers."""
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion."""
        ...
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion."""
        ...
    
    @property
    def model_name(self) -> str:
        """Model identifier."""
        ...

@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol for agent tools."""
    
    @property
    def name(self) -> str: ...
    
    @property
    def description(self) -> str: ...
    
    def execute(self, **kwargs) -> "ToolResult": ...

@runtime_checkable
class EventPublisherProtocol(Protocol):
    """Protocol for event-driven architecture (future use)."""
    
    def publish(self, event: Event) -> None: ...
    def subscribe(self, event_type: str, handler: Callable) -> None: ...
```

### Type Checking

```python
def process_with_llm(llm: LLMProtocol, prompt: str) -> str:
    # Type checker ensures llm has complete() method
    return llm.complete(prompt)

# Works with any implementation
process_with_llm(VLLMAdapter())
process_with_llm(OllamaAdapter())
process_with_llm(OpenAIAdapter())
```

---

## 5. Exception Hierarchy

**Location**: `src/workflow_composer/infrastructure/exceptions.py`

Unified error handling with error codes and context.

### Hierarchy

```
BioPipelinesError
├── ConfigurationError
│   ├── MissingConfigError
│   └── InvalidConfigError
├── ToolNotFoundError
├── LLMError
│   ├── LLMConnectionError
│   ├── LLMRateLimitError
│   └── LLMModelError
├── SLURMError
│   ├── JobSubmissionError
│   └── JobNotFoundError
├── WorkflowError
│   ├── WorkflowGenerationError
│   └── WorkflowValidationError
├── DataError
│   ├── DataNotFoundError
│   └── DataValidationError
└── DiagnosisError
```

### Error Codes

| Code Range | Category |
|------------|----------|
| 1000-1999 | Configuration |
| 2000-2999 | Tool/Agent |
| 3000-3999 | LLM |
| 4000-4999 | SLURM/Execution |
| 5000-5999 | Workflow |
| 6000-6999 | Data |
| 7000-7999 | Diagnosis |

### Usage

```python
from workflow_composer.infrastructure import (
    LLMError, 
    ErrorCode,
    handle_error
)

try:
    result = llm.complete(prompt)
except LLMError as e:
    print(f"Error [{e.error_code}]: {e.message}")
    if e.recoverable:
        # Attempt recovery
        result = fallback_llm.complete(prompt)
```

---

## 6. Settings Management

**Location**: `src/workflow_composer/infrastructure/settings.py`

Pydantic-settings based configuration with validation.

### Configuration Hierarchy

```python
class Settings(BaseSettings):
    """Root configuration."""
    
    # Application
    app_name: str = "BioPipelines"
    version: str = "2.0.0"
    debug: bool = False
    
    # Nested configurations
    llm: LLMSettings
    slurm: SLURMSettings
    paths: PathSettings
    
    class Config:
        env_prefix = "BIOPIPELINES_"
        env_nested_delimiter = "__"
```

### Environment Variables

```bash
# Example .env configuration
BIOPIPELINES_DEBUG=true
BIOPIPELINES_LLM__PROVIDER=vllm
BIOPIPELINES_LLM__VLLM_URL=http://localhost:8000
BIOPIPELINES_SLURM__PARTITION=gpu
BIOPIPELINES_SLURM__DEFAULT_TIME=4:00:00
```

### Usage

```python
from workflow_composer.infrastructure import Settings, get_settings

# Get singleton settings
settings = get_settings()

# Access configuration
print(settings.llm.provider)
print(settings.slurm.partition)
print(settings.paths.data_dir)
```

---

## 7. Structured Logging

**Location**: `src/workflow_composer/infrastructure/logging.py`

Structured logging with correlation IDs for request tracing.

### Features

- **Correlation IDs**: Track requests across components
- **Operation Context**: Automatic timing and status
- **Structured Output**: JSON-compatible log format
- **Log Levels**: DEBUG, INFO, WARNING, ERROR

### Usage

```python
from workflow_composer.infrastructure import (
    get_logger, 
    operation_context,
    set_correlation_id
)

logger = get_logger(__name__)

# Set correlation ID for request tracing
set_correlation_id("req-12345")

# Log with context
logger.info("Processing query", query=query, user="researcher")

# Operation context for timing
with operation_context("generate_workflow", query=query) as ctx:
    workflow = composer.generate(query)
    ctx.set_result(workflow_id=workflow.id)
# Automatically logs: operation.completed, duration_ms=1234
```

---

## 8. Tool System

**Location**: `src/workflow_composer/agents/tools/`

Modular tool system for agent capabilities.

### Tool Result

```python
@dataclass
class ToolResult:
    """Standard result from tool execution."""
    success: bool
    tool_name: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None
```

### Tool Categories

| Category | Module | Tools |
|----------|--------|-------|
| Data Discovery | `data_discovery/` | scan_data, search_databases, describe_files |
| Data Management | `data_management/` | download_dataset, download_reference |
| Workflow | `workflow/` | generate_workflow, check_references |
| Execution | `execution/` | submit_job, get_status, cancel_job |
| Diagnostics | `diagnostics/` | diagnose_error, recover_error |
| Education | `education/` | explain_concept, compare_samples |

### Adding New Tools

```python
# 1. Define patterns
MY_TOOL_PATTERNS = [
    r"my tool (\w+)",
    r"run my operation on (.+)",
]

# 2. Implement function
def my_tool_impl(arg: str) -> ToolResult:
    return ToolResult(
        success=True,
        tool_name="my_tool",
        message=f"Processed: {arg}",
        data={"result": arg}
    )

# 3. Register in __init__.py
```

---

## Component Interactions

```
User Query
    │
    ▼
┌─────────────────┐
│  BioPipelines   │◄──────────────────────────────────────┐
│    Facade       │                                        │
└────────┬────────┘                                        │
         │                                                 │
         ▼                                                 │
┌─────────────────┐     ┌─────────────────┐               │
│  UnifiedAgent   │────►│    AgentTools   │               │
│                 │     │                 │               │
│  - Classify     │     │  - Execute      │               │
│  - Route        │     │  - Validate     │               │
│  - Control      │     │  - Format       │               │
└────────┬────────┘     └────────┬────────┘               │
         │                       │                         │
         ▼                       ▼                         │
┌─────────────────┐     ┌─────────────────┐               │
│  LLM Adapters   │     │   Tool Impls    │               │
│                 │     │                 │               │
│  - vLLM         │     │  - Workflow     │───────────────┘
│  - Ollama       │     │  - SLURM        │
│  - OpenAI       │     │  - Diagnosis    │
└─────────────────┘     └─────────────────┘
```

---

## Testing

Each component has corresponding tests:

| Component | Test Location |
|-----------|---------------|
| Infrastructure | `tests/unit/test_infrastructure.py` |
| Unified Agent | `tests/test_unified_agent.py` |
| LLM Adapters | `tests/unit/test_llm_adapters.py` |
| Data Discovery | `tests/unit/test_data_discovery.py` |
| Diagnosis | `tests/unit/test_diagnosis.py` |
| Results | `tests/unit/test_results.py` |

Run tests:
```bash
pytest tests/unit/test_infrastructure.py -v
pytest tests/ -v --tb=short
```
