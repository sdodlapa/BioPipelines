# BioPipelines v2.0 Architecture

**Version**: 2.0.0  
**Date**: November 29, 2025  
**Status**: Production

---

## Overview

BioPipelines is an AI-powered bioinformatics workflow automation platform that enables researchers to compose, execute, and monitor genomics analysis pipelines through natural language interaction.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BioPipelines v2.0                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Web UI    │  │   CLI       │  │  API        │  │  BioPipelines       │ │
│  │  (Gradio)   │  │             │  │             │  │  Facade             │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                │                     │           │
│         └────────────────┼────────────────┼─────────────────────┘           │
│                          │                │                                 │
│  ┌───────────────────────▼────────────────▼─────────────────────────────┐   │
│  │                      Unified Agent Layer                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐    │   │
│  │  │ Query Router │  │ Tool Selector│  │ Autonomy Controller      │    │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘    │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐   │
│  │                         Tool Categories                                │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │   │
│  │  │  Data   │ │Workflow │ │Execution│ │Diagnosis│ │    Education    │  │   │
│  │  │Discovery│ │Generator│ │ (SLURM) │ │  Agent  │ │                 │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐   │
│  │                    Infrastructure Layer                                │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │   │
│  │  │Container│ │Protocols│ │ Logging │ │Settings │ │   Exceptions    │  │   │
│  │  │  (DI)   │ │         │ │         │ │         │ │                 │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐   │
│  │                    External Services                                   │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │   │
│  │  │  vLLM   │ │ Ollama  │ │ OpenAI  │ │Anthropic│ │    Lightning    │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Principles

### 1. **Facade Pattern** - Single Entry Point
All external interactions go through the `BioPipelines` facade class, providing a clean, versioned API.

```python
from workflow_composer import BioPipelines

pipeline = BioPipelines()
result = pipeline.chat("Analyze RNA-seq data in /data/samples")
```

### 2. **Protocol-Based Interfaces** - Duck Typing with Type Safety
Components communicate through Python Protocols, enabling loose coupling and easy testing.

```python
from workflow_composer.infrastructure import LLMProtocol

class CustomLLM(LLMProtocol):
    def complete(self, prompt: str) -> str:
        return "response"
```

### 3. **Dependency Injection** - Testable Components
The DI container manages component lifecycles and dependencies.

```python
from workflow_composer.infrastructure import Container, Scope

container = Container()
container.register("llm", lambda: VLLMAdapter(), scope=Scope.SINGLETON)
llm = container.resolve("llm")
```

### 4. **Unified Error Hierarchy** - Consistent Error Handling
All errors inherit from `BioPipelinesError` with error codes and context.

```python
from workflow_composer.infrastructure import LLMError, ErrorCode

try:
    result = llm.complete(prompt)
except LLMError as e:
    print(f"[{e.error_code}] {e.message}")
    # Auto-recovery may trigger
```

---

## Directory Structure

```
src/workflow_composer/
├── __init__.py              # Package exports
├── facade.py                # BioPipelines entry point
│
├── infrastructure/          # Cross-cutting concerns
│   ├── __init__.py
│   ├── container.py         # Dependency injection
│   ├── protocols.py         # Interface definitions
│   ├── exceptions.py        # Error hierarchy
│   ├── logging.py           # Structured logging
│   └── settings.py          # Configuration management
│
├── agents/                  # AI agent system
│   ├── unified_agent.py     # Main agent orchestrator
│   ├── self_healing.py      # Auto-recovery agent
│   └── tools/               # Agent tool implementations
│       ├── base.py          # ToolResult, ToolName
│       ├── registry.py      # Tool registration
│       ├── data_discovery/  # Data scanning tools
│       ├── workflow/        # Workflow generation
│       ├── execution/       # SLURM job management
│       ├── diagnostics/     # Error diagnosis
│       └── education/       # Concept explanation
│
├── composer/                # Workflow composition engine
│   ├── composer.py          # Main composition logic
│   ├── module_mapper.py     # Tool → module mapping
│   ├── template_engine.py   # Nextflow templates
│   └── tool_selector.py     # LLM-based tool selection
│
├── llm/                     # LLM adapters
│   ├── base.py              # BaseLLMAdapter
│   ├── factory.py           # Adapter factory
│   ├── vllm_adapter.py      # Local vLLM
│   ├── ollama_adapter.py    # Local Ollama
│   ├── openai_adapter.py    # OpenAI API
│   └── anthropic_adapter.py # Claude API
│
├── data/                    # Data management
│   ├── discovery/           # Multi-source search
│   ├── scanner.py           # Local file scanning
│   ├── downloader.py        # Dataset download
│   └── reference_manager.py # Reference genomes
│
├── diagnosis/               # Error diagnosis system
│   ├── agent.py             # Diagnosis agent
│   ├── patterns.py          # 50+ error patterns
│   ├── auto_fix.py          # Automated fixes
│   └── history.py           # Learning from past
│
├── results/                 # Results management
│   ├── collector.py         # Result collection
│   ├── viewer.py            # Result visualization
│   └── archiver.py          # Result archiving
│
└── web/                     # Web interface
    ├── app.py               # Gradio application
    └── chat_handler.py      # Chat processing
```

---

## Key Components

### 1. BioPipelines Facade (`facade.py`)

The unified entry point for all BioPipelines functionality.

| Method | Description |
|--------|-------------|
| `chat(query)` | Process natural language queries |
| `generate_workflow(type, input_dir)` | Create analysis pipelines |
| `submit(workflow_dir)` | Submit jobs to SLURM |
| `status(job_id)` | Check job status |
| `diagnose(job_id)` | Analyze job failures |
| `scan_data(path)` | Discover data files |
| `search_databases(query)` | Search GEO, ENCODE, SRA |

### 2. Unified Agent (`agents/unified_agent.py`)

Orchestrates all AI-powered operations with:
- **Query Classification**: Determines query intent and required tools
- **Tool Selection**: Chooses appropriate tools for the task
- **Autonomy Levels**: SUPERVISED, GUIDED, AUTONOMOUS, FULL_AUTO
- **Permission System**: read, write, execute permissions per tool

### 3. Infrastructure Layer (`infrastructure/`)

| Module | Purpose |
|--------|---------|
| `container.py` | Thread-safe DI container with SINGLETON/TRANSIENT/SCOPED lifecycles |
| `protocols.py` | LLMProtocol, ToolProtocol, EventPublisherProtocol |
| `exceptions.py` | BioPipelinesError hierarchy with error codes |
| `logging.py` | Structured logging with correlation IDs |
| `settings.py` | Pydantic-settings configuration with validation |

### 4. Tool System (`agents/tools/`)

30+ tools organized by category:

| Category | Tools |
|----------|-------|
| **Data Discovery** | scan_data, search_databases, describe_files, validate_dataset |
| **Data Management** | download_dataset, download_reference, build_index |
| **Workflow** | generate_workflow, list_workflows, check_references |
| **Execution** | submit_job, get_job_status, cancel_job, resubmit_job |
| **Diagnostics** | diagnose_error, recover_error, analyze_results |
| **Education** | explain_concept, compare_samples, get_help |

### 5. LLM Adapters (`llm/`)

Unified interface for multiple LLM providers:

| Adapter | Provider | Use Case |
|---------|----------|----------|
| `VLLMAdapter` | Local vLLM | Primary (GPU cluster) |
| `OllamaAdapter` | Local Ollama | Fallback (lightweight) |
| `OpenAIAdapter` | OpenAI API | Cloud option |
| `AnthropicAdapter` | Claude API | Cloud option |
| `LightningAdapter` | Lightning AI | Managed cloud |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Web UI** | Gradio 4.x |
| **Workflow Engine** | Nextflow DSL2, Snakemake |
| **Job Scheduler** | SLURM |
| **Containers** | Singularity/Apptainer |
| **LLM Runtime** | vLLM, Ollama |
| **Configuration** | Pydantic-settings, YAML |
| **Testing** | pytest, pytest-asyncio |
| **Logging** | structlog-compatible |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | Nov 2025 | Architecture modernization - DI, Protocols, Facade |
| 1.0.0 | Oct 2025 | Initial release with unified agent |
