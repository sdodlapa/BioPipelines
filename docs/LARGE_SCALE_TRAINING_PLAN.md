# Large-Scale Workflow Training & Model Fine-Tuning Plan

**Date:** December 2, 2025  
**Status:** Planning  
**Goal:** Run 1000+ bioinformatics workflows, collect structured training data, fine-tune agents/models

---

## Executive Summary

This document outlines a systematic approach to:
1. **Generate** 1000+ diverse bioinformatics workflows via the agentic system
2. **Execute** workflows on SLURM cluster with comprehensive logging
3. **Collect** structured data (logs, errors, outcomes, trajectories)
4. **Curate** high-quality training datasets
5. **Fine-tune** relevant models (Orchestrator-8B, CodeGen, Validator)

### Key Outcomes

| Outcome | Target | Metric |
|---------|--------|--------|
| Workflow coverage | 15+ analysis types | Coverage map |
| Training examples | 10,000+ trajectories | Dataset size |
| Model improvement | 20%+ accuracy gain | Benchmark scores |
| Cost reduction | 50%+ with fine-tuned routing | Cost tracking |

---

## Architecture: Training Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA COLLECTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────┐     ┌──────────────────┐     ┌────────────────────────────┐ │
│  │ Query Generator│     │  Orchestrated    │     │    Workflow Executor       │ │
│  │                │────►│  Supervisor      │────►│    (Nextflow + SLURM)      │ │
│  │ • Templates    │     │  + Logging       │     │    • Container runtime     │ │
│  │ • Variations   │     │                  │     │    • Resource tracking     │ │
│  │ • Edge cases   │     │                  │     │                            │ │
│  └────────────────┘     └──────────────────┘     └─────────────┬──────────────┘ │
│                                │                               │                 │
│                                │ Logs                          │ Results         │
│                                ▼                               ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        TRAINING DATA STORE                                   ││
│  │                                                                              ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  ││
│  │  │ Agent        │  │ Workflow     │  │ Execution    │  │ Error          │  ││
│  │  │ Trajectories │  │ Artifacts    │  │ Logs         │  │ Patterns       │  ││
│  │  │              │  │              │  │              │  │                │  ││
│  │  │ • Query      │  │ • main.nf    │  │ • .command   │  │ • Type         │  ││
│  │  │ • Plan       │  │ • config     │  │ • .exitcode  │  │ • Message      │  ││
│  │  │ • Code       │  │ • README     │  │ • slurm logs │  │ • Resolution   │  ││
│  │  │ • Validation │  │ • containers │  │ • metrics    │  │ • Outcome      │  ││
│  │  │ • Outcome    │  │              │  │              │  │                │  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘  ││
│  │                                                                              ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                │                                                 │
│                                ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         FINE-TUNING PIPELINE                                 ││
│  │                                                                              ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  ││
│  │  │ Orchestrator │  │ CodeGen      │  │ Validator    │  │ Error Recovery │  ││
│  │  │ Fine-tuning  │  │ Fine-tuning  │  │ Fine-tuning  │  │ Fine-tuning    │  ││
│  │  │              │  │              │  │              │  │                │  ││
│  │  │ Task: Which  │  │ Task: Query  │  │ Task: Code   │  │ Task: Error    │  ││
│  │  │ model tier?  │  │ → Nextflow   │  │ → Issues     │  │ → Fix          │  ││
│  │  │              │  │              │  │              │  │                │  ││
│  │  │ Data: Query+ │  │ Data: Plan+  │  │ Data: Code+  │  │ Data: Error+   │  ││
│  │  │ Optimal tier │  │ Working code │  │ Feedback     │  │ Resolution     │  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘  ││
│  │                                                                              ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Query Generation & Diversity (Week 1-2)

### 1.1 Query Template System

**File to Create:** `src/workflow_composer/training/query_generator.py`

```python
"""
Query Generator for Training Data Collection
=============================================

Generates diverse bioinformatics queries across:
- 15+ analysis types
- 10+ organisms
- Multiple complexity levels
- Various parameter combinations
- Edge cases and error scenarios
"""

ANALYSIS_TEMPLATES = {
    "rna_seq": [
        "RNA-seq differential expression analysis for {organism}",
        "{organism} RNA-seq workflow with {aligner} alignment",
        "Bulk RNA-seq comparing {condition1} vs {condition2} in {organism}",
        "RNA-seq with {quantifier} quantification for {organism} samples",
        "Stranded RNA-seq analysis using {library_type} library prep",
    ],
    "chip_seq": [
        "ChIP-seq peak calling for {target} in {organism}",
        "{organism} ChIP-seq with {peak_caller} and {organism} reference",
        "Histone modification ChIP-seq for {histone_mark} in {cell_type}",
        "Transcription factor ChIP-seq for {tf} binding sites",
    ],
    "atac_seq": [
        "ATAC-seq chromatin accessibility for {organism}",
        "Single-cell ATAC-seq analysis for {cell_type}",
        "ATAC-seq with {peak_caller} peak calling",
    ],
    "wgs": [
        "Whole genome sequencing variant calling for {organism}",
        "WGS germline variant detection with {caller}",
        "WGS somatic mutation calling for {tumor_type}",
        "Structural variant detection from WGS data",
    ],
    "wes": [
        "Whole exome sequencing for {organism}",
        "WES variant annotation with {annotator}",
        "Clinical exome analysis with {panel}",
    ],
    "methylation": [
        "Bisulfite sequencing methylation analysis",
        "RRBS methylation profiling for {organism}",
        "WGBS with {aligner} alignment",
        "Differential methylation analysis",
    ],
    "scrna_seq": [
        "Single-cell RNA-seq with {platform}",
        "scRNA-seq clustering and annotation for {cell_type}",
        "10x Genomics scRNA-seq processing",
        "scRNA-seq trajectory analysis",
    ],
    "spatial": [
        "Spatial transcriptomics analysis for {tissue}",
        "Visium spatial gene expression",
        "MERFISH/seqFISH analysis",
    ],
    "metagenomics": [
        "16S rRNA amplicon analysis",
        "Shotgun metagenomics with {assembler}",
        "Metagenomic taxonomic profiling",
        "Antimicrobial resistance detection",
    ],
    "hic": [
        "Hi-C chromatin interaction analysis",
        "Hi-C contact matrix generation",
        "TAD boundary detection from Hi-C",
    ],
    "long_read": [
        "Oxford Nanopore sequencing analysis",
        "PacBio long-read assembly",
        "Long-read RNA isoform detection",
    ],
    "multiomics": [
        "Integrated RNA-seq and ATAC-seq analysis",
        "Multi-modal single-cell analysis (RNA + ATAC)",
        "Proteogenomics integration",
    ],
    "crispr": [
        "CRISPR screen analysis with MAGeCK",
        "CRISPR guide RNA design and analysis",
    ],
    "phylogenetics": [
        "Phylogenetic tree construction from {marker}",
        "Comparative genomics analysis",
    ],
}

ORGANISMS = [
    "human", "mouse", "rat", "zebrafish", "drosophila",
    "c_elegans", "yeast", "arabidopsis", "e_coli", "custom"
]

COMPLEXITY_LEVELS = {
    "simple": {
        "samples": "2-4",
        "conditions": "2",
        "replicates": "2",
        "compute": "low",
    },
    "medium": {
        "samples": "12-24",
        "conditions": "3-4",
        "replicates": "3",
        "compute": "medium",
    },
    "complex": {
        "samples": "48-96",
        "conditions": "6+",
        "replicates": "4+",
        "compute": "high",
    },
}

PARAMETER_VARIATIONS = {
    "aligner": ["STAR", "HISAT2", "BWA-MEM2", "minimap2", "salmon"],
    "quantifier": ["salmon", "kallisto", "RSEM", "featureCounts"],
    "peak_caller": ["MACS2", "MACS3", "HOMER", "SICER2"],
    "variant_caller": ["GATK", "DeepVariant", "Strelka2", "FreeBayes"],
    "container_runtime": ["singularity", "docker", "apptainer"],
    "executor": ["slurm", "local", "awsbatch", "google-batch"],
}
```

### 1.2 Query Diversity Matrix

| Analysis Type | Organisms | Variations | Error Scenarios | Total Queries |
|---------------|-----------|------------|-----------------|---------------|
| RNA-seq | 10 | 25 | 10 | 350 |
| ChIP-seq | 8 | 15 | 8 | 184 |
| ATAC-seq | 6 | 10 | 6 | 96 |
| WGS | 5 | 20 | 10 | 150 |
| WES | 5 | 15 | 8 | 115 |
| Methylation | 4 | 12 | 6 | 72 |
| scRNA-seq | 6 | 15 | 8 | 138 |
| Metagenomics | 1 | 20 | 10 | 30 |
| Hi-C | 3 | 8 | 5 | 39 |
| Long-read | 4 | 10 | 6 | 64 |
| Multiomics | 3 | 8 | 4 | 36 |
| **Total** | - | - | - | **~1,300** |

---

## Phase 2: Enhanced Logging Infrastructure (Week 2-3)

### 2.1 Agent Trajectory Logging

**File to Create:** `src/workflow_composer/training/trajectory_logger.py`

```python
"""
Agent Trajectory Logger
=======================

Logs complete agent execution trajectories for training:
- Query → Plan → Code → Validation → Outcome
- Model routing decisions
- Token usage per step
- Latency measurements
- Error recovery attempts
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import sqlite3

logger = logging.getLogger(__name__)


class TrajectoryOutcome(Enum):
    """Final outcome of a workflow generation."""
    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    EXECUTION_FAILED = "execution_failed"
    TIMEOUT = "timeout"
    USER_CANCELLED = "user_cancelled"
    ERROR = "error"


@dataclass
class AgentStep:
    """Single step in agent trajectory."""
    step_id: int
    agent: str  # planner, codegen, validator, etc.
    action: str  # e.g., "generate_plan", "generate_code", "validate"
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_used: str
    model_tier: str  # local_small, local_large, cloud_small, cloud_large
    cost_usd: float
    success: bool
    input_summary: str  # Truncated input for training
    output_summary: str  # Truncated output for training
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrchestratorDecision:
    """Routing decision from Orchestrator-8B."""
    query_summary: str
    selected_tier: str
    selected_model: str
    reasoning: str
    alternatives_considered: List[str]
    cost_estimate: float
    actual_cost: float
    was_optimal: bool  # Determined post-hoc


@dataclass
class TrajectoryRecord:
    """Complete trajectory for one workflow generation."""
    trajectory_id: str
    query: str
    analysis_type: str
    organism: Optional[str]
    complexity: str
    
    # Agent steps
    steps: List[AgentStep] = field(default_factory=list)
    
    # Orchestrator decisions
    routing_decisions: List[OrchestratorDecision] = field(default_factory=list)
    
    # Artifacts
    plan_json: Optional[str] = None
    generated_code: Optional[str] = None
    config_code: Optional[str] = None
    readme: Optional[str] = None
    
    # Validation
    validation_iterations: int = 0
    validation_issues: List[str] = field(default_factory=list)
    final_validation_passed: bool = False
    
    # Execution (if run)
    execution_submitted: bool = False
    execution_completed: bool = False
    execution_exit_code: Optional[int] = None
    execution_logs_path: Optional[str] = None
    
    # Outcome
    outcome: TrajectoryOutcome = TrajectoryOutcome.ERROR
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    
    # Metadata
    session_id: Optional[str] = None
    user_feedback: Optional[str] = None  # positive, negative, neutral
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


class TrajectoryLogger:
    """
    Logs agent trajectories to SQLite for training data collection.
    
    Usage:
        logger = TrajectoryLogger()
        trajectory = logger.start_trajectory(query, analysis_type)
        
        # During execution...
        logger.log_step(trajectory.trajectory_id, step)
        logger.log_routing_decision(trajectory.trajectory_id, decision)
        
        # On completion
        logger.complete_trajectory(trajectory.trajectory_id, outcome)
    """
    
    def __init__(self, db_path: str = "~/.biopipelines/training/trajectories.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    organism TEXT,
                    complexity TEXT,
                    outcome TEXT,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost_usd REAL DEFAULT 0,
                    total_latency_ms REAL DEFAULT 0,
                    validation_iterations INTEGER DEFAULT 0,
                    final_validation_passed INTEGER DEFAULT 0,
                    execution_exit_code INTEGER,
                    user_feedback TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    data_json TEXT
                );
                
                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trajectory_id TEXT NOT NULL,
                    step_id INTEGER NOT NULL,
                    agent TEXT NOT NULL,
                    action TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    latency_ms REAL,
                    model_used TEXT,
                    model_tier TEXT,
                    cost_usd REAL,
                    success INTEGER,
                    error TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
                );
                
                CREATE TABLE IF NOT EXISTS routing_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trajectory_id TEXT NOT NULL,
                    selected_tier TEXT,
                    selected_model TEXT,
                    reasoning TEXT,
                    cost_estimate REAL,
                    actual_cost REAL,
                    was_optimal INTEGER,
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_traj_outcome ON trajectories(outcome);
                CREATE INDEX IF NOT EXISTS idx_traj_analysis ON trajectories(analysis_type);
                CREATE INDEX IF NOT EXISTS idx_steps_traj ON steps(trajectory_id);
            """)
```

### 2.2 Execution Log Collector

**File to Create:** `src/workflow_composer/training/execution_collector.py`

```python
"""
Execution Log Collector
=======================

Collects logs from Nextflow workflow executions:
- .command.log files
- .exitcode files
- trace.txt metrics
- SLURM logs
- MultiQC reports
"""

@dataclass
class ExecutionMetrics:
    """Metrics from a workflow execution."""
    workflow_id: str
    total_processes: int
    completed_processes: int
    failed_processes: int
    total_cpu_hours: float
    total_memory_gb_hours: float
    wall_time_minutes: float
    
    # Per-process breakdown
    process_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Errors
    error_processes: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    
    # Data
    input_files: int = 0
    output_files: int = 0
    total_input_size_gb: float = 0.0
    total_output_size_gb: float = 0.0


class ExecutionCollector:
    """Collect execution logs and metrics from Nextflow runs."""
    
    def collect_from_work_dir(
        self, 
        work_dir: Path,
        trace_file: Optional[Path] = None
    ) -> ExecutionMetrics:
        """
        Collect metrics from a completed Nextflow work directory.
        
        Args:
            work_dir: Nextflow work directory
            trace_file: Optional trace.txt file path
            
        Returns:
            ExecutionMetrics with collected data
        """
        pass
    
    def parse_trace_file(self, trace_path: Path) -> List[Dict[str, Any]]:
        """Parse Nextflow trace.txt file."""
        pass
    
    def collect_slurm_logs(
        self, 
        job_ids: List[str],
        log_dir: Path = Path("logs/slurm")
    ) -> Dict[str, str]:
        """Collect SLURM job logs by job ID."""
        pass
    
    def extract_error_patterns(
        self, 
        log_content: str
    ) -> List[Dict[str, str]]:
        """Extract error patterns from log content."""
        pass
```

### 2.3 Enhanced Observability Schema

Extend `src/workflow_composer/observability/metrics.py`:

```python
# Additional tables for training data

TRAINING_TABLES = """
CREATE TABLE IF NOT EXISTS agent_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT,
    agent TEXT NOT NULL,
    decision_type TEXT NOT NULL,  -- model_selection, retry, fallback
    input_context TEXT,
    decision TEXT,
    outcome TEXT,  -- success, failure, timeout
    latency_ms REAL,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS error_resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT,
    error_type TEXT NOT NULL,
    error_message TEXT,
    resolution_attempted TEXT,
    resolution_success INTEGER,
    iterations INTEGER,
    final_state TEXT
);

CREATE TABLE IF NOT EXISTS model_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL,
    model_a TEXT,
    model_b TEXT,
    output_a TEXT,
    output_b TEXT,
    quality_a REAL,  -- Human or automated rating
    quality_b REAL,
    cost_a REAL,
    cost_b REAL,
    winner TEXT,  -- a, b, tie
    timestamp TEXT
);
"""
```

---

## Phase 3: Training Data Curation (Week 4-5)

### 3.1 Dataset Formats

**Orchestrator Fine-tuning (Routing)**:
```json
{
  "query": "RNA-seq differential expression for human samples",
  "context": {
    "available_models": ["llama-7b", "llama-70b", "gpt-4o", "claude-3.5"],
    "user_preference": "balanced",
    "budget_remaining": 5.0
  },
  "optimal_tier": "LOCAL_LARGE",
  "optimal_model": "llama-70b",
  "reasoning": "Code generation task benefits from larger local model. No need for cloud for standard RNA-seq workflow."
}
```

**CodeGen Fine-tuning**:
```json
{
  "plan": {
    "analysis_type": "rna_seq",
    "steps": ["fastqc", "trimgalore", "star", "featurecounts", "deseq2"],
    "organism": "human",
    "reference": "GRCh38"
  },
  "generated_code": "// ... Nextflow DSL2 code ...",
  "validation_passed": true,
  "execution_success": true
}
```

**Validator Fine-tuning**:
```json
{
  "code": "// ... Nextflow code with issues ...",
  "issues": [
    {"line": 42, "type": "missing_container", "message": "Process STAR_ALIGN missing container directive"},
    {"line": 67, "type": "hardcoded_path", "message": "Avoid hardcoded paths, use params"}
  ],
  "severity": "error",
  "auto_fixable": true
}
```

**Error Recovery Fine-tuning**:
```json
{
  "error_type": "MEMORY",
  "error_message": "java.lang.OutOfMemoryError: GC overhead limit exceeded",
  "context": {
    "process": "STAR_ALIGN",
    "current_memory": "16 GB",
    "input_size_gb": 50
  },
  "resolution": {
    "action": "increase_memory",
    "new_value": "48 GB",
    "config_change": "process { withName: 'STAR_ALIGN' { memory = '48 GB' } }"
  },
  "outcome": "success"
}
```

### 3.2 Data Quality Filters

```python
class DataQualityFilter:
    """Filter training data for quality."""
    
    def filter_trajectories(
        self, 
        trajectories: List[TrajectoryRecord]
    ) -> List[TrajectoryRecord]:
        """
        Filter trajectories for training quality.
        
        Filters:
        1. Remove incomplete trajectories
        2. Remove very short/trivial queries
        3. Remove duplicate queries (fuzzy)
        4. Balance success/failure ratio
        5. Ensure diversity across analysis types
        """
        pass
    
    def score_trajectory(self, trajectory: TrajectoryRecord) -> float:
        """
        Score trajectory quality for training value.
        
        High value:
        - Complex queries with successful completion
        - Error recovery that succeeded
        - Multi-iteration validation fixes
        - Diverse organisms/tools
        
        Low value:
        - Trivial queries
        - Immediate failures without recovery
        - Duplicate/near-duplicate queries
        """
        pass
```

### 3.3 Dataset Statistics Dashboard

```python
class DatasetStats:
    """Generate statistics for training dataset."""
    
    def generate_report(self) -> Dict[str, Any]:
        return {
            "total_trajectories": 0,
            "by_outcome": {
                "success": 0,
                "validation_failed": 0,
                "execution_failed": 0,
            },
            "by_analysis_type": {},
            "by_organism": {},
            "by_complexity": {},
            "token_distribution": {
                "mean": 0,
                "median": 0,
                "p95": 0,
            },
            "cost_distribution": {},
            "error_patterns": {},
            "model_usage": {},
        }
```

---

## Phase 4: Fine-Tuning Pipeline (Week 6-8)

### 4.1 Fine-Tuning Targets

| Model | Task | Dataset Size | Method | Hardware |
|-------|------|--------------|--------|----------|
| Orchestrator-8B | Routing | 5,000+ examples | LoRA | 1x A100 |
| CodeLlama-7B | CodeGen | 2,000+ examples | LoRA | 1x A100 |
| Qwen2.5-7B | Validation | 3,000+ examples | LoRA | 1x A100 |
| Mistral-7B | Error Recovery | 1,000+ examples | LoRA | 1x A100 |

### 4.2 Orchestrator-8B Fine-Tuning

**Purpose:** Improve routing decisions for BioPipelines-specific queries.

```python
"""
Orchestrator-8B Fine-Tuning Script
==================================

Fine-tune on BioPipelines routing decisions.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch

def prepare_routing_dataset(trajectories: List[TrajectoryRecord]) -> Dataset:
    """Prepare routing examples from trajectories."""
    examples = []
    
    for traj in trajectories:
        for decision in traj.routing_decisions:
            if decision.was_optimal:  # Only train on good decisions
                examples.append({
                    "query": traj.query,
                    "context": {
                        "analysis_type": traj.analysis_type,
                        "complexity": traj.complexity,
                    },
                    "target_tier": decision.selected_tier,
                    "target_model": decision.selected_model,
                    "reasoning": decision.reasoning,
                })
    
    return Dataset.from_list(examples)


def fine_tune_orchestrator(
    base_model: str = "nvidia/Orchestrator-8B",
    dataset: Dataset = None,
    output_dir: str = "models/orchestrator-biopipelines",
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
):
    """Fine-tune Orchestrator-8B with LoRA."""
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Training loop...
    # (use standard HuggingFace Trainer)
```

### 4.3 CodeGen Fine-Tuning

```python
def prepare_codegen_dataset(trajectories: List[TrajectoryRecord]) -> Dataset:
    """Prepare code generation examples."""
    examples = []
    
    for traj in trajectories:
        if traj.outcome == TrajectoryOutcome.SUCCESS and traj.generated_code:
            examples.append({
                "plan": traj.plan_json,
                "code": traj.generated_code,
                "config": traj.config_code,
            })
    
    return Dataset.from_list(examples)
```

### 4.4 Validation Reward Model

For RLHF/DPO on validation quality:

```python
def prepare_validation_preferences(trajectories: List[TrajectoryRecord]) -> Dataset:
    """
    Create preference dataset for validation.
    
    Format: (query, code, chosen_feedback, rejected_feedback)
    
    Chosen: Validation that caught real issues
    Rejected: False positives or missed issues
    """
    pass
```

---

## Phase 5: Workflow Execution at Scale (Parallel)

### 5.1 Batch Execution Manager

**File to Create:** `src/workflow_composer/training/batch_executor.py`

```python
"""
Batch Workflow Executor
=======================

Execute generated workflows at scale on SLURM cluster.

Features:
- Parallel submission with throttling
- Resource quota management
- Automatic retry on transient failures
- Comprehensive logging
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class BatchConfig:
    """Configuration for batch execution."""
    max_concurrent_jobs: int = 50
    max_jobs_per_hour: int = 100
    partition: str = "main"
    account: str = "biopipelines"
    default_time: str = "4:00:00"
    default_memory: str = "32G"
    work_base_dir: Path = Path("/scratch/biopipelines/training")
    log_dir: Path = Path("logs/training_runs")


class BatchExecutor:
    """Execute workflows in batches on SLURM."""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.active_jobs: Dict[str, str] = {}  # trajectory_id -> job_id
        self.completed: List[str] = []
        self.failed: List[str] = []
    
    async def submit_batch(
        self,
        workflows: List[Tuple[str, Path]],  # (trajectory_id, workflow_dir)
        dry_run: bool = False,
    ) -> Dict[str, str]:
        """
        Submit a batch of workflows for execution.
        
        Returns:
            Mapping of trajectory_id -> SLURM job_id
        """
        pass
    
    async def monitor_jobs(self) -> Dict[str, str]:
        """Monitor active jobs and collect completions."""
        pass
    
    async def collect_results(
        self,
        trajectory_id: str,
        job_id: str,
    ) -> ExecutionMetrics:
        """Collect execution results for a completed job."""
        pass
```

### 5.2 Test Data Management

```python
"""
Test Data Manager
=================

Manage reference datasets for workflow testing.
"""

@dataclass
class TestDataset:
    """Reference dataset for testing."""
    name: str
    analysis_type: str
    organism: str
    size: str  # small, medium, large
    samples: int
    fastq_paths: List[Path]
    reference_path: Path
    expected_outputs: Dict[str, Any]


# Pre-configured test datasets
TEST_DATASETS = {
    "rnaseq_human_small": TestDataset(
        name="E-MTAB-2836 subset",
        analysis_type="rna_seq",
        organism="human",
        size="small",
        samples=4,
        fastq_paths=[...],
        reference_path=Path("/references/GRCh38"),
        expected_outputs={"genes_detected": ">15000", "mapping_rate": ">80%"},
    ),
    "chipseq_mouse_small": TestDataset(...),
    # ... more datasets
}
```

---

## Phase 6: Cyclic Refinement (Puppeteer Integration)

Based on the Puppeteer paper analysis, implement cyclic agent structures:

### 6.1 Cyclic Validation Loop

**File to Modify:** `src/workflow_composer/agents/specialists/supervisor.py`

```python
class CyclicSupervisor(SupervisorAgent):
    """
    Supervisor with cyclic refinement capability.
    
    Implements Puppeteer-style cyclic structures:
    - Validator can send back to Planner (not just CodeGen)
    - Failed execution triggers diagnostic agent
    - Successful patterns reinforce agent paths
    """
    
    async def execute_with_cycles(
        self,
        query: str,
        max_cycles: int = 3,
        output_dir: Optional[Path] = None,
    ) -> WorkflowResult:
        """
        Execute with cyclic refinement.
        
        Cycles:
        1. Planner → CodeGen → Validator
           If validation fails: → CodeGen (fix) or → Planner (redesign)
        2. Executor → DiagnosticAgent (on failure)
           → CodeGen (fix) or → Planner (alternative approach)
        """
        cycle = 0
        context = GenerationContext(query=query)
        
        while cycle < max_cycles:
            # Forward pass
            plan_result = await self._plan(context)
            if not plan_result.success:
                cycle += 1
                continue
            
            code_result = await self._generate(context)
            validation_result = await self._validate(context)
            
            if validation_result.passed:
                # Success path
                await self._document(context)
                await self._qc(context)
                return self._build_result(context)
            
            # Cyclic decision: Can CodeGen fix, or need Planner redesign?
            if self._can_codegen_fix(validation_result):
                context.feedback = validation_result.issues
                cycle += 1
            else:
                # Escalate to Planner
                context.planner_feedback = self._summarize_for_planner(
                    validation_result
                )
                context.plan = None  # Reset plan
                cycle += 1
        
        return self._build_failed_result(context, "max_cycles_exceeded")
```

### 6.2 Reflect Agent

New agent for analyzing failed trajectories:

```python
"""
Reflect Agent
=============

Analyzes failed workflows to improve future generations.
Inspired by Puppeteer's "reflect" agent type.
"""

class ReflectAgent:
    """Analyze failed trajectories and propose improvements."""
    
    async def analyze_failure(
        self,
        trajectory: TrajectoryRecord,
        execution_logs: Optional[str] = None,
    ) -> ReflectionResult:
        """
        Analyze why a trajectory failed.
        
        Returns:
        - Root cause analysis
        - What could have been done differently
        - Recommendations for similar queries
        """
        prompt = f"""
        Analyze this failed bioinformatics workflow generation:
        
        Query: {trajectory.query}
        Analysis Type: {trajectory.analysis_type}
        Outcome: {trajectory.outcome}
        
        Validation Issues:
        {trajectory.validation_issues}
        
        Execution Logs (if available):
        {execution_logs[:2000] if execution_logs else "N/A"}
        
        Provide:
        1. ROOT CAUSE: What fundamentally went wrong?
        2. PREVENTION: How could this have been avoided?
        3. ALTERNATIVE: What approach would work better?
        4. LEARNING: What should future generations learn from this?
        """
        
        # ... LLM call and parse
```

---

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Query Generation | QueryGenerator, 1300+ query templates |
| 2-3 | Logging Infrastructure | TrajectoryLogger, ExecutionCollector |
| 3-4 | Integration | Enhanced SupervisorAgent with logging |
| 4-5 | Data Curation | Quality filters, dataset formats |
| 5-6 | Batch Execution | BatchExecutor, 500+ workflow runs |
| 6-7 | Fine-Tuning | LoRA scripts, initial fine-tuning |
| 7-8 | Evaluation | Benchmarks, A/B testing |
| 8+ | Cyclic Refinement | CyclicSupervisor, ReflectAgent |

---

## Resource Requirements

### Compute

| Task | Resources | Duration | Cost Estimate |
|------|-----------|----------|---------------|
| Workflow Generation | 1 GPU node | Continuous | Included in cluster |
| Workflow Execution | 50 nodes (peak) | 2 weeks | Included in cluster |
| Fine-Tuning | 1x A100 (80GB) | 1 week | ~$500-1000 |
| Evaluation | 1 GPU node | 1 week | Included |

### Storage

| Data Type | Estimated Size |
|-----------|----------------|
| Trajectories (SQLite) | ~5 GB |
| Generated workflows | ~50 GB |
| Execution logs | ~100 GB |
| Training datasets | ~10 GB |
| Fine-tuned models | ~50 GB (LoRA adapters) |

### Personnel

- 1 ML Engineer (fine-tuning pipeline)
- 1 DevOps (execution infrastructure)
- 1 Domain Expert (quality review)

---

## Success Metrics

### Phase 1-3 (Data Collection)

| Metric | Target |
|--------|--------|
| Unique queries generated | 1,000+ |
| Successful workflow generations | 80%+ |
| Trajectories with full logging | 95%+ |
| Coverage of analysis types | 15/15 |

### Phase 4 (Fine-Tuning)

| Metric | Target |
|--------|--------|
| Orchestrator routing accuracy | +20% over baseline |
| CodeGen validation pass rate | +15% over baseline |
| Validator precision | +25% over baseline |
| Error recovery success rate | +30% over baseline |

### Phase 5-6 (Execution & Refinement)

| Metric | Target |
|--------|--------|
| Workflows executed successfully | 70%+ |
| Execution logs collected | 95%+ |
| Cyclic refinement success rate | 85%+ |
| Cost reduction from routing | 50%+ |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Low query diversity | Biased models | Use systematic template expansion |
| Execution failures dominate | Poor training signal | Balance with synthetic successes |
| Fine-tuning overfits | Poor generalization | Hold-out test set, early stopping |
| Compute constraints | Delayed timeline | Prioritize high-value fine-tuning |
| Data quality issues | Noisy training | Aggressive filtering, human review sample |

---

## Next Steps (Immediate)

1. **Create `src/workflow_composer/training/` module structure**
2. **Implement QueryGenerator with 100 initial templates**
3. **Add trajectory logging to SupervisorAgent**
4. **Set up test dataset (small RNA-seq)**
5. **Run pilot batch of 50 workflows**

---

## References

- ToolOrchestra Paper: https://arxiv.org/abs/2511.21689
- Puppeteer Paper: https://arxiv.org/abs/2505.19591
- DeepCode Repository: https://github.com/HKUDS/DeepCode
- Nextflow Documentation: https://nextflow.io/docs/latest/
- LoRA Fine-tuning: https://arxiv.org/abs/2106.09685
