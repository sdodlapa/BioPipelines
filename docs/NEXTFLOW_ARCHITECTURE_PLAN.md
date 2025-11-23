# Nextflow Pipeline Architecture Plan

**Date**: November 23, 2025  
**Purpose**: Design a new AI-driven, container-based bioinformatics platform using Nextflow  
**Status**: Planning Phase  
**Environment**: Google Cloud HPC (8x H100 80GB GPUs per node)  
**Target Users**: ~10 users/week at rollout  
**Development Philosophy**: Quality over speed - build best version at our own pace

---

## 1. Executive Summary

### Vision
Build a **modern, AI-agentic bioinformatics pipeline platform** that dynamically generates and executes workflows based on user specifications, using Nextflow as the orchestration engine with containerized tools. Leverage GPU-accelerated AI models for intelligent pipeline design while maintaining parallel operation with existing Snakemake infrastructure.

### Key Differentiators from Current System
- **Dynamic Pipeline Generation**: AI agents create workflows on-demand vs static Snakemake rules
- **GPU-Accelerated AI**: Self-hosted LLMs on H100 GPUs (no API costs, data stays on-premise)
- **Nextflow DSL2**: Modern workflow language with better parallelization and cloud integration
- **Microservices Architecture**: Modular tool containers that can be composed flexibly
- **User-Centric**: Pipelines tailored to specific research questions, not fixed templates
- **Cloud-Native**: Native GCP integration with Google Batch and Cloud Storage
- **Parallel Systems**: Coexists with Snakemake - users choose best tool for their needs

### Strategic Goals
1. **Flexibility**: Generate custom pipelines for novel research questions
2. **Scalability**: Handle 1-1000s of samples with automatic parallelization
3. **Reproducibility**: Containerized tools + versioned workflows + data provenance
4. **AI Integration**: Natural language → executable pipeline translation using on-premise GPU inference
5. **Efficiency**: Minimize storage, maximize compute utilization
6. **User Experience**: Simple enough for new users, powerful enough for experts

---

## 2. Architecture Overview

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│  - CLI (Python Click/Typer)                                 │
│  - Web API (FastAPI) [Future]                               │
│  - Natural Language Interface (LLM Integration)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  AI AGENT ORCHESTRATION                      │
│  - Pipeline Planner: Analyze user query → workflow design   │
│  - Tool Selector: Choose optimal tools for each step        │
│  - Parameter Optimizer: Set tool parameters based on data   │
│  - Resource Manager: Estimate compute/storage requirements  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              NEXTFLOW WORKFLOW ENGINE                        │
│  - DSL2 Pipeline Generator                                  │
│  - Executor: SLURM, Google Batch, AWS Batch, Local         │
│  - Resume/Cache: Automatic checkpoint recovery             │
│  - Tower Integration: Monitoring & logging                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 CONTAINER REGISTRY                           │
│  - Tool Containers: 100+ bioinformatics tools               │
│  - Base Images: Language runtimes (Python, R, Conda)        │
│  - Workflow Templates: Pre-built module library             │
│  - Version Control: Semantic versioning, immutable tags     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA MANAGEMENT LAYER                       │
│  - Storage: /scratch (fast), GCS (cloud), /home (persistent)│
│  - Staging: Automatic data transfer & caching               │
│  - Metadata: Sample sheets, provenance, results tracking    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Workflow Engine** | Nextflow 24.x (DSL2) | Industry standard, excellent GCP support, active nf-core community |
| **Container Runtime** | Singularity/Apptainer | HPC-friendly, rootless, works with SLURM on GCP |
| **Container Registry** | GCP Artifact Registry + Local SIF | Fast local caching, cloud-backed distribution |
| **AI Inference** | vLLM + Llama 3.3 70B / Qwen 2.5 72B | Self-hosted on H100 GPUs, <100ms latency, no API costs |
| **AI Framework** | LangChain + Custom Agents | Structured prompts, tool use, multi-agent orchestration |
| **GPU Hardware** | 8x H100 80GB per node | 3.2TB HBM3 bandwidth, tensor core acceleration |
| **Scheduling** | SLURM + Google Batch | Primary HPC scheduler + cloud burst for large jobs |
| **Data Storage** | NVMe (hot) + HDD (warm) + GCS (cold) | Tiered storage: compute → staging → archival |
| **Programming** | Nextflow DSL2 + Python 3.11 | Workflow definition + AI agent logic |
| **Configuration** | YAML + TOML | Human-readable, validation-friendly |
| **Monitoring** | Nextflow Tower + Cloud Logging | Real-time pipeline monitoring, resource tracking |

---

## 3. Core Capabilities

### 3.1 Pipeline Types (Modular Design)

Each pipeline is a **composition of reusable modules** rather than monolithic workflows:

#### Quality Control & Preprocessing
- **FastQC Module**: Quality metrics for raw sequencing data
- **Trimming Module**: Adapter removal (Trimmomatic, Cutadapt, fastp)
- **Decontamination Module**: Remove host/contaminant reads
- **Normalization Module**: Depth normalization, batch correction

#### Genomics Pipelines
1. **DNA-Seq (Variant Calling)**
   - Modules: BWA/Bowtie2 → GATK/FreeBayes → VEP Annotation → VCF filtering
   - Use Cases: WGS, WES, targeted panels, population genetics

2. **RNA-Seq (Transcriptomics)**
   - Modules: STAR/Salmon → DESeq2/edgeR → GSEA → Visualization
   - Use Cases: Differential expression, isoform analysis, fusion detection

3. **scRNA-Seq (Single-Cell)**
   - Modules: CellRanger/STARsolo → Seurat/Scanpy → Trajectory/Clustering
   - Use Cases: Cell type identification, developmental trajectories, spatial

4. **ChIP-Seq / ATAC-Seq (Epigenomics)**
   - Modules: Bowtie2 → MACS2/HOMER → Peak annotation → Motif analysis
   - Use Cases: TF binding, chromatin accessibility, histone marks

5. **Hi-C (3D Genome)**
   - Modules: HiC-Pro/Juicer → Cooler → TAD calling → Loop detection
   - Use Cases: Chromatin interactions, structural variants

#### Advanced Genomics
6. **Long-Read Sequencing**
   - Modules: Minimap2 → Flye/Canu → Medaka/Arrow → SV calling
   - Use Cases: De novo assembly, structural variants, phasing

7. **Metagenomics**
   - Modules: Kraken2 → MetaPhlAn → Assembly → Binning → Annotation
   - Use Cases: Microbiome profiling, pathogen detection, functional analysis

8. **Structural Variants**
   - Modules: Manta/Delly/SURVIVOR → Filtering → Annotation → Prioritization
   - Use Cases: Cancer genomics, rare disease, population SVs

9. **Methylation**
   - Modules: Bismark → MethylKit → DMR calling → Annotation
   - Use Cases: WGBS, RRBS, targeted bisulfite sequencing

10. **Variant Annotation & Interpretation**
    - Modules: VEP/SnpEff → ClinVar → ACMG classification → Report generation
    - Use Cases: Clinical interpretation, pathogenicity assessment

### 3.2 AI-Driven Features

#### Natural Language Pipeline Design
```python
# User Input (Natural Language)
"I have 50 paired-end RNA-seq samples from tumor and normal tissue. 
I want to find differentially expressed genes and perform pathway enrichment."

# AI Agent Output (Executable Pipeline)
Pipeline Design:
1. FastQC: Quality assessment
2. STAR: Genome alignment (2-pass mode)
3. featureCounts: Gene quantification
4. DESeq2: Differential expression (tumor vs normal)
5. GSEA: Pathway enrichment (Hallmark + GO)
6. MultiQC: Unified report

Resources:
- Samples: 50 (25 tumor, 25 normal)
- Estimated time: 12 hours (parallel)
- Storage: 2TB (alignments) + 500GB (results)
- Cost: 800 CPU-hours

Confirm pipeline? [Y/n]
```

#### Dynamic Tool Selection
- **Best Practice**: AI selects optimal tools based on:
  - Data type (short/long reads, single/paired-end)
  - Organism (human, mouse, non-model)
  - Research question (discovery vs validation)
  - Available resources (compute, time, budget)

- **Example**: For variant calling
  - Human WGS → GATK HaplotypeCaller (gold standard)
  - Non-human WGS → FreeBayes (no training data needed)
  - RNA-seq variants → GATK RNA-seq mode
  - Long reads → Clair3/DeepVariant

#### Intelligent Parameter Tuning
```yaml
# AI-optimized parameters based on data characteristics
alignment:
  tool: STAR
  threads: 16  # Auto-scaled based on node availability
  genomeDir: /scratch/references/GRCh38_STAR_index
  readFilesCommand: zcat
  outSAMtype: BAM SortedByCoordinate
  outSAMattributes: NH HI AS nM  # Optimized for downstream variant calling
  
  # AI-suggested parameters based on read length distribution
  alignIntronMin: 20  # Short introns detected in data
  alignIntronMax: 1000000
  alignMatesGapMax: 1000000
```

#### Adaptive Resource Allocation
- **Profile-Based**: Learn from past runs to predict resource needs
- **Data-Driven**: Estimate based on input file sizes and complexity
- **Cost-Aware**: Balance speed vs cost for cloud execution

---

## 4. Technical Implementation

### 4.1 Directory Structure (New Codebase)

```
nextflow-pipelines/
├── README.md
├── LICENSE
├── pyproject.toml                  # Python dependencies (AI agents, CLI)
├── nextflow.config                 # Global Nextflow configuration
├── .gitignore
│
├── bin/                            # Executable scripts
│   ├── nfp                        # Main CLI entry point
│   ├── pipeline_generator.py      # AI agent for pipeline generation
│   └── resource_estimator.py      # Compute/storage prediction
│
├── src/                           # Python source code
│   ├── agents/                    # AI agent implementations
│   │   ├── planner.py            # Workflow design agent
│   │   ├── selector.py           # Tool selection agent
│   │   ├── optimizer.py          # Parameter tuning agent
│   │   └── validator.py          # Pipeline validation agent
│   ├── api/                       # API interfaces
│   │   ├── cli.py                # Command-line interface (Click/Typer)
│   │   └── rest.py               # REST API (FastAPI) [Future]
│   ├── core/                      # Core logic
│   │   ├── pipeline.py           # Pipeline object model
│   │   ├── module.py             # Module/process definitions
│   │   └── config.py             # Configuration management
│   └── utils/                     # Utilities
│       ├── storage.py            # Data staging & caching
│       ├── slurm.py              # SLURM integration
│       └── validators.py         # Input validation
│
├── modules/                       # Nextflow DSL2 modules (reusable)
│   ├── qc/
│   │   ├── fastqc.nf             # FastQC module
│   │   ├── multiqc.nf            # MultiQC aggregation
│   │   └── trimming.nf           # Adapter trimming
│   ├── alignment/
│   │   ├── bwa.nf
│   │   ├── star.nf
│   │   ├── minimap2.nf
│   │   └── bowtie2.nf
│   ├── variants/
│   │   ├── gatk_haplotypecaller.nf
│   │   ├── freebayes.nf
│   │   ├── annotation.nf
│   │   └── filtering.nf
│   ├── expression/
│   │   ├── featurecounts.nf
│   │   ├── salmon.nf
│   │   ├── deseq2.nf
│   │   └── gsea.nf
│   └── ... (more modules)
│
├── workflows/                     # Complete pipeline workflows
│   ├── rnaseq.nf                 # RNA-seq reference workflow
│   ├── dnaseq.nf                 # DNA-seq reference workflow
│   ├── scrnaseq.nf               # Single-cell RNA-seq
│   └── custom/                   # AI-generated custom pipelines
│       └── .gitkeep              # Generated dynamically
│
├── containers/                    # Container definitions
│   ├── Singularity.base          # Base container
│   ├── tools/                    # Individual tool containers
│   │   ├── fastqc.def
│   │   ├── star.def
│   │   ├── gatk.def
│   │   └── ... (100+ tools)
│   ├── modules/                  # Module-level containers (grouped tools)
│   │   ├── qc_suite.def         # FastQC + MultiQC + Trim
│   │   └── variant_calling.def   # BWA + GATK + VEP
│   └── images/                   # Built SIF files
│       └── .gitkeep
│
├── config/                        # Configuration files
│   ├── profiles/                  # Execution profiles
│   │   ├── slurm.config          # SLURM cluster settings
│   │   ├── google.config         # Google Cloud Batch
│   │   ├── aws.config            # AWS Batch
│   │   └── local.config          # Local execution
│   ├── resources/                 # Resource requirements
│   │   ├── standard.yaml         # Default resource specs
│   │   └── optimized.yaml        # AI-tuned resources
│   └── references/                # Reference genome configs
│       ├── hg38.yaml
│       ├── mm10.yaml
│       └── custom.yaml
│
├── data/                          # Data directory (symlinks)
│   ├── raw -> /scratch/.../raw
│   ├── references -> /scratch/.../references
│   └── results -> /scratch/.../results
│
├── scripts/                       # Helper scripts
│   ├── setup_environment.sh      # Install dependencies
│   ├── build_containers.sh       # Batch container building
│   └── download_references.sh    # Reference genome setup
│
├── tests/                         # Testing suite
│   ├── unit/                     # Unit tests (pytest)
│   ├── integration/              # Integration tests
│   └── data/                     # Test datasets
│       └── small_fastq/
│
├── docs/                          # Documentation
│   ├── installation.md
│   ├── quickstart.md
│   ├── modules.md                # Module documentation
│   ├── ai_agents.md              # AI agent design
│   └── examples/                 # Usage examples
│
└── logs/                          # Execution logs
    ├── .nextflow.log
    ├── pipelines/                # Per-pipeline logs
    └── agents/                   # AI agent decision logs
```

### 4.2 Nextflow Module Example

```nextflow
// modules/alignment/star.nf

process STAR_ALIGN {
    tag "${sample_id}"
    label 'high_cpu'
    container "${params.containers.star}"
    
    publishDir "${params.outdir}/${sample_id}/alignment", 
               mode: 'copy',
               pattern: "*.bam*"
    
    input:
    tuple val(sample_id), path(reads)
    path genome_index
    
    output:
    tuple val(sample_id), path("${sample_id}.Aligned.sortedByCoord.out.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}.Aligned.sortedByCoord.out.bam.bai"), emit: bai
    path "${sample_id}.Log.final.out", emit: log
    path "${sample_id}.SJ.out.tab", emit: splice_junctions
    
    script:
    def read_files = reads instanceof List ? reads.join(' ') : reads
    def avail_mem = task.memory ? "--limitBAMsortRAM ${task.memory.toBytes()}" : ''
    """
    STAR \\
        --runThreadN ${task.cpus} \\
        --genomeDir ${genome_index} \\
        --readFilesIn ${read_files} \\
        --readFilesCommand zcat \\
        --outFileNamePrefix ${sample_id}. \\
        --outSAMtype BAM SortedByCoordinate \\
        --outSAMattributes NH HI AS nM MD \\
        --quantMode GeneCounts \\
        ${avail_mem}
    
    samtools index ${sample_id}.Aligned.sortedByCoord.out.bam
    """
}
```

### 4.3 AI Agent Implementation Example

```python
# src/agents/planner.py

from typing import List, Dict, Any
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class PipelineStep(BaseModel):
    """A single step in the pipeline."""
    name: str = Field(description="Step name (e.g., 'Quality Control')")
    tool: str = Field(description="Primary tool (e.g., 'FastQC')")
    module: str = Field(description="Nextflow module path (e.g., 'qc/fastqc')")
    inputs: List[str] = Field(description="Required inputs")
    outputs: List[str] = Field(description="Generated outputs")
    depends_on: List[str] = Field(default=[], description="Dependencies")

class PipelineDesign(BaseModel):
    """Complete pipeline design."""
    name: str
    description: str
    steps: List[PipelineStep]
    estimated_time: str
    estimated_storage: str
    estimated_cost: float

class PipelinePlannerAgent:
    """AI agent that designs pipelines from natural language queries."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.llm = ChatAnthropic(model=model, temperature=0)
        self.prompt = self._create_prompt()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert bioinformatics pipeline designer.
            Given a user's research question, design an optimal analysis pipeline.
            
            Available pipeline types:
            - DNA-Seq: Variant calling from genomic DNA
            - RNA-Seq: Transcriptome analysis, differential expression
            - scRNA-Seq: Single-cell RNA sequencing analysis
            - ChIP-Seq: Chromatin immunoprecipitation sequencing
            - ATAC-Seq: Chromatin accessibility
            - Hi-C: 3D genome structure
            - Long-Read: PacBio/Nanopore sequencing
            - Metagenomics: Microbiome analysis
            - Methylation: DNA methylation analysis
            
            Available modules in modules/ directory:
            - qc/: fastqc, multiqc, trimming
            - alignment/: bwa, star, minimap2, bowtie2
            - variants/: gatk, freebayes, annotation
            - expression/: featurecounts, salmon, deseq2, gsea
            - epigenomics/: macs2, homer, deeptools
            - assembly/: spades, flye, canu
            - annotation/: vep, snpeff, annovar
            
            Design a pipeline that:
            1. Follows bioinformatics best practices
            2. Uses appropriate tools for the data type
            3. Includes quality control and validation steps
            4. Minimizes intermediate storage
            5. Maximizes parallelization opportunities
            
            Return a structured PipelineDesign object."""),
            ("user", "{query}")
        ])
    
    def plan_pipeline(self, user_query: str, 
                     data_info: Dict[str, Any] = None) -> PipelineDesign:
        """
        Design a pipeline from a natural language query.
        
        Args:
            user_query: User's research question or analysis goal
            data_info: Optional metadata about input data
                - sample_count: Number of samples
                - read_type: 'single' or 'paired'
                - read_length: Average read length
                - organism: 'human', 'mouse', etc.
        
        Returns:
            PipelineDesign object with complete workflow specification
        """
        # Enhance query with data info
        enhanced_query = user_query
        if data_info:
            enhanced_query += f"\n\nData characteristics:\n"
            for key, value in data_info.items():
                enhanced_query += f"- {key}: {value}\n"
        
        # Generate pipeline design using structured output
        structured_llm = self.llm.with_structured_output(PipelineDesign)
        chain = self.prompt | structured_llm
        
        design = chain.invoke({"query": enhanced_query})
        
        # Validate design
        self._validate_design(design)
        
        return design
    
    def _validate_design(self, design: PipelineDesign) -> None:
        """Validate that the pipeline design is feasible."""
        # Check that all module paths exist
        import os
        base_path = "modules"
        for step in design.steps:
            module_path = os.path.join(base_path, f"{step.module}.nf")
            if not os.path.exists(module_path):
                raise ValueError(f"Module not found: {module_path}")
        
        # Check for circular dependencies
        # ... (dependency graph validation)
        
        # Check that inputs/outputs match between steps
        # ... (data flow validation)

# Usage Example
if __name__ == "__main__":
    agent = PipelinePlannerAgent()
    
    query = """
    I have RNA-seq data from 20 cancer patients and 20 healthy controls.
    I want to identify genes that are differentially expressed and understand
    which biological pathways are affected.
    """
    
    data_info = {
        "sample_count": 40,
        "read_type": "paired",
        "read_length": 150,
        "organism": "human",
        "sequencing_depth": "50M reads/sample"
    }
    
    design = agent.plan_pipeline(query, data_info)
    
    print(f"Pipeline: {design.name}")
    print(f"Description: {design.description}")
    print(f"\nSteps ({len(design.steps)}):")
    for i, step in enumerate(design.steps, 1):
        print(f"{i}. {step.name} ({step.tool})")
    
    print(f"\nEstimates:")
    print(f"- Time: {design.estimated_time}")
    print(f"- Storage: {design.estimated_storage}")
    print(f"- Cost: ${design.estimated_cost}")
```

### 4.4 CLI Interface Example

```python
# src/api/cli.py

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
from typing import Optional

app = typer.Typer(help="Nextflow Pipeline Generator - AI-driven bioinformatics workflows")
console = Console()

@app.command()
def plan(
    query: str = typer.Argument(..., help="Describe your analysis goal"),
    samples: Optional[int] = typer.Option(None, "--samples", "-n", help="Number of samples"),
    organism: Optional[str] = typer.Option("human", "--organism", "-o", help="Organism"),
    output: Optional[Path] = typer.Option(None, "--output", "-O", help="Save pipeline design to file")
):
    """Design a pipeline from natural language description."""
    
    from src.agents.planner import PipelinePlannerAgent
    
    console.print(f"[bold blue]Analyzing query:[/bold blue] {query}")
    
    # Gather data info
    data_info = {"organism": organism}
    if samples:
        data_info["sample_count"] = samples
    
    # Plan pipeline
    with console.status("[bold green]Designing pipeline..."):
        agent = PipelinePlannerAgent()
        design = agent.plan_pipeline(query, data_info)
    
    # Display results
    console.print(f"\n[bold green]✓ Pipeline Design Complete[/bold green]")
    console.print(f"[bold]{design.name}[/bold]")
    console.print(f"{design.description}\n")
    
    # Steps table
    table = Table(title="Pipeline Steps")
    table.add_column("Step", style="cyan")
    table.add_column("Tool", style="magenta")
    table.add_column("Module", style="green")
    
    for i, step in enumerate(design.steps, 1):
        table.add_row(f"{i}. {step.name}", step.tool, step.module)
    
    console.print(table)
    
    # Estimates
    console.print(f"\n[bold]Resource Estimates:[/bold]")
    console.print(f"  Time: {design.estimated_time}")
    console.print(f"  Storage: {design.estimated_storage}")
    console.print(f"  Cost: ${design.estimated_cost:.2f}")
    
    # Save if requested
    if output:
        import json
        output.write_text(design.model_dump_json(indent=2))
        console.print(f"\n[green]✓ Design saved to {output}[/green]")
    
    # Prompt to generate
    if typer.confirm("\nGenerate Nextflow pipeline?"):
        generate_pipeline(design)

@app.command()
def run(
    pipeline: Path = typer.Argument(..., help="Path to pipeline.nf or design.json"),
    samples: Path = typer.Argument(..., help="Path to sample sheet (CSV)"),
    profile: str = typer.Option("slurm", "--profile", "-p", help="Execution profile"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume previous run")
):
    """Execute a Nextflow pipeline."""
    
    console.print(f"[bold blue]Running pipeline:[/bold blue] {pipeline}")
    
    # Build nextflow command
    cmd = f"nextflow run {pipeline} "
    cmd += f"--samples {samples} "
    cmd += f"-profile {profile} "
    if resume:
        cmd += "-resume "
    
    # Execute
    import subprocess
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        console.print("[bold green]✓ Pipeline completed successfully[/bold green]")
    else:
        console.print("[bold red]✗ Pipeline failed[/bold red]", err=True)

@app.command()
def list_modules():
    """List all available pipeline modules."""
    
    from pathlib import Path
    
    modules_dir = Path("modules")
    
    table = Table(title="Available Modules")
    table.add_column("Category", style="cyan")
    table.add_column("Module", style="magenta")
    table.add_column("Description", style="white")
    
    for category in sorted(modules_dir.iterdir()):
        if category.is_dir():
            for module in sorted(category.glob("*.nf")):
                # Extract description from module file
                desc = "..."  # Parse from module comments
                table.add_row(category.name, module.stem, desc)
    
    console.print(table)

if __name__ == "__main__":
    app()
```

### Usage Examples

```bash
# 1. Plan a pipeline from natural language
nfp plan "Find differentially expressed genes in tumor vs normal RNA-seq" \
    --samples 40 \
    --organism human \
    --output designs/rnaseq_tumor_normal.json

# 2. Generate pipeline from plan
nfp generate designs/rnaseq_tumor_normal.json \
    --output workflows/custom/tumor_normal_rnaseq.nf

# 3. Run the pipeline
nfp run workflows/custom/tumor_normal_rnaseq.nf \
    samples.csv \
    --profile slurm \
    --resume

# 4. List available modules
nfp list-modules

# 5. Interactive mode
nfp interactive
> I have ChIP-seq data for a transcription factor. What analysis should I do?
[AI suggests pipeline...]
> Generate the pipeline
[Pipeline created...]
> Run it with samples in /scratch/data/chipseq/
[Execution starts...]
```

---

## 5. Implementation Roadmap

**Timeline**: 12-week phased development with quality checkpoints  
**Philosophy**: Build best version at our own pace - quality over speed  
**Strategy**: Parallel systems - Snakemake continues production while Nextflow develops

### Phase 1: Foundation & GPU Infrastructure (Weeks 1-3)
**Goal**: Core infrastructure and self-hosted AI

**GPU Infrastructure Setup**
- [ ] Deploy vLLM inference server on H100 node
- [ ] Test models: Llama 3.3 70B, Qwen 2.5 72B, DeepSeek-V3
- [ ] Benchmark latency: target <500ms for pipeline generation
- [ ] Set up model quantization (FP8/INT4) for optimal throughput
- [ ] Configure GPU SLURM partition with proper isolation

**Nextflow Foundation**
- [ ] Create directory: `BioPipelines/nextflow-platform/`
- [ ] Install Nextflow (24.x) and configure for SLURM + Google Batch
- [ ] Set up GCP Artifact Registry for container distribution
- [ ] Configure nf-core tooling and templates
- [ ] Build base container with Nextflow + Python 3.11

**CLI & Basic Modules**
- [ ] Implement CLI skeleton (Typer + Rich for beautiful output)
- [ ] Create 3 reference modules:
  - `qc/fastqc.nf`
  - `alignment/star.nf`
  - `expression/deseq2.nf`
- [ ] Write first AI agent: `PipelinePlannerAgent` (connecting to local vLLM)
- [ ] Test: "RNA-seq differential expression" → generated pipeline

**Checkpoint**: Can we generate a valid Nextflow pipeline from natural language? (Pass/Fail)

**Deliverable**: Functional proof-of-concept with GPU-accelerated AI

---

### Phase 2: Core Pipeline Tier 1 (Weeks 4-6)
**Goal**: Build 3 complete, production-ready pipelines

**RNA-seq Pipeline** (Priority 1)
- [ ] Modules: FastQC → STAR/Salmon → featureCounts → DESeq2 → GSEA → MultiQC
- [ ] Support: Single/paired-end, stranded/unstranded, multiple references
- [ ] AI agent: Auto-detect strandedness, choose quantification method
- [ ] Testing: ENCODE test dataset, compare to Snakemake results

**DNA-seq Variant Calling** (Priority 2)
- [ ] Modules: BWA-MEM2 → GATK HaplotypeCaller → VEP annotation → Filtering
- [ ] Support: WGS, WES, targeted panels
- [ ] AI agent: Choose GATK vs FreeBayes based on organism/coverage
- [ ] Testing: Genome in a Bottle reference samples

**scRNA-seq** (Priority 3)
- [ ] Modules: CellRanger/STARsolo → Scanpy/Seurat → Clustering → Annotation
- [ ] Support: 10x, Drop-seq, Smart-seq2
- [ ] AI agent: Recommend normalization and batch correction strategies
- [ ] Testing: PBMC 3k dataset from 10x Genomics

**Infrastructure**
- [ ] Create module containers (shared across pipelines)
- [ ] Implement resource profiles (small/medium/large datasets)
- [ ] Add GCS data staging automation
- [ ] Write comprehensive module documentation

**Checkpoint**: Are pipelines as good or better than Snakemake versions? (Quality gate)

**Deliverable**: 3 production-ready pipelines, full documentation

---

### Phase 3: AI Agent Intelligence (Weeks 7-9)
**Goal**: Multi-agent system with expert-level decision making

**Core Agents**
- [ ] **PlannerAgent**: High-level workflow design (Llama 3.3 70B)
- [ ] **ToolSelectorAgent**: Choose optimal tools per step (rule-based + LLM)
- [ ] **ParameterOptimizerAgent**: Literature-based parameter tuning
- [ ] **ResourceManagerAgent**: Predict compute/storage from data characteristics
- [ ] **ValidatorAgent**: Check pipeline correctness (syntax + logic)
- [ ] **CostEstimatorAgent**: Estimate GCP costs before execution

**Agent Orchestration**
- [ ] Implement hierarchical coordination (planner → specialists)
- [ ] Add agent memory: learn from past pipeline runs
- [ ] Create decision logging (why agent chose tool X over Y)
- [ ] Build feedback loop: user corrections → agent improvement

**Advanced Features**
- [ ] Interactive mode: conversational pipeline refinement
- [ ] Explain mode: "Why did you choose STAR over HISAT2?"
- [ ] Alternative mode: "Show me 3 different approaches"
- [ ] Benchmark mode: "Compare GATK vs FreeBayes on my data"

**Testing**
- [ ] Create test suite: 50 natural language queries
- [ ] Target accuracy: 80% correct pipelines without human correction
- [ ] Measure: generation time, resource accuracy, user satisfaction

**Checkpoint**: Do AI agents reliably generate correct pipelines? (80% success target)

**Deliverable**: Intelligent multi-agent system with learning capability

---

### Phase 4: Additional Pipelines Tier 2 (Weeks 10-11)
**Goal**: Expand pipeline library to cover more use cases

**Additional Pipelines** (Choose 3 based on user demand)
- [ ] **ATAC-seq**: Chromatin accessibility analysis
- [ ] **ChIP-seq**: TF binding and histone modifications (shares modules with ATAC-seq)
- [ ] **Long-read sequencing**: PacBio/Nanopore analysis and assembly
- [ ] **Metagenomics**: Microbiome profiling and functional analysis
- [ ] **Methylation**: WGBS and RRBS analysis

**Module Reuse**
- [ ] Maximize module sharing across pipelines
- [ ] Create meta-modules: combined tool groups
- [ ] Document module compatibility matrix

**nf-core Integration**
- [ ] Import stable nf-core modules where appropriate
- [ ] Contribute novel modules back to nf-core community
- [ ] Follow nf-core best practices and standards

**Deliverable**: 6 total pipelines covering major analysis types

---

### Phase 5: Production Features (Week 12)
**Goal**: Enterprise-ready platform

**Cloud Integration**
- [ ] Google Batch executor for burst capacity
- [ ] Automatic GCS archival (results older than 30 days)
- [ ] Cloud Storage FUSE for transparent data access
- [ ] Preemptible VM support for cost optimization

**Monitoring & Observability**
- [ ] Nextflow Tower integration
- [ ] Cloud Logging for centralized logs
- [ ] Prometheus metrics: job success rate, resource utilization
- [ ] Alert system: email on pipeline failure

**User Experience**
- [ ] Sample sheet templates and validation
- [ ] Pipeline resumption from any failure point
- [ ] Progress indicators and ETA estimates
- [ ] Result summaries with key findings highlighted

**Documentation**
- [ ] Comprehensive user guide
- [ ] Video tutorials (pipeline generation, execution, troubleshooting)
- [ ] API documentation for programmatic access
- [ ] FAQ from beta testing

**Deliverable**: Production-ready platform with full observability

---

### Phase 6: Testing & Validation (Week 13)
**Goal**: Ensure correctness and reliability

**Validation Testing**
- [ ] Compare outputs vs Snakemake pipelines (identical results)
- [ ] Run benchmark datasets: ENCODE, TCGA, GIAB references
- [ ] Test edge cases: single sample, 1000 samples, missing data
- [ ] Stress test: concurrent pipeline submissions
- [ ] Failure recovery: kill jobs mid-run, verify resume works

**Performance Benchmarking**
- [ ] Measure: wall time, CPU hours, cost per pipeline
- [ ] Compare: Nextflow vs Snakemake efficiency
- [ ] Optimize: identify and fix bottlenecks
- [ ] Document: performance characteristics and scaling limits

**Security Audit**
- [ ] Container vulnerability scanning
- [ ] Data access permissions review
- [ ] Secrets management (API keys, credentials)
- [ ] Compliance: HIPAA considerations for human data

**User Acceptance Testing**
- [ ] Beta testers: 3-5 power users
- [ ] Real projects: not just test data
- [ ] Gather feedback: usability issues, feature requests
- [ ] Iterate: fix critical issues before launch

**Checkpoint**: Would we confidently use this for production research? (Go/No-Go decision)

**Deliverable**: Validated, secure, performant platform

---

### Phase 7: Deployment & Rollout (Week 14)
**Goal**: Launch to users and establish support model

**Deployment**
- [ ] Production deployment on cluster
- [ ] Load balancing for AI inference (multiple H100 nodes)
- [ ] Backup and disaster recovery plan
- [ ] Documentation website (Sphinx or MkDocs)

**Training & Onboarding**
- [ ] User workshop: 2-hour hands-on training
- [ ] Create tutorial notebooks (Jupyter)
- [ ] Office hours: weekly drop-in support
- [ ] Slack/Teams channel for questions

**Launch Strategy**
- [ ] Soft launch: announce to research group
- [ ] Target: 10 users/week capacity
- [ ] Monitor: usage patterns, common issues
- [ ] Support: quick response to early adopter problems

**Metrics & KPIs**
- [ ] Track: pipelines generated, success rate, user satisfaction
- [ ] Weekly reports: usage statistics and trends
- [ ] Feedback loop: prioritize improvements based on data

**Deliverable**: Live platform with active users and support infrastructure

---

### Phase 8: Continuous Improvement (Ongoing)
**Goal**: Iterate based on real-world usage

**Weeks 15-20: Enhancement Cycle**
- [ ] Add most-requested pipeline types
- [ ] Improve AI agents based on user feedback
- [ ] Optimize resource allocation from actual usage data
- [ ] Expand module library (community contributions)

**Technical Debt**
- [ ] Refactor based on lessons learned
- [ ] Improve test coverage to 80%+
- [ ] Performance optimization rounds
- [ ] Documentation updates

**Community Building**
- [ ] Internal user group meetings (monthly)
- [ ] Share success stories and use cases
- [ ] Contribute to nf-core community
- [ ] Consider open-source release (if appropriate)

**Deliverable**: Mature platform with growing user base and feature set

---

## 6. Key Decisions & Trade-offs

### 6.1 Nextflow vs Other Orchestrators

| Feature | Nextflow | Snakemake | Cromwell | Our Choice |
|---------|----------|-----------|----------|------------|
| **DSL** | Groovy-based DSL2 | Python | WDL | **Nextflow** |
| **Cloud Native** | Excellent (AWS, GCP, Azure) | Limited | Good | ✓ |
| **HPC Support** | Native SLURM | Native SLURM | Via backends | ✓ |
| **Resume/Cache** | Automatic | Automatic | Limited | ✓ |
| **Ecosystem** | nf-core (1000+ pipelines) | Snakemake-wrappers | BioWDL | ✓ |
| **Learning Curve** | Moderate | Easy (Python) | Moderate | ✓ |
| **Container Support** | Docker, Singularity, Podman | Conda, Singularity | Docker | ✓ |

**Decision**: **Nextflow** for superior cloud integration, active community, and nf-core ecosystem

### 6.2 Container Strategy

**Option A: Monolithic Containers** (Current Approach)
- Pros: Simple deployment, one container per pipeline
- Cons: Large size (~10GB), slow builds, poor modularity

**Option B: Micro-Containers** (Individual Tools)
- Pros: Small, reusable, fast builds
- Cons: Many containers to manage (~100+), startup overhead

**Option C: Module-Level Containers** (Grouped Tools)
- Pros: Balance of modularity and efficiency
- Cons: Some redundancy

**Decision**: **Hybrid Approach**
- **Base container**: Python, R, Conda (shared by all)
- **Module containers**: Grouped by function (qc_suite, alignment_suite, etc.)
- **Specialty containers**: Large/complex tools (GATK, CellRanger)

Target: ~20 containers vs current 12, better reuse

### 6.3 AI Agent Architecture

**Option A: Single LLM Agent**
- Pros: Simple, low latency
- Cons: Limited reasoning, poor at complex planning

**Option B: Multi-Agent System** (Specialist Agents)
- Pros: Expert knowledge per domain, better reasoning
- Cons: Higher latency, coordination complexity

**Option C: Hierarchical Agents** (Planner → Executors)
- Pros: Balance of expertise and efficiency
- Cons: Moderate complexity

**Decision**: **Multi-Agent with Hierarchical Coordination + GPU Acceleration**
- **Planner Agent**: High-level workflow design (Llama 3.3 70B on H100)
- **Specialist Agents**: Tool selection, parameter tuning (Qwen 2.5 72B or smaller models)
- **Validator Agent**: Check correctness (rule-based + LLM verification)
- **Inference**: vLLM server on 8x H100 GPUs (~100-500ms latency per query)
- **Scaling**: Multiple H100 nodes for concurrent users (10 users/week → 1 GPU per user)

**GPU Resource Allocation**:
```yaml
AI Inference Server:
  Hardware: 1-2 H100 nodes (8 GPUs each)
  Model Loading:
    - Llama 3.3 70B: 4 GPUs (tensor parallelism)
    - Qwen 2.5 72B: 4 GPUs (tensor parallelism)
  Throughput: 50-100 tokens/sec per model
  Concurrent Users: 10-20 simultaneous queries
  
Pipeline Execution:
  Remains on CPU nodes (SLURM)
  GPU-accelerated tools (optional): DeepVariant, AlphaFold
```

### 6.4 Data Management

**Current**: `/scratch` for everything (volatile, not backed up)

**Proposed**: GCP-Native Tiered Storage
1. **Hot (Compute)**: Local NVMe on compute nodes - active jobs only (auto-cleanup)
2. **Warm (Staging)**: Persistent Disk (SSD) - recent results (30-day retention)
3. **Cold (Archive)**: Google Cloud Storage Standard - long-term storage (lifecycle management)
4. **Deep Archive**: GCS Nearline/Coldline - rarely accessed data (cost-optimized)

**Implementation**:
- Nextflow publishes results to GCS automatically (via publishDir)
- Cloud Storage FUSE mounts GCS as local filesystem (transparent access)
- AI agent estimates data lifecycle and storage tier
- Automatic lifecycle rules:
  - 30 days: Standard → Nearline (50% cost reduction)
  - 90 days: Nearline → Coldline (75% cost reduction)
  - Never delete: Researchers decide retention
  
**Cost Optimization**:
```yaml
Storage_Costs_GCP:
  Standard: $0.020/GB/month (first 30 days)
  Nearline: $0.010/GB/month (30-90 days)
  Coldline: $0.004/GB/month (90+ days)
  
Example_Project:
  Raw_Data: 500GB (Archive immediately, rarely accessed)
  Alignments: 2TB (Keep 30 days, then Nearline)
  Results: 50GB (Keep indefinitely in Standard)
  Total_Monthly: ~$25-30 vs $41 all-Standard
```

---

## 7. Success Metrics

### 7.1 Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Pipeline Generation Time** | < 30 seconds | AI agent response time |
| **Execution Efficiency** | 90% of optimal | CPU utilization, wall time |
| **Resource Accuracy** | ±20% | Predicted vs actual compute/storage |
| **Failure Rate** | < 5% | Failed runs / total runs |
| **Resume Success** | > 95% | Successful resumes after failure |
| **Storage Efficiency** | 50% reduction | vs storing all intermediates |

### 7.2 User Experience Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Time to First Result** | < 5 minutes | Setup → running pipeline |
| **Learning Curve** | < 1 hour | Onboarding → first custom pipeline |
| **User Satisfaction** | 4.5/5 | Post-use survey |
| **Query Success Rate** | > 80% | Natural language → correct pipeline |
| **Documentation Clarity** | 4.5/5 | User feedback |

### 7.3 Scientific Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Result Reproducibility** | 100% | Re-runs produce identical outputs |
| **Best Practice Adherence** | > 90% | Compliance with field standards |
| **Tool Version Control** | 100% | All tools versioned, containers tagged |
| **Provenance Tracking** | 100% | Full lineage from raw data → results |

---

## 8. Risk Analysis & Mitigation

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Nextflow learning curve** | Medium | Medium | Use nf-core templates, extensive docs |
| **AI hallucinations** | High | High | Validator agent, human review step |
| **Container build failures** | Low | Medium | CI/CD testing, rollback mechanism |
| **SLURM incompatibility** | Low | High | Test on cluster early, use nf-core configs |
| **Storage quota exceeded** | Medium | High | Tiered storage, auto-cleanup policies |

### 8.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **User adoption resistance** | Medium | High | Training, demo success cases, gradual rollout |
| **Maintenance burden** | High | Medium | Good documentation, modular design |
| **Dependency updates** | High | Low | Automated testing, container versioning |
| **Cost overruns (cloud)** | Medium | Medium | Budget alerts, cost estimation before runs |

### 8.3 Scientific Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Incorrect tool selection** | Low | Critical | Validator agent, expert review option |
| **Parameter errors** | Medium | High | Literature-based defaults, QC checks |
| **Reproducibility failure** | Low | Critical | Container immutability, version locking |
| **Data loss** | Low | Critical | Redundant storage, backup to GCS |

---

## 9. Future Enhancements (Post-Launch)

### Year 1: Consolidation
- [ ] Support 50+ modules covering all major analysis types
- [ ] Cloud deployment (Google Batch, AWS Batch)
- [ ] Web UI for non-CLI users
- [ ] Integration with Galaxy, LIMS systems
- [ ] Real-time collaboration (shared pipelines)

### Year 2: Intelligence
- [ ] Learn from user feedback (reinforcement learning)
- [ ] Automatic benchmark comparison (tool A vs B)
- [ ] Cost optimization recommendations
- [ ] Anomaly detection (QC failures, outliers)
- [ ] Scientific literature integration (auto-update best practices)

### Year 3: Ecosystem
- [ ] Public pipeline repository (share with community)
- [ ] Marketplace for custom modules
- [ ] Integration with data repositories (GEO, SRA, ENA)
- [ ] Multi-omics integration (joint RNA+ATAC+HiC analysis)
- [ ] Federated analysis (multi-site collaborations)

---

## 10. Comparison with Current System

| Aspect | Current (Snakemake) | New (Nextflow + AI) | Improvement |
|--------|---------------------|---------------------|-------------|
| **Pipeline Design** | Manual rule writing | AI-generated from NL query | 10x faster |
| **Flexibility** | Fixed 10 pipelines | Infinite custom pipelines | ∞ |
| **Modularity** | Monolithic workflows | Composable modules | High reuse |
| **Cloud Support** | Limited | Native (GCP, AWS, Azure) | Future-proof |
| **Resource Mgmt** | Static SLURM config | AI-optimized allocation | 20% cost ↓ |
| **Resumption** | Snakemake checkpoints | Nextflow work/ dir | Robust |
| **Monitoring** | SLURM logs | Tower dashboards | Real-time |
| **Learning Curve** | Python + Snakemake | Natural language | Accessible |
| **Reproducibility** | Good (containers) | Excellent (containers + provenance) | Auditable |
| **Community** | Snakemake ecosystem | nf-core + custom | Larger |

---

## 11. Stakeholder Decisions & Requirements

### Infrastructure (Confirmed ✓)
1. **GPU Access**: 8x H100 80GB GPUs per node - CONFIRMED
   - Use for self-hosted LLM inference (vLLM)
   - No API costs, data stays on-premise
   - ~100-500ms latency for pipeline generation

2. **Cloud Environment**: Google Cloud Platform - CONFIRMED
   - Native GCP integration (Google Batch, Cloud Storage)
   - No budget constraints for development
   - Cost optimization through lifecycle management

3. **Container Registry**: GCP Artifact Registry - DECIDED
   - Private registry for proprietary tools
   - Fast distribution across compute nodes
   - Integration with Cloud Build for CI/CD

4. **LIMS Integration**: Defer to Phase 8 (post-launch)
   - Focus on core functionality first
   - Add integrations based on user demand

### User Requirements (Confirmed ✓)
1. **Target Users**: ~10 users/week at rollout - CONFIRMED
   - Gradual rollout strategy
   - Close monitoring and support in early weeks
   - Scale infrastructure as adoption grows

2. **Analysis Priorities**: 
   - Tier 1: RNA-seq, DNA-seq variant calling, scRNA-seq
   - Tier 2: ATAC-seq, ChIP-seq, Long-read sequencing
   - Tier 3: Add based on user requests

3. **Interface**: CLI-first, Web UI optional
   - Power users prefer command-line control
   - Web UI for visualization/monitoring (future)
   - Jupyter notebook integration for exploratory analysis

4. **AI Trust Level**: High tolerance with validation
   - Users comfortable with AI-generated pipelines
   - Validation step before execution
   - Explain mode to understand AI decisions

### Development Strategy (Confirmed ✓)
1. **Timeline**: 14-week phased development - DECIDED
   - Quality over speed: "build best version at our own pace"
   - Checkpoints at weeks 3, 9, 13 for quality gates
   - Flexible schedule based on real progress

2. **Resources**: 
   - Dedicated development time (not rushing)
   - H100 GPUs available for AI inference
   - GCP credits unlimited for development

3. **Success Criteria**:
   - Week 3: Can generate valid Nextflow pipeline from NL
   - Week 9: AI agents 80% accurate on test queries
   - Week 13: Would use for real research (user validation)
   - Post-launch: 80%+ user satisfaction, <5% failure rate

4. **System Architecture**: Parallel systems - DECIDED
   - Keep Snakemake running (production)
   - Build Nextflow alongside (development)
   - No forced migration, users choose
   - Long-term: Natural adoption of superior system

### Public Release (TBD)
- Internal use for first 6 months
- Evaluate open-source release after validation
- Potential nf-core contributions
- Decision deferred until system matures

---

## 12. Next Immediate Steps

### Week 1: GPU & Nextflow Foundation
**Days 1-2: GPU Infrastructure**
1. **Deploy vLLM server** on H100 node
   ```bash
   # Test models and benchmark
   vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4
   # Measure: latency, throughput, memory usage
   ```
2. **Test inference APIs**: REST endpoint, Python client
3. **Benchmark models**: Llama 3.3 70B, Qwen 2.5 72B, DeepSeek-V3
4. **Document**: GPU setup guide, model selection criteria

**Days 3-4: Nextflow Setup**
1. **Create directory**: `BioPipelines/nextflow-platform/`
   ```bash
   cd ~/BioPipelines
   mkdir -p nextflow-platform/{bin,src,modules,workflows,containers,config,tests,docs}
   ```
2. **Install Nextflow**: Latest stable (24.x) on login and compute nodes
3. **Configure profiles**: 
   - `slurm.config`: Local SLURM execution
   - `google.config`: Google Batch for burst capacity
4. **GCP setup**: Artifact Registry, IAM permissions, Cloud Storage buckets

**Days 5-7: First Module + AI Agent**
1. **Create reference module**: `modules/qc/fastqc.nf`
2. **Build simple workflow**: Single FastQC run end-to-end
3. **Implement basic AI agent**: 
   - Input: "Run quality control on my fastq files"
   - Output: Generated fastqc.nf workflow
   - Backend: vLLM server on H100
4. **Test**: Submit real FastQ → Nextflow executes → results generated

**Deliverable**: Working proof-of-concept (NL query → GPU AI → Nextflow execution)

---

### Week 2-3: Expand to Complete RNA-seq
1. **Add modules**: STAR alignment, featureCounts, DESeq2, MultiQC
2. **Create containers**: Build Singularity images for each module
3. **Test integration**: Complete RNA-seq pipeline end-to-end
4. **Benchmark**: Compare to Snakemake RNA-seq (speed, accuracy)
5. **Document**: User guide for first pipeline

**CHECKPOINT (Week 3)**: 
- ✓ Can we generate a complete, correct pipeline from NL?
- ✓ Is Nextflow faster/better than Snakemake?
- ✓ Does GPU AI inference work reliably?
- **Decision**: Proceed to Phase 2 or pivot strategy

---

### Communication Plan
**Stakeholder Updates**:
- Week 3: Checkpoint review (POC results)
- Week 6: Tier 1 pipelines demo
- Week 9: AI agents demo (interactive session)
- Week 13: Pre-launch user testing
- Week 14: Launch announcement

**Documentation**:
- Living document: Update this plan as decisions are made
- Decision log: Track what we tried, what worked, what failed
- Weekly progress notes: Keep stakeholders informed

---

### Resource Requirements (Phase 1)

**Compute**:
- 1 H100 node (8 GPUs): AI inference server (dedicated)
- 2-4 CPU nodes: Nextflow testing and development
- 10-20 CPU cores: Parallel module testing

**Storage**:
- 500GB persistent disk: Code, containers, small test data
- 2TB scratch: Large test datasets (ENCODE, TCGA subsets)
- 1TB GCS: Archival of test results and benchmarks

**Personnel**:
- Primary developer: Full-time on Nextflow platform
- Domain expert: Part-time for pipeline validation
- System admin: Support for GCP and GPU setup (as needed)

---

## Conclusion

This new Nextflow-based platform represents a **fundamental architectural shift** from fixed pipelines to **AI-driven, dynamic workflow generation**, leveraging Google Cloud's H100 GPU infrastructure for on-premise AI inference. Key advantages:

1. **GPU-Accelerated AI**: Self-hosted LLMs on H100s (no API costs, data privacy, <500ms latency)
2. **User-Centric**: Pipelines tailored to research questions, not forced into templates
3. **Cloud-Native GCP**: Native integration with Google Batch, Cloud Storage, lifecycle management
4. **Parallel Systems**: Coexists with Snakemake - users choose best tool for their needs
5. **Scalable**: Modular architecture supports infinite pipeline combinations (10 users/week → 100s)
6. **Intelligent**: Multi-agent system with learning from usage patterns
7. **Maintainable**: Small, tested modules following nf-core best practices
8. **Quality-Focused**: 14-week thoughtful development over rushed 10-week sprint

### Strategic Positioning

**Current State (Snakemake)**:
- 10 fixed pipelines, fully containerized
- Works well for standard analyses
- Difficult to customize or extend
- Manual parameter tuning required

**Future State (Nextflow + AI)**:
- Infinite custom pipelines from natural language
- AI-optimized tool selection and parameters
- Self-service for researchers (reduced bioinformatics bottleneck)
- Cloud-ready for scale-out when needed

**Transition Strategy**:
- No disruption: Snakemake continues production use
- Gradual adoption: Early adopters test Nextflow platform
- Natural migration: Users choose superior system over time
- No forced timeline: Quality and user satisfaction drive adoption

### Success Metrics (6-Month Post-Launch)

**Technical Excellence**:
- 80%+ AI-generated pipelines run successfully without modification
- 95%+ pipeline resumption success after failures
- Results match or exceed Snakemake quality (validated on benchmarks)
- <1 hour from query to running pipeline (including AI generation)

**User Adoption**:
- 10 active users/week generating custom pipelines
- 80%+ user satisfaction score
- 50%+ of new projects choose Nextflow over Snakemake
- 20+ unique pipeline types generated (vs 10 fixed Snakemake)

**Operational Efficiency**:
- 30%+ reduction in bioinformatics support time (self-service)
- 20%+ cost reduction through GCS lifecycle management
- Zero data loss incidents
- 99%+ system uptime

### Risk Mitigation

**Technical Risks**: Addressed via checkpoints at weeks 3, 9, 13 (go/no-go decisions)
**User Adoption**: Parallel systems reduce risk, no forced migration
**GPU Availability**: Dedicated H100 node for AI, no competition with compute jobs
**Data Privacy**: Self-hosted AI keeps all data on-premise (no cloud APIs)
**Maintenance Burden**: Modular design and comprehensive docs reduce long-term costs

### Next Decision Points

1. **Week 3**: Is GPU AI + Nextflow viable? (Technical validation)
2. **Week 9**: Are AI agents good enough? (Quality gate: 80% accuracy)
3. **Week 13**: Ready for production? (User acceptance testing)
4. **Week 20**: Scale up or pivot? (Based on 6 weeks of real usage)

**Recommendation**: Proceed with 14-week phased development, building the best possible system at our own pace. Quality and user trust are more valuable than speed.

---

**Document Status**: Living Document - Update as implementation progresses  
**Last Updated**: November 23, 2025  
**Next Review**: After Week 3 Checkpoint (GPU + Nextflow POC)  
**Approved By**: Development team (stakeholder sign-off pending)

---

## Appendix: Quick Reference

### Repository Structure
```
BioPipelines/
├── pipelines/              # Current Snakemake (keep as-is)
├── containers/             # Shared containers (both systems)
├── nextflow-platform/      # NEW - Nextflow + AI system
│   ├── bin/nfp            # Main CLI
│   ├── modules/           # Reusable Nextflow modules
│   ├── workflows/         # Complete pipelines
│   └── src/agents/        # AI agent implementations
└── data/                  # Shared data (raw, references, results)
```

### Key Commands (Future)
```bash
# Plan a pipeline
nfp plan "RNA-seq differential expression, tumor vs normal"

# Generate Nextflow code
nfp generate design.json --output rnaseq.nf

# Execute pipeline
nfp run rnaseq.nf samples.csv --profile slurm

# Monitor progress
nfp status <run-id>

# Interactive mode
nfp chat
> I have 50 samples, what pipeline should I use?
```

### Technology Choices Summary
| Component | Technology | Why |
|-----------|-----------|-----|
| Workflow Engine | Nextflow DSL2 | GCP-native, nf-core ecosystem |
| AI Inference | vLLM on H100 | Self-hosted, <500ms latency |
| Model | Llama 3.3 70B | Open source, strong reasoning |
| Container | Singularity/Apptainer | HPC-friendly, SLURM compatible |
| Storage | GCS Standard/Nearline/Coldline | Tiered costs, lifecycle automation |
| Scheduling | SLURM + Google Batch | Local + cloud burst capacity |
| Monitoring | Nextflow Tower | Real-time tracking, resource viz |
