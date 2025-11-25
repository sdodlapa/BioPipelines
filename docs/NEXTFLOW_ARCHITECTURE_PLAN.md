# Nextflow Pipeline Architecture Plan

**Date**: November 25, 2025 (Updated - Strategic Reset)  
**Purpose**: Design a new AI-driven, container-based bioinformatics platform using Nextflow  
**Status**: Phase 2 Reset - Leverage Existing Assets, Focus on Composition  
**Environment**: Google Cloud HPC (8x H100 80GB GPUs per node)  
**Target Users**: ~10 users/week at rollout  
**Development Philosophy**: Use what works, build what's missing  
**Critical Update**: Pivoted from building new containers to using existing 12 containers + dynamic workflow composition

---

## 1. Executive Summary

### What Changed (November 25, 2025 - Strategic Reset)

**The Failed Approach** (November 23-24):
- Attempted to build new "Tier 2" container architecture from scratch
- Source compilation of tools (STAR, BWA, MACS2, etc.)
- 10+ container build failures over 2 days
- Exit code 127, timeout issues, dependency problems
- **Result**: Zero working containers built

**The Discovery**:
- 12 comprehensive containers **already exist** in containers/images/
- Each contains 500-700 conda-installed, tested tools
- Covers 95% of common bioinformatics workflows
- ~20 GB total (reasonable size)
- Already referenced in Nextflow configurations

**The Realization**:
We were solving the **wrong problem**:
- ❌ Trying to build perfect containers
- ❌ Reinventing what already works
- ❌ Container construction as focus

**What users actually need**:
- ✅ Dynamic workflow composition ("I want STAR + featureCounts")
- ✅ Tool-level granularity (mix any tools)
- ✅ Fast response (minutes, not hours)
- ✅ No bioinformatics expertise required

**The Solution**:
1. **Use existing 12 containers** - Stop building new ones
2. **Create module library** - One module per tool (star.nf, bwa.nf, etc.)
3. **Build AI composer** - Translate requests → Nextflow workflows
4. **Enable customization** - Support user scripts via overlays

**Fundamental Change**:
- **Before**: Container-first architecture (build containers, then use them)
- **After**: Composition-first architecture (use existing containers, compose flexibly)

---

### Vision
Build a **modern, AI-driven bioinformatics platform** using Nextflow DSL2 for workflow orchestration with **existing containerized tools**. Enable dynamic workflow composition where users describe their analysis in natural language and the system automatically generates, optimizes, and executes the appropriate pipeline. Leverage 12 existing comprehensive containers (covering 95% of common tools) rather than building new ones.

### Strategic Pivot (November 25, 2025)

**Original Plan**: Build new multi-tier container architecture (Tier 1: base, Tier 2: domain modules, Tier 3: custom)
- **Problem**: Spent 2 days failing to build containers (10+ failures, source compilation too fragile)
- **Discovery**: 12 working containers already exist with 500-700 tools each
- **Realization**: We're solving the wrong problem - users need workflow composition, not perfect containers

**New Plan**: Dynamic workflow composition using existing containers
- **Foundation**: Use 12 existing containers (rna-seq, dna-seq, chip-seq, etc.) - ~20GB total, comprehensive coverage
- **Module Library**: Create tool-specific Nextflow modules (star.nf, bwa.nf, etc.) that reference existing containers
- **AI Composer**: Build system to translate user requests → Nextflow workflows by composing modules
- **Custom Integration**: Support user scripts via overlays/extensions when needed

### Key Differentiators from Current System
- **AI-Driven Composition**: "I want STAR + featureCounts" → automatic workflow generation
- **Tool-Level Granularity**: Compose workflows from individual tools, not pre-built pipelines
- **Existing Container Leverage**: Use proven 12-container library (~500-700 tools each, conda-based, reliable)
- **Module Library**: Reusable Nextflow processes enable infinite workflow combinations
- **Dynamic Assembly**: Generate custom pipelines on-demand, no pre-programming needed
- **Nextflow DSL2**: Modern workflow language with better parallelization and cloud integration
- **Cloud-Native**: Native GCP integration with Google Batch and Cloud Storage

## 2. Core Infrastructure (What We Have vs What We Need)

### 2.1 Existing Assets ✅

**12 Comprehensive Containers** (Already Built, Already Working)
```
Location: /home/sdodl001_odu_edu/BioPipelines/containers/images/

rna-seq_1.0.0.sif        (1.7 GB, 643 tools)
├── STAR 2.7.11a, HISAT2 2.2.1 (alignment)
├── Salmon 1.10.3, featureCounts 2.0.6, HTSeq (quantification)
├── DESeq2, EdgeR (differential expression)
├── FastQC, MultiQC, Picard (QC)
└── 636 other conda-installed tools

dna-seq_1.0.0.sif        (2.1 GB, ~600 tools)
├── BWA, Bowtie2 (alignment)
├── GATK, FreeBayes, DeepVariant (variant calling)
├── bcftools, VEP, SnpEff (annotation)
└── Full variant analysis pipeline

chipseq_1.0.0.sif        (1.8 GB, ~550 tools)
├── Bowtie2, BWA (alignment)
├── MACS2, HOMER (peak calling)
├── deepTools, bedtools (analysis)
└── ChIP/ATAC-seq complete toolkit

scrna-seq_1.0.0.sif      (2.0 GB, ~700 tools)
├── STARsolo, Alevin (quantification)
├── Seurat, Scanpy (analysis)
├── Monocle3 (trajectory)
└── Single-cell complete ecosystem

... plus 8 more covering:
- ATAC-seq, Hi-C, long-read, metagenomics
- methylation, structural variants
- Base and workflow-engine containers

Total: ~20 GB, 5000-7000 tools, all conda-based, all tested
```

**Nextflow Platform** (Validated in Phase 1)
- ✅ 7-10 concurrent workflows tested successfully
- ✅ DSL2 module architecture established  
- ✅ SLURM integration working
- ✅ Cloud-ready (GCP compatible)
- ✅ Configuration system (containers.config)

**Module Library Foundation**
- Location: `/nextflow-pipelines/modules/`
- Initial modules: qc/, alignment/, quantification/
- Pattern: One process per module, parameterized, tested
- Ready to expand to 30-50 modules

### 2.2 What We Need to Build 🔨

**Tool Catalog** (Week 2 - Priority 1)
```json
{
  "catalog_version": "1.0.0",
  "tools": {
    "STAR": {
      "container": "rna-seq_1.0.0.sif",
      "version": "2.7.11a",
      "module": "modules/alignment/star.nf",
      "category": "alignment",
      "data_types": ["rna-seq"],
      "input": "FASTQ",
      "output": "BAM"
    },
    "featureCounts": {
      "container": "rna-seq_1.0.0.sif",
      "version": "2.0.6",
      "module": "modules/quantification/featurecounts.nf",
      "category": "quantification",
      "data_types": ["rna-seq"],
      "input": "BAM",
      "output": "counts"
    }
    // ... 5000+ tools mapped
  }
}
```

**Module Library Expansion** (Weeks 2-3 - Priority 1)
```
Target: 30-50 tool-specific modules

Priority 1 (Week 2):
✓ alignment/: star.nf, hisat2.nf, bwa.nf, bowtie2.nf, salmon.nf
✓ quantification/: featurecounts.nf, htseq.nf, rsem.nf, kallisto.nf
✓ qc/: fastqc.nf, multiqc.nf, picard.nf
✓ peaks/: macs2.nf, homer.nf

Priority 2 (Week 3):
✓ variants/: gatk.nf, freebayes.nf, bcftools.nf
✓ annotation/: vep.nf, snpeff.nf
✓ scrna/: starsolo.nf, seurat.nf, scanpy.nf
✓ assembly/: spades.nf, flye.nf
```

**AI Workflow Composer** (Weeks 4-5 - Core Feature)
```python
class WorkflowComposer:
    """Translate natural language → Nextflow workflow"""
    
    def compose(self, user_request: str) -> WorkflowExecution:
        # 1. Parse intent
        intent = self.intent_parser.parse(user_request)
        # Example: {"task": "rnaseq", "tools": ["STAR", "featureCounts"]}
        
        # 2. Select modules from catalog
        modules = self.tool_catalog.find_modules(intent['tools'])
        
        # 3. Generate Nextflow workflow
        workflow_code = self.workflow_generator.generate(modules, intent)
        
        # 4. Validate workflow
        self.validator.check(workflow_code)
        
        # 5. Execute
        return self.executor.run(workflow_code, intent['parameters'])
```

### Strategic Goals (Revised - Focus on Composition, Not Construction)

**Phase 1 - Validation (Week 1) ✅ COMPLETE**:
1. ✅ **Validated Nextflow**: Proved better than Snakemake (7 concurrent workflows, no locking)
2. ✅ **Learned Platform**: Mastered DSL2, modules, executors, cloud integration
3. ✅ **Translated Pipelines**: 8-10 Nextflow workflows from Snakemake
4. ✅ **Discovered Assets**: 12 existing containers with comprehensive tool coverage

**Phase 2 - Composition Infrastructure (Weeks 2-3) 🔄 REVISED**:
1. **Tool Catalog**: Inventory 12 containers, map 5000+ tools to containers and categories
2. **Module Library**: Create 30-50 tool-specific modules (star.nf, bwa.nf, featurecounts.nf, etc.)
3. **Test Workflows**: Build 10+ example workflows by composing modules
4. **Documentation**: User guide for manual composition, module templates

**Phase 3 - AI Integration (Weeks 4-6) 🎯 CORE FEATURE**:
1. **Intent Parser**: NLP system to understand user requests
2. **Module Selector**: Choose appropriate modules based on task and tools requested
3. **Workflow Generator**: Compose Nextflow code from selected modules
4. **Execution Engine**: Run generated workflows, monitor progress, return results
5. **Iteration Support**: Allow parameter tuning, tool swapping, comparison studies

**Phase 4 - Custom Integration (Weeks 7-8) 🚀 ADVANCED**:
1. **Overlay System**: Mount user scripts into existing containers (30 seconds)
2. **Extension Builder**: Add pip/conda packages to containers (2-5 minutes)
3. **Custom Containers**: Build from scratch only when necessary (10-30 minutes)
4. **Smart Caching**: Reuse custom containers across users and sessions
7. **AI Model Selection**: Evaluate open source LLMs based on real needs
8. **Parameter Assistant**: AI suggests optimal settings (not generates code)
9. **User Experience**: Simple CLI with optional AI assistance

---

## 2. Architecture Overview

### 2.1 System Components (Revised - Phased)

```
PHASE 1: Foundation (Weeks 1-4)
┌─────────────────────────────────────────────────────────────┐
│                  NEXTFLOW WORKFLOW ENGINE                    │
│  - DSL2 Pipeline Translation from Snakemake                 │
│  - Executor: SLURM (primary)                                │
│  - Resume/Cache: Automatic checkpoint recovery             │
│  - Container: Reuse existing Singularity images            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             EXISTING CONTAINER LIBRARY (Reuse)               │
│  - 12 pipeline containers already built                     │
│  - Proven tools: STAR, GATK, CellRanger, etc.              │
│  - Immediate compatibility                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA MANAGEMENT (Existing)                      │
│  - Storage: /scratch (fast), /home (persistent)             │
│  - Same structure as Snakemake pipelines                    │
└─────────────────────────────────────────────────────────────┘

PHASE 2: Expansion (Weeks 5-10)
┌─────────────────────────────────────────────────────────────┐
│                MODULAR PROCESS LIBRARY                       │
│  - qc/: FastQC, MultiQC, Trimming                          │
│  - alignment/: STAR, BWA, Bowtie2                          │
│  - quantification/: featureCounts, Salmon                   │
│  - variants/: GATK, FreeBayes, Annotation                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               CLOUD INTEGRATION (Optional)                   │
│  - GCS: Results archival                                   │
│  - Google Batch: Burst capacity                            │
└─────────────────────────────────────────────────────────────┘

PHASE 3: Intelligence (Weeks 11+)
┌─────────────────────────────────────────────────────────────┐
│              AI PARAMETER ASSISTANT (Future)                 │
│  - Open source LLM (model TBD based on needs)              │
│  - Parameter suggestion, not code generation                │
│  - Human-in-loop approval                                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack (Revised + Container Strategy)

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Workflow Engine** | Nextflow 24.x (DSL2) - **NATIVE** | Industry standard, excellent GCP support, active nf-core community |
| **Container Runtime** | Singularity/Apptainer | HPC-friendly, rootless, works with SLURM - **Multi-tier architecture** |
| **Container Strategy** | Base + Modules + Microservices + JIT | Dynamic generation, 75% storage reduction vs monolithic |
| **Container Cache** | /scratch/container_cache (~500 GB) | Local cache with TTL-based cleanup, shared read-only |
| **Container Builds** | SLURM compute nodes (fakeroot) | Isolated builds, parallel execution, 3-30 min per container |
| **Scheduling** | SLURM (primary) + Google Batch (future) | Current HPC scheduler, cloud burst optional |
| **Data Storage** | /scratch (hot) + /home (persistent) + GCS (future) | Existing storage, add tiering later |
| **Programming** | Nextflow DSL2 + Bash/Python scripts | Workflow definition + existing tool scripts |
| **Configuration** | nextflow.config + params.yaml | Native Nextflow configuration |
| **Monitoring** | SLURM logs + Nextflow reports | Built-in, no additional tools needed initially |
| **AI/LLM** | **Phase 3 Decision** | Evaluate open source models (Llama, Qwen, Mixtral) after Nextflow validated |
| **AI Container Builder** | Python agents + Singularity | Automated container generation based on workflow needs |

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
│   │   ├── validator.py          # Pipeline validation agent
│   │   ├── container_strategy.py  # Container tier selection (NEW)
│   │   └── container_builder.py   # Container build orchestration (NEW)
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
│       ├── validators.py         # Input validation
│       ├── container_cache.py    # Container cache management (NEW)
│       └── container_ttl.py      # TTL enforcement & promotion (NEW)
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
├── containers/                    # Container definitions & build scripts
│   ├── base/
│   │   └── base.def              # Tier 1: 2 GB foundation (Python, R, samtools)
│   ├── modules/                  # Tier 2: Domain-specific modules
│   │   ├── alignment_short_read.def  # STAR, Bowtie2, BWA, Salmon
│   │   ├── alignment_long_read.def   # Minimap2, NGMLR
│   │   ├── variant_calling.def       # GATK, FreeBayes, VEP
│   │   ├── peak_calling.def          # MACS2, MACS3, HOMER
│   │   ├── assembly.def              # SPAdes, Velvet, Canu
│   │   ├── quantification.def        # featureCounts, HTSeq, Kallisto
│   │   └── scrna_analysis.def        # CellRanger, Seurat, Scanpy
│   ├── microservices/            # Tier 3B: Single-tool containers
│   │   ├── templates/
│   │   │   ├── bioconda_tool.def     # Template for Bioconda tools
│   │   │   └── custom_script.def     # Template for user scripts
│   │   └── README.md             # How to create microservices
│   ├── overlays/                 # Tier 3A: Overlay filesystem configs
│   │   └── overlay_template.def  # Template for version overlays
│   ├── custom/                   # Tier 3C: User JIT builds (runtime only)
│   │   └── .gitkeep              # Not tracked, generated on-demand
│   ├── build_container.sh        # SLURM job script for container builds
│   └── images/                   # Legacy monolithic containers (migration)
│       └── .gitkeep
│
├── cache/                         # Container cache (symlink to /scratch)
│   └── README.md                 # Points to /scratch/container_cache/
│
# Actual container cache on SLURM (not in repo):
# /scratch/container_cache/
#   ├── base/
#   │   └── foundation_v1.0.sif             # 2 GB, pre-built
#   ├── modules/
#   │   ├── alignment_short_read_v1.0.sif   # 8 GB, pre-built
#   │   ├── alignment_long_read_v1.0.sif    # 4 GB, pre-built
#   │   ├── variant_calling_v1.0.sif        # 8 GB, pre-built
#   │   └── ... (~40 GB total, 10 modules)
#   ├── overlays/
#   │   ├── user1/
#   │   │   └── 20250115_star_2.7.11b.sif   # 50 MB, TTL=30 days
#   │   └── user2/
#   │       └── 20250114_salmon_1.10.sif    # 100 MB, TTL=30 days
#   ├── microservices/
#   │   ├── star_2.7.11b_abc123.sif         # 400 MB, TTL=30 days
#   │   └── kallisto_0.50.1_def456.sif      # 300 MB, TTL=30 days
#   └── custom/
#       ├── user1/
#       │   └── job_12345_custom_norm.sif   # 800 MB, TTL=7 days, private
#       └── shared/
#           └── validated_tool_xyz.sif      # 600 MB, opt-in shared
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

## 5. Implementation Roadmap (REVISED)

**Timeline**: 10-12 week phased development with validation checkpoints  
**Philosophy**: Build working Nextflow foundation FIRST, add AI intelligence LATER  
**Strategy**: Parallel systems - Snakemake continues production while Nextflow develops

### Phase 1: Nextflow Foundation (Weeks 1-4) 🎯 CURRENT FOCUS
**Goal**: Prove Nextflow is viable replacement for Snakemake

**Week 1: Setup & Learning**
- [ ] Install Nextflow 24.x on cluster (login and compute nodes)
- [ ] Complete Nextflow training: https://training.nextflow.io
- [ ] Study nf-core RNA-seq as reference: https://nf-co.re/rnaseq
- [ ] Create directory structure: `nextflow-pipelines/`
- [ ] Configure SLURM executor in `nextflow.config`
- [ ] Test: "Hello World" pipeline on cluster

**Week 2: RNA-seq Translation (Part 1)**
- [ ] Translate Snakemake RNA-seq to Nextflow processes:
  - FastQC module
  - STAR alignment module  
  - featureCounts quantification module
- [ ] Reuse existing containers: `/home/.../containers/images/rna-seq_1.0.0.sif`
- [ ] Test each module independently
- [ ] Document: Module interface contracts (inputs/outputs)

**Week 3: RNA-seq Complete Workflow**
- [ ] Add remaining modules: DESeq2, MultiQC
- [ ] Connect processes into complete workflow
- [ ] Implement publishDir for results
- [ ] Add resume capability testing
- [ ] Run on same test data as Snakemake version

**Week 4: Validation & Comparison**
- [ ] Run Nextflow RNA-seq on full dataset
- [ ] Compare outputs: Nextflow vs Snakemake (must be identical)
- [ ] Benchmark: wall time, CPU hours, ease of debugging
- [ ] User testing: 1-2 researchers try Nextflow version
- [ ] Document: "Nextflow vs Snakemake - RNA-seq Comparison Report"

**Checkpoint Week 4**: 
- ✓ Does Nextflow produce identical results to Snakemake?
- ✓ Is it faster/easier to use?
- ✓ Can users run it without issues?
- **Decision**: Proceed to Phase 2 or pivot?

**Deliverable**: One production-ready Nextflow pipeline (RNA-seq) with comparison report

---

### Phase 2: Module Library + Container Migration (Weeks 5-10)
**Goal**: Build modular process library, translate 3 more pipelines, migrate to multi-tier containers

**Week 5-6: DNA-seq + Container Migration Planning**
- [ ] Translate BWA + GATK workflow from Snakemake
- [ ] Test with legacy `dna-seq.sif` container (monolithic)
- [ ] **NEW**: Design Tier 2 module containers:
  - `alignment_short_read.def`: STAR, Bowtie2, BWA, Salmon (8 GB)
  - `variant_calling.def`: GATK, FreeBayes, VEP, SnpEff (8 GB)
  - `peak_calling.def`: MACS2, MACS3, HOMER (3 GB)
- [ ] **NEW**: Test Singularity fakeroot builds on SLURM compute nodes
- [ ] Extract reusable modules: bwa_align, gatk_haplotypecaller, variant_annotation

**Week 7-8: scRNA-seq + Container Build Infrastructure**
- [ ] Translate CellRanger workflow
- [ ] Test with legacy `scrna-seq.sif` container
- [ ] **NEW**: Build container infrastructure:
  - Create `containers/build_container.sh` (SLURM job script)
  - Implement container cache at `/scratch/container_cache/`
  - Set up TTL enforcement (daily cron job)
- [ ] **NEW**: Build first Tier 2 modules:
  - `alignment_short_read_v1.0.sif` (test with RNA-seq, DNA-seq)
  - `scrna_analysis_v1.0.sif` (CellRanger, Seurat, Scanpy)
- [ ] Extract reusable modules: cellranger_count, scanpy_qc, seurat_normalize

**Week 9-10: Third Pipeline + Container Migration**
- [ ] Pick based on user demand: ChIP-seq, ATAC-seq, or Long-read
- [ ] Translate using NEW Tier 2 module containers (not monolithic)
- [ ] **NEW**: Complete container migration:
  - Build remaining Tier 2 modules (~10 total, ~40 GB)
  - Update all Nextflow workflows to use module containers
  - Deprecate monolithic containers (keep for 30-day transition)
- [ ] **NEW**: Implement container cache management:
  - `src/utils/container_cache.py`: Cache lookup, metadata storage
  - `src/utils/container_ttl.py`: TTL cleanup, promotion logic
- [ ] Refactor: Identify common modules across all pipelines
- [ ] Create module library: `modules/{qc,alignment,quantification,variants}/`

**Documentation**
- [ ] Module documentation: Each module's purpose, inputs, outputs
- [ ] **NEW**: Container documentation: How to use Tier 1-2, when to build Tier 3
- [ ] User guides: How to run each pipeline type
- [ ] Developer guide: How to add new modules/pipelines, build containers

**Checkpoint Week 10**:
- ✓ Are 4 Nextflow pipelines working in production?
- ✓ Do users prefer Nextflow or Snakemake?
- ✓ Is module library reusable across pipelines?
- ✓ **NEW**: Are Tier 2 containers working (smaller, faster, modular)?
- ✓ **NEW**: Is container build infrastructure reliable (fakeroot builds)?
- **Decision**: Continue to Phase 3 (AI + dynamic containers) or focus on more pipelines?

**Deliverable**: 4 production pipelines, modular process library, Tier 1-2 container cache, build infrastructure

---

### Phase 3: AI Integration + Dynamic Container Generation (Weeks 11-14) 🚀 FUTURE
**Goal**: Add intelligent parameter suggestion + on-demand container builds (Tier 3A-3C)

**Week 11: AI Model Selection & Container Strategy Agent**
- [ ] Test open source models on H100 GPUs:
  - Llama 3.3 70B vs 8B (size vs performance tradeoff)
  - Qwen 2.5 72B vs 32B  
  - Mixtral 8x7B (mixture of experts)
  - DeepSeek-V3 (if available)
- [ ] Benchmark: Latency, accuracy on parameter suggestion tasks
- [ ] Test: "Given RNA-seq data, suggest STAR parameters"
- [ ] **NEW**: Build ContainerStrategyAgent:
  - Input: User query ("Use STAR 2.7.11b", "Compare aligners", "Run my script")
  - Logic: Check cache → Select tier (3A overlay, 3B microservice, 3C JIT)
  - Output: Container path or build specification
- [ ] Choose: Best model for our use case

**Week 12: AI Parameter Assistant + Container Builder Agent**
- [ ] Build CLI tool: `nextflow-assist`
- [ ] Input: User describes data (# samples, organism, read length, etc.)
- [ ] AI Output: Suggested parameter file (nextflow.params.json)
- [ ] **NEW**: Build ContainerBuilderAgent:
  - Generate Singularity definition files
  - Submit SLURM build jobs (`sbatch containers/build_container.sh`)
  - Monitor build progress, validate results
  - Cache with metadata (TTL, usage stats, privacy settings)
- [ ] **NEW**: Implement Tier 3 container builds:
  - **Tier 3A (Overlays)**: Quick version updates (2-3 min builds)
  - **Tier 3B (Microservices)**: Single-tool containers (3-5 min builds)
  - **Tier 3C (Custom JIT)**: User scripts (10-30 min builds, queued)
- [ ] Human review: User confirms/edits before execution

**Week 13: Integration & Testing**
- [ ] Integrate AI assistant with existing pipelines
- [ ] **NEW**: Test dynamic container workflows:
  ```bash
  # Example 1: Version-specific tool
  nextflow-assist describe --pipeline rnaseq --star-version 2.7.11b
  # AI generates overlay container (2-3 min) + parameter file
  nextflow run rnaseq.nf -params-file suggested.json
  
  # Example 2: Tool comparison
  nextflow-assist compare --tools "star,salmon,kallisto"
  # AI creates 3 microservices (3-5 min each) + comparison workflow
  
  # Example 3: Custom script
  nextflow-assist custom --script my_normalization.R --deps "DESeq2,ggplot2"
  # AI builds JIT container (10-30 min, queued) + wraps in Nextflow process
  ```
- [ ] Beta testing with 3-5 users
- [ ] Collect feedback: What's helpful? What's confusing?
- [ ] **NEW**: Test container cache management:
  - TTL cleanup (daily cron)
  - Promotion (popular microservices → Tier 2 modules)
  - Privacy enforcement (user custom scripts stay private)

**Week 14: Refinement & Documentation**
- [ ] Fix issues from beta testing
- [ ] Add "explain" mode: Why AI suggested these parameters/containers
- [ ] **NEW**: Smart queuing for long builds:
  - Quick builds (<5 min): Wait for completion
  - Long builds (>5 min): Queue + notify when ready
  - User choice: `--wait` or `--queue` flag
- [ ] **NEW**: Container sharing workflow:
  - Default: Private containers (user-specific)
  - Opt-in: Share after validation (admin approval)
  - Automated validation: Test imports, version checks, security scan
- [ ] Documentation: How to use AI assistant + dynamic containers, when to override
- [ ] Fallback: All pipelines work WITHOUT AI (always)

**Checkpoint Week 14**:
- ✓ Does AI assistant save time vs manual configuration?
- ✓ Are suggestions accurate (>80%)?
- ✓ Do users trust AI recommendations?
- ✓ **NEW**: Do dynamic containers work reliably (build success >95%)?
- ✓ **NEW**: Are build times acceptable (90% < 10 min)?
- ✓ **NEW**: Is storage usage under budget (140 GB steady-state)?
- **Decision**: Expand AI features or focus elsewhere?

**Deliverable**: AI parameter assistant + dynamic container generation (Tier 3A-3C), smart queuing, cache management

---

### Phase 4: Production Features (Weeks 15-16) - Optional
**Goal**: Enterprise-ready platform

**Cloud Integration** (If needed)
- [ ] Google Batch executor for burst capacity
- [ ] GCS integration for results archival
- [ ] Cost tracking and optimization

**Monitoring** (If helpful)
- [ ] Nextflow Tower for real-time monitoring
- [ ] Automated email alerts on failures
- [ ] Resource utilization dashboards

**User Experience**
- [ ] Sample sheet validators
- [ ] Progress indicators and ETAs
- [ ] Result summaries with key findings

**Deliverable**: Production platform with observability

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

### 6.2 Container Strategy (DECIDED - Multi-Tier Architecture)

> **Status**: APPROVED after comprehensive evaluation (see CRITICAL_EVALUATION.md and DYNAMIC_CONTAINER_STRATEGY.md)
> **Problem Solved**: Static monolithic containers (120 GB) cannot support AI-generated custom workflows requiring version-specific tools, user scripts, or novel tool combinations
> **Result**: 75% storage reduction (140 GB vs 1.2 TB per-user duplication) with infinite flexibility

#### **Multi-Tier Architecture**

**Tier 1: Base Foundation** (2 GB, Pre-built)
- **Purpose**: Core utilities shared across all workflows
- **Contents**: samtools, bedtools, FastQC, Python 3.11, R 4.3, basic plotting
- **Build**: One-time, maintained by admins
- **Location**: `/scratch/container_cache/base/foundation_v1.0.sif`
- **Already exists**: `containers/base/base.def` (validated)

**Tier 2: Domain Modules** (3-8 GB each, ~40 GB total, Pre-built)
- **Purpose**: Grouped tools for common workflows (covers 90% of standard cases)
- **Modules**:
  - `alignment_short_read` (STAR, Bowtie2, BWA, Salmon) - 8 GB
  - `alignment_long_read` (Minimap2, NGMLR) - 4 GB
  - `variant_calling` (GATK, FreeBayes, BCFtools, VEP, SnpEff) - 8 GB
  - `peak_calling` (MACS2, MACS3, HOMER) - 3 GB
  - `assembly` (SPAdes, Velvet, Canu) - 6 GB
  - `quantification` (featureCounts, HTSeq, Kallisto) - 3 GB
  - `scrna_analysis` (CellRanger, Seurat, Scanpy) - 8 GB
- **Build**: Quarterly updates, CI/CD tested
- **Location**: `/scratch/container_cache/modules/{module_name}_v{version}.sif`

**Tier 3A: Overlays** (50-200 MB each, ~25 GB total, AI-Generated)
- **Purpose**: Quick version updates or single-tool additions to modules
- **Build Time**: 2-3 minutes
- **Examples**:
  - User: "Use STAR 2.7.11b not 2.7.10a" → overlay on `alignment_short_read`
  - User: "Add featureCounts" → overlay on existing container
- **TTL**: 30 days (auto-cleanup if not used)
- **Location**: `/scratch/container_cache/overlays/{user}/{timestamp}_{tool}.sif`
- **Technology**: Singularity overlay filesystem (writable layer on base)

**Tier 3B: Microservices** (300-500 MB each, ~30 GB total, AI-Generated)
- **Purpose**: Single-tool containers for comparisons or niche tools
- **Build Time**: 3-5 minutes
- **Examples**:
  - User: "Compare STAR vs Salmon vs Kallisto alignments" → 3 microservices
  - User: "Use obscure_tool from Bioconda" → single microservice
- **TTL**: 30 days (promote to module if popular)
- **Location**: `/scratch/container_cache/microservices/{tool}_{version}_{hash}.sif`
- **Template**: `containers/microservices/templates/tool_template.def`

**Tier 3C: Custom JIT (Just-In-Time)** (500 MB - 2 GB each, ~40 GB total, AI-Generated)
- **Purpose**: User-provided scripts, custom environments, novel pipelines
- **Build Time**: 10-30 minutes (includes compilation, testing)
- **Examples**:
  - User: "Run my custom R normalization script" → JIT with dependencies
  - User: "Python pipeline with specific package versions" → JIT from requirements.txt
- **TTL**: 7 days (private to user)
- **Location**: `/scratch/container_cache/custom/{user}/{job_id}_{hash}.sif`
- **Build**: Singularity fakeroot on SLURM compute nodes

#### **Storage Budget & Efficiency**

```yaml
Storage_Comparison:
  Old_Monolithic:
    Containers: 12 pipeline-specific × 10 GB = 120 GB
    Per_User_Duplication: 120 GB × 10 users = 1.2 TB
    Flexibility: Fixed (cannot customize)
  
  New_Multi_Tier:
    Tier_1_Base: 2 GB (1 container)
    Tier_2_Modules: 40 GB (10 modules, shared)
    Tier_3A_Overlays: 25 GB (50 active overlays)
    Tier_3B_Microservices: 30 GB (60 active microservices)
    Tier_3C_Custom: 40 GB (20 active JIT containers)
    Buffer: 3 GB
    Total_Steady_State: 140 GB (500 GB budget approved)
    
  Result:
    Storage_Savings: 75% (140 GB vs 1.2 TB)
    Flexibility: Infinite (any tool, any version, any combination)
    Users_Supported: 10-100+ (shared cache, no duplication)
```

#### **Build Infrastructure (SLURM-Based)**

**Location**: Compute nodes (not login nodes)
- **Method**: Singularity fakeroot (rootless builds, no sudo required)
- **Parallelization**: 4-8 concurrent builds (SLURM job array)
- **Queuing Strategy**: Smart (Option C from Q3)
  - **Quick builds** (<5 min estimated): Wait for completion
  - **Long builds** (>5 min estimated): Queue job, notify user when ready
  - **User choice**: `--wait` or `--queue` flag to override
- **Build nodes**: `partition=build, nodes=4, cpus-per-task=8, mem=32GB`
- **Cache warming**: Pre-build popular tool versions during off-hours

#### **AI Agent Integration**

**ContainerStrategyAgent**: Analyzes user query → selects optimal tier
- Input: User query ("Use STAR 2.7.11b", "Run my script", "Compare aligners")
- Logic: 
  1. Check cache for exact match → return existing
  2. Check base + overlay feasibility → Tier 3A (2-3 min)
  3. Check microservice template → Tier 3B (3-5 min)
  4. Fall back to JIT → Tier 3C (10-30 min)
- Output: Container path or build job ID

**ContainerBuilderAgent**: Executes builds on SLURM
- Input: Build specification (tool, version, dependencies)
- Actions:
  1. Generate Singularity definition file
  2. Submit SLURM job (`sbatch containers/build_container.sh`)
  3. Monitor build progress
  4. Validate container (test imports, version check)
  5. Cache result with metadata (TTL, usage stats)
- Output: Container path + validation report

**Caching & Promotion**:
- **TTL enforcement**: Daily cron job cleans expired containers
- **Usage tracking**: Track container access frequency
- **Promotion**: Popular Tier 3 containers (>10 users) promoted to Tier 2 modules
- **Privacy**: User's custom scripts stay private (opt-in sharing after validation)

#### **Build Time Estimates**

| Tier | Typical Build Time | User Wait Time | Example |
|------|-------------------|----------------|---------|
| Tier 1 | N/A (pre-built) | 0 seconds | Use base container |
| Tier 2 | N/A (pre-built) | 0 seconds | Use alignment module |
| Tier 3A | 2-3 minutes | 2-3 minutes (wait) | Add STAR 2.7.11b overlay |
| Tier 3B | 3-5 minutes | 3-5 minutes (wait) | Build Kallisto microservice |
| Tier 3C | 10-30 minutes | Queue + notify | Custom R pipeline |

**User Experience**:
- 90% of queries: Use Tier 1-2 (instant, 0 wait)
- 8% of queries: Tier 3A-3B (2-5 min, acceptable wait)
- 2% of queries: Tier 3C (queued, notification when ready)

**Target**: ~20 base containers + unlimited dynamic containers vs previous 12 monolithic

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
| **Container Build Success** | > 95% | **NEW**: Successful builds / total build attempts |
| **Container Build Time** | < 10 min (90%) | **NEW**: Build duration for Tier 3A-3C |
| **Container Cache Hit Rate** | > 80% | **NEW**: Cache hits / total container requests |
| **Storage Budget Compliance** | < 500 GB | **NEW**: Total container cache size |

### 7.2 User Experience Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Time to First Result** | < 5 minutes | Setup → running pipeline |
| **Learning Curve** | < 1 hour | Onboarding → first custom pipeline |
| **User Satisfaction** | 4.5/5 | Post-use survey |
| **Query Success Rate** | > 80% | Natural language → correct pipeline |
| **Documentation Clarity** | 4.5/5 | User feedback |
| **Container Wait Satisfaction** | 4/5 | **NEW**: User feedback on build times + queuing |

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
| **Container build failures** | Medium | Medium | **UPDATED**: Robust fallback: use Tier 1-2 if Tier 3 build fails, retry mechanism, pre-build validation |
| **SLURM incompatibility** | Low | High | Test on cluster early, use nf-core configs |
| **Storage quota exceeded** | Low | High | **UPDATED**: Multi-tier strategy (140 GB steady-state vs 500 GB budget), TTL cleanup, promotion system |
| **Dynamic container security** | Medium | High | **NEW**: Singularity security (no root, namespaces), user script validation, opt-in sharing only |
| **Build queue congestion** | Low | Medium | **NEW**: Smart queuing (quick builds wait, long builds queue), 4-8 parallel builds, off-hours pre-building |
| **Overlay filesystem limits** | Low | Low | **NEW**: Singularity overlays limited to ~500 MB, fall back to microservices for larger changes |

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

**Document Status**: Living Document - Update as implementation progresses  
**Last Updated**: November 24, 2025  
**Next Review**: After Phase 1 Week 4 Checkpoint (8 Nextflow pipelines complete)  
**Major Updates**: Nov 24 - Added multi-tier container strategy, removed obsolete GPU-first section, updated current status  
**Current Status**: Phase 1 Day 2 - 2/10 pipelines complete, 5/10 running, multi-user validated (7 concurrent workflows)  
**Approved By**: Development team (stakeholder sign-off pending)

---

## 12. Implementation Summary & Key Decisions

### 12.1 Strategic Architecture Decisions

**✓ DECIDED: Continue Nextflow (Maintain Snakemake in Parallel)**
- **Rationale**: Comprehensive evaluation (see CRITICAL_EVALUATION.md) proved Nextflow superiority:
  - Multi-user concurrent execution: PROVEN with 7 workflows, 13 jobs, 1h+ runtime
  - DSL2 modularity: Programmatically composable for AI agents (vs file-based includes)
  - Cloud-native design: Google Batch integration, GCS support, future-proof architecture
  - Community ecosystem: nf-core modules, 1000+ contributors, active development
- **Risk Mitigation**: Snakemake stays in production (no disruption), gradual user migration
- **Decision Point**: After Week 4 checkpoint (8 pipelines validated)

**✓ DECIDED: Native Workflow Engine + Containerized Tools**
- **Nextflow Installation**: Native on login nodes (NOT containerized)
  - Simpler deployment (standard HPC pattern)
  - Faster execution (no container overhead for orchestration)
  - Easier debugging (direct access to Nextflow commands)
- **Tool Execution**: Containerized on compute nodes (Singularity)
  - Reproducible environments (version-locked dependencies)
  - Isolated execution (no package conflicts)
  - Portable across systems (dev → staging → production)
- **workflow-engine.sif**: Development/testing only (not for production HPC)

**✓ DECIDED: Multi-Tier Container Strategy**
- **Problem Solved**: Static monolithic containers (120 GB) cannot support:
  - Version-specific tools: "Use STAR 2.7.11b not 2.7.10a"
  - User custom scripts: "Run my normalization algorithm"
  - Tool comparisons: "Compare STAR vs Salmon vs Kallisto"
  - Novel combinations: "RNA-seq + ChIP-seq integration"
- **Solution**: Dynamic 3-tier architecture (see DYNAMIC_CONTAINER_STRATEGY.md):
  - **Tier 1 (Base)**: 2 GB foundation - Python, R, samtools, bedtools
  - **Tier 2 (Modules)**: 40 GB total - 10 domain-specific suites (alignment, variants, etc.)
  - **Tier 3A (Overlays)**: 25 GB - 2-3 min builds, version updates, 30-day TTL
  - **Tier 3B (Microservices)**: 30 GB - 3-5 min builds, single-tool containers, 30-day TTL
  - **Tier 3C (Custom JIT)**: 40 GB - 10-30 min builds, user scripts, 7-day TTL
- **Storage Impact**: 140 GB steady-state (vs 1.2 TB per-user duplication) = **75% reduction**
- **Flexibility Impact**: Infinite combinations (vs 12 fixed pipelines) = **∞ improvement**

### 12.2 Infrastructure & Implementation Decisions

**Q1: Storage Budget - ANSWERED: 500 GB**
- Current monolithic: 120 GB (12 × 10 GB containers)
- New multi-tier: 140 GB steady-state + 360 GB buffer
- Per-user Snakemake: 1.2 TB (10 users × 120 GB)
- Result: **75% storage savings + infinite flexibility**

**Q2: Container Build Location - ANSWERED: SLURM Compute Nodes**
- Method: Singularity fakeroot (rootless builds, no sudo required)
- Parallelization: 4-8 concurrent builds
- Resources: `partition=build, cpus=8, mem=32GB` per build
- Benefit: No load on login nodes, faster builds with more CPU/RAM

**Q3: Build Queuing Strategy - ANSWERED: Option C (Smart Queuing)**
- Quick builds (<5 min estimated): Wait for completion, show progress bar
- Long builds (>5 min estimated): Queue SLURM job, notify when ready
- User override: `--wait` or `--queue` flag to force behavior
- Expected: 90% of builds <10 min, 98% < 30 min (acceptable UX)

**Q4: Container Privacy - ANSWERED: Opt-In Sharing**
- Default: Private containers (user-specific cache)
- Sharing: Opt-in after validation (admin approval required)
- Validation: Test imports, version checks, security scan
- Cache: Shared for common tools (Tier 1-2), private for custom scripts (Tier 3C)

**Build Infrastructure Details**:
- Location: `/scratch/container_cache/` (fast local storage)
- TTL Enforcement: Daily cron job cleans expired containers
- Promotion: Popular Tier 3 containers (>10 users) → Tier 2 modules
- Cache Warming: Pre-build popular tool versions during off-hours

### 12.3 AI Agent Architecture Decisions

**ContainerStrategyAgent**: Selects optimal container tier
- Input: User query analysis (tool requirements, version constraints, custom scripts)
- Logic:
  1. Check cache for exact match → return existing container (instant)
  2. Check base + overlay feasibility → Tier 3A (2-3 min)
  3. Check microservice template → Tier 3B (3-5 min)
  4. Fall back to JIT → Tier 3C (10-30 min, queued)
- Output: Container path or build specification + estimated build time

**ContainerBuilderAgent**: Executes container builds
- Actions:
  1. Generate Singularity definition file (tool-specific template)
  2. Submit SLURM job (`sbatch containers/build_container.sh`)
  3. Monitor build progress (log tailing, status updates)
  4. Validate container (test imports, version check, security scan)
  5. Cache with metadata (TTL, usage stats, privacy settings)
- Fallback: If build fails → use Tier 1-2 base containers → retry with adjustments
- Notification: Email/Slack when long builds complete

**Multi-Agent Coordination**:
- PipelinePlannerAgent: High-level workflow design (Llama 3.3 70B)
- ToolSelectorAgent: Chooses best tools for each step (Qwen 2.5 72B)
- ContainerStrategyAgent: Determines container requirements (rule-based + LLM)
- ContainerBuilderAgent: Builds containers on-demand (orchestration)
- ValidatorAgent: Checks pipeline correctness (rule-based + LLM)

### 12.4 Implementation Timeline & Phases (UPDATED)

**Phase 1: Nextflow Validation (Week 1)** - ✅ COMPLETE
- Goal: Prove Nextflow viability, validate concurrent execution
- Achieved: 7-10 concurrent workflows without locking issues
- Translated: 8-10 pipelines from Snakemake to Nextflow
- Validated: Multi-user architecture works
- Discovered: 12 existing comprehensive containers
- **Outcome**: Nextflow proven superior to Snakemake

**Phase 2: Composition Infrastructure (Weeks 2-3)** - 🔄 IN PROGRESS (REVISED)
- **DISCARDED**: Building Tier 2 container modules (failed approach)
- **NEW APPROACH**: Tool catalog + module library using existing containers
- Week 2 Goals:
  - Create tool catalog (5000+ tools → container mapping)
  - Build 15-20 core modules (STAR, BWA, featureCounts, etc.)
  - Test manual workflow composition
  - Document usage patterns
- Week 3 Goals:
  - Expand to 30-50 modules  
  - Create 10+ example workflows
  - User documentation
  - Validate all existing containers work
- Deliverable: Module library ready for AI integration

**Phase 3: AI Workflow Composer (Weeks 4-6)** - 🎯 CORE FEATURE
- Goal: Enable dynamic workflow generation from natural language
- Week 4: Intent parser + tool selector
  - Parse user requests ("I want STAR + featureCounts")
  - Map to appropriate modules and containers
  - Validate feasibility
- Week 5: Workflow generator + executor
  - Generate Nextflow DSL2 code from modules
  - Parameter optimization
  - Execute and monitor
- Week 6: Testing + iteration
  - Test with 10-20 real user scenarios
  - Refine intent parsing
  - Handle edge cases
- Deliverable: Working AI-driven pipeline generation

**Phase 4: Custom Tool Integration (Weeks 7-8)** - 🚀 ADVANCED
- Goal: Support user scripts and custom tools
- Week 7: Overlay system
  - Mount user scripts into existing containers
  - 30-second turnaround for simple cases
  - Documentation and examples
- Week 8: Extension builder
  - pip/conda extension of containers (2-5 min)
  - Custom container builds (last resort, 10-30 min)
  - Smart caching and reuse
- Deliverable: Complete custom integration system

**Phase 5: Production & Scale (Weeks 9-10)** - 📈 OPTIONAL
- Monitoring (Nextflow Tower, dashboards)
- Cloud burst (Google Batch integration)
- User training and onboarding
- Performance optimization

### 12.5 Expected Outcomes & Benefits

**For Researchers**:
- Self-service pipeline creation (no bioinformatics bottleneck)
- Version-specific tools (test reproducibility with exact versions)
- Custom script integration (run proprietary algorithms)
- Tool comparisons (evaluate multiple methods scientifically)
- Faster iteration (2-5 min container builds vs hours of environment setup)

**For Administrators**:
- 75% storage reduction (140 GB vs 1.2 TB)
- 30% support time reduction (self-service reduces tickets)
- Scalable architecture (10-100+ concurrent users)
- Cloud-ready (future GCP burst capacity)
- Automated maintenance (TTL cleanup, cache warming)

**For Platform**:
- Infinite pipeline flexibility (vs 12 fixed pipelines)
- Multi-user concurrent execution (proven with Nextflow)
- Modular architecture (reusable components)
- AI-driven intelligence (parameter optimization, tool selection)
- Production-ready design (robust error handling, validation, monitoring)

### 12.6 Success Criteria (Phase 3 Completion)

**Technical Excellence**:
- ✓ 8-10 Nextflow pipelines in production (Phase 1)
- ✓ 10 Tier 2 module containers built (Phase 2)
- ✓ 95%+ container build success rate (Phase 3)
- ✓ 90%+ builds complete in <10 min (Phase 3)
- ✓ Storage under 500 GB budget (all phases)

**User Experience**:
- ✓ 80%+ query success rate (AI → correct pipeline)
- ✓ <5 min time to first result (Phase 1-2)
- ✓ 4/5 satisfaction with container wait times (Phase 3)
- ✓ 50%+ of new projects choose Nextflow (Phase 3-4)

**Scientific Quality**:
- ✓ 100% result reproducibility (Nextflow = Snakemake)
- ✓ 100% tool version tracking (container immutability)
- ✓ 100% provenance tracking (Nextflow work/ directory)

### 12.7 Reference Documents

**Core Architecture**:
- `NEXTFLOW_ARCHITECTURE_PLAN.md` (this document) - Master architecture plan
- `CRITICAL_EVALUATION.md` - Snakemake vs Nextflow comparison (14 sections, Nextflow wins 5/5 criteria)
- `DYNAMIC_CONTAINER_STRATEGY.md` - Multi-tier container strategy (Q1-Q4 decisions, AI agents, caching)

**Implementation Guides**:
- `containers/modules/README.md` - How to build Tier 2 modules (Phase 2)
- `containers/microservices/README.md` - Microservice templates (Phase 3)
- `src/agents/README.md` - AI agent implementation guide (Phase 3)

**Status Tracking**:
- `PIPELINE_STATUS_FINAL.md` - Current Snakemake pipeline status
- `nextflow-pipelines/STATUS.md` - Nextflow translation progress (Phase 1)

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
