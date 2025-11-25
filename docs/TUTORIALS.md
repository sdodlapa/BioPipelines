# BioPipelines Tutorials

## Tutorial 1: Your First RNA-seq Workflow

This tutorial walks through creating a complete RNA-seq differential expression workflow.

### Prerequisites

- BioPipelines installed
- Python 3.8+
- Nextflow installed (for running workflows)

### Step 1: Import and Initialize

```python
from src.workflow_composer import Composer

# Initialize composer (uses default config)
composer = Composer()

# Check available providers
from src.workflow_composer.llm import check_providers
available = check_providers()
print(f"Available LLM providers: {available}")
```

### Step 2: Describe Your Analysis

```python
# Natural language description
description = """
RNA-seq differential expression analysis:
- Organism: Mouse (Mus musculus)
- Genome: GRCm39
- Data: Paired-end Illumina reads
- Comparison: Treatment vs Control (3 replicates each)
- Steps needed:
  1. Quality control with FastQC
  2. Adapter trimming with fastp
  3. Alignment with STAR to mouse genome
  4. Read counting with featureCounts
  5. Differential expression with DESeq2
  6. Generate MultiQC report
"""
```

### Step 3: Generate the Workflow

```python
# Generate workflow
workflow = composer.generate(
    description,
    output_dir="tutorials/rnaseq_de/"
)

print(f"Workflow generated: {workflow.name}")
print(f"Tools used: {workflow.tools}")
```

### Step 4: Review Generated Files

```
tutorials/rnaseq_de/
├── main.nf           # Main workflow
├── nextflow.config   # Configuration
├── params.yaml       # Default parameters
└── modules/          # Symlinked modules
```

### Step 5: Customize Parameters

Edit `params.yaml`:

```yaml
# Input/Output
input: "samplesheet.csv"
outdir: "results"

# Reference
genome: "GRCm39"
gtf: "/path/to/Mus_musculus.GRCm39.109.gtf"
star_index: "/path/to/star_index/"

# Analysis options
strandedness: "reverse"
min_reads: 1000000
fdr_cutoff: 0.05
```

### Step 6: Create Sample Sheet

Create `samplesheet.csv`:

```csv
sample,fastq_1,fastq_2,condition
control_1,data/control_1_R1.fq.gz,data/control_1_R2.fq.gz,control
control_2,data/control_2_R1.fq.gz,data/control_2_R2.fq.gz,control
control_3,data/control_3_R1.fq.gz,data/control_3_R2.fq.gz,control
treatment_1,data/treatment_1_R1.fq.gz,data/treatment_1_R2.fq.gz,treatment
treatment_2,data/treatment_2_R1.fq.gz,data/treatment_2_R2.fq.gz,treatment
treatment_3,data/treatment_3_R1.fq.gz,data/treatment_3_R2.fq.gz,treatment
```

### Step 7: Run the Workflow

```bash
# Local execution
nextflow run tutorials/rnaseq_de/main.nf \
  -profile singularity \
  --input samplesheet.csv \
  --outdir results/

# On SLURM cluster
nextflow run tutorials/rnaseq_de/main.nf \
  -profile slurm,singularity \
  --input samplesheet.csv \
  --outdir results/
```

### Step 8: View Results

```
results/
├── fastqc/           # QC reports
├── fastp/            # Trimming logs
├── star/             # Alignments
├── featurecounts/    # Count matrices
├── deseq2/           # DE results
│   ├── normalized_counts.tsv
│   ├── differential_expression.tsv
│   ├── ma_plot.pdf
│   └── volcano_plot.pdf
└── multiqc/
    └── multiqc_report.html
```

---

## Tutorial 2: ChIP-seq Peak Calling

### Goal
Identify H3K4me3 peaks in human cells with input control.

### Generate Workflow

```python
from src.workflow_composer import Composer

composer = Composer()

workflow = composer.generate("""
ChIP-seq peak calling analysis:
- Human samples (GRCh38)
- Histone mark: H3K4me3 (narrow peaks)
- Single-end 50bp reads
- Have input control samples
- Steps:
  1. FastQC quality control
  2. Trim adapters with Trim Galore
  3. Align with Bowtie2
  4. Remove duplicates
  5. Call peaks with MACS2
  6. Annotate peaks with HOMER
  7. Generate signal tracks with deepTools
""", output_dir="tutorials/chipseq_peaks/")
```

### Sample Sheet Format

```csv
sample,fastq_1,antibody,control
H3K4me3_rep1,chip_rep1.fq.gz,H3K4me3,input_rep1
H3K4me3_rep2,chip_rep2.fq.gz,H3K4me3,input_rep2
input_rep1,input_rep1.fq.gz,input,
input_rep2,input_rep2.fq.gz,input,
```

### Run

```bash
nextflow run tutorials/chipseq_peaks/main.nf \
  -profile singularity \
  --input samplesheet.csv \
  --genome GRCh38
```

---

## Tutorial 3: Variant Calling Pipeline

### Goal
Germline variant calling from whole genome sequencing data.

### Generate Workflow

```python
workflow = composer.generate("""
WGS germline variant calling:
- Human samples, GRCh38 reference
- Paired-end 150bp Illumina
- Pipeline:
  1. FastQC on raw reads
  2. fastp trimming
  3. BWA-MEM alignment
  4. Picard MarkDuplicates
  5. GATK BaseRecalibrator (BQSR)
  6. GATK HaplotypeCaller
  7. GATK VariantFiltration
  8. Variant annotation (optional)
  9. MultiQC summary
""", output_dir="tutorials/wgs_variants/")
```

### Required References

```yaml
# params.yaml
genome: "GRCh38"
fasta: "/references/GRCh38/genome.fa"
known_sites:
  - "/references/GRCh38/dbsnp_146.hg38.vcf.gz"
  - "/references/GRCh38/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz"
intervals: "/references/GRCh38/wgs_calling_regions.interval_list"
```

---

## Tutorial 4: Single-cell RNA-seq

### Goal
Process 10X Genomics single-cell data.

### Generate Workflow

```python
workflow = composer.generate("""
10X Genomics single-cell RNA-seq analysis:
- Human PBMC samples
- 10X Genomics 3' v3 chemistry
- Steps:
  1. Run STARsolo for alignment and counting
  2. Quality control (cell filtering)
  3. Normalization and scaling
  4. Dimensionality reduction (PCA, UMAP)
  5. Clustering
  6. Marker gene identification
  7. Cell type annotation
""", output_dir="tutorials/scrna_10x/")
```

### Sample Sheet

```csv
sample,fastq_dir,expected_cells
pbmc_sample1,/data/pbmc1/,5000
pbmc_sample2,/data/pbmc2/,5000
```

---

## Tutorial 5: Metagenomics Analysis

### Goal
Taxonomic profiling of microbiome samples.

### Generate Workflow

```python
workflow = composer.generate("""
Shotgun metagenomics analysis:
- Human gut microbiome samples
- Paired-end Illumina data
- Pipeline:
  1. fastp quality control and trimming
  2. Host (human) read removal with Bowtie2
  3. Taxonomic classification with Kraken2
  4. Abundance estimation with Bracken
  5. Functional profiling with HUMAnN3
  6. Diversity analysis
  7. MultiQC report
""", output_dir="tutorials/metagenomics/")
```

### Required Databases

```yaml
kraken2_db: "/databases/kraken2/k2_standard/"
bracken_db: "/databases/bracken/"
human_genome: "/references/GRCh38/genome.fa"
```

---

## Tutorial 6: Using the CLI

The `biocomposer` command-line tool provides quick access to workflow generation.

### Generate Workflow

```bash
# Basic generation
biocomposer generate "RNA-seq DE analysis for mouse"

# With output directory
biocomposer generate "ChIP-seq peak calling" --output chipseq_workflow/

# Using specific LLM provider
biocomposer generate "WGS variant calling" --provider openai
```

### Search Tools

```bash
# Search all tools
biocomposer tools --search "alignment"

# Filter by container
biocomposer tools --search "variant" --container dna-seq

# List all tools in container
biocomposer tools --container rna-seq --limit 50
```

### List Modules

```bash
# All modules
biocomposer modules

# By category
biocomposer modules --category alignment
biocomposer modules --category variant_calling
```

### Interactive Chat

```bash
biocomposer chat

# Chat session example:
> I need to analyze ATAC-seq data
Assistant: I can help you create an ATAC-seq workflow. What organism 
are you working with, and do you need peak calling, differential 
accessibility analysis, or both?

> Mouse samples, I need both peak calling and differential analysis
Assistant: I'll create a workflow with:
1. FastQC for quality control
2. Trim Galore for adapter trimming
3. Bowtie2 for alignment
4. Picard for duplicate removal
5. MACS2 for peak calling
6. DiffBind for differential accessibility
7. deepTools for signal visualization

Shall I generate this workflow?

> yes
[Workflow generated and saved to atac_mouse_workflow/]
```

---

## Tutorial 7: Customizing Workflows

### Modifying Generated Workflows

After generation, you can customize the workflow:

#### Add a New Step

Edit `main.nf` to add a process:

```nextflow
// Add after existing imports
include { CUSTOM_ANALYSIS } from './modules/custom/main.nf'

// Add to workflow
workflow {
    // ... existing steps ...
    
    CUSTOM_ANALYSIS(previous_output)
}
```

#### Change Parameters

Edit `nextflow.config`:

```nextflow
params {
    // Modify defaults
    min_reads = 500000
    
    // Add new parameters
    custom_threshold = 0.01
}

process {
    // Change resource allocation
    withName: 'STAR_ALIGN' {
        cpus = 16
        memory = '64.GB'
    }
}
```

### Creating Custom Modules

Create a new module in `nextflow-pipelines/modules/`:

```nextflow
// modules/custom/myanalysis/main.nf
process MY_ANALYSIS {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(input_file)
    
    output:
    tuple val(meta), path("*.results.txt"), emit: results
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    my_tool --input $input_file --output ${prefix}.results.txt
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mytool: \$(my_tool --version)
    END_VERSIONS
    """
}
```

---

## Tutorial 8: Running on HPC

### SLURM Configuration

The generated workflows include SLURM profiles:

```bash
# Run with SLURM
nextflow run main.nf -profile slurm,singularity

# With custom config
nextflow run main.nf -profile slurm,singularity -c my_cluster.config
```

### Custom Cluster Config

Create `my_cluster.config`:

```nextflow
process {
    executor = 'slurm'
    queue = 'normal'
    
    withLabel: 'process_low' {
        cpus = 2
        memory = '8.GB'
        time = '4.h'
    }
    
    withLabel: 'process_medium' {
        cpus = 8
        memory = '32.GB'
        time = '12.h'
    }
    
    withLabel: 'process_high' {
        cpus = 16
        memory = '64.GB'
        time = '24.h'
        queue = 'large'
    }
}

singularity {
    enabled = true
    autoMounts = true
    cacheDir = '/scratch/singularity_cache'
}
```

### Monitor Jobs

```bash
# Nextflow monitoring
nextflow log

# View running processes
squeue -u $USER

# Resume failed run
nextflow run main.nf -resume
```

---

## Troubleshooting

### Common Issues

**1. "LLM provider not available"**
```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Or set OpenAI key
export OPENAI_API_KEY="sk-..."
```

**2. "Module not found for tool X"**
```python
# Check available modules
from src.workflow_composer.core import ModuleMapper
mapper = ModuleMapper("nextflow-pipelines/modules")
print(mapper.list_modules())

# Check aliases
print(mapper.TOOL_ALIASES)
```

**3. "Container not available"**
```bash
# List available containers
ls containers/images/

# Build missing container
cd containers/rna-seq
singularity build rna-seq.sif Singularity.def
```

**4. "Nextflow execution failed"**
```bash
# Check logs
cat .nextflow.log

# Run with debug
nextflow run main.nf -with-trace -with-report

# Resume from failure
nextflow run main.nf -resume
```
