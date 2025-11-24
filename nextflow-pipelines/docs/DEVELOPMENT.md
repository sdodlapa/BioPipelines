# Development Guidelines - Nextflow Pipelines

**Project**: BioPipelines Nextflow Migration  
**Strategy**: Phased approach - Foundation → Expansion → Intelligence  
**Current Phase**: Phase 1 (Weeks 1-4)

---

## Project Philosophy

### Why This Phased Approach?

**Original Risk**: GPU + AI + Nextflow simultaneously  
- 3 unknown technologies at once
- High complexity, high failure risk
- Difficult to debug when things break

**Current Strategy**: Validate incrementally  
- Phase 1: Prove Nextflow works (no AI)
- Phase 2: Build pipeline library (no AI)
- Phase 3: Add AI intelligence (informed by real usage)

**Rationale**:
- Lower risk, faster validation
- Learn Nextflow deeply before adding complexity
- Choose AI models based on observed needs, not assumptions
- Easier to debug (single technology at a time)

---

## Code Standards

### DSL2 Module Structure

All modules follow nf-core conventions:

```
modules/
└── category/
    └── tool/
        └── main.nf
```

**Example**: `modules/qc/fastqc/main.nf`

```groovy
#!/usr/bin/env nextflow

process FASTQC {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.zip") , emit: zip
    path "versions.yml"            , emit: versions
    
    script:
    """
    fastqc --threads $task.cpus --quiet $reads
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastqc: \$(fastqc --version | sed 's/FastQC v//')
    END_VERSIONS
    """
}
```

### Key Conventions

1. **Process Names**: UPPERCASE (e.g., `FASTQC`, `STAR_ALIGN`)
2. **Meta Map**: First element in tuple for sample metadata
3. **Tag Directive**: Show sample ID in logs (`tag "$meta.id"`)
4. **Labels**: Resource categories (`process_low`, `process_medium`, `process_high`)
5. **Emit Names**: Named outputs for clarity (`.html`, `.zip`)
6. **Version Tracking**: YAML file with tool versions

### Workflow Structure

```groovy
#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Include modules
include { FASTQC } from '../modules/qc/fastqc/main'
include { STAR_ALIGN } from '../modules/alignment/star/main'

// Parameters
params.input = null
params.genome = null
params.outdir = './results'

// Workflow
workflow {
    // Input validation
    if (!params.input) error "Missing --input"
    if (!params.genome) error "Missing --genome"
    
    // Create channels
    input_ch = Channel
        .fromPath(params.input)
        .splitCsv(header:true)
        .map { row -> create_input_tuple(row) }
    
    // Run processes
    FASTQC(input_ch)
    STAR_ALIGN(FASTQC.out.reads, params.genome)
    
    // Publish results
    STAR_ALIGN.out.bam
        .collectFile(name: 'alignment_summary.txt')
}

// Helper functions
def create_input_tuple(row) {
    def meta = [id: row.sample, single_end: false]
    def reads = [file(row.fastq_1), file(row.fastq_2)]
    return [meta, reads]
}

// Completion message
workflow.onComplete {
    println "Completed at: $workflow.complete"
    println "Duration: $workflow.duration"
    println "Status: ${workflow.success ? 'SUCCESS' : 'FAILED'}"
}
```

---

## Container Strategy

### Phase 1-2: Reuse Existing Containers

**Location**: `/home/sdodl001_odu_edu/BioPipelines/containers/images/`

**Available Containers**:
- `rna-seq_1.0.0.sif` (1.9GB) - STAR, featureCounts, DESeq2
- `dna-seq_1.0.0.sif` (2.8GB) - BWA, GATK, samtools
- `scrna-seq_1.0.0.sif` (2.6GB) - CellRanger, Scanpy
- `atac-seq_1.0.0.sif` (1.7GB) - Bowtie2, MACS2, HOMER
- `chip-seq_1.0.0.sif` (1.6GB) - Bowtie2, MACS2, deepTools
- `long-read_1.0.0.sif` (1.5GB) - Minimap2, NanoPlot, Flye
- `hic_1.0.0.sif` (1.8GB) - HiCExplorer, cooler, chromosight
- `methylation_1.0.0.sif` (2.0GB) - Bismark, MethylDackel
- `metagenomics_1.0.0.sif` (3.2GB) - Kraken2, MetaPhlAn
- `structural-variants_1.0.0.sif` (1.4GB) - SURVIVOR, Manta, Lumpy

**Configuration** (`config/containers.config`):
```groovy
params {
    containers {
        rnaseq = "${params.container_base}/rna-seq_1.0.0.sif"
        dnaseq = "${params.container_base}/dna-seq_1.0.0.sif"
        // etc.
    }
}
```

**Usage in Modules**:
```groovy
process STAR_ALIGN {
    container "${params.containers.rnaseq}"
    // ...
}
```

### Phase 3+: Modular Containers (Future)

When performance requires tool-specific containers:
- One tool per container (e.g., `star-2.7.10a.sif`)
- Automated builds via GitHub Actions
- Store in GCP Artifact Registry
- Version pinning for reproducibility

---

## Testing Strategy

### Unit Tests (Module Level)

Test each module in isolation:

```bash
# Test FastQC module
nextflow run tests/test_fastqc.nf

# Test STAR module
nextflow run tests/test_star.nf
```

**Test File Example** (`tests/test_fastqc.nf`):
```groovy
#!/usr/bin/env nextflow

include { FASTQC } from '../modules/qc/fastqc/main'

workflow {
    input_ch = Channel.of([
        [id: 'test_sample', single_end: false],
        [
            file('data/sample_R1.fastq.gz'),
            file('data/sample_R2.fastq.gz')
        ]
    ])
    
    FASTQC(input_ch)
    
    // Verify outputs exist
    FASTQC.out.html
        .map { meta, html -> 
            assert html.exists()
            return html
        }
        .view { "✅ FastQC HTML: $it" }
}
```

### Integration Tests (Workflow Level)

Test complete workflows:

```bash
# RNA-seq end-to-end test (small test dataset)
nextflow run workflows/rnaseq.nf \
    --input tests/data/rnaseq_test.csv \
    --genome GRCh38 \
    --outdir results/test \
    -profile test
```

### Validation Tests (Phase Checkpoints)

Compare Nextflow vs Snakemake outputs:

```bash
# Generate outputs from both systems
snakemake -s pipelines/rna-seq/Snakefile --configfile config.yaml
nextflow run workflows/rnaseq.nf --input samples.csv --genome GRCh38

# Compare results
diff -r results/snakemake/ results/nextflow/
md5sum results/snakemake/counts.txt results/nextflow/counts.txt
```

**Acceptance Criteria**:
- Count matrices must be identical (MD5 match)
- BAM files may differ slightly (different tools) but same alignment rate
- Plots/reports visual inspection for equivalence

---

## Resource Management

### Process Labels

Define in `config/base.config`:

```groovy
process {
    withLabel: process_low {
        cpus = 2
        memory = 4.GB
        time = 1.h
    }
    withLabel: process_medium {
        cpus = 4
        memory = 8.GB
        time = 4.h
    }
    withLabel: process_high {
        cpus = 8
        memory = 16.GB
        time = 12.h
    }
    withLabel: process_long {
        time = 48.h
    }
}
```

### Process-Specific Overrides

When needed:

```groovy
process {
    withName: 'STAR_ALIGN' {
        cpus = 12
        memory = 48.GB
        time = 6.h
    }
    withName: 'CELLRANGER_COUNT' {
        cpus = 16
        memory = 64.GB
        time = 12.h
    }
}
```

### Dynamic Resources

Scale based on input:

```groovy
process STAR_ALIGN {
    cpus { 8 * task.attempt }
    memory { 32.GB * task.attempt }
    time { 6.h * task.attempt }
    errorStrategy 'retry'
    maxRetries 3
    
    // ...
}
```

---

## Error Handling

### Retry on Transient Failures

```groovy
process {
    // Retry on specific exit codes (OOM, timeout, etc.)
    errorStrategy = { task.exitStatus in [143,137,104,134,139] ? 'retry' : 'finish' }
    maxRetries = 3
}
```

### Ignore Optional Steps

```groovy
process OPTIONAL_QC {
    errorStrategy 'ignore'  // Continue pipeline even if this fails
    // ...
}
```

### Custom Error Messages

```groovy
workflow {
    if (!params.input) {
        error """
        ERROR: Missing required parameter --input
        
        Usage:
          nextflow run workflows/rnaseq.nf --input samples.csv --genome GRCh38
        
        See docs/USAGE.md for details.
        """
    }
}
```

---

## Documentation Standards

### Module Documentation

Each module includes inline comments:

```groovy
process STAR_ALIGN {
    // Align RNA-seq reads to reference genome using STAR
    //
    // Input:
    //   meta: Map with sample metadata (id, single_end, etc.)
    //   reads: List of FASTQ files [R1, R2]
    //   index: Path to STAR genome index directory
    //
    // Output:
    //   bam: Aligned BAM file (coordinate sorted)
    //   log: STAR alignment log with mapping statistics
    //
    // Resources:
    //   Label: process_high (8 CPUs, 16 GB RAM, 12h)
    //   Container: rna-seq_1.0.0.sif (includes STAR 2.7.10a)
    
    tag "$meta.id"
    label 'process_high'
    // ...
}
```

### Workflow Documentation

Each workflow has README:

```markdown
# RNA-seq Workflow

## Overview
Bulk RNA-seq pipeline: QC → Alignment → Quantification → Differential Expression

## Usage
\`\`\`bash
nextflow run workflows/rnaseq.nf \\
    --input samples.csv \\
    --genome GRCh38 \\
    --outdir results/
\`\`\`

## Input Format (samples.csv)
\`\`\`csv
sample,fastq_1,fastq_2,condition
WT_rep1,/path/to/WT_rep1_R1.fastq.gz,/path/to/WT_rep1_R2.fastq.gz,WT
WT_rep2,/path/to/WT_rep2_R1.fastq.gz,/path/to/WT_rep2_R2.fastq.gz,WT
KO_rep1,/path/to/KO_rep1_R1.fastq.gz,/path/to/KO_rep1_R2.fastq.gz,KO
KO_rep2,/path/to/KO_rep2_R1.fastq.gz,/path/to/KO_rep2_R2.fastq.gz,KO
\`\`\`

## Outputs
- `fastqc/`: Quality control reports
- `star/`: Aligned BAM files
- `counts/`: Gene count matrix
- `deseq2/`: Differential expression results
- `multiqc/`: Aggregated QC report

## Parameters
- `--input`: CSV file with sample information (required)
- `--genome`: Reference genome (GRCh38, GRCm39) (required)
- `--outdir`: Output directory (default: ./results)
- `--single_end`: Single-end mode (default: false)

## Requirements
- Nextflow >= 24.0.0
- Singularity/Apptainer
- SLURM cluster

## Runtime
~2-4 hours for 8 samples (20M reads each) on 8 CPUs
```

---

## Git Workflow

### Branch Strategy

- `main`: Stable, production-ready code
- `dev`: Active development (Phase 1-2-3)
- `feature/*`: Individual features (e.g., `feature/rnaseq-workflow`)

### Commit Messages

```
type(scope): short description

Longer explanation if needed.

- Bullet points for details
- Reference issues: #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code restructure (no behavior change)
- `chore`: Maintenance (dependencies, configs)

**Examples**:
```
feat(rnaseq): add STAR alignment module

- Created modules/alignment/star/main.nf
- Supports paired-end and single-end reads
- Uses rna-seq_1.0.0.sif container
- Closes #45

test(fastqc): add unit test for FastQC module

- Created tests/test_fastqc.nf
- Validates HTML and ZIP outputs
```

### Pull Requests (If Multi-Developer)

1. Create feature branch: `git checkout -b feature/my-feature`
2. Implement and test locally
3. Commit with clear messages
4. Push: `git push origin feature/my-feature`
5. Create PR with description and test results
6. Code review
7. Merge to `dev` after approval

---

## Phase Checkpoints

### Phase 1 Completion Criteria (Week 4)

**Must Have**:
- ✅ Nextflow installed and configured
- ✅ RNA-seq workflow complete (QC → Align → Count → DESeq2)
- ✅ Outputs match Snakemake version (MD5 validation)
- ✅ All modules tested independently
- ✅ Documentation complete

**Validation**:
- Run both Snakemake and Nextflow on same data
- Compare outputs (must be identical)
- User testing: 1-2 researchers run workflow
- Performance benchmark: speed, resource usage

**Decision**:
- ✅ PASS → Proceed to Phase 2
- ❌ FAIL → Debug issues, extend Phase 1

### Phase 2 Completion Criteria (Week 10)

**Must Have**:
- ✅ DNA-seq workflow (BWA + GATK variant calling)
- ✅ scRNA-seq workflow (CellRanger + Scanpy)
- ✅ One additional workflow (user choice)
- ✅ Modular process library (20+ reusable modules)
- ✅ User documentation (tutorials, examples)

**Validation**:
- All workflows tested on real data
- User feedback from 3-5 researchers
- Performance acceptable for production
- Module reuse demonstrated across workflows

**Decision**:
- ✅ PASS → Proceed to Phase 3 (AI)
- ❌ FAIL → Extend Phase 2, improve workflows

### Phase 3 Completion Criteria (Week 14)

**Must Have**:
- ✅ AI model selected (open source)
- ✅ Parameter suggestion assistant working
- ✅ Human-in-loop approval workflow
- ✅ Beta testing with 3-5 users
- ✅ AI integration documented

**Validation**:
- AI suggestions are helpful (user survey)
- No incorrect suggestions in critical parameters
- Performance overhead acceptable (<10% slower)
- Users prefer AI-assisted vs manual parameter tuning

**Decision**:
- ✅ PASS → Production deployment
- ❌ FAIL → Improve AI or deploy without AI

---

## Common Patterns

### Processing Multiple Samples

```groovy
workflow {
    // Read sample sheet
    samples_ch = Channel
        .fromPath(params.input)
        .splitCsv(header:true)
        .map { row -> create_sample_tuple(row) }
    
    // Process each sample
    FASTQC(samples_ch)
    ALIGN(FASTQC.out.reads, params.genome)
    
    // Aggregate results
    MULTIQC(ALIGN.out.stats.collect())
}
```

### Conditional Processing

```groovy
workflow {
    // Quality filter reads
    FASTQC(input_ch)
    
    // Trim only if quality is low
    FASTQC.out.qc_stats
        .branch {
            good: it.mean_quality > 30
            bad: it.mean_quality <= 30
        }
        .set { qc_results }
    
    // Good quality: skip trimming
    // Bad quality: trim first
    trimmed_ch = qc_results.bad | TRIMMING
    all_reads_ch = qc_results.good.mix(trimmed_ch)
    
    ALIGN(all_reads_ch, params.genome)
}
```

### Grouping by Metadata

```groovy
workflow {
    // Process samples
    ALIGN(input_ch, params.genome)
    
    // Group by condition for differential expression
    grouped_ch = ALIGN.out.counts
        .map { meta, counts -> [meta.condition, meta, counts] }
        .groupTuple(by: 0)
    
    // Run DE analysis per condition pair
    DESEQ2(grouped_ch)
}
```

---

## Performance Tips

### 1. Use `resume`

Always run with `-resume` during development:

```bash
nextflow run workflow.nf -resume
```

Reuses cached results from previous runs.

### 2. Optimize Channels

Avoid unnecessary file copying:

```groovy
// Good: Use file() or path() for existing files
input_ch = Channel.fromPath('/scratch/data/*.fastq.gz')

// Bad: Don't stage files unnecessarily
// Nextflow will stage inputs automatically
```

### 3. Parallelize Wisely

```groovy
// Good: Parallel processing per sample
samples_ch.map { sample -> PROCESS(sample) }

// Bad: Sequential processing
samples_ch.each { sample -> PROCESS(sample) }
```

### 4. Clean Work Directory

```bash
# After successful runs
nextflow clean -f

# Or set retention policy in config
cleanup = true
```

---

## Debugging Tips

### 1. Check Work Directory

```bash
# Find failed task
ls -lt work/*/* | head -20

# Inspect failed task
cd work/a1/b2c3d4e5f6...
cat .command.sh      # Command executed
cat .command.log     # Output
cat .command.err     # Errors
cat .exitcode        # Exit code
```

### 2. Run Interactively

```bash
# Get into failed task environment
cd work/a1/b2c3d4e5f6...
bash .command.sh  # Run command directly
```

### 3. Enable Debug Logging

```bash
nextflow run workflow.nf -debug
```

### 4. Check Configuration

```bash
# Show merged config
nextflow config

# Show specific profile
nextflow config -profile slurm
```

---

## Phase Transition Guidelines

### When to Move to Next Phase?

**DO NOT** proceed if:
- ❌ Current phase checklist incomplete
- ❌ Validation tests failing
- ❌ User testing reveals major issues
- ❌ Performance significantly worse than Snakemake

**PROCEED** if:
- ✅ All phase criteria met
- ✅ Validation tests passing
- ✅ Users satisfied with functionality
- ✅ Performance acceptable for production

### Gap Between Phases

Allow 1-2 weeks between phases for:
- User testing and feedback collection
- Bug fixes and refinements
- Documentation improvements
- Team training

---

## Contact & Support

- **Project Lead**: BioPipelines Team
- **Questions**: Open GitHub issue or Slack #nextflow-migration
- **Training**: See `docs/WEEK1_GUIDE.md` for onboarding

---

**Last Updated**: November 24, 2025  
**Phase**: Phase 1 - Foundation  
**Next Milestone**: Week 1 completion (Nextflow setup & training)
