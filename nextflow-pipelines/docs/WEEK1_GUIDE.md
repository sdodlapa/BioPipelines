# Week 1 Implementation Guide: Nextflow Setup & Learning

**Duration**: 5-7 days  
**Goal**: Install Nextflow, configure SLURM, complete training, create first module  
**Success Criteria**: Running "Hello World" pipeline on SLURM cluster

---

## Day 1: Installation & Environment Setup

### Install Nextflow

```bash
# On login node
cd ~
curl -s https://get.nextflow.io | bash

# Make executable and move to PATH
chmod +x nextflow
sudo mv nextflow /usr/local/bin/

# Verify installation
nextflow -version
# Expected: nextflow version 24.x.x

# Create Nextflow config directory
mkdir -p ~/.nextflow
```

### Configure Nextflow for SLURM

Create `~/.nextflow/config`:

```groovy
// Global Nextflow configuration
process {
    executor = 'slurm'
    queue = 'default'  // Adjust to your partition name
    
    // Default resources
    cpus = 2
    memory = '8 GB'
    time = '2h'
}

// Singularity configuration
singularity {
    enabled = true
    autoMounts = true
    cacheDir = '/scratch/sdodl001/BioPipelines/cache'
}

// Work directory
workDir = '/scratch/sdodl001/BioPipelines/work'

// Resume on failure
resume = true
```

### Test Nextflow Installation

```bash
# Hello World test (local executor)
nextflow run hello

# Expected output:
# N E X T F L O W  ~  version 24.x.x
# Launching `nextflow-io/hello` [...]
# executor >  local (4)
# [d7/d0a26b] process > sayHello (4) [100%] 4 of 4 ✔
# Bonjour world!
# Hello world!
# Hola world!
# Ciao world!
```

**Checkpoint**: If this works, Nextflow is correctly installed.

---

## Day 2-3: Complete Nextflow Training

### Required Training Modules

Work through https://training.nextflow.io in order:

1. **Basic Training** (4-6 hours):
   - Introduction to Nextflow
   - Channels and operators
   - Processes and workflows
   - Configuration files
   - Executors and containers

2. **Advanced Training** (2-4 hours):
   - DSL2 syntax (THIS IS WHAT WE USE)
   - Modules and subworkflows
   - Publishing results
   - Error handling and debugging

### Hands-On Exercises

Create practice scripts in `~/BioPipelines/nextflow-pipelines/tests/`:

#### Exercise 1: Simple Process

```groovy
// test_simple.nf
#!/usr/bin/env nextflow

process sayHello {
    input:
    val name

    output:
    stdout

    script:
    """
    echo "Hello, $name!"
    """
}

workflow {
    names = Channel.of('Alice', 'Bob', 'Charlie')
    sayHello(names) | view
}
```

Run: `nextflow run test_simple.nf`

#### Exercise 2: File Processing

```groovy
// test_files.nf
#!/usr/bin/env nextflow

process countLines {
    input:
    path file

    output:
    stdout

    script:
    """
    wc -l $file
    """
}

workflow {
    files = Channel.fromPath('*.txt')
    countLines(files) | view
}
```

#### Exercise 3: With SLURM

```groovy
// test_slurm.nf
#!/usr/bin/env nextflow

process heavyComputation {
    cpus 4
    memory '16 GB'
    time '10m'
    
    input:
    val x

    output:
    stdout

    script:
    """
    echo "Computing $x with 4 CPUs"
    sleep 30
    echo "Done!"
    """
}

workflow {
    Channel.of(1, 2, 3, 4) | heavyComputation | view
}
```

Run: `nextflow run test_slurm.nf`

Check SLURM: `squeue -u $USER`

**Checkpoint**: Can you submit jobs to SLURM via Nextflow?

---

## Day 4: Study nf-core RNA-seq

### Clone and Examine nf-core/rnaseq

```bash
cd ~/BioPipelines/nextflow-pipelines/docs/
git clone https://github.com/nf-core/rnaseq.git nf-core-rnaseq-reference

cd nf-core-rnaseq-reference
```

### Key Files to Study

1. **workflows/rnaseq.nf**: Main workflow
2. **modules/nf-core/fastqc/**: FastQC module structure
3. **modules/nf-core/star/align/**: STAR alignment module
4. **conf/base.config**: Resource configurations
5. **nextflow.config**: Main configuration

### Take Notes On

- Module structure (main.nf in each module directory)
- Input/output declarations
- Container specifications
- Resource labels (low, medium, high)
- publishDir usage for results
- Meta map pattern for sample information

**Goal**: Understand nf-core conventions (we'll adapt them)

---

## Day 5: Create First Module - FastQC

### Module Structure

Create `modules/qc/fastqc/main.nf`:

```groovy
#!/usr/bin/env nextflow

process FASTQC {
    tag "$meta.id"
    label 'process_medium'
    
    container '/home/sdodl001_odu_edu/BioPipelines/containers/images/rna-seq_1.0.0.sif'
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.zip") , emit: zip
    path "versions.yml"            , emit: versions
    
    script:
    def prefix = "${meta.id}"
    """
    fastqc \\
        --threads $task.cpus \\
        --quiet \\
        $reads
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastqc: \$(fastqc --version | sed 's/FastQC v//')
    END_VERSIONS
    """
}
```

### Test the Module

Create `tests/test_fastqc.nf`:

```groovy
#!/usr/bin/env nextflow

include { FASTQC } from '../modules/qc/fastqc/main'

workflow {
    // Test with RNA-seq sample
    input_ch = Channel.of([
        [id: 'sample1', single_end: false],
        file('/scratch/sdodl001/BioPipelines/data/raw/rna-seq/sample1_1.fastq.gz')
    ])
    
    FASTQC(input_ch)
    
    FASTQC.out.html | view
}
```

Run: `nextflow run tests/test_fastqc.nf`

**Checkpoint**: Does FastQC run successfully and produce HTML report?

---

## Day 6: Create Configuration Files

### Main Configuration

Create `nextflow-pipelines/nextflow.config`:

```groovy
// Main Nextflow configuration for BioPipelines

// Manifest
manifest {
    name            = 'BioPipelines/nextflow-pipelines'
    author          = 'BioPipelines Team'
    homePage        = 'https://github.com/yourorg/BioPipelines'
    description     = 'Modular bioinformatics workflows'
    mainScript      = 'main.nf'
    nextflowVersion = '>=24.0.0'
    version         = '0.1.0'
}

// Include configuration files
includeConfig 'config/base.config'
includeConfig 'config/containers.config'

// Profiles
profiles {
    slurm {
        includeConfig 'config/slurm.config'
    }
    local {
        process.executor = 'local'
        process.cpus = 2
        process.memory = '4 GB'
    }
    test {
        includeConfig 'config/test.config'
    }
}

// Default profile
profiles.default = 'slurm'

// Global defaults
params {
    outdir = './results'
    tracedir = "${params.outdir}/pipeline_info"
    
    // Resource defaults
    max_cpus = 32
    max_memory = '128 GB'
    max_time = '72h'
}

// Reporting
timeline {
    enabled = true
    file = "${params.tracedir}/timeline.html"
}

report {
    enabled = true
    file = "${params.tracedir}/report.html"
}

trace {
    enabled = true
    file = "${params.tracedir}/trace.txt"
}

dag {
    enabled = true
    file = "${params.tracedir}/dag.svg"
}
```

### Base Configuration

Create `config/base.config`:

```groovy
// Base process configuration

process {
    // Error strategy
    errorStrategy = { task.exitStatus in [143,137,104,134,139] ? 'retry' : 'finish' }
    maxRetries = 3
    maxErrors = '-1'
    
    // Resource labels
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

### Container Configuration

Create `config/containers.config`:

```groovy
// Container paths for BioPipelines

singularity {
    enabled = true
    autoMounts = true
    cacheDir = '/scratch/sdodl001/BioPipelines/cache'
}

params {
    container_base = '/home/sdodl001_odu_edu/BioPipelines/containers/images'
    
    containers {
        rnaseq = "${params.container_base}/rna-seq_1.0.0.sif"
        dnaseq = "${params.container_base}/dna-seq_1.0.0.sif"
        scrnaseq = "${params.container_base}/scrna-seq_1.0.0.sif"
        atacseq = "${params.container_base}/atac-seq_1.0.0.sif"
        chipseq = "${params.container_base}/chip-seq_1.0.0.sif"
        longread = "${params.container_base}/long-read_1.0.0.sif"
        hic = "${params.container_base}/hic_1.0.0.sif"
        methylation = "${params.container_base}/methylation_1.0.0.sif"
        metagenomics = "${params.container_base}/metagenomics_1.0.0.sif"
        structural = "${params.container_base}/structural-variants_1.0.0.sif"
    }
}
```

### SLURM Configuration

Create `config/slurm.config`:

```groovy
// SLURM executor configuration

process {
    executor = 'slurm'
    queue = 'default'  // Update with your partition name
    
    // SLURM options
    clusterOptions = '--account=YOUR_ACCOUNT'  // Update if needed
    
    // Job submission settings
    queueSize = 50
    submitRateLimit = '10/1min'
}

executor {
    queueSize = 50
    pollInterval = '30 sec'
}

singularity {
    enabled = true
    runOptions = '--bind /scratch'
}
```

---

## Day 7: Create "Hello Bioinformatics" Test Pipeline

### Complete Test Workflow

Create `workflows/test_hello.nf`:

```groovy
#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Include modules
include { FASTQC } from '../modules/qc/fastqc/main'

// Parameters
params.input = null
params.outdir = './results'

// Workflow
workflow {
    // Check required params
    if (!params.input) {
        error "Please provide --input parameter"
    }
    
    // Create input channel
    // Format: CSV with columns: sample,fastq_1,fastq_2
    input_ch = Channel
        .fromPath(params.input)
        .splitCsv(header:true)
        .map { row ->
            def meta = [id: row.sample, single_end: false]
            def reads = [file(row.fastq_1), file(row.fastq_2)]
            return [meta, reads]
        }
    
    // Run FastQC
    FASTQC(input_ch)
    
    // Publish results
    FASTQC.out.html
        .map { meta, html -> html }
        .collectFile(name: 'fastqc_reports.txt', newLine: true)
        .view { "FastQC reports: $it" }
}

workflow.onComplete {
    println "Pipeline completed at: $workflow.complete"
    println "Execution status: $workflow.success"
    println "Duration: $workflow.duration"
}
```

### Test Input File

Create `tests/samples_test.csv`:

```csv
sample,fastq_1,fastq_2
sample1,/scratch/sdodl001/BioPipelines/data/raw/rna-seq/sample1_1.fastq.gz,/scratch/sdodl001/BioPipelines/data/raw/rna-seq/sample1_2.fastq.gz
```

### Run Test Pipeline

```bash
cd ~/BioPipelines/nextflow-pipelines

nextflow run workflows/test_hello.nf \
    --input tests/samples_test.csv \
    --outdir results/test_hello \
    -profile slurm \
    -resume

# Monitor SLURM jobs
watch -n 5 squeue -u $USER

# Check results
ls -lh results/test_hello/
ls -lh work/  # Temporary work directory
```

### Validate Outputs

```bash
# Check FastQC HTML reports
find results/test_hello -name "*.html" -ls

# Check execution report
firefox results/test_hello/pipeline_info/report.html

# Check timeline
firefox results/test_hello/pipeline_info/timeline.html

# Check trace for resource usage
less results/test_hello/pipeline_info/trace.txt
```

---

## Week 1 Completion Checklist

### ✅ Installation
- [ ] Nextflow installed and on PATH
- [ ] Version 24.x confirmed
- [ ] SLURM executor configured

### ✅ Training
- [ ] Completed basic Nextflow training
- [ ] Completed DSL2 training
- [ ] Understand channels, processes, workflows
- [ ] Understand modules and subworkflows

### ✅ Configuration
- [ ] Created nextflow.config
- [ ] Created base.config (resource labels)
- [ ] Created containers.config (reuse existing SIFs)
- [ ] Created slurm.config (SLURM settings)

### ✅ First Module
- [ ] FastQC module created with proper structure
- [ ] Tested FastQC module in isolation
- [ ] FastQC produces HTML and ZIP outputs

### ✅ Test Pipeline
- [ ] "Hello Bioinformatics" workflow runs successfully
- [ ] Jobs submitted to SLURM correctly
- [ ] Results published to output directory
- [ ] Pipeline reports generated (timeline, report, trace, DAG)

### ✅ Validation
- [ ] Nextflow can resume interrupted runs
- [ ] Singularity containers work with Nextflow
- [ ] Resource limits respected (CPUs, memory)
- [ ] SLURM queue correctly configured

---

## Week 1 Success Criteria

**You have successfully completed Week 1 if:**

1. ✅ Nextflow is installed and running on your cluster
2. ✅ You can submit jobs to SLURM via Nextflow
3. ✅ You understand DSL2 syntax and module structure
4. ✅ You have created and tested your first Nextflow module (FastQC)
5. ✅ You can run a simple multi-sample workflow
6. ✅ Pipeline produces execution reports and trace files

**If all criteria met**: ✅ **PROCEED TO WEEK 2** (RNA-seq translation)

**If criteria not met**: Spend additional 2-3 days on training and debugging

---

## Common Issues & Solutions

### Issue 1: Nextflow Not Found After Install
```bash
# Solution: Add to PATH permanently
echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
source ~/.bashrc
```

### Issue 2: SLURM Jobs Not Submitting
```bash
# Check SLURM config
sinfo  # Should list partitions

# Test manual SLURM submission
sbatch --wrap="echo Hello" --output=test.out

# Check Nextflow config
nextflow config  # Shows merged configuration
```

### Issue 3: Singularity Container Not Found
```bash
# Verify container exists
ls -lh /home/sdodl001_odu_edu/BioPipelines/containers/images/rna-seq_1.0.0.sif

# Test container manually
singularity exec rna-seq_1.0.0.sif fastqc --version
```

### Issue 4: Work Directory Filling Up /scratch
```bash
# Clean old work directories
nextflow clean -f  # Force clean

# Or manually
rm -rf work/
```

---

## Resources for Week 1

### Documentation
- Nextflow Docs: https://www.nextflow.io/docs/latest/
- DSL2 Guide: https://www.nextflow.io/docs/latest/dsl2.html
- SLURM Executor: https://www.nextflow.io/docs/latest/executor.html#slurm

### Training
- Nextflow Training: https://training.nextflow.io
- nf-core Tutorials: https://nf-co.re/docs/usage/tutorials

### Community
- Nextflow Slack: https://nextflow.io/slack-invite.html
- nf-core Slack: https://nf-co.re/join
- Seqera Community Forum: https://community.seqera.io

---

## Next Steps (Week 2 Preview)

Once Week 1 is complete:

1. **Translate Snakemake RNA-seq rules** to Nextflow processes
2. **Create modules**: STAR alignment, featureCounts, MultiQC
3. **Build complete RNA-seq workflow** using modules
4. **Test on same data** as Snakemake version
5. **Compare outputs** (must be identical)

**Goal**: Fully functional RNA-seq pipeline ready for validation in Week 4

---

**Last Updated**: November 24, 2025  
**Status**: Ready for Week 1 Implementation
