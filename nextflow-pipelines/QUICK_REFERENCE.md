# Week 1 Quick Reference Card

**Goal**: Install Nextflow, configure SLURM, complete training, create first module  
**Duration**: 5-7 days  
**Success**: Running "Hello Bioinformatics" pipeline on SLURM

---

## Installation (30 minutes)

```bash
# Install Nextflow
cd ~
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
nextflow -version  # Should show 24.x

# Configure SLURM
mkdir -p ~/.nextflow
cat > ~/.nextflow/config << 'EOF'
process.executor = 'slurm'
process.queue = 'default'
singularity.enabled = true
singularity.autoMounts = true
singularity.cacheDir = '/scratch/sdodl001/BioPipelines/cache'
workDir = '/scratch/sdodl001/BioPipelines/work'
resume = true
EOF

# Test installation
nextflow run hello  # Should print hello world messages
```

---

## Training (6-10 hours)

**Required**: https://training.nextflow.io

**Focus On**:
- ✅ Channels and operators
- ✅ Processes and scripts
- ✅ **DSL2 syntax** (THIS IS WHAT WE USE)
- ✅ Modules and subworkflows
- ✅ Configuration files
- ✅ Executors (SLURM)

**Hands-On Exercises**: Create in `~/BioPipelines/nextflow-pipelines/tests/`

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

---

## Study nf-core (2-3 hours)

```bash
cd ~/BioPipelines/nextflow-pipelines/docs/
git clone https://github.com/nf-core/rnaseq.git nf-core-rnaseq-reference
cd nf-core-rnaseq-reference
```

**Study These Files**:
- `workflows/rnaseq.nf` - Main workflow structure
- `modules/nf-core/fastqc/` - Module structure
- `modules/nf-core/star/align/` - STAR alignment module
- `conf/base.config` - Resource configurations

**Take Notes On**:
- Module directory structure (main.nf pattern)
- Input/output declarations (meta map pattern)
- Container specifications
- publishDir usage for results

---

## Create FastQC Module (2-4 hours)

Create `~/BioPipelines/nextflow-pipelines/modules/qc/fastqc/main.nf`:

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

**Test It**:

Create `tests/test_fastqc.nf`:

```groovy
#!/usr/bin/env nextflow

include { FASTQC } from '../modules/qc/fastqc/main'

workflow {
    input_ch = Channel.of([
        [id: 'sample1', single_end: false],
        file('/scratch/sdodl001/BioPipelines/data/raw/rna-seq/sample1_1.fastq.gz')
    ])
    
    FASTQC(input_ch)
    FASTQC.out.html | view
}
```

Run: `nextflow run tests/test_fastqc.nf`

---

## Test Pipeline (1-2 hours)

Create `workflows/test_hello.nf`:

```groovy
#!/usr/bin/env nextflow

nextflow.enable.dsl=2

include { FASTQC } from '../modules/qc/fastqc/main'

params.input = null
params.outdir = './results'

workflow {
    if (!params.input) error "Please provide --input parameter"
    
    input_ch = Channel
        .fromPath(params.input)
        .splitCsv(header:true)
        .map { row ->
            def meta = [id: row.sample, single_end: false]
            def reads = [file(row.fastq_1), file(row.fastq_2)]
            return [meta, reads]
        }
    
    FASTQC(input_ch)
    
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

Create `tests/samples_test.csv`:

```csv
sample,fastq_1,fastq_2
sample1,/scratch/sdodl001/BioPipelines/data/raw/rna-seq/sample1_1.fastq.gz,/scratch/sdodl001/BioPipelines/data/raw/rna-seq/sample1_2.fastq.gz
```

**Run Test**:

```bash
cd ~/BioPipelines/nextflow-pipelines

nextflow run workflows/test_hello.nf \
    --input tests/samples_test.csv \
    --outdir results/test_hello \
    -profile slurm \
    -resume

# Monitor
watch -n 5 squeue -u $USER

# Check results
ls -lh results/test_hello/
firefox results/test_hello/pipeline_info/report.html
```

---

## Week 1 Checklist

- [ ] Nextflow installed (`nextflow -version` shows 24.x)
- [ ] SLURM configured (`~/.nextflow/config` exists)
- [ ] Test passed (`nextflow run hello` works)
- [ ] Training completed (basic + DSL2)
- [ ] nf-core/rnaseq studied (notes taken)
- [ ] FastQC module created (`modules/qc/fastqc/main.nf`)
- [ ] FastQC tested (`tests/test_fastqc.nf` runs)
- [ ] Test pipeline runs (`workflows/test_hello.nf` works)
- [ ] SLURM submission works (jobs appear in `squeue`)
- [ ] Reports generated (timeline, report, trace, DAG)

**All checked?** ✅ **PROCEED TO WEEK 2** (RNA-seq translation)

---

## Common Commands

```bash
# Run workflow
nextflow run workflow.nf -profile slurm -resume

# Check configuration
nextflow config
nextflow config -profile slurm

# Clean work directory
nextflow clean -f

# Monitor SLURM jobs
squeue -u $USER
watch -n 5 squeue -u $USER

# Check job details
sacct -j JOBID

# View reports
firefox results/pipeline_info/report.html
firefox results/pipeline_info/timeline.html
less results/pipeline_info/trace.txt

# Debug failed task
cd work/a1/b2c3d4...
cat .command.sh    # Command executed
cat .command.log   # Output
cat .command.err   # Errors
bash .command.sh   # Run interactively
```

---

## Key Concepts

### DSL2 Process Structure
```groovy
process PROCESS_NAME {
    tag "$meta.id"              // Show in logs
    label 'process_medium'      // Resource label
    container 'path/to.sif'     // Singularity container
    
    input:
    tuple val(meta), path(reads)  // Sample info + files
    
    output:
    tuple val(meta), path("*.bam"), emit: bam
    path "versions.yml", emit: versions
    
    script:
    """
    command --input $reads --output output.bam
    """
}
```

### Workflow Structure
```groovy
workflow {
    // Create input channel
    input_ch = Channel.fromPath(params.input)
    
    // Run processes
    PROCESS1(input_ch)
    PROCESS2(PROCESS1.out)
    
    // Publish results
    PROCESS2.out.bam | collectFile
}
```

### Meta Map Pattern
```groovy
// Meta map contains sample metadata
def meta = [
    id: 'sample1',
    single_end: false,
    condition: 'WT'
]

// Always first element in tuple
tuple val(meta), path(reads)
```

---

## Resources

### Documentation
- **Nextflow Docs**: https://www.nextflow.io/docs/latest/
- **DSL2 Guide**: https://www.nextflow.io/docs/latest/dsl2.html
- **SLURM Executor**: https://www.nextflow.io/docs/latest/executor.html#slurm

### Training
- **Nextflow Training**: https://training.nextflow.io ⭐ REQUIRED
- **nf-core Tutorials**: https://nf-co.re/docs/usage/tutorials

### Reference
- **nf-core/rnaseq**: https://nf-co.re/rnaseq
- **nf-core/modules**: https://github.com/nf-core/modules

### Community
- **Nextflow Slack**: https://nextflow.io/slack-invite.html
- **nf-core Slack**: https://nf-co.re/join

---

## Troubleshooting

### Nextflow Not Found
```bash
echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
source ~/.bashrc
```

### SLURM Jobs Not Submitting
```bash
# Check SLURM status
sinfo

# Test manual submission
sbatch --wrap="echo Hello" --output=test.out

# Check Nextflow config
nextflow config
```

### Container Not Found
```bash
# Verify container exists
ls -lh /home/sdodl001_odu_edu/BioPipelines/containers/images/rna-seq_1.0.0.sif

# Test manually
singularity exec rna-seq_1.0.0.sif fastqc --version
```

### Work Directory Full
```bash
nextflow clean -f
# Or manually: rm -rf work/
```

---

## Success Criteria

**You have successfully completed Week 1 if:**

1. ✅ Nextflow is installed and running
2. ✅ You can submit jobs to SLURM via Nextflow
3. ✅ You understand DSL2 syntax and module structure
4. ✅ You have created and tested your first module (FastQC)
5. ✅ You can run a simple multi-sample workflow
6. ✅ Pipeline produces execution reports (timeline, trace, DAG)

**All criteria met?** ✅ **PROCEED TO WEEK 2**

**Not met?** Spend 2-3 more days on training and debugging

---

## Week 2 Preview

**Goal**: Translate Snakemake RNA-seq to Nextflow

**Tasks**:
1. Create STAR alignment module
2. Create featureCounts module
3. Create DESeq2 module
4. Create MultiQC module
5. Build complete RNA-seq workflow
6. Test on same data as Snakemake

**Resource**: See RNA-seq Snakefile at `pipelines/rna-seq/Snakefile`

---

**Detailed Guide**: See `docs/WEEK1_GUIDE.md`  
**Code Standards**: See `docs/DEVELOPMENT.md`  
**Getting Started**: See `GETTING_STARTED.md`

**Good luck! 🚀**
