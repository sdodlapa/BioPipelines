# nf-core/rnaseq Study Notes

**Purpose**: Learn nf-core conventions and RNA-seq module patterns before implementing our own.

---

## Key Files to Study

### 1. Main Workflow
- `workflows/rnaseq.nf` - Main workflow orchestration
- `main.nf` - Entry point with parameter validation

### 2. Module Examples
- `modules/nf-core/fastqc/` - Quality control module
- `modules/nf-core/star/align/` - STAR alignment module
- `modules/nf-core/salmon/quant/` - Quantification module

### 3. Configuration
- `conf/base.config` - Resource configurations
- `conf/modules.config` - Module-specific settings
- `nextflow.config` - Main configuration

### 4. Subworkflows
- `subworkflows/nf-core/` - Reusable workflow components

---

## Study Checklist

### Module Structure Pattern
- [ ] Understand directory structure: `modules/<category>/<tool>/main.nf`
- [ ] Meta map usage: `tuple val(meta), path(reads)`
- [ ] Container specification
- [ ] Input/output declarations with `emit` names
- [ ] Version tracking in YAML format

### Process Patterns
- [ ] `tag` directive for sample IDs
- [ ] `label` for resource management
- [ ] Script blocks with proper variable interpolation
- [ ] Error handling strategies

### Channel Operations
- [ ] Input channel creation from samplesheet
- [ ] Channel transformations (map, filter, branch)
- [ ] Mixing and joining channels
- [ ] Output collection

### Configuration Patterns
- [ ] Process selectors: `withName:`, `withLabel:`
- [ ] Resource profiles (standard, test, etc.)
- [ ] Container paths
- [ ] Parameter validation

---

## Quick Overview

```bash
cd nf-core-rnaseq-reference

# View main workflow
cat workflows/rnaseq.nf | less

# Study FastQC module
cat modules/nf-core/fastqc/main.nf

# Check resource configuration
cat conf/base.config

# See module config
cat conf/modules.config | grep fastqc
```

---

## Notes Section

### FastQC Module Pattern
```groovy
// Location: modules/nf-core/fastqc/main.nf

process FASTQC {
    tag "$meta.id"
    label 'process_medium'
    
    conda "bioconda::fastqc=0.12.1"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/fastqc:0.12.1--hdfd78af_0' :
        'biocontainers/fastqc:0.12.1--hdfd78af_0' }"
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.zip") , emit: zip
    path  "versions.yml"           , emit: versions
    
    script:
    // Process implementation
}
```

**Key Observations**:
1. Meta map always first in tuple
2. Named outputs with `emit:`
3. Version tracking
4. Container specifications for both Singularity and Docker
5. Label for resource management

### Workflow Pattern
```groovy
// From workflows/rnaseq.nf

include { FASTQC } from '../modules/nf-core/fastqc/main'

workflow NFCORE_RNASEQ {
    take:
    ch_samplesheet // channel: samplesheet read in from --input
    
    main:
    ch_versions = Channel.empty()
    
    // QC
    FASTQC ( ch_samplesheet )
    ch_versions = ch_versions.mix(FASTQC.out.versions.first())
    
    // More processes...
    
    emit:
    versions = ch_versions
}
```

**Key Observations**:
1. Named workflows with `take:`, `main:`, `emit:` blocks
2. Version collection pattern
3. Module inclusion at top
4. Clear input/output contracts

---

## Differences from Our Approach

### nf-core:
- Uses conda environments AND containers
- Separate Docker/Singularity container URLs
- Complex parameter validation system
- nf-schema for samplesheet validation
- Multi-profile support (docker, singularity, conda, etc.)

### Our Approach (Simpler):
- Reuse existing Singularity containers
- Single container per pipeline type
- Simple CSV samplesheet
- Focus on SLURM execution
- Fewer profiles (slurm, local, test)

**Decision**: Take module structure and patterns, simplify container and config approach.

---

## Action Items

After studying nf-core/rnaseq:

1. [ ] Create our simplified module template
2. [ ] Design our samplesheet format (CSV with meta columns)
3. [ ] Create FastQC module following nf-core pattern
4. [ ] Create STAR alignment module
5. [ ] Create simple workflow connecting modules
6. [ ] Test with actual RNA-seq data

---

## Questions to Answer

1. How do they handle single-end vs paired-end detection?
   - Answer: Meta map has `single_end: true/false`

2. How do they structure subworkflows?
   - Answer: Reusable components in `subworkflows/`

3. How do they manage publishDir?
   - Answer: Via `conf/modules.config` with process-specific settings

4. How do they collect MultiQC inputs?
   - Answer: Collect all QC outputs into channels, pass to MultiQC

---

**Status**: Week 1 Day 4 - Studying nf-core/rnaseq  
**Next**: Create FastQC module (Day 5)
