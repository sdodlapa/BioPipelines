# Nextflow Pipelines - BioPipelines Platform

Modern, modular bioinformatics workflows built with Nextflow DSL2.

## Project Status

**Current Phase**: Phase 1 - Foundation (Weeks 1-4)  
**Goal**: Translate RNA-seq pipeline from Snakemake to Nextflow  
**Target**: Production-ready RNA-seq workflow with validation

---

## Quick Start

### Prerequisites
- Nextflow 24.x installed
- Singularity/Apptainer available
- Access to SLURM cluster
- Existing Singularity containers from Snakemake system

### Installation

```bash
# Install Nextflow
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/

# Verify installation
nextflow -version

# Clone repository
cd ~/BioPipelines/nextflow-pipelines
```

### Running a Pipeline

```bash
# RNA-seq example (Phase 1 - In Development)
nextflow run workflows/rnaseq.nf \
  --input samples.csv \
  --genome GRCh38 \
  --outdir results/ \
  -profile slurm
```

---

## Directory Structure

```
nextflow-pipelines/
├── workflows/           # Complete pipeline workflows
│   └── rnaseq.nf       # RNA-seq (Phase 1 - Current)
├── modules/            # Reusable process modules
│   ├── qc/            # FastQC, MultiQC, trimming
│   ├── alignment/     # STAR, BWA, Bowtie2
│   ├── quantification/ # featureCounts, Salmon
│   └── variants/      # GATK, FreeBayes
├── config/            # Configuration files
│   ├── nextflow.config  # Main config
│   ├── slurm.config    # SLURM executor
│   └── containers.config # Container paths
├── bin/               # Helper scripts
├── docs/              # Documentation
└── tests/             # Test data and scripts
```

---

## Development Roadmap

### ✅ Phase 1: Foundation (Weeks 1-4) - CURRENT
- **Week 1**: Setup & Learning
  - [x] Install Nextflow
  - [x] Create directory structure
  - [ ] Complete Nextflow training
  - [ ] Configure SLURM executor
  
- **Week 2-3**: RNA-seq Translation
  - [ ] Translate Snakemake rules to Nextflow processes
  - [ ] Reuse existing containers
  - [ ] Test individual modules
  - [ ] Build complete workflow
  
- **Week 4**: Validation
  - [ ] Compare outputs with Snakemake
  - [ ] Benchmark performance
  - [ ] User testing
  - [ ] Document findings

### 🔜 Phase 2: Expansion (Weeks 5-10)
- DNA-seq variant calling
- scRNA-seq (CellRanger)
- Additional pipeline (ChIP-seq/ATAC-seq/Long-read)
- Modular process library

### 🚀 Phase 3: Intelligence (Weeks 11-14)
- AI model selection and testing
- Parameter suggestion assistant
- Integration and refinement

---

## Container Strategy

**Reuse Existing Containers** (Phase 1-2):
- `/home/sdodl001_odu_edu/BioPipelines/containers/images/*.sif`
- 12 proven containers from Snakemake system
- No rebuilding needed initially

**Future** (Phase 3+):
- Modular containers for specific tools
- GCP Artifact Registry integration
- Automated container builds

---

## Configuration

### SLURM Executor
Primary execution environment:
- Queue: Default SLURM partitions
- Resources: Per-process resource requirements
- Resume: Automatic checkpoint recovery

### Data Locations
- Raw data: `/scratch/sdodl001/BioPipelines/data/raw/`
- References: `/scratch/sdodl001/BioPipelines/data/references/`
- Results: `/scratch/sdodl001/BioPipelines/data/results/`

---

## Comparison with Snakemake

| Feature | Snakemake (Current) | Nextflow (New) |
|---------|---------------------|----------------|
| **Language** | Python-based | Groovy-based DSL2 |
| **Parallelization** | Good | Excellent (dataflow) |
| **Cloud Support** | Limited | Native (GCP, AWS, Azure) |
| **Resume** | Checkpoints | Work directory caching |
| **Modularity** | Rules in Snakefile | Separate process modules |
| **Community** | Snakemake-wrappers | nf-core (1000+ pipelines) |

---

## Resources

### Learning Nextflow
- Official Training: https://training.nextflow.io
- Documentation: https://www.nextflow.io/docs/latest/
- nf-core: https://nf-co.re (community pipelines)
- Seqera Labs: https://seqera.io (Nextflow creators)

### Reference Pipelines
- nf-core/rnaseq: https://nf-co.re/rnaseq
- nf-core/sarek: https://nf-co.re/sarek (variant calling)
- nf-core/scrnaseq: https://nf-co.re/scrnaseq

---

## Contributing

This is an internal development project. See `docs/DEVELOPMENT.md` for guidelines.

---

## License

Same as parent BioPipelines project.

---

## Contact

Development Team: BioPipelines Project  
Phase 1 Lead: [Your Name]  
Questions: [Contact Info]

---

**Last Updated**: November 24, 2025  
**Next Milestone**: Week 1 completion (Nextflow setup & training)
