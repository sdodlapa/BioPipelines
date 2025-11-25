# Archived Snakemake Pipelines

⚠️ **ARCHIVED** - These are the original Snakemake pipelines, kept for reference.

## Current Pipelines

The active bioinformatics pipelines are now in **Nextflow DSL2** format:

```
nextflow-pipelines/
├── workflows/      # Main pipeline workflows (rnaseq.nf, chipseq.nf, etc.)
├── modules/        # Reusable process modules
├── config/         # Configuration files
└── bin/           # Helper scripts
```

## Why Archived?

- Nextflow provides better support for:
  - Container orchestration (Singularity/Docker)
  - Cloud execution (GCP, AWS)
  - HPC integration (SLURM)
  - Workflow resumption
  - nf-core compatibility

## Using the New Pipelines

```bash
# Example: Run ChIP-seq pipeline
cd nextflow-pipelines
nextflow run workflows/chipseq.nf -profile slurm,singularity

# Example: Run RNA-seq pipeline  
nextflow run workflows/rnaseq_simple.nf -profile slurm,singularity
```

## Archived Date

November 25, 2025
