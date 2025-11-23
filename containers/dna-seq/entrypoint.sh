#!/bin/bash
set -euo pipefail

# DNA-seq Pipeline Entrypoint
# Handles parameter passing and pipeline execution

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    cat << EOF
DNA-seq Pipeline Container

Usage:
  singularity run dna-seq.sif --input <input_dir> --output <output_dir> [options]

Required Arguments:
  --input DIR       Directory containing FASTQ files
  --output DIR      Output directory for results
  --reference FILE  Reference genome FASTA file

Optional Arguments:
  --threads N       Number of threads (default: 4)
  --memory GB       Memory limit in GB (default: 16)
  --mode MODE       Analysis mode: alignment, variant_calling, full (default: full)
  --known-sites VCF Known variant sites for BQSR (optional)
  --help            Show this help message

Examples:
  # Full DNA-seq analysis
  singularity run dna-seq.sif --input /data/fastq --output /results --reference hg38.fa

  # Alignment only
  singularity run dna-seq.sif --input /data/fastq --output /results --reference hg38.fa --mode alignment

Environment Variables:
  PIPELINE_NAME:    dna-seq
  PIPELINE_VERSION: 1.0.0
EOF
    exit 0
fi

# Execute pipeline script with all arguments
exec /opt/biopipelines/run_pipeline.sh "$@"
