#!/bin/bash
set -euo pipefail

# ATAC-seq Pipeline Entrypoint

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    cat << EOF
ATAC-seq Pipeline Container

Usage:
  singularity run atac-seq.sif --input <input_dir> --output <output_dir> [options]

Required Arguments:
  --input DIR       Directory containing FASTQ files
  --output DIR      Output directory for results
  --reference FILE  Reference genome FASTA file

Optional Arguments:
  --threads N       Number of threads (default: 4)
  --memory GB       Memory limit in GB (default: 16)
  --peak-caller     Peak caller: macs2, genrich (default: macs2)
  --genome SIZE     Genome size (e.g., hs, mm, 2.7e9)
  --shift-size N    Tn5 shift size (default: auto)
  --help            Show this help message

Examples:
  # Basic ATAC-seq analysis
  singularity run atac-seq.sif --input /data/atac --output /results --reference hg38.fa

  # With custom peak caller
  singularity run atac-seq.sif --input /data/atac --output /results --reference hg38.fa --peak-caller genrich

Environment Variables:
  PIPELINE_NAME:    atac-seq
  PIPELINE_VERSION: 1.0.0
EOF
    exit 0
fi

exec /opt/biopipelines/run_pipeline.sh "$@"
