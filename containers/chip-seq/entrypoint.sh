#!/bin/bash
set -euo pipefail

# ChIP-seq Pipeline Entrypoint

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    cat << EOF
ChIP-seq Pipeline Container

Usage:
  singularity run chip-seq.sif --input <input_dir> --output <output_dir> [options]

Required Arguments:
  --input DIR       Directory containing FASTQ files
  --output DIR      Output directory for results
  --reference FILE  Reference genome FASTA file

Optional Arguments:
  --control FILE    Input/control sample (optional)
  --threads N       Number of threads (default: 4)
  --memory GB       Memory limit in GB (default: 16)
  --peak-caller     Peak caller: macs2, homer (default: macs2)
  --genome SIZE     Genome size for MACS2 (e.g., hs, mm, 2.7e9)
  --help            Show this help message

Examples:
  # ChIP-seq with control
  singularity run chip-seq.sif --input /data/chip --output /results --reference hg38.fa --control /data/input.bam

  # Without control
  singularity run chip-seq.sif --input /data/chip --output /results --reference hg38.fa --genome hs

Environment Variables:
  PIPELINE_NAME:    chip-seq
  PIPELINE_VERSION: 1.0.0
EOF
    exit 0
fi

exec /opt/biopipelines/run_pipeline.sh "$@"
