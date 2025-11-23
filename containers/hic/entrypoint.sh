#!/bin/bash
set -euo pipefail

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Hi-C Pipeline Container v1.0.0"
    echo "Usage: singularity run hic.sif --input <dir> --output <dir> --reference <file>"
    exit 0
fi

exec /opt/biopipelines/run_pipeline.sh "$@"
