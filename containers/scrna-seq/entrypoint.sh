#!/bin/bash
set -euo pipefail

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "scRNA-seq Pipeline Container v1.0.0"
    echo "Usage: singularity run scrna-seq.sif --input <dir> --output <dir>"
    exit 0
fi

exec /opt/biopipelines/run_pipeline.sh "$@"
