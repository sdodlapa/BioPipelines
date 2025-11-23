#!/bin/bash
# Master script to build all pipeline containers in parallel

set -euo pipefail

echo "════════════════════════════════════════════════════════"
echo "Submitting Container Build Jobs"
echo "Start time: $(date)"
echo "════════════════════════════════════════════════════════"

SCRIPT_DIR="$HOME/BioPipelines/scripts/containers"
declare -a JOB_IDS

# Submit all build jobs
for pipeline in dna-seq chip-seq atac-seq hic long-read metagenomics methylation scrna-seq structural-variants; do
    if [[ -f "$SCRIPT_DIR/build_${pipeline//-/_}_container.slurm" ]]; then
        JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/build_${pipeline//-/_}_container.slurm")
        JOB_IDS+=($JOB_ID)
        echo "✓ Submitted $pipeline build job: $JOB_ID"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "Submitted ${#JOB_IDS[@]} container build jobs"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs: ls -lht ~/BioPipelines/logs/build_*"
echo "════════════════════════════════════════════════════════"
