#!/bin/bash
#SBATCH --job-name=build_base
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/build_base_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/build_base_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

set -euo pipefail

echo "════════════════════════════════════════════════════════"
echo "Building Base Container"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "════════════════════════════════════════════════════════"

# Configuration
CONTAINER_DIR="/home/sdodl001_odu_edu/BioPipelines/containers/base"
OUTPUT="/home/sdodl001_odu_edu/BioPipelines/containers/images/base_1.0.0.sif"
LOG_FILE="/tmp/base_build_${SLURM_JOB_ID}.log"

# Build container
cd "$CONTAINER_DIR"
echo "Building base container from: $CONTAINER_DIR/base.def"
echo "Output: $OUTPUT"
echo "Log: $LOG_FILE"
echo ""

singularity build --fakeroot "$OUTPUT" base.def 2>&1 | tee "$LOG_FILE"

# Verify build
if [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "✓ Build successful!"
    ls -lh "$OUTPUT"
    
    # Quick validation
    echo ""
    echo "Running validation tests..."
    singularity exec "$OUTPUT" micromamba --version
    singularity exec "$OUTPUT" samtools --version | head -2
    
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "✓ Base container ready: $OUTPUT"
    echo "End time: $(date)"
    echo "════════════════════════════════════════════════════════"
else
    echo "✗ Build failed! Check log: $LOG_FILE"
    exit 1
fi
