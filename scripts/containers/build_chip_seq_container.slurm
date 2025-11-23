#!/bin/bash
#SBATCH --job-name=build_chip_seq
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/build_chip_seq_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/build_chip_seq_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpuspot

set -euo pipefail

echo "════════════════════════════════════════════════════════"
echo "Building ChIP-seq Container"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "════════════════════════════════════════════════════════"

# Configuration
CONTAINER_DIR="/home/sdodl001_odu_edu/BioPipelines/containers/chip-seq"
OUTPUT="/scratch/sdodl001/containers/chip-seq_1.0.0.sif"
LOG_FILE="/tmp/chip_seq_build_${SLURM_JOB_ID}.log"

# Build container
cd "$CONTAINER_DIR"
echo "Building ChIP-seq container from: $CONTAINER_DIR/chip-seq.def"
echo "Output: $OUTPUT"
echo "Log: $LOG_FILE"
echo ""

singularity build --fakeroot "$OUTPUT" chip-seq.def 2>&1 | tee "$LOG_FILE"

# Verify build
if [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "✓ Build successful!"
    ls -lh "$OUTPUT"
    
    # Quick validation
    echo ""
    echo "Running validation tests..."
    singularity exec "$OUTPUT" bowtie2 --version | head -1
    singularity exec "$OUTPUT" macs2 --version
    
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "✓ ChIP-seq container ready: $OUTPUT"
    echo "End time: $(date)"
    echo "════════════════════════════════════════════════════════"
else
    echo "✗ Build failed! Check log: $LOG_FILE"
    exit 1
fi
