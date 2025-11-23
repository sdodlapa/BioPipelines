#!/bin/bash
#SBATCH --job-name=build_workflow_engine
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/build_workflow_engine.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/build_workflow_engine.err

set -euo pipefail

echo "════════════════════════════════════════"
echo "Building Workflow Engine Container"
echo "Started: $(date)"
echo "════════════════════════════════════════"
echo ""

CONTAINER_DIR="/home/sdodl001_odu_edu/BioPipelines/containers/images"
BUILD_DIR="/home/sdodl001_odu_edu/BioPipelines/containers/workflow-engine"
OUTPUT_IMAGE="${CONTAINER_DIR}/workflow-engine_1.0.0.sif"

cd "$BUILD_DIR"

# Build container
echo "Building container..."
singularity build --fakeroot "$OUTPUT_IMAGE" workflow-engine.def

echo ""
echo "════════════════════════════════════════"
echo "✓ Build successful!"
ls -lh "$OUTPUT_IMAGE"
echo ""
echo "Running validation tests..."
singularity run "$OUTPUT_IMAGE" --version
echo ""
echo "Testing Nextflow..."
singularity exec "$OUTPUT_IMAGE" nextflow -version
echo "════════════════════════════════════════"
