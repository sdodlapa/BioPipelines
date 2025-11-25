#!/bin/bash
#SBATCH --job-name=build_conda_peak
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/conda_peak_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/conda_peak_%j.err
#SBATCH --time=30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpuspot

set -euo pipefail

echo "=========================================="
echo "Building Peak Calling (Conda-Based)"
echo "=========================================="
time singularity build --fakeroot \
    /scratch/sdodl001/BioPipelines/containers/tier2/peak_calling_conda.sif \
    containers/tier2/peak_calling_conda.def

echo ""
echo "=========================================="
echo "Testing Peak Calling Container"
echo "=========================================="
singularity test /scratch/sdodl001/BioPipelines/containers/tier2/peak_calling_conda.sif

echo ""
echo "Build completed successfully!"
ls -lh /scratch/sdodl001/BioPipelines/containers/tier2/peak_calling_conda.sif
