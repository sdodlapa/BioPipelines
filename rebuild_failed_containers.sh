#!/bin/bash
#SBATCH --job-name=rebuild_containers
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rebuild_containers_%j.out
#SBATCH --error=logs/rebuild_containers_%j.err

set -e

CONTAINER_DIR="/scratch/sdodl001/containers"

echo "Rebuilding failed containers..."

# Rebuild HiC container
echo "Building HiC container..."
cd ~/BioPipelines/containers/hic
singularity build --fakeroot ${CONTAINER_DIR}/hic_1.0.0.sif hic.def

# Rebuild scRNA-seq container  
echo "Building scRNA-seq container..."
cd ~/BioPipelines/containers/scrna-seq
singularity build --fakeroot ${CONTAINER_DIR}/scrna-seq_1.0.0.sif scrna-seq.def

echo "Container rebuilds complete!"
