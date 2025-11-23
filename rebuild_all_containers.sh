#!/bin/bash
#SBATCH --job-name=rebuild_all_containers
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rebuild_all_%j.out
#SBATCH --error=logs/rebuild_all_%j.err

set -e

CONTAINER_DIR="/home/sdodl001_odu_edu/BioPipelines/containers/images"
mkdir -p ${CONTAINER_DIR}

echo "Building base container..."
cd ~/BioPipelines/containers/base
singularity build --fakeroot ${CONTAINER_DIR}/base_1.0.0.sif base.def

echo "Building pipeline containers..."

# RNA-seq
echo "Building RNA-seq container..."
cd ~/BioPipelines/containers/rna-seq
singularity build --fakeroot ${CONTAINER_DIR}/rna-seq_1.0.0.sif rna-seq.def

# DNA-seq
echo "Building DNA-seq container..."
cd ~/BioPipelines/containers/dna-seq
singularity build --fakeroot ${CONTAINER_DIR}/dna-seq_1.0.0.sif dna-seq.def

# ChIP-seq
echo "Building ChIP-seq container..."
cd ~/BioPipelines/containers/chip-seq
singularity build --fakeroot ${CONTAINER_DIR}/chip-seq_1.0.0.sif chip-seq.def

# ATAC-seq
echo "Building ATAC-seq container..."
cd ~/BioPipelines/containers/atac-seq
singularity build --fakeroot ${CONTAINER_DIR}/atac-seq_1.0.0.sif atac-seq.def

# Hi-C
echo "Building HiC container..."
cd ~/BioPipelines/containers/hic
singularity build --fakeroot ${CONTAINER_DIR}/hic_1.0.0.sif hic.def

# scRNA-seq
echo "Building scRNA-seq container..."
cd ~/BioPipelines/containers/scrna-seq
singularity build --fakeroot ${CONTAINER_DIR}/scrna-seq_1.0.0.sif scrna-seq.def

# Methylation
echo "Building Methylation container..."
cd ~/BioPipelines/containers/methylation
singularity build --fakeroot ${CONTAINER_DIR}/methylation_1.0.0.sif methylation.def

# Long-read
echo "Building Long-read container..."
cd ~/BioPipelines/containers/long-read
singularity build --fakeroot ${CONTAINER_DIR}/long-read_1.0.0.sif long-read.def

# Metagenomics
echo "Building Metagenomics container..."
cd ~/BioPipelines/containers/metagenomics
singularity build --fakeroot ${CONTAINER_DIR}/metagenomics_1.0.0.sif metagenomics.def

# Structural variants
echo "Building Structural-variants container..."
cd ~/BioPipelines/containers/structural-variants
singularity build --fakeroot ${CONTAINER_DIR}/structural-variants_1.0.0.sif structural-variants.def

# Workflow engine
echo "Building Workflow-engine container..."
cd ~/BioPipelines/containers/workflow-engine
singularity build --fakeroot ${CONTAINER_DIR}/workflow-engine_1.0.0.sif workflow-engine.def

echo "All container rebuilds complete!"
