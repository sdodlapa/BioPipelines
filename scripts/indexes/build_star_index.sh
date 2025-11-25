#!/bin/bash
#SBATCH --job-name=star_index
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=star_index_%j.out
#SBATCH --error=star_index_%j.err

# Build STAR genome index for RNA-seq alignment

echo "========================================="
echo "Building STAR Genome Index"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Activate conda environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Set paths
GENOME_FA="/home/sdodl001_odu_edu/references/genomes/hg38/hg38.fa"
GTF="/home/sdodl001_odu_edu/references/annotations/gencode.v45.annotation.gtf"
INDEX_DIR="/home/sdodl001_odu_edu/references/indexes/star"

# Create index directory
mkdir -p $INDEX_DIR

echo "Starting STAR genome index generation..."
echo "Genome: $GENOME_FA"
echo "GTF: $GTF"
echo "Output: $INDEX_DIR"

# Generate STAR index
STAR \
    --runMode genomeGenerate \
    --runThreadN $SLURM_CPUS_PER_TASK \
    --genomeDir $INDEX_DIR \
    --genomeFastaFiles $GENOME_FA \
    --sjdbGTFfile $GTF \
    --sjdbOverhang 99 \
    --genomeSAindexNbases 14

echo "========================================="
echo "STAR index generation complete!"
echo "Index location: $INDEX_DIR"
echo "========================================="
