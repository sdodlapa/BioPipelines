#!/bin/bash
#SBATCH --job-name=bwa_index
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/bwa_index_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/bwa_index_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

echo "Building BWA index for hg38..."
date

# Activate conda environment
source /home/sdodl001_odu_edu/envs/biopipelines/bin/activate
conda activate biopipelines

cd /scratch/sdodl001/BioPipelines/data/references

# Build BWA index (takes ~30-45 minutes for hg38)
bwa index hg38.fa

echo "BWA index build complete"
date
ls -lh hg38.fa.*
