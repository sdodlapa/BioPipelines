#!/bin/bash
#SBATCH --job-name=snpeff_db
#SBATCH --output=snpeff_download_%j.out
#SBATCH --error=snpeff_download_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpuspot

# Activate conda environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Print environment info
echo "Starting SnpEff database download at $(date)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "SnpEff version: $(snpEff -version 2>&1 | head -1)"

# Download hg38 database
echo ""
echo "Downloading hg38 database..."
snpEff download -v hg38

# Check if download was successful
if [ -d ~/envs/biopipelines/share/snpeff-5.1-3/data/hg38 ]; then
    echo ""
    echo "SUCCESS: hg38 database downloaded successfully!"
    echo "Database location: ~/envs/biopipelines/share/snpeff-5.1-3/data/hg38"
    ls -lh ~/envs/biopipelines/share/snpeff-5.1-3/data/hg38/
else
    echo ""
    echo "ERROR: Database download failed or incomplete"
    exit 1
fi

echo ""
echo "Download completed at $(date)"
