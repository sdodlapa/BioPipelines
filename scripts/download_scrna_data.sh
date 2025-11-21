#!/bin/bash
#SBATCH --job-name=download_scrna
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00

# Download scRNA-seq test data
# Using 10x Genomics PBMC data from SRA

echo "Downloading scRNA-seq test data (10x PBMC)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/sdodl001_odu_edu/envs/biopipelines

# Output directory
output_dir=~/BioPipelines/data/raw/scrna_seq
mkdir -p $output_dir
cd $output_dir

echo "="*80
echo "Downloading SRR8206317 (10x PBMC, ~5k cells)"
echo "Size: ~1-2 GB (paired-end)"
echo "="*80

# Download using fasterq-dump (faster than fastq-dump)
fasterq-dump SRR8206317 \
    --split-files \
    --threads 2 \
    --progress \
    --outdir $output_dir

# Check if download succeeded
if [ $? -eq 0 ]; then
    echo "✓ Download complete"
    
    # Compress with pigz for speed
    echo "Compressing FASTQ files..."
    pigz -p 2 SRR8206317_1.fastq &
    pigz -p 2 SRR8206317_2.fastq &
    wait
    
    # Rename to standard format
    mv SRR8206317_1.fastq.gz sample1_R1.fastq.gz
    mv SRR8206317_2.fastq.gz sample1_R2.fastq.gz
    
    echo "✓ Files ready:"
    ls -lh sample1_R*.fastq.gz
else
    echo "✗ Download failed"
    exit 1
fi

echo "End: $(date)"
echo "✓ scRNA-seq test data ready!"
