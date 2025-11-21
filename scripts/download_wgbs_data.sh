#!/bin/bash
#SBATCH --job-name=download_wgbs
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00

echo "Downloading WGBS data using proper Python module"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/sdodl001_odu_edu/envs/biopipelines

# Navigate to project root
cd ~/BioPipelines

# Use the proper download module
python << 'PYTHON_SCRIPT'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from biopipelines.data_download.methylation_downloader import MethylationDownloader

# Setup output directory
output_dir = Path("data/raw/methylation")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize downloader
downloader = MethylationDownloader(output_dir=str(output_dir))

print("=" * 80)
print("Downloading WGBS experiment: ENCSR765JPC")
print("Description: Whole Genome Bisulfite Sequencing")
print("Cell line: H1 embryonic stem cells")
print("Expected size: ~156 GB total (2 paired-end files)")
print("=" * 80)

try:
    files = downloader.download_encode_experiment(
        experiment_id="ENCSR765JPC",
        file_type="fastq"
    )
    
    print(f"\n✓ Successfully downloaded {len(files)} files:")
    for f in files:
        print(f"  - {f}")
        
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    sys.exit(1)

# Rename to standard names
import os
import glob

fastq_files = sorted(glob.glob(str(output_dir / "*.fastq.gz")))
if len(fastq_files) >= 2:
    os.rename(fastq_files[0], str(output_dir / "sample1_R1.fastq.gz"))
    os.rename(fastq_files[1], str(output_dir / "sample1_R2.fastq.gz"))
    print("\n✓ Renamed to sample1_R1.fastq.gz and sample1_R2.fastq.gz")

PYTHON_SCRIPT

echo "End time: $(date)"
echo "Download complete!"
