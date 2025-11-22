#!/usr/bin/env python3
"""
Download test data for long-read sequencing pipeline
Supports Oxford Nanopore and PacBio HiFi data from SRA
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.sra_downloader import SRADownloader

def main():
    # Output directory
    output_dir = Path("/scratch/sdodl001/BioPipelines/data/raw/long_read")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Long-Read Sequencing Data Downloader")
    print("=" * 80)
    
    # Initialize downloader
    downloader = SRADownloader(output_dir=str(output_dir))
    
    # Test datasets
    # ONT: E. coli (small test dataset)
    ont_accession = "SRR6702603"  # E. coli K12 ONT sequencing (~500MB)
    
    # PacBio HiFi: Human CHM13 chromosome 20 subset (small test)
    pacbio_accession = "SRR10382244"  # PacBio HiFi test data (~1GB)
    
    print(f"\nDownloading Oxford Nanopore test data: {ont_accession}")
    print(f"Expected size: ~500MB")
    print(f"Dataset: E. coli K12 genome")
    
    try:
        # Download ONT data
        print("\nStarting ONT data download...")
        ont_files = downloader.download_run(ont_accession)
        
        # Rename to sample1.fastq.gz
        if ont_files:
            for f in ont_files:
                if os.path.exists(f):
                    new_name = output_dir / "sample1.fastq.gz"
                    os.rename(f, new_name)
                    print(f"✓ Downloaded and renamed to: {new_name}")
                    
                    # Get file size
                    size_mb = os.path.getsize(new_name) / (1024 * 1024)
                    print(f"  File size: {size_mb:.1f} MB")
        
        print("\n" + "=" * 80)
        print("Download completed successfully!")
        print("=" * 80)
        print(f"\nData location: {output_dir}")
        print("\nNext steps:")
        print("1. Review config.yaml and update platform/parameters if needed")
        print("2. Submit pipeline: sbatch scripts/submit_long_read.sh")
        print("3. Monitor progress: tail -f long_read_sv_*.err")
        
    except Exception as e:
        print(f"\n✗ Error downloading data: {e}")
        print("\nAlternative: Manual download from SRA")
        print(f"  fastq-dump --gzip --split-files {ont_accession}")
        print(f"  mv {ont_accession}.fastq.gz {output_dir}/sample1.fastq.gz")
        sys.exit(1)

if __name__ == "__main__":
    main()
