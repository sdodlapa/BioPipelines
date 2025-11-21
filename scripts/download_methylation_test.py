#!/usr/bin/env python3
"""
Download test datasets for DNA methylation pipeline validation.
Uses the MethylationDownloader module to fetch ENCODE WGBS data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.methylation_downloader import MethylationDownloader
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Initialize downloader
    downloader = MethylationDownloader(
        output_dir="/home/sdodl001_odu_edu/BioPipelines/data/raw/methylation"
    )
    
    print("=" * 80)
    print("Downloading ENCODE WGBS test dataset")
    print("=" * 80)
    
    # Download a known good ENCODE experiment (GM12878 WGBS)
    # ENCSR765JPC - GM12878 cell line, small dataset, good for testing
    experiment_id = "ENCSR765JPC"
    
    print(f"\nDownloading experiment: {experiment_id}")
    print("Cell type: GM12878 (human lymphoblastoid)")
    print("Organism: Homo sapiens")
    print("Assay: WGBS (whole-genome bisulfite sequencing)\n")
    
    try:
        files = downloader.download_encode_experiment(
            experiment_id=experiment_id,
            file_type="fastq"
        )
        
        print(f"\n{'=' * 80}")
        print(f"Downloaded {len(files)} files successfully!")
        print(f"{'=' * 80}")
        for f in files:
            print(f"  - {f}")
        
        print("\nUpdate config.yaml with these sample names:")
        for f in files:
            sample_name = f.stem.replace('.fastq', '').replace('.gz', '')
            print(f"  - {sample_name}")
            
    except Exception as e:
        print(f"\nError downloading {experiment_id}: {e}")
        print("\nTrying alternative experiment: ENCSR890UQO")
        
        # Backup: try another experiment
        files = downloader.download_encode_experiment(
            experiment_id="ENCSR890UQO",
            file_type="fastq"
        )
        
        print(f"\n{'=' * 80}")
        print(f"Downloaded {len(files)} files successfully!")
        print(f"{'=' * 80}")
        for f in files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
