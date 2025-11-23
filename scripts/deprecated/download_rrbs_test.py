#!/usr/bin/env python3
"""
Download RRBS test data from ENCODE

RRBS (Reduced Representation Bisulfite Sequencing) is much smaller than WGBS:
- RRBS: ~1-2 GB per experiment (focuses on CpG-rich regions)
- WGBS: ~78 GB per file (whole genome coverage)

This script downloads ENCSR000DGH: RRBS on human GM19239 cell line
Total size: ~1.43 GB (2 fastq files)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.methylation_downloader import MethylationDownloader

def main():
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "methylation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize downloader
    downloader = MethylationDownloader(output_dir=str(output_dir))
    
    # Download RRBS experiment ENCSR000DGH
    # This is a GM19239 cell line with only 1.43 GB total
    print("=" * 80)
    print("Downloading RRBS test data: ENCSR000DGH (GM19239 cell line)")
    print("Expected size: ~1.43 GB (2 fastq files)")
    print("Estimated time: 5-10 minutes")
    print("=" * 80)
    
    try:
        files = downloader.download_encode_experiment(
            experiment_id="ENCSR000DGH",
            file_type="fastq"
        )
        
        print(f"\n✓ Successfully downloaded {len(files)} files:")
        for f in files:
            print(f"  - {f}")
            
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
