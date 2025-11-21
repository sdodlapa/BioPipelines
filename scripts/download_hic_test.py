#!/usr/bin/env python3
"""
Download test datasets for Hi-C pipeline validation.
Uses the HiCDownloader module to fetch ENCODE Hi-C data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.hic_downloader import HiCDownloader
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Initialize downloader
    downloader = HiCDownloader(
        output_dir="/home/sdodl001_odu_edu/BioPipelines/data/raw/hic"
    )
    
    print("=" * 80)
    print("Downloading ENCODE Hi-C test dataset")
    print("=" * 80)
    
    # Download a known good ENCODE Hi-C experiment
    # ENCSR312KHQ - GM12878, in situ Hi-C, good depth
    experiment_id = "ENCSR312KHQ"
    
    print(f"\nDownloading experiment: {experiment_id}")
    print("Cell type: GM12878 (human lymphoblastoid)")
    print("Protocol: in situ Hi-C")
    print("Enzyme: MboI")
    print("Organism: Homo sapiens\n")
    
    try:
        # Download fastq files for full pipeline test
        files = downloader.download_encode_hic(
            experiment_id=experiment_id,
            file_format="fastq"
        )
        
        print(f"\n{'=' * 80}")
        print(f"Downloaded {len(files)} files successfully!")
        print(f"{'=' * 80}")
        for f in files:
            print(f"  - {f}")
        
        print("\nUpdate config.yaml with these sample names:")
        # Extract unique sample names (remove _R1/_R2 suffixes)
        samples = set()
        for f in files:
            sample_name = f.stem.replace('.fastq', '').replace('.gz', '')
            # Remove _R1 or _R2 suffix
            if sample_name.endswith('_R1') or sample_name.endswith('_R2'):
                sample_name = sample_name[:-3]
            samples.add(sample_name)
        
        for sample in sorted(samples):
            print(f"  - {sample}")
            
    except Exception as e:
        print(f"\nError downloading {experiment_id}: {e}")
        print("\nNote: Hi-C files are large. Consider downloading from 4DN Portal instead.")
        print("Visit: https://data.4dnucleome.org/")


if __name__ == "__main__":
    main()
