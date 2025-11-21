#!/usr/bin/env python3
"""
Download test metagenomics datasets for pipeline validation

Downloads small, well-characterized metagenomic samples:
1. Mock community (ZymoBIOMICS or similar) - known composition
2. Human gut metagenome - representative real-world sample

Total size: ~5-10 GB
"""

import sys
import subprocess
from pathlib import Path

def download_with_sra_toolkit(accession, output_dir):
    """Download FASTQ files from SRA using fasterq-dump"""
    print(f"\nDownloading {accession}...")
    cmd = [
        "fasterq-dump",
        accession,
        "--outdir", str(output_dir),
        "--split-files",  # Split into R1/R2
        "--progress"
    ]
    subprocess.run(cmd, check=True)
    
    # Compress files
    print(f"Compressing {accession}...")
    for fastq in output_dir.glob(f"{accession}*.fastq"):
        subprocess.run(["gzip", str(fastq)], check=True)

def main():
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "metagenomics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Downloading Metagenomics Test Datasets")
    print("=" * 80)
    
    # Dataset 1: Human gut metagenome (small, well-studied)
    # SRR1927149: Human gut metagenome, ~2GB, good quality
    print("\nDataset 1: Human gut metagenome sample")
    print("  Accession: SRR1927149")
    print("  Description: Human gut microbiome")
    print("  Size: ~2 GB")
    print("  Expected: Diverse bacterial community")
    
    try:
        download_with_sra_toolkit("SRR1927149", output_dir)
        
        # Rename to standard format
        (output_dir / "SRR1927149_1.fastq.gz").rename(output_dir / "sample1_R1.fastq.gz")
        (output_dir / "SRR1927149_2.fastq.gz").rename(output_dir / "sample1_R2.fastq.gz")
        
        print("✓ Downloaded and renamed to sample1_R1/R2.fastq.gz")
        
    except Exception as e:
        print(f"Error downloading SRR1927149: {e}")
        print("\nAlternative: Download manually from:")
        print("  https://www.ebi.ac.uk/ena/browser/view/SRR1927149")
        return 1
    
    # Dataset 2: Mock community (optional, for validation)
    print("\n" + "=" * 80)
    print("Optional: ZymoBIOMICS Mock Community")
    print("  For validation with known composition")
    print("  Accession: SRR17913526 (~3 GB)")
    print("  Skip for now, uncomment if needed")
    print("=" * 80)
    
    # Uncomment to download mock community:
    # try:
    #     download_with_sra_toolkit("SRR17913526", output_dir)
    #     (output_dir / "SRR17913526_1.fastq.gz").rename(output_dir / "sample2_R1.fastq.gz")
    #     (output_dir / "SRR17913526_2.fastq.gz").rename(output_dir / "sample2_R2.fastq.gz")
    #     print("✓ Downloaded mock community sample")
    # except Exception as e:
    #     print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print("Download Complete!")
    print("=" * 80)
    print(f"Files located in: {output_dir}")
    print("\nNext steps:")
    print("1. Download Kraken2 database:")
    print("   kraken2-build --standard --db /scratch/.../kraken2_db")
    print("2. Update config.yaml with database paths")
    print("3. Submit pipeline: sbatch scripts/submit_metagenomics.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
