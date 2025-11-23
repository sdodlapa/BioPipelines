#!/usr/bin/env python3
"""
Unified data download CLI for BioPipelines
Replaces 25+ scattered download scripts with a single entry point
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.encode_downloader import EncodeDownloader
from biopipelines.data_download.sra_downloader import SRADownloader
from biopipelines.data_download.methylation_downloader import MethylationDownloader
from biopipelines.data_download.hic_downloader import HiCDownloader


def download_chipseq(args):
    """Download ChIP-seq data from ENCODE"""
    downloader = EncodeDownloader(output_dir=args.output)
    
    if args.accession:
        downloader.download_experiment(args.accession)
    elif args.file_id:
        downloader.download_file(args.file_id)
    else:
        print("Error: Provide either --accession or --file-id")
        sys.exit(1)
    
    print(f"✓ ChIP-seq data downloaded to {args.output}")


def download_rnaseq(args):
    """Download RNA-seq data from SRA/GEO"""
    downloader = SRADownloader(output_dir=args.output)
    
    if args.sra:
        downloader.download_sra(args.sra, paired=args.paired)
    elif args.geo:
        downloader.download_geo(args.geo)
    else:
        print("Error: Provide either --sra or --geo")
        sys.exit(1)
    
    print(f"✓ RNA-seq data downloaded to {args.output}")


def download_atacseq(args):
    """Download ATAC-seq data from ENCODE"""
    downloader = EncodeDownloader(output_dir=args.output)
    
    if args.accession:
        downloader.download_experiment(args.accession)
    else:
        print("Error: Provide --accession")
        sys.exit(1)
    
    print(f"✓ ATAC-seq data downloaded to {args.output}")


def download_methylation(args):
    """Download methylation/bisulfite sequencing data"""
    downloader = MethylationDownloader(output_dir=args.output)
    
    if args.test:
        size = args.test_size or "small"
        downloader.download_test_data(size=size)
    elif args.accession:
        downloader.download_experiment(args.accession)
    else:
        print("Error: Provide --accession or --test")
        sys.exit(1)
    
    print(f"✓ Methylation data downloaded to {args.output}")


def download_hic(args):
    """Download Hi-C data"""
    downloader = HiCDownloader(output_dir=args.output)
    
    if args.test:
        downloader.download_test_data()
    elif args.accession:
        downloader.download_experiment(args.accession)
    else:
        print("Error: Provide --accession or --test")
        sys.exit(1)
    
    print(f"✓ Hi-C data downloaded to {args.output}")


def download_metagenomics(args):
    """Download metagenomics data"""
    downloader = SRADownloader(output_dir=args.output)
    
    if args.test:
        # Download small metagenomics test dataset
        downloader.download_sra("ERR2984773", paired=True)  # Small mock community
    elif args.sra:
        downloader.download_sra(args.sra, paired=args.paired)
    else:
        print("Error: Provide --sra or --test")
        sys.exit(1)
    
    print(f"✓ Metagenomics data downloaded to {args.output}")


def download_longread(args):
    """Download long-read sequencing data"""
    downloader = SRADownloader(output_dir=args.output)
    
    if args.test:
        # Download Nanopore test data
        downloader.download_sra("SRR10971019", paired=False)
    elif args.sra:
        downloader.download_sra(args.sra, paired=False)
    else:
        print("Error: Provide --sra or --test")
        sys.exit(1)
    
    print(f"✓ Long-read data downloaded to {args.output}")


def download_scrna(args):
    """Download single-cell RNA-seq data"""
    downloader = SRADownloader(output_dir=args.output)
    
    if args.test:
        # Download small scRNA-seq test dataset
        downloader.download_sra("SRR9330304", paired=True)
    elif args.sra:
        downloader.download_sra(args.sra, paired=args.paired)
    elif args.geo:
        downloader.download_geo(args.geo)
    else:
        print("Error: Provide --sra, --geo, or --test")
        sys.exit(1)
    
    print(f"✓ scRNA-seq data downloaded to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified data download CLI for BioPipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download ChIP-seq from ENCODE
  %(prog)s chipseq --accession ENCSR000AKP --output data/raw/chip_seq/

  # Download RNA-seq from SRA
  %(prog)s rnaseq --sra SRR000001 --paired --output data/raw/rna_seq/

  # Download test data for any pipeline
  %(prog)s methylation --test --test-size small --output data/raw/methylation/
  %(prog)s hic --test --output data/raw/hic/

  # Download metagenomics data
  %(prog)s metagenomics --sra ERR2984773 --paired --output data/raw/metagenomics/

For more information, see: docs/tutorials/
        """
    )
    
    subparsers = parser.add_subparsers(dest='pipeline', help='Pipeline type')
    
    # ChIP-seq
    chipseq_parser = subparsers.add_parser('chipseq', help='Download ChIP-seq data')
    chipseq_parser.add_argument('--accession', help='ENCODE experiment accession')
    chipseq_parser.add_argument('--file-id', help='ENCODE file ID')
    chipseq_parser.add_argument('--output', default='data/raw/chip_seq/', help='Output directory')
    chipseq_parser.set_defaults(func=download_chipseq)
    
    # RNA-seq
    rnaseq_parser = subparsers.add_parser('rnaseq', help='Download RNA-seq data')
    rnaseq_parser.add_argument('--sra', help='SRA accession (SRR/ERR/DRR)')
    rnaseq_parser.add_argument('--geo', help='GEO accession (GSM/GSE)')
    rnaseq_parser.add_argument('--paired', action='store_true', help='Paired-end data')
    rnaseq_parser.add_argument('--output', default='data/raw/rna_seq/', help='Output directory')
    rnaseq_parser.set_defaults(func=download_rnaseq)
    
    # ATAC-seq
    atacseq_parser = subparsers.add_parser('atacseq', help='Download ATAC-seq data')
    atacseq_parser.add_argument('--accession', required=True, help='ENCODE experiment accession')
    atacseq_parser.add_argument('--output', default='data/raw/atac_seq/', help='Output directory')
    atacseq_parser.set_defaults(func=download_atacseq)
    
    # Methylation
    methylation_parser = subparsers.add_parser('methylation', help='Download methylation data')
    methylation_parser.add_argument('--accession', help='ENCODE experiment accession')
    methylation_parser.add_argument('--test', action='store_true', help='Download test data')
    methylation_parser.add_argument('--test-size', choices=['small', 'medium', 'large'], 
                                   help='Test data size')
    methylation_parser.add_argument('--output', default='data/raw/methylation/', help='Output directory')
    methylation_parser.set_defaults(func=download_methylation)
    
    # Hi-C
    hic_parser = subparsers.add_parser('hic', help='Download Hi-C data')
    hic_parser.add_argument('--accession', help='ENCODE experiment accession')
    hic_parser.add_argument('--test', action='store_true', help='Download test data')
    hic_parser.add_argument('--output', default='data/raw/hic/', help='Output directory')
    hic_parser.set_defaults(func=download_hic)
    
    # Metagenomics
    metagenomics_parser = subparsers.add_parser('metagenomics', help='Download metagenomics data')
    metagenomics_parser.add_argument('--sra', help='SRA accession')
    metagenomics_parser.add_argument('--paired', action='store_true', help='Paired-end data')
    metagenomics_parser.add_argument('--test', action='store_true', help='Download test data')
    metagenomics_parser.add_argument('--output', default='data/raw/metagenomics/', help='Output directory')
    metagenomics_parser.set_defaults(func=download_metagenomics)
    
    # Long-read
    longread_parser = subparsers.add_parser('longread', help='Download long-read data')
    longread_parser.add_argument('--sra', help='SRA accession')
    longread_parser.add_argument('--test', action='store_true', help='Download test data')
    longread_parser.add_argument('--output', default='data/raw/long_read/', help='Output directory')
    longread_parser.set_defaults(func=download_longread)
    
    # scRNA-seq
    scrna_parser = subparsers.add_parser('scrna', help='Download scRNA-seq data')
    scrna_parser.add_argument('--sra', help='SRA accession')
    scrna_parser.add_argument('--geo', help='GEO accession')
    scrna_parser.add_argument('--paired', action='store_true', help='Paired-end data')
    scrna_parser.add_argument('--test', action='store_true', help='Download test data')
    scrna_parser.add_argument('--output', default='data/raw/scrna_seq/', help='Output directory')
    scrna_parser.set_defaults(func=download_scrna)
    
    args = parser.parse_args()
    
    if not args.pipeline:
        parser.print_help()
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Call the appropriate function
    args.func(args)


if __name__ == '__main__':
    main()
