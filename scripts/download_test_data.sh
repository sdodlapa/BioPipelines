#!/bin/bash
#
# Download small test datasets for each pipeline
# These can be downloaded independently and in parallel
#

set -e

source ~/miniconda3/bin/activate ~/envs/biopipelines

DATA_DIR="$HOME/BioPipelines/data/raw"
mkdir -p "${DATA_DIR}"/{dna_seq,rna_seq,chip_seq,atac_seq}

echo "========================================="
echo "Downloading Test Datasets"
echo "========================================="

# Option 1: Download from SRA (small datasets)
# You'll need SRA toolkit installed

# Option 2: Use pre-subsampled datasets
# We'll create a manifest of small public datasets

echo ""
echo "[1/4] DNA-seq test data (paired-end, ~100MB)..."
cd "${DATA_DIR}/dna_seq"

# Example: NA12878 chr22 subset (if available)
# Or subsample from 1000 Genomes
echo "  Option: Download 1000 Genomes chr22 subset"
echo "  Command: wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/NA12878/sequence_read/..."

echo ""
echo "[2/4] RNA-seq test data (paired-end, ~200MB per sample)..."
cd "${DATA_DIR}/rna_seq"
echo "  Option: Download from ENCODE or GEO"
echo "  Suggested: Small paired treatment/control experiment"

echo ""
echo "[3/4] ChIP-seq test data (~100MB)..."
cd "${DATA_DIR}/chip_seq"
echo "  Option: Download ENCODE H3K4me3 ChIP + input"
echo "  Suggested: Small mammalian dataset"

echo ""
echo "[4/4] ATAC-seq test data (~150MB)..."
cd "${DATA_DIR}/atac_seq"
echo "  Option: Download from ENCODE ATAC-seq"

echo ""
echo "========================================="
echo "Test Data Download Options Created"
echo "========================================="
echo ""
echo "For actual downloads, you can use:"
echo "1. SRA Toolkit: fastq-dump SRR..."
echo "2. ENCODE portal: https://www.encodeproject.org"
echo "3. GEO datasets: https://www.ncbi.nlm.nih.gov/geo/"
echo ""
echo "Suggested approach:"
echo "- Use pre-processed, subsampled data (chr22 only)"
echo "- Download ~500MB total across all pipelines"
echo "- Focus on quality over quantity for testing"
echo ""
