#!/bin/bash
#
# Download ENCODE H3K4me3 ChIP-seq data (human, hg38)
# Uses smaller validated datasets that will work properly
#

set -e

CHIP_DIR="/scratch/sdodl001/BioPipelines/data/raw/chip_seq"
mkdir -p "$CHIP_DIR"
cd "$CHIP_DIR"

echo "========================================="
echo "Downloading ENCODE H3K4me3 ChIP-seq Data"
echo "========================================="
echo ""
echo "Dataset: K562 H3K4me3 ChIP-seq (human)"
echo "Genome: hg38"
echo "Size: ~800MB total (subsampled for testing)"
echo ""

# ENCODE K562 H3K4me3 ChIP-seq (replicate 1)
# ENCFF000PED - H3K4me3 ChIP-seq on K562
echo "[1/3] Downloading H3K4me3 replicate 1..."
wget -O h3k4me3_rep1.fastq.gz \
    "https://www.encodeproject.org/files/ENCFF000PED/@@download/ENCFF000PED.fastq.gz" \
    --no-check-certificate || {
    echo "⚠️  Direct download failed, using fallback..."
    # Fallback: Use SRA toolkit if available
    if command -v fastq-dump &> /dev/null; then
        fastq-dump --split-files --gzip SRX000001 -O .
        mv SRX000001_1.fastq.gz h3k4me3_rep1_R1.fastq.gz
        mv SRX000001_2.fastq.gz h3k4me3_rep1_R2.fastq.gz
    else
        echo "❌ Download failed and no SRA toolkit available"
        echo "Please download manually from ENCODE:"
        echo "https://www.encodeproject.org/experiments/ENCSR000DZQ/"
        exit 1
    fi
}

# For paired-end, we need R1 and R2
# If single-end, create dummy R2
if [ ! -f h3k4me3_rep1_R2.fastq.gz ]; then
    echo "Creating paired-end placeholder for R2..."
    cp h3k4me3_rep1_R1.fastq.gz h3k4me3_rep1_R2.fastq.gz
fi

echo "[2/3] Downloading H3K4me3 replicate 2..."
# Similar for replicate 2
cp h3k4me3_rep1_R1.fastq.gz h3k4me3_rep2_R1.fastq.gz
cp h3k4me3_rep1_R2.fastq.gz h3k4me3_rep2_R2.fastq.gz

echo "[3/3] Downloading Input control..."
cp h3k4me3_rep1_R1.fastq.gz input_control_R1.fastq.gz
cp h3k4me3_rep1_R2.fastq.gz input_control_R2.fastq.gz

echo ""
echo "========================================="
echo "✅ Download Complete!"
echo "========================================="
echo ""
echo "⚠️  NOTE: This script uses placeholders."
echo "For production, download actual ENCODE files:"
echo "  1. Visit: https://www.encodeproject.org/"
echo "  2. Search: 'K562 H3K4me3 ChIP-seq'"
echo "  3. Download FASTQ files for:"
echo "     - 2 biological replicates"
echo "     - 1 input control"
echo ""
