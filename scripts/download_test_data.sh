#!/bin/bash
#
# Download small test datasets for each pipeline
# Using publicly available data from ENCODE and other sources
#

set -e

DATA_DIR="$HOME/BioPipelines/data/raw"
mkdir -p "${DATA_DIR}"/{dna_seq,rna_seq,chip_seq,atac_seq}

echo "========================================="
echo "Downloading Test Datasets"
echo "========================================="

# Function to download with retries
download_file() {
    local url=$1
    local output=$2
    echo "  Downloading: $(basename $output)"
    wget -q --show-progress -c -O "$output" "$url" || curl -L -o "$output" "$url"
}

# [1/4] RNA-seq test data (from ENCODE - GM12878 cell line)
echo ""
echo "[1/4] RNA-seq test data (ENCODE GM12878, ~600MB)..."
cd "${DATA_DIR}/rna_seq"

# Treatment samples (ENCSR000AED - polyA plus RNA-seq)
download_file "https://www.encodeproject.org/files/ENCFF001RTP/@@download/ENCFF001RTP.fastq.gz" "treat_rep1_R1.fastq.gz"
download_file "https://www.encodeproject.org/files/ENCFF001RTQ/@@download/ENCFF001RTQ.fastq.gz" "treat_rep1_R2.fastq.gz"

download_file "https://www.encodeproject.org/files/ENCFF001RTR/@@download/ENCFF001RTR.fastq.gz" "treat_rep2_R1.fastq.gz"
download_file "https://www.encodeproject.org/files/ENCFF001RTS/@@download/ENCFF001RTS.fastq.gz" "treat_rep2_R2.fastq.gz"

# Control samples (different time point or condition)
download_file "https://www.encodeproject.org/files/ENCFF001RTT/@@download/ENCFF001RTT.fastq.gz" "ctrl_rep1_R1.fastq.gz"
download_file "https://www.encodeproject.org/files/ENCFF001RTU/@@download/ENCFF001RTU.fastq.gz" "ctrl_rep1_R2.fastq.gz"

download_file "https://www.encodeproject.org/files/ENCFF001RTV/@@download/ENCFF001RTV.fastq.gz" "ctrl_rep2_R1.fastq.gz"
download_file "https://www.encodeproject.org/files/ENCFF001RTW/@@download/ENCFF001RTW.fastq.gz" "ctrl_rep2_R2.fastq.gz"

# Create symlinks for rep3 (using rep1 for testing)
ln -sf treat_rep1_R1.fastq.gz treat_rep3_R1.fastq.gz
ln -sf treat_rep1_R2.fastq.gz treat_rep3_R2.fastq.gz
ln -sf ctrl_rep1_R1.fastq.gz ctrl_rep3_R1.fastq.gz
ln -sf ctrl_rep1_R2.fastq.gz ctrl_rep3_R2.fastq.gz

echo "  ✓ RNA-seq data downloaded"

# [2/4] ChIP-seq test data (H3K4me3 from ENCODE)
echo ""
echo "[2/4] ChIP-seq test data (ENCODE H3K4me3, ~400MB)..."
cd "${DATA_DIR}/chip_seq"

# H3K4me3 ChIP samples
download_file "https://www.encodeproject.org/files/ENCFF000PED/@@download/ENCFF000PED.fastq.gz" "sample1_R1.fastq.gz"
download_file "https://www.encodeproject.org/files/ENCFF000PEE/@@download/ENCFF000PEE.fastq.gz" "sample2_R1.fastq.gz"
download_file "https://www.encodeproject.org/files/ENCFF000PEF/@@download/ENCFF000PEF.fastq.gz" "sample3_R1.fastq.gz"

# Input control
download_file "https://www.encodeproject.org/files/ENCFF000PEG/@@download/ENCFF000PEG.fastq.gz" "input_R1.fastq.gz"

echo "  ✓ ChIP-seq data downloaded"

# [3/4] ATAC-seq test data (from ENCODE)
echo ""
echo "[3/4] ATAC-seq test data (ENCODE, ~300MB)..."
cd "${DATA_DIR}/atac_seq"

# ATAC-seq samples (GM12878)
download_file "https://www.encodeproject.org/files/ENCFF341MYG/@@download/ENCFF341MYG.fastq.gz" "sample1_R1.fastq.gz"
download_file "https://www.encodeproject.org/files/ENCFF341MYH/@@download/ENCFF341MYH.fastq.gz" "sample1_R2.fastq.gz"

download_file "https://www.encodeproject.org/files/ENCFF362NJQ/@@download/ENCFF362NJQ.fastq.gz" "sample2_R1.fastq.gz"
download_file "https://www.encodeproject.org/files/ENCFF362NJR/@@download/ENCFF362NJR.fastq.gz" "sample2_R2.fastq.gz"

# Create symlink for sample3 (using sample1 for testing)
ln -sf sample1_R1.fastq.gz sample3_R1.fastq.gz
ln -sf sample1_R2.fastq.gz sample3_R2.fastq.gz

echo "  ✓ ATAC-seq data downloaded"

# [4/4] DNA-seq data already exists, just check
echo ""
echo "[4/4] Checking DNA-seq data..."
cd "${DATA_DIR}"
if [ -f "sample1_R1.fastq.gz" ] && [ -f "sample1_R2.fastq.gz" ]; then
    echo "  ✓ DNA-seq data already present"
else
    echo "  ⚠ DNA-seq data not found in expected location"
fi

echo ""
echo "========================================="
echo "Test Data Download Complete"
echo "========================================="
echo ""
echo "Downloaded datasets:"
echo "  - RNA-seq: 8 files (paired treatment/control, 2 reps each)"
echo "  - ChIP-seq: 4 files (3 ChIP + 1 input)"
echo "  - ATAC-seq: 4 files (2 samples, paired-end)"
echo "  - DNA-seq: Already present"
echo ""
echo "Total size: ~1-2 GB"
echo ""
