#!/bin/bash
#
# Download small, validated test datasets from public repositories
# Sources: ENA/EBI, SRA, UCSC, 1000 Genomes Project
#
# This script uses direct FTP/HTTP downloads from reliable public archives
# that don't require authentication or API keys
#

set -e

DATA_DIR="$HOME/BioPipelines/data/raw"
mkdir -p "${DATA_DIR}"/{dna_seq,rna_seq,chip_seq,atac_seq}

echo "========================================="
echo "Downloading Public Test Datasets"
echo "========================================="
echo ""
echo "Sources:"
echo "  - RNA-seq: ENA (European Nucleotide Archive)"
echo "  - ChIP-seq: ENA H3K4me3 datasets"
echo "  - ATAC-seq: ENA ATAC-seq samples"
echo "  - DNA-seq: 1000 Genomes Project"
echo ""
echo "Total size: ~3-4GB"
echo "Estimated time: 20-40 minutes"
echo ""

# Function to download with progress and retries
download_file() {
    local url=$1
    local output=$2
    local size=$3
    
    if [ -f "$output" ] && [ -s "$output" ]; then
        echo "  ✓ $(basename $output) already exists ($(du -h $output | cut -f1))"
        return 0
    fi
    
    echo "  Downloading: $(basename $output) (~$size)"
    
    # Try wget first, fallback to curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress --tries=3 --timeout=60 -c -O "$output" "$url" || {
            echo "  ⚠ wget failed, trying curl..."
            curl -L --retry 3 --retry-delay 5 -o "$output" "$url"
        }
    else
        curl -L --retry 3 --retry-delay 5 -o "$output" "$url"
    fi
    
    # Verify download
    if [ ! -s "$output" ]; then
        echo "  ✗ Download failed or file is empty: $output"
        rm -f "$output"
        return 1
    fi
    
    echo "  ✓ Downloaded: $(du -h $output | cut -f1)"
}

# ========================================
# [1/4] RNA-seq - Small E. coli dataset for testing
# ========================================
echo ""
echo "[1/4] RNA-seq test data (E. coli, small dataset for quick testing)..."
cd "${DATA_DIR}/rna_seq"

# Using SRA ERR458493-ERR458502 (E. coli, paired-end, ~50-100MB each)
# Direct ENA FTP links (much faster than ENCODE)

# Treatment samples (e.g., exponential phase)
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR458/ERR458493/ERR458493_1.fastq.gz" "treat_rep1_R1.fastq.gz" "50MB"
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR458/ERR458493/ERR458493_2.fastq.gz" "treat_rep1_R2.fastq.gz" "50MB"

download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR458/ERR458494/ERR458494_1.fastq.gz" "treat_rep2_R1.fastq.gz" "50MB"
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR458/ERR458494/ERR458494_2.fastq.gz" "treat_rep2_R2.fastq.gz" "50MB"

# Control samples (e.g., stationary phase)  
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR458/ERR458495/ERR458495_1.fastq.gz" "ctrl_rep1_R1.fastq.gz" "50MB"
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR458/ERR458495/ERR458495_2.fastq.gz" "ctrl_rep1_R2.fastq.gz" "50MB"

download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR458/ERR458496/ERR458496_1.fastq.gz" "ctrl_rep2_R1.fastq.gz" "50MB"
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR458/ERR458496/ERR458496_2.fastq.gz" "ctrl_rep2_R2.fastq.gz" "50MB"

# Create symlinks for rep3 (reuse rep1 for testing purposes)
ln -sf treat_rep1_R1.fastq.gz treat_rep3_R1.fastq.gz
ln -sf treat_rep1_R2.fastq.gz treat_rep3_R2.fastq.gz
ln -sf ctrl_rep1_R1.fastq.gz ctrl_rep3_R1.fastq.gz
ln -sf ctrl_rep1_R2.fastq.gz ctrl_rep3_R2.fastq.gz

echo "  ✓ RNA-seq data downloaded"

# ========================================
# [2/4] ChIP-seq - H3K4me3 from ENA
# ========================================
echo ""
echo "[2/4] ChIP-seq test data (H3K4me3 ChIP-seq from ENA, ~200-500MB)..."
cd "${DATA_DIR}/chip_seq"

# Using SRR datasets from ENA (single-end ChIP-seq)
# SRR5344681-SRR5344683: H3K4me3 ChIP-seq, human

download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR534/001/SRR5344681/SRR5344681.fastq.gz" "sample1.fastq.gz" "200MB"
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR534/002/SRR5344682/SRR5344682.fastq.gz" "sample2.fastq.gz" "200MB"

# Input control
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR534/003/SRR5344683/SRR5344683.fastq.gz" "input.fastq.gz" "180MB"

echo "  ✓ ChIP-seq data downloaded"

# ========================================
# [3/4] ATAC-seq from ENA
# ========================================
echo ""
echo "[3/4] ATAC-seq test data (paired-end from ENA, ~300-600MB)..."
cd "${DATA_DIR}/atac_seq"

# Using SRR datasets: SRR891268-SRR891269 (ATAC-seq, paired-end)
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR891/SRR891268/SRR891268_1.fastq.gz" "sample1_R1.fastq.gz" "300MB"
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR891/SRR891268/SRR891268_2.fastq.gz" "sample1_R2.fastq.gz" "300MB"

download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR891/SRR891269/SRR891269_1.fastq.gz" "sample2_R1.fastq.gz" "300MB"
download_file "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR891/SRR891269/SRR891269_2.fastq.gz" "sample2_R2.fastq.gz" "300MB"

echo "  ✓ ATAC-seq data downloaded"

# ========================================
# [4/4] DNA-seq from 1000 Genomes
# ========================================
echo ""
echo "[4/4] DNA-seq test data (1000 Genomes, NA12878, subsampled ~500MB)..."
cd "${DATA_DIR}/dna_seq"

# Using 1000 Genomes Phase 3 - NA12878 (CEU population)
# Small exome subset for testing
download_file "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/NA12878/exome_alignment/NA12878.chrom20.ILLUMINA.bwa.CEU.exome.20121211.bam" "sample1.bam" "500MB"

# Convert BAM to FASTQ for pipeline testing
if command -v samtools &> /dev/null; then
    echo "  Converting BAM to FASTQ..."
    samtools fastq -1 sample1_R1.fastq.gz -2 sample1_R2.fastq.gz -0 /dev/null -s /dev/null sample1.bam
    rm sample1.bam
    echo "  ✓ Converted to FASTQ"
else
    echo "  ⚠ samtools not found, keeping BAM format"
    echo "    Install samtools to convert: conda install -c bioconda samtools"
fi

echo "  ✓ DNA-seq data downloaded"

# ========================================
# Summary and Verification
# ========================================
echo ""
echo "========================================="
echo "✓ Download Complete!"
echo "========================================="
echo ""
echo "📊 Downloaded datasets:"
echo ""
echo "RNA-seq (${DATA_DIR}/rna_seq):"
ls -lh "${DATA_DIR}/rna_seq" | grep -E "fastq.gz$" | awk '{print "  - " $9 ": " $5}'

echo ""
echo "ChIP-seq (${DATA_DIR}/chip_seq):"
ls -lh "${DATA_DIR}/chip_seq" | grep -E "fastq.gz$" | awk '{print "  - " $9 ": " $5}'

echo ""
echo "ATAC-seq (${DATA_DIR}/atac_seq):"
ls -lh "${DATA_DIR}/atac_seq" | grep -E "fastq.gz$" | awk '{print "  - " $9 ": " $5}'

echo ""
echo "DNA-seq (${DATA_DIR}/dna_seq):"
ls -lh "${DATA_DIR}/dna_seq" | grep -E "fastq.gz$\|bam$" | awk '{print "  - " $9 ": " $5}'

echo ""
echo "📦 Total size:"
du -sh "${DATA_DIR}"

echo ""
echo "🔍 Verify data integrity:"
echo "  # Check file types"
echo "  file ${DATA_DIR}/rna_seq/*.fastq.gz | head -3"
echo ""
echo "  # Check read counts"
echo "  zcat ${DATA_DIR}/rna_seq/treat_rep1_R1.fastq.gz | wc -l"
echo ""
echo "⚠️  NOTE: These are smaller test datasets suitable for pipeline validation."
echo "   For production analysis, download full-size datasets from:"
echo "   - ENCODE: https://www.encodeproject.org/"
echo "   - GEO/SRA: https://www.ncbi.nlm.nih.gov/geo/"
echo "   - 1000 Genomes: http://www.internationalgenome.org/"
echo "   - ENA: https://www.ebi.ac.uk/ena"
echo ""
echo "📚 Next steps:"
echo "  1. Update pipeline configs with sample names"
echo "  2. Run pipelines: sbatch scripts/submit_*.sh"
echo "  3. Monitor with: squeue --me"
echo ""

