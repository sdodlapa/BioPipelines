#!/bin/bash
#
# Download dbSNP and GENCODE annotations (independent of BWA index)
#

set -e

# Activate conda
source ~/miniconda3/bin/activate ~/envs/biopipelines

REF_DIR="$HOME/references"
KNOWN_SITES_DIR="${REF_DIR}/known_sites"
ANNOT_DIR="${REF_DIR}/annotations"

mkdir -p "${KNOWN_SITES_DIR}"
mkdir -p "${ANNOT_DIR}"

echo "========================================="
echo "Downloading Additional References"
echo "========================================="

# 1. Download dbSNP (corrected URL)
echo ""
echo "[1/2] Downloading dbSNP known sites (~11GB)..."
cd "${KNOWN_SITES_DIR}"

if [ ! -s "dbsnp_155.hg38.vcf.gz" ]; then
    # Use the correct NCBI dbSNP URL
    wget -c https://ftp.ncbi.nlm.nih.gov/snp/archive/b155/VCF/GCF_000001405.40.gz \
        -O dbsnp_155.hg38.vcf.gz || \
    wget -c https://ftp.ncbi.nlm.nih.gov/snp/latest_release/VCF/GCF_000001405.40.gz \
        -O dbsnp_155.hg38.vcf.gz
    
    # Index the VCF
    tabix -p vcf dbsnp_155.hg38.vcf.gz
    echo "✓ dbSNP downloaded and indexed"
else
    echo "✓ dbSNP already exists"
fi

# 2. Download GENCODE gene annotations
echo ""
echo "[2/2] Downloading GENCODE gene annotations..."
cd "${ANNOT_DIR}"

if [ ! -f "gencode.v45.annotation.gtf" ]; then
    wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz
    gunzip gencode.v45.annotation.gtf.gz
    echo "✓ GENCODE annotations downloaded"
else
    echo "✓ GENCODE annotations already exist"
fi

echo ""
echo "========================================="
echo "✓ Additional references complete!"
echo "========================================="
echo ""
echo "Reference locations:"
echo "  dbSNP:       ${KNOWN_SITES_DIR}/dbsnp_155.hg38.vcf.gz"
echo "  Annotations: ${ANNOT_DIR}/gencode.v45.annotation.gtf"
echo ""
