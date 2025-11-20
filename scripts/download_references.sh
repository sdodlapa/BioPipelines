#!/bin/bash
#
# BioPipelines Reference Genome Download Script
# Downloads hg38 reference genome, indexes, and known sites for variant calling
#
# Usage: ./download_references.sh
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
REF_DIR="$HOME/references"
GENOME_DIR="${REF_DIR}/genomes/hg38"
KNOWN_SITES_DIR="${REF_DIR}/known_sites"
ANNOT_DIR="${REF_DIR}/annotations"

# Create directories
echo "Creating reference directories..."
mkdir -p "${GENOME_DIR}"
mkdir -p "${KNOWN_SITES_DIR}"
mkdir -p "${ANNOT_DIR}"

# Activate conda environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

echo "========================================="
echo "BioPipelines Reference Download"
echo "========================================="
echo "Target directory: ${REF_DIR}"
echo "Estimated total size: ~25GB"
echo "Estimated time: 1-3 hours (depends on network)"
echo "========================================="

# 1. Download hg38 reference genome from UCSC
echo ""
echo "[1/5] Downloading hg38 reference genome (~3GB)..."
cd "${GENOME_DIR}"

if [ ! -f "hg38.fa" ]; then
    wget -c http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
    gunzip hg38.fa.gz
    echo "✓ hg38 genome downloaded"
else
    echo "✓ hg38.fa already exists, skipping"
fi

# 2. Create FASTA index
echo ""
echo "[2/5] Creating FASTA index..."
if [ ! -f "hg38.fa.fai" ]; then
    samtools faidx hg38.fa
    echo "✓ FASTA index created"
else
    echo "✓ hg38.fa.fai already exists, skipping"
fi

# 3. Create sequence dictionary for GATK
echo ""
echo "[3/5] Creating sequence dictionary..."
if [ ! -f "hg38.dict" ]; then
    gatk CreateSequenceDictionary -R hg38.fa -O hg38.dict
    echo "✓ Sequence dictionary created"
else
    echo "✓ hg38.dict already exists, skipping"
fi

# 4. Create BWA index
echo ""
echo "[4/5] Creating BWA index (~1 hour)..."
if [ ! -f "hg38.fa.bwt" ]; then
    bwa index hg38.fa
    echo "✓ BWA index created"
else
    echo "✓ BWA index already exists, skipping"
fi

# 5. Download dbSNP known sites for BQSR
echo ""
echo "[5/5] Downloading dbSNP known sites (~10GB)..."
cd "${KNOWN_SITES_DIR}"

if [ ! -f "dbsnp_155.hg38.vcf.gz" ]; then
    wget -c https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b155_GRCh38p7/VCF/00-All.vcf.gz \
        -O dbsnp_155.hg38.vcf.gz
    
    # Index the VCF
    tabix -p vcf dbsnp_155.hg38.vcf.gz
    echo "✓ dbSNP downloaded and indexed"
else
    echo "✓ dbsnp_155.hg38.vcf.gz already exists, skipping"
fi

# 6. Download gene annotations (GTF) for RNA-seq
echo ""
echo "[6/6] Downloading GENCODE gene annotations..."
cd "${ANNOT_DIR}"

if [ ! -f "gencode.v45.annotation.gtf.gz" ]; then
    wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz
    gunzip gencode.v45.annotation.gtf.gz
    echo "✓ GENCODE annotations downloaded"
else
    echo "✓ gencode.v45.annotation.gtf.gz already exists, skipping"
fi

echo ""
echo "========================================="
echo "✓ Reference download complete!"
echo "========================================="
echo ""
echo "Reference locations:"
echo "  Genome:      ${GENOME_DIR}/hg38.fa"
echo "  Known sites: ${KNOWN_SITES_DIR}/dbsnp_155.hg38.vcf.gz"
echo "  Annotations: ${ANNOT_DIR}/gencode.v45.annotation.gtf"
echo ""
echo "Disk usage:"
du -sh "${REF_DIR}"
echo ""
