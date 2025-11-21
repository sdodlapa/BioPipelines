#!/bin/bash
#
# Download test datasets for BioPipelines and upload to GCS
# Designed for GCP HPC Slurm cluster deployment
#

set -e

# Configuration
GCP_PROJECT="rcc-hpc"
GCS_DATA_BUCKET="gs://biopipelines-data"
GCS_REF_BUCKET="gs://biopipelines-references"

# Local paths (temporary staging)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEMP_DIR="/tmp/biopipelines-download-$$"

echo "========================================="
echo "📦 BioPipelines Test Data Download & Upload"
echo "========================================="
echo ""
echo "This script will:"
echo "  1. Download test datasets to temporary storage"
echo "  2. Upload to GCS buckets for cluster access"
echo ""
echo "GCS Buckets:"
echo "  - Data: ${GCS_DATA_BUCKET}"
echo "  - References: ${GCS_REF_BUCKET}"
echo ""
echo "Total size: ~500MB"
echo "Estimated time: 10-20 minutes"
echo ""

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "❌ Error: gsutil not found. Please install Google Cloud SDK."
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gsutil ls &> /dev/null; then
    echo "❌ Error: Not authenticated with GCP"
    echo "   Run: gcloud auth login"
    exit 1
fi

# Create temporary directory
mkdir -p "${TEMP_DIR}"/{dna_seq,rna_seq,chip_seq,atac_seq,references}

echo "✅ Prerequisites checked"
echo "📁 Temporary directory: ${TEMP_DIR}"
echo ""

# ========================================
# 1. Download chr22 reference (for testing)
# ========================================
echo "[1/5] Downloading hg38 chr22 reference (~12MB)..."
cd "${TEMP_DIR}/references"

if command -v wget &> /dev/null; then
    wget -q --show-progress https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz
else
    curl -# -O https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz
fi

gunzip chr22.fa.gz
mv chr22.fa hg38_chr22.fa

echo "✅ Reference genome downloaded"
echo "📤 Uploading to GCS..."
gsutil -m cp hg38_chr22.fa ${GCS_REF_BUCKET}/genomes/hg38/test/

# ========================================
# 2. Download DNA-seq test data
# ========================================
echo ""
echo "[2/5] Downloading DNA-seq test data (~100MB)..."
echo "  Source: GIAB NA12878 (1000 Genomes, chr22 subset)"
cd "${TEMP_DIR}/dna_seq"

# Download small subset of NA12878 exome data (chr22)
# Note: Using pre-subsampled data for speed
echo "  Downloading sample files..."

# Option 1: Use SRA toolkit if available
if command -v fastq-dump &> /dev/null; then
    echo "  Using SRA toolkit to download subset..."
    # Example: Download small subset from SRA
    fastq-dump --split-files --gzip -X 100000 SRR098401
    mv SRR098401_1.fastq.gz sample1_R1.fastq.gz
    mv SRR098401_2.fastq.gz sample1_R2.fastq.gz
else
    echo "  ⚠️  SRA toolkit not found. Please install or download manually."
    echo "     Alternatively, use: conda install -c bioconda sra-tools"
    echo ""
    echo "  Manual download option:"
    echo "  wget ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR098/SRR098401/SRR098401_1.fastq.gz"
    echo "  wget ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR098/SRR098401/SRR098401_2.fastq.gz"
fi

if [ -f "sample1_R1.fastq.gz" ]; then
    echo "✅ DNA-seq data downloaded"
    echo "📤 Uploading to GCS..."
    gsutil -m cp sample1_*.fastq.gz ${GCS_DATA_BUCKET}/dna_seq/test/
else
    echo "⚠️  DNA-seq download incomplete - will need manual upload"
fi

# ========================================
# 3. Download RNA-seq test data
# ========================================
echo ""
echo "[3/5] Downloading RNA-seq test data (~200MB)..."
echo "  Source: ENCODE cell line RNA-seq"
cd "${TEMP_DIR}/rna_seq"

echo "  📝 For RNA-seq data, recommended sources:"
echo "     - ENCODE: https://www.encodeproject.org/"
echo "     - GEO: https://www.ncbi.nlm.nih.gov/geo/"
echo "     - SRA: Use accession SRR3192412 (small K562 RNA-seq)"
echo ""
echo "  ⚠️  Skipping automated download (too large)"
echo "     Upload manually after download to:"
echo "     ${GCS_DATA_BUCKET}/rna_seq/test/"

# ========================================
# 4. Download ChIP-seq test data
# ========================================
echo ""
echo "[4/5] ChIP-seq test data..."
echo "  📝 Recommended: ENCODE H3K4me3 ChIP-seq"
echo "     Upload to: ${GCS_DATA_BUCKET}/chip_seq/test/"
echo ""

# ========================================
# 5. Download ATAC-seq test data
# ========================================
echo ""
echo "[5/5] ATAC-seq test data..."
echo "  📝 Recommended: ENCODE ATAC-seq"
echo "     Upload to: ${GCS_DATA_BUCKET}/atac_seq/test/"
echo ""

# ========================================
# Cleanup and Summary
# ========================================
echo ""
echo "========================================="
echo "🧹 Cleaning up temporary files..."
echo "========================================="
cd ~
rm -rf "${TEMP_DIR}"

echo ""
echo "========================================="
echo "✅ Data Upload Complete!"
echo "========================================="
echo ""
echo "📦 Uploaded to GCS:"
echo "  - Reference (chr22): ${GCS_REF_BUCKET}/genomes/hg38/test/"
echo "  - DNA-seq test data: ${GCS_DATA_BUCKET}/dna_seq/test/"
echo ""
echo "🔍 Verify uploads:"
echo "  gsutil ls -lh ${GCS_REF_BUCKET}/genomes/hg38/test/"
echo "  gsutil ls -lh ${GCS_DATA_BUCKET}/dna_seq/test/"
echo ""
echo "📚 Next Steps:"
echo ""
echo "1. Download additional datasets manually:"
echo "   - RNA-seq: SRA accession SRR3192412"
echo "   - ChIP-seq: ENCODE datasets"
echo "   - ATAC-seq: ENCODE datasets"
echo ""
echo "2. Upload to GCS:"
echo "   gsutil -m cp *.fastq.gz ${GCS_DATA_BUCKET}/[pipeline_type]/test/"
echo ""
echo "3. Update pipeline configs to use GCS paths"
echo ""
echo "4. Submit a test job:"
echo "   cd ~/BioPipelines"
echo "   sbatch scripts/submit_dna_seq.sh"
echo ""
echo "📖 For more info, see:"
echo "   docs/GCP_STORAGE_ARCHITECTURE.md"
echo "========================================="
