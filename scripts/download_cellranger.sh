#!/bin/bash
#SBATCH --job-name=download_cellranger
#SBATCH --output=logs/download_cellranger_%j.out
#SBATCH --error=logs/download_cellranger_%j.err
#SBATCH --partition=cpuspot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00

# Download and install CellRanger for scRNA-seq analysis
# ======================================================

echo "Starting CellRanger download and installation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

TOOLS_DIR="/scratch/sdodl001/BioPipelines/tools"
CELLRANGER_VERSION="9.0.0"
CELLRANGER_URL="https://cf.10xgenomics.com/releases/cell-exp/cellranger-${CELLRANGER_VERSION}.tar.gz"

mkdir -p $TOOLS_DIR
cd $TOOLS_DIR

echo ""
echo "Downloading CellRanger ${CELLRANGER_VERSION}..."
echo "URL: ${CELLRANGER_URL}"
echo ""

# Download CellRanger
wget -O cellranger-${CELLRANGER_VERSION}.tar.gz "${CELLRANGER_URL}"

if [ $? -eq 0 ]; then
    echo ""
    echo "Download successful!"
    echo "Extracting..."
    tar -xzf cellranger-${CELLRANGER_VERSION}.tar.gz
    
    echo ""
    echo "CellRanger installed at: ${TOOLS_DIR}/cellranger-${CELLRANGER_VERSION}"
    echo "Add to PATH: export PATH=${TOOLS_DIR}/cellranger-${CELLRANGER_VERSION}:$PATH"
    echo ""
    echo "Cleaning up tarball..."
    rm cellranger-${CELLRANGER_VERSION}.tar.gz
    
    echo ""
    echo "Installation complete!"
    ls -lh ${TOOLS_DIR}/cellranger-${CELLRANGER_VERSION}/ | head -10
else
    echo ""
    echo "Download failed! Check if URL is valid or requires authentication."
    echo "Manual download: https://www.10xgenomics.com/support/software/cell-ranger/downloads"
    exit 1
fi

echo ""
echo "End time: $(date)"
