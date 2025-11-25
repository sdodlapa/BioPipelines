#!/bin/bash
#SBATCH --job-name=build_peak_calling
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpuspot
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/scripts/../logs/tier2_builds/build_peak_calling_20251124_232148.log
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/scripts/../logs/tier2_builds/build_peak_calling_20251124_232148.err

set -e

echo "=========================================="
echo "Building: peak_calling"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="
echo ""

# Load Singularity module if available
module load singularity 2>/dev/null || true

# Build container using fakeroot
echo "Starting build with fakeroot..."
singularity build --fakeroot \
    /scratch/sdodl001/BioPipelines/containers/tier2/peak_calling.sif \
    /home/sdodl001_odu_edu/BioPipelines/scripts/../containers/tier2/peak_calling.def

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Build completed successfully"
    echo "=========================================="
    
    # Get container info
    echo ""
    echo "Container information:"
    singularity inspect /scratch/sdodl001/BioPipelines/containers/tier2/peak_calling.sif
    
    # Run validation tests
    echo ""
    echo "Running validation tests..."
    singularity test /scratch/sdodl001/BioPipelines/containers/tier2/peak_calling.sif
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ Validation passed"
        echo "=========================================="
        
        # Set permissions
        chmod 755 /scratch/sdodl001/BioPipelines/containers/tier2/peak_calling.sif
        
        # Get final size
        SIZE=$(du -h /scratch/sdodl001/BioPipelines/containers/tier2/peak_calling.sif | cut -f1)
        echo ""
        echo "Container size: ${SIZE}"
        echo "Location: /scratch/sdodl001/BioPipelines/containers/tier2/peak_calling.sif"
        
        exit 0
    else
        echo ""
        echo "=========================================="
        echo "❌ Validation failed"
        echo "=========================================="
        exit 1
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Build failed"
    echo "=========================================="
    exit 1
fi
