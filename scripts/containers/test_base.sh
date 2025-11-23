#!/bin/bash
# Test base container functionality

set -euo pipefail

CONTAINER="/home/sdodl001_odu_edu/BioPipelines/containers/images/base_1.0.0.sif"

echo "════════════════════════════════════════"
echo "Testing Base Container"
echo "════════════════════════════════════════"

if [[ ! -f "$CONTAINER" ]]; then
    echo "❌ Container not found: $CONTAINER"
    exit 1
fi

echo "✓ Container found: $CONTAINER"
echo "  Size: $(du -h $CONTAINER | cut -f1)"
echo ""

# Test 1: samtools
echo "Test 1: samtools"
singularity exec "$CONTAINER" samtools --version | head -1
echo "✓ samtools works"
echo ""

# Test 2: bcftools
echo "Test 2: bcftools"
singularity exec "$CONTAINER" bcftools --version | head -1
echo "✓ bcftools works"
echo ""

# Test 3: bedtools
echo "Test 3: bedtools"
singularity exec "$CONTAINER" bedtools --version
echo "✓ bedtools works"
echo ""

# Test 4: fastqc
echo "Test 4: fastqc"
singularity exec "$CONTAINER" fastqc --version
echo "✓ fastqc works"
echo ""

# Test 5: multiqc
echo "Test 5: multiqc"
singularity exec "$CONTAINER" multiqc --version
echo "✓ multiqc works"
echo ""

# Test 6: Python and libraries
echo "Test 6: Python and libraries"
singularity exec "$CONTAINER" python3 -c "
import numpy as np
import pandas as pd
import scipy
import matplotlib
import seaborn
import Bio
import pysam
print('Python:', __import__('sys').version.split()[0])
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('✓ All Python libraries work')
"
echo ""

echo "════════════════════════════════════════"
echo "✓ All base container tests passed!"
echo "════════════════════════════════════════"
