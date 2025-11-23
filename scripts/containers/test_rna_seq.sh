#!/bin/bash
# Test RNA-seq container functionality

set -euo pipefail

CONTAINER="/scratch/sdodl001/containers/rna-seq_1.0.0.sif"

echo "════════════════════════════════════════"
echo "Testing RNA-seq Container"
echo "════════════════════════════════════════"

if [[ ! -f "$CONTAINER" ]]; then
    echo "❌ Container not found: $CONTAINER"
    exit 1
fi

echo "✓ Container found: $CONTAINER"
echo "  Size: $(du -h $CONTAINER | cut -f1)"
echo ""

# Test 1: Base tools (inherited)
echo "Test 1: Base tools"
singularity exec "$CONTAINER" samtools --version | head -1
singularity exec "$CONTAINER" fastqc --version
echo "✓ Base tools work"
echo ""

# Test 2: fastp
echo "Test 2: fastp"
singularity exec "$CONTAINER" fastp --version 2>&1 | head -1
echo "✓ fastp works"
echo ""

# Test 3: STAR
echo "Test 3: STAR aligner"
singularity exec "$CONTAINER" STAR --version
echo "✓ STAR works"
echo ""

# Test 4: Salmon
echo "Test 4: Salmon"
singularity exec "$CONTAINER" salmon --version
echo "✓ Salmon works"
echo ""

# Test 5: featureCounts
echo "Test 5: featureCounts"
singularity exec "$CONTAINER" featureCounts -v 2>&1 | head -1
echo "✓ featureCounts works"
echo ""

# Test 6: R and Bioconductor
echo "Test 6: R and DESeq2"
singularity exec "$CONTAINER" R --version | head -1
singularity exec "$CONTAINER" Rscript -e "library(DESeq2); cat('DESeq2 version:', as.character(packageVersion('DESeq2')), '\n')"
echo "✓ R and DESeq2 work"
echo ""

# Test 7: Entrypoint help
echo "Test 7: Container entrypoint"
singularity run "$CONTAINER" --help | head -5
echo "✓ Entrypoint works"
echo ""

echo "════════════════════════════════════════"
echo "✓ All RNA-seq container tests passed!"
echo "════════════════════════════════════════"
