#!/bin/bash
# Quick Start Test Script for BioPipelines
# This script helps you validate that the DNA-seq pipeline works

set -e

echo "🧬 BioPipelines Quick Start Test"
echo "================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install conda/miniconda first"
    exit 1
fi

echo "✅ Conda found"

# Check if environment exists
if conda env list | grep -q "^biopipelines "; then
    echo "✅ biopipelines environment exists"
else
    echo "📦 Creating biopipelines environment..."
    conda env create -f environment.yml
fi

# Activate environment
echo "🔧 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate biopipelines

# Install Python package in development mode
echo "📦 Installing Python package..."
pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Download test data: ./scripts/download_test_data.sh"
echo "2. Test DNA-seq pipeline:"
echo "   cd pipelines/dna_seq/variant_calling"
echo "   # Edit config.yaml with your sample information"
echo "   snakemake --cores 4 --use-conda -n  # Dry run"
echo "   snakemake --cores 4 --use-conda     # Real run"
echo ""
echo "📚 Documentation:"
echo "   - See DEVELOPMENT_STATUS.md for project status"
echo "   - See TODO.md for next steps"
echo "   - See README.md for full documentation"

