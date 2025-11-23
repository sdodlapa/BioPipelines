#!/bin/bash
#SBATCH --job-name=test_micromamba
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/test_micromamba_%j.out
#SBATCH --error=logs/test_micromamba_%j.err

echo "Testing on compute node: $(hostname)"
echo ""

echo "1. Checking micromamba binary:"
ls -lh ~/bin/micromamba 2>&1

echo ""
echo "2. Testing micromamba activation:"
export MAMBA_EXE="$HOME/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)" 2>&1

echo ""
echo "3. Activating base environment:"
micromamba activate 2>&1

echo ""
echo "4. Checking Snakemake:"
which snakemake 2>&1
snakemake --version 2>&1

echo ""
echo "5. Checking Python:"
which python 2>&1
python --version 2>&1

echo ""
echo "6. Checking Singularity:"
which singularity 2>&1
singularity --version 2>&1

echo ""
echo "TEST COMPLETE"
