#!/bin/bash
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# BioPipelines - Nextflow Workflow Submission Script
# This script submits a Nextflow workflow to SLURM
# The workflow itself will launch additional SLURM jobs via the executor

# Usage: Use scripts/submit_workflow.sh instead for automatic job naming
# Direct usage: sbatch scripts/submit_nextflow.sh <workflow.nf> [nextflow args...]
# Example: sbatch scripts/submit_nextflow.sh tests/test_fastqc_real.nf -resume

set -euo pipefail

# Activate nextflow environment (for Java 17)
eval "$(micromamba shell hook --shell bash)"
micromamba activate nextflow

# Create logs directory
mkdir -p logs

# Get workflow file from first argument
WORKFLOW_FILE="${1:-}"
if [ -z "$WORKFLOW_FILE" ]; then
    echo "ERROR: No workflow file specified"
    echo "Usage: sbatch scripts/submit_nextflow.sh <workflow.nf> [nextflow args...]"
    exit 1
fi
shift  # Remove first argument, rest are nextflow args

# Check if workflow file exists
if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "ERROR: Workflow file not found: $WORKFLOW_FILE"
    exit 1
fi

# Print job info
echo "=========================================="
echo "Nextflow Workflow Submission"
echo "=========================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Node:             $SLURMD_NODENAME"
echo "Start Time:       $(date)"
echo "Workflow:         $WORKFLOW_FILE"
echo "Extra Args:       $@"
echo "=========================================="
echo ""

# Run Nextflow workflow
# The workflow will submit its own SLURM jobs via the executor
cd ~/BioPipelines/nextflow-pipelines

nextflow run "$WORKFLOW_FILE" \
    -c config/base.config \
    -c config/containers.config \
    "$@"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Workflow completed with exit code: $EXIT_CODE"
echo "End Time:         $(date)"
echo "=========================================="

exit $EXIT_CODE
