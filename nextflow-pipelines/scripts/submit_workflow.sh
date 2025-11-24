#!/bin/bash
# Wrapper script to submit Nextflow workflows with custom job names
# Usage: ./scripts/submit_workflow.sh <workflow.nf> [nextflow args...]

set -euo pipefail

WORKFLOW_FILE="${1:-}"
if [ -z "$WORKFLOW_FILE" ]; then
    echo "ERROR: No workflow file specified"
    echo "Usage: ./scripts/submit_workflow.sh <workflow.nf> [nextflow args...]"
    exit 1
fi

if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "ERROR: Workflow file not found: $WORKFLOW_FILE"
    exit 1
fi

# Extract workflow name from file path
WORKFLOW_NAME=$(basename "$WORKFLOW_FILE" .nf)

# Create logs directory
mkdir -p logs

# Submit with custom job name
sbatch --job-name="nf_${WORKFLOW_NAME}" \
       --output="logs/${WORKFLOW_NAME}_%j.out" \
       --error="logs/${WORKFLOW_NAME}_%j.err" \
       scripts/submit_nextflow.sh "$@"
