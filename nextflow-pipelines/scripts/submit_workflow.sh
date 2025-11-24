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

# Generate unique run name with timestamp to prevent session conflicts
RUN_NAME="${WORKFLOW_NAME}_$(date +%Y%m%d_%H%M%S)"

# Create logs directory
mkdir -p logs

echo "Submitting workflow: $WORKFLOW_NAME"
echo "Unique run name: $RUN_NAME"

# Submit with custom job name and unique session name
# The -name flag creates a unique Nextflow session, preventing lock conflicts
sbatch --job-name="nf_${WORKFLOW_NAME}" \
       --output="logs/${WORKFLOW_NAME}_%j.out" \
       --error="logs/${WORKFLOW_NAME}_%j.err" \
       scripts/submit_nextflow.sh "$WORKFLOW_FILE" -name "$RUN_NAME" "${@:2}"
