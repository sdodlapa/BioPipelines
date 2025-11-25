#!/bin/bash
#SBATCH --job-name=test_rnaseq_validated
#SBATCH --output=logs/rnaseq_test_%j.out
#SBATCH --error=logs/rnaseq_test_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpuspot

set -euo pipefail

echo "=========================================="
echo "Testing RNA-seq with Existing Container"
echo "=========================================="
echo "Container: /home/sdodl001_odu_edu/BioPipelines/containers/images/rna-seq_1.0.0.sif"
echo "Workflow: workflows/rnaseq_simple.nf"
echo "Data: mut_rep1 (paired-end)"
echo ""

# Use workflow engine container for Nextflow
WORKFLOW_ENGINE=/scratch/sdodl001/BioPipelines/containers/workflow-engine.sif
WORK_DIR=/scratch/sdodl001/BioPipelines/test_rnaseq_$$

mkdir -p $WORK_DIR

singularity exec $WORKFLOW_ENGINE nextflow run workflows/rnaseq_simple.nf \
    -work-dir $WORK_DIR \
    -process.executor local \
    -with-report ${WORK_DIR}/report.html \
    -with-timeline ${WORK_DIR}/timeline.html

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo "Check outputs in: $WORK_DIR"
ls -lh $WORK_DIR/
