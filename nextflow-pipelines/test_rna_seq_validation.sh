#!/bin/bash
#SBATCH --job-name=rnaseq_validate
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/rnaseq_validate_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/rnaseq_validate_%j.err
#SBATCH --partition=cpuspot
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Validate existing RNA-seq container with Nextflow
# Purpose: Prove existing containers work (no need to build new ones)

WORKFLOW_ENGINE=/scratch/sdodl001/BioPipelines/containers/workflow-engine.sif
WORK_DIR="work_rnaseq_validate_${SLURM_JOB_ID}"

echo "=========================================="
echo "RNA-seq Container Validation Test"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Workflow Engine: ${WORKFLOW_ENGINE}"
echo "Working Directory: ${WORK_DIR}"
echo "=========================================="

cd /home/sdodl001_odu_edu/BioPipelines/nextflow-pipelines

# Clean up any previous test artifacts
rm -rf ${WORK_DIR} .nextflow* rnaseq_validate_*.{html,json} 2>/dev/null

# Run RNA-seq simple workflow using containerized Nextflow
# Key change: Remove -process.executor local to use default SLURM executor
# This allows processes to submit as SLURM jobs with access to singularity
singularity exec ${WORKFLOW_ENGINE} nextflow run \
    rnaseq_simple.nf \
    -work-dir ${WORK_DIR} \
    -with-trace rnaseq_validate_trace.txt \
    -with-timeline rnaseq_validate_timeline.html \
    -with-report rnaseq_validate_report.html

EXIT_CODE=$?

echo "=========================================="
echo "Workflow completed with exit code: ${EXIT_CODE}"
echo "=========================================="

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ SUCCESS: Existing container infrastructure validated!"
    echo "Next steps: Create tool catalog and expand module library"
else
    echo "❌ FAILED: Workflow execution failed"
    echo "Check logs for details"
fi

exit ${EXIT_CODE}
