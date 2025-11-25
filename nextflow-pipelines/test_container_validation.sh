#!/bin/bash
#SBATCH --job-name=container_test
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/container_test_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/container_test_%j.err
#SBATCH --partition=cpuspot
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Test existing containers with Nextflow
# This validates the entire infrastructure

WORKFLOW_ENGINE=/scratch/sdodl001/BioPipelines/containers/workflow-engine.sif
WORK_DIR="work_container_test_${SLURM_JOB_ID}"

echo "=========================================="
echo "Container Validation Test"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

cd /home/sdodl001_odu_edu/BioPipelines/nextflow-pipelines

# Clean up
rm -rf ${WORK_DIR} .nextflow* container_test_*.{html,json} 2>/dev/null

# Run validation workflow
singularity exec ${WORKFLOW_ENGINE} nextflow run \
    container_validation.nf \
    -work-dir ${WORK_DIR} \
    -with-trace container_test_trace.txt \
    -with-report container_test_report.html

EXIT_CODE=$?

echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ SUCCESS: Container infrastructure validated!"
    cat results/container_validation/tests/*.txt
else
    echo "❌ FAILED: Exit code ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
