#!/bin/bash
#SBATCH --job-name=container_simple_test
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/container_simple_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/container_simple_%j.err
#SBATCH --partition=cpuspot
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Simple direct test - just run containerized Nextflow with bind mounts for SLURM
# This validates containers work without nested SLURM submission

WORKFLOW_ENGINE=/scratch/sdodl001/BioPipelines/containers/workflow-engine.sif
WORK_DIR="work_simple_${SLURM_JOB_ID}"

echo "=========================================="
echo "Simple Container Test (No nested SLURM)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

cd /home/sdodl001_odu_edu/BioPipelines/nextflow-pipelines

# Clean up
rm -rf ${WORK_DIR} .nextflow* container_simple_*.{html,json} 2>/dev/null

# Run with local executor AND bind mount SLURM binaries so Nextflow can call singularity
singularity exec \
    --bind /usr/bin/sbatch:/usr/bin/sbatch \
    --bind /usr/bin/squeue:/usr/bin/squeue \
    --bind /usr/bin/scancel:/usr/bin/scancel \
    --bind /usr/bin/singularity:/usr/bin/singularity \
    ${WORKFLOW_ENGINE} \
    nextflow run container_validation.nf \
    -work-dir ${WORK_DIR} \
    -process.executor local \
    -with-singularity

EXIT_CODE=$?

echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ SUCCESS: Containers validated!"
    echo ""
    cat results/container_validation/tests/*.txt 2>/dev/null || echo "Output files in work directory"
else
    echo "❌ FAILED: Exit code ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
