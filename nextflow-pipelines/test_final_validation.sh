#!/bin/bash
#SBATCH --job-name=final_validation
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/final_validation_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/final_validation_%j.err
#SBATCH --partition=cpuspot
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Final validation: Use native Nextflow with local executor
# Containers called by Nextflow for each process

echo "=========================================="
echo "Final Container Validation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Approach: Native Nextflow + Local Executor + Process Containers"
echo "=========================================="

cd /home/sdodl001_odu_edu/BioPipelines/nextflow-pipelines

# Load module for Nextflow if available, or use workflow-engine on login node
module load nextflow 2>/dev/null || export PATH="/cm/shared/apps/nextflow/24.10.0/bin:$PATH"

WORK_DIR="work_final_${SLURM_JOB_ID}"

# Clean up
rm -rf ${WORK_DIR} .nextflow* final_validation_*.{html,json} 2>/dev/null

echo "Nextflow version:"
nextflow -version

echo ""
echo "Running validation workflow..."
echo ""

# Run with local executor - processes run locally but IN their containers
nextflow run container_validation.nf \
    -work-dir ${WORK_DIR} \
    -process.executor local \
    -with-singularity \
    -with-trace final_validation_trace.txt \
    -with-report final_validation_report.html

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ SUCCESS: Containers validated!"
    echo ""
    echo "RNA-seq container tools:"
    cat results/container_validation/tests/rnaseq_tools.txt 2>/dev/null || echo "  (check work directory)"
    echo ""
    echo "DNA-seq container tools:"
    cat results/container_validation/tests/dnaseq_tools.txt 2>/dev/null || echo "  (check work directory)"
else
    echo "❌ FAILED: Exit code ${EXIT_CODE}"
    echo "Check .nextflow.log for details"
fi
echo "=========================================="

exit ${EXIT_CODE}
