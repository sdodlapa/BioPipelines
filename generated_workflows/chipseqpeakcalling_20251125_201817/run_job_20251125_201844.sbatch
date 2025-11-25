#!/bin/bash
#SBATCH --job-name=nf_chipseqpeakcalling_20251125_201817
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/generated_workflows/chipseqpeakcalling_20251125_201817/logs/slurm_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/generated_workflows/chipseqpeakcalling_20251125_201817/logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpuspot

# Load required modules
module load nextflow 2>/dev/null || true
module load singularity 2>/dev/null || true

# Activate conda if available
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate biopipelines 2>/dev/null || true

# Change to workflow directory
cd /home/sdodl001_odu_edu/BioPipelines/generated_workflows/chipseqpeakcalling_20251125_201817

# Run Nextflow
nextflow run /home/sdodl001_odu_edu/BioPipelines/generated_workflows/chipseqpeakcalling_20251125_201817/main.nf -profile slurm -with-report /home/sdodl001_odu_edu/BioPipelines/generated_workflows/chipseqpeakcalling_20251125_201817/logs/report.html -with-timeline /home/sdodl001_odu_edu/BioPipelines/generated_workflows/chipseqpeakcalling_20251125_201817/logs/timeline.html -with-dag /home/sdodl001_odu_edu/BioPipelines/generated_workflows/chipseqpeakcalling_20251125_201817/logs/dag.png --outdir /home/sdodl001_odu_edu/BioPipelines/generated_workflows/chipseqpeakcalling_20251125_201817/results 2>&1 | tee /home/sdodl001_odu_edu/BioPipelines/generated_workflows/chipseqpeakcalling_20251125_201817/logs/nextflow_job_20251125_201844.log

echo "Pipeline finished at $(date)"
