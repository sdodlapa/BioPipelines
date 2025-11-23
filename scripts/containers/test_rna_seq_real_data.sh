#!/bin/bash
#SBATCH --job-name=test_rna_container
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/test_rna_container_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/test_rna_container_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

set -euo pipefail

echo "════════════════════════════════════════════════════════"
echo "Testing RNA-seq Container with Real Data"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "════════════════════════════════════════════════════════"

# Setup
CONTAINER="/home/sdodl001_odu_edu/BioPipelines/containers/images/rna-seq_1.0.0.sif"
INPUT_DIR="/scratch/sdodl001/BioPipelines/data/raw/rna_seq"
OUTPUT_DIR="/scratch/sdodl001/BioPipelines/test_container_output/rna_seq"
REF_DIR="/scratch/sdodl001/BioPipelines/data/references"
THREADS=8

# Create output directory
mkdir -p "$OUTPUT_DIR"/{fastqc,trimmed,aligned,counts}

echo ""
echo "Test 1: FastQC on raw reads"
echo "─────────────────────────────────────────"
singularity exec \
    --bind "$INPUT_DIR:/data:ro" \
    --bind "$OUTPUT_DIR:/results" \
    "$CONTAINER" \
    fastqc \
        -t $THREADS \
        -o /results/fastqc \
        /data/mut_rep1_R1.fastq.gz \
        /data/mut_rep1_R2.fastq.gz

echo "✓ FastQC completed"
ls -lh "$OUTPUT_DIR/fastqc"

echo ""
echo "Test 2: Fastp quality trimming"
echo "─────────────────────────────────────────"
singularity exec \
    --bind "$INPUT_DIR:/data:ro" \
    --bind "$OUTPUT_DIR:/results" \
    "$CONTAINER" \
    fastp \
        -i /data/mut_rep1_R1.fastq.gz \
        -I /data/mut_rep1_R2.fastq.gz \
        -o /results/trimmed/mut_rep1_R1_trimmed.fastq.gz \
        -O /results/trimmed/mut_rep1_R2_trimmed.fastq.gz \
        -h /results/trimmed/mut_rep1_fastp.html \
        -j /results/trimmed/mut_rep1_fastp.json \
        -w $THREADS

echo "✓ Fastp completed"
ls -lh "$OUTPUT_DIR/trimmed"

echo ""
echo "Test 3: Salmon quantification (quasi-mapping mode)"
echo "─────────────────────────────────────────"
# Check if salmon index exists
if [[ ! -d "$REF_DIR/salmon_index" ]]; then
    echo "⚠ Salmon index not found - skipping alignment test"
    echo "To run full test, build salmon index first:"
    echo "  singularity exec $CONTAINER salmon index -t transcripts.fa -i salmon_index"
else
    singularity exec \
        --bind "$OUTPUT_DIR:/results" \
        --bind "$REF_DIR:/references:ro" \
        "$CONTAINER" \
        salmon quant \
            -i /references/salmon_index \
            -l A \
            -1 /results/trimmed/mut_rep1_R1_trimmed.fastq.gz \
            -2 /results/trimmed/mut_rep1_R2_trimmed.fastq.gz \
            -p $THREADS \
            -o /results/aligned/mut_rep1_salmon

    echo "✓ Salmon quantification completed"
    ls -lh "$OUTPUT_DIR/aligned/mut_rep1_salmon"
fi

echo ""
echo "Test 4: MultiQC report generation"
echo "─────────────────────────────────────────"
singularity exec \
    --bind "$OUTPUT_DIR:/results" \
    "$CONTAINER" \
    multiqc \
        /results \
        -o /results \
        -n container_test_multiqc_report

echo "✓ MultiQC completed"
ls -lh "$OUTPUT_DIR"/*.html

echo ""
echo "════════════════════════════════════════════════════════"
echo "✓ All container tests passed!"
echo "End time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════"
