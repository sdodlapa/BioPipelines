#!/bin/bash
# Monitor Methylation (356) and scRNA-seq (375) to completion
# Log outputs to file for later review

LOG_FILE="/home/sdodl001_odu_edu/BioPipelines/logs/final_monitoring_$(date +%Y%m%d_%H%M%S).log"
METHYLATION_JOB=356
SCRNA_JOB=375

echo "=== Final Pipeline Monitoring Started: $(date) ===" | tee -a "$LOG_FILE"
echo "Methylation Job: $METHYLATION_JOB" | tee -a "$LOG_FILE"
echo "scRNA-seq Job: $SCRNA_JOB" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to check if job is still running
is_running() {
    squeue --me | grep -q "$1"
}

# Function to get job status
get_status() {
    sacct -j "$1" --format=State,Elapsed,ExitCode --noheader | head -1
}

# Monitor loop
ITERATION=0
MAX_ITERATIONS=120  # 120 * 3 min = 6 hours max

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))
    
    echo "=== Check #$ITERATION at $(date) ===" | tee -a "$LOG_FILE"
    
    # Check Methylation
    METH_STATUS=$(get_status $METHYLATION_JOB)
    echo "Methylation ($METHYLATION_JOB): $METH_STATUS" | tee -a "$LOG_FILE"
    
    # Count sequences in methylation log if running
    if is_running $METHYLATION_JOB; then
        METH_SEQS=$(grep -oP 'Sequences processed: \K\d+' ~/BioPipelines/logs/methylation_${METHYLATION_JOB}.out 2>/dev/null | tail -1)
        if [ -n "$METH_SEQS" ]; then
            echo "  └─ Sequences processed: $METH_SEQS" | tee -a "$LOG_FILE"
        fi
        TRIM_PROCS=$(pgrep -u $USER trim_galore | wc -l)
        echo "  └─ trim_galore processes: $TRIM_PROCS" | tee -a "$LOG_FILE"
    fi
    
    # Check scRNA-seq
    SCRNA_STATUS=$(get_status $SCRNA_JOB)
    echo "scRNA-seq ($SCRNA_JOB): $SCRNA_STATUS" | tee -a "$LOG_FILE"
    
    # Check scRNA-seq progress from snakemake log
    if is_running $SCRNA_JOB; then
        SCRNA_STEP=$(tail -50 ~/BioPipelines/logs/scrna_seq_${SCRNA_JOB}.out 2>/dev/null | grep -E "rule (starsolo|initial_qc|detect_doublets|filter_cells|normalize|feature_selection|umap|cluster|annotate|report):" | tail -1)
        if [ -n "$SCRNA_STEP" ]; then
            echo "  └─ Current step: $SCRNA_STEP" | tee -a "$LOG_FILE"
        fi
    fi
    
    echo "" | tee -a "$LOG_FILE"
    
    # Check if both jobs completed or failed
    METH_RUNNING=$(is_running $METHYLATION_JOB && echo "yes" || echo "no")
    SCRNA_RUNNING=$(is_running $SCRNA_JOB && echo "yes" || echo "no")
    
    if [ "$METH_RUNNING" = "no" ] && [ "$SCRNA_RUNNING" = "no" ]; then
        echo "=== Both jobs completed at $(date) ===" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "Final status:" | tee -a "$LOG_FILE"
        sacct -j ${METHYLATION_JOB},${SCRNA_JOB} --format=JobID,JobName,State,Elapsed,ExitCode | tee -a "$LOG_FILE"
        break
    fi
    
    # Wait 3 minutes before next check
    sleep 180
done

if [ $ITERATION -eq $MAX_ITERATIONS ]; then
    echo "=== Monitoring timeout after 6 hours ===" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Monitoring log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
