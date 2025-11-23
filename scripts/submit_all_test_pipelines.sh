#!/bin/bash
# Submit test runs for all successfully built containers
# Uses minimal test data for validation

set -euo pipefail

CONTAINER_DIR="/home/sdodl001_odu_edu/BioPipelines/containers/images"
DATA_DIR="/scratch/sdodl001/BioPipelines/data"
REFS_DIR="/scratch/sdodl001/BioPipelines/data/references"
LOG_DIR="$HOME/BioPipelines/logs/pipeline_tests"
PARTITION="cpuspot"

mkdir -p "$LOG_DIR"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════"
echo "  BioPipelines Container Test Submission"
echo "════════════════════════════════════════════════════"
echo ""

# Check which containers are available
declare -a AVAILABLE_CONTAINERS=()
declare -a PIPELINES=(
    "rna-seq"
    "dna-seq" 
    "chip-seq"
    "atac-seq"
    "hic"
    "long-read"
    "scrna-seq"
    "metagenomics"
    "methylation"
    "structural-variants"
)

echo "Checking available containers..."
for pipeline in "${PIPELINES[@]}"; do
    container="${CONTAINER_DIR}/${pipeline}_1.0.0.sif"
    if [[ -f "$container" ]]; then
        AVAILABLE_CONTAINERS+=("$pipeline")
        echo -e "  ${GREEN}✓${NC} $pipeline"
    else
        echo -e "  ${YELLOW}⏳${NC} $pipeline (container not ready)"
    fi
done

echo ""
echo "════════════════════════════════════════════════════"
echo "Found ${#AVAILABLE_CONTAINERS[@]} ready containers"
echo "════════════════════════════════════════════════════"
echo ""

# Function to submit a test job
submit_test() {
    local pipeline=$1
    local input_dir=$2
    local output_dir=$3
    local extra_args=${4:-}
    
    local container="${CONTAINER_DIR}/${pipeline}_1.0.0.sif"
    local job_name="test_${pipeline}_$(date +%H%M%S)"
    
    # Check if input data exists
    if [[ ! -d "$input_dir" ]]; then
        echo -e "${YELLOW}⊘${NC} $pipeline: No test data in $input_dir"
        return 1
    fi
    
    # Check if directory is empty
    if [[ -z "$(ls -A "$input_dir" 2>/dev/null)" ]]; then
        echo -e "${YELLOW}⊘${NC} $pipeline: Empty test data directory"
        return 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Submit via sbatch
    cat <<EOF | sbatch --parsable
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=$LOG_DIR/${job_name}.out
#SBATCH --error=$LOG_DIR/${job_name}.err

set -euo pipefail

echo "════════════════════════════════════════"
echo "Pipeline: $pipeline"
echo "Container: $container"
echo "Input: $input_dir"
echo "Output: $output_dir"
echo "════════════════════════════════════════"
echo ""

cd \$SLURM_SUBMIT_DIR

# Test container execution
singularity exec \\
    --bind $input_dir:/data/input:ro \\
    --bind $output_dir:/data/output \\
    $container \\
    bash -c "echo 'Container loaded successfully' && ls /data/input | head -5"

EXIT_CODE=\$?

echo ""
echo "════════════════════════════════════════"
if [[ \$EXIT_CODE -eq 0 ]]; then
    echo "✓ Container test passed: $pipeline"
else
    echo "✗ Container test failed: $pipeline"
fi
echo "════════════════════════════════════════"

exit \$EXIT_CODE
EOF
}

# Submit tests for available pipelines
declare -a SUBMITTED_JOBS=()

echo "Submitting test jobs..."
echo ""

# RNA-seq
if [[ " ${AVAILABLE_CONTAINERS[@]} " =~ " rna-seq " ]]; then
    jobid=$(submit_test "rna-seq" \
        "$DATA_DIR/raw/rna_seq" \
        "$DATA_DIR/results/rna_seq/test_$(date +%Y%m%d)")
    if [[ -n "$jobid" ]]; then
        SUBMITTED_JOBS+=("$jobid:rna-seq")
        echo -e "${GREEN}✓${NC} rna-seq (Job $jobid)"
    fi
fi

# DNA-seq
if [[ " ${AVAILABLE_CONTAINERS[@]} " =~ " dna-seq " ]]; then
    jobid=$(submit_test "dna-seq" \
        "$DATA_DIR/raw/dna_seq" \
        "$DATA_DIR/results/dna_seq/test_$(date +%Y%m%d)")
    if [[ -n "$jobid" ]]; then
        SUBMITTED_JOBS+=("$jobid:dna-seq")
        echo -e "${GREEN}✓${NC} dna-seq (Job $jobid)"
    fi
fi

# ChIP-seq
if [[ " ${AVAILABLE_CONTAINERS[@]} " =~ " chip-seq " ]]; then
    jobid=$(submit_test "chip-seq" \
        "$DATA_DIR/raw/chip_seq" \
        "$DATA_DIR/results/chip_seq/test_$(date +%Y%m%d)")
    if [[ -n "$jobid" ]]; then
        SUBMITTED_JOBS+=("$jobid:chip-seq")
        echo -e "${GREEN}✓${NC} chip-seq (Job $jobid)"
    fi
fi

# ATAC-seq
if [[ " ${AVAILABLE_CONTAINERS[@]} " =~ " atac-seq " ]]; then
    jobid=$(submit_test "atac-seq" \
        "$DATA_DIR/raw/atac_seq" \
        "$DATA_DIR/results/atac_seq/test_$(date +%Y%m%d)")
    if [[ -n "$jobid" ]]; then
        SUBMITTED_JOBS+=("$jobid:atac-seq")
        echo -e "${GREEN}✓${NC} atac-seq (Job $jobid)"
    fi
fi

# Hi-C
if [[ " ${AVAILABLE_CONTAINERS[@]} " =~ " hic " ]]; then
    jobid=$(submit_test "hic" \
        "$DATA_DIR/raw/hic" \
        "$DATA_DIR/results/hic/test_$(date +%Y%m%d)")
    if [[ -n "$jobid" ]]; then
        SUBMITTED_JOBS+=("$jobid:hic")
        echo -e "${GREEN}✓${NC} hic (Job $jobid)"
    fi
fi

# Long-read
if [[ " ${AVAILABLE_CONTAINERS[@]} " =~ " long-read " ]]; then
    jobid=$(submit_test "long-read" \
        "$DATA_DIR/raw/long_read" \
        "$DATA_DIR/results/long_read/test_$(date +%Y%m%d)")
    if [[ -n "$jobid" ]]; then
        SUBMITTED_JOBS+=("$jobid:long-read")
        echo -e "${GREEN}✓${NC} long-read (Job $jobid)"
    fi
fi

# scRNA-seq
if [[ " ${AVAILABLE_CONTAINERS[@]} " =~ " scrna-seq " ]]; then
    jobid=$(submit_test "scrna-seq" \
        "$DATA_DIR/raw/scrna_seq" \
        "$DATA_DIR/results/scrna_seq/test_$(date +%Y%m%d)")
    if [[ -n "$jobid" ]]; then
        SUBMITTED_JOBS+=("$jobid:scrna-seq")
        echo -e "${GREEN}✓${NC} scrna-seq (Job $jobid)"
    fi
fi

# Metagenomics
if [[ " ${AVAILABLE_CONTAINERS[@]} " =~ " metagenomics " ]]; then
    jobid=$(submit_test "metagenomics" \
        "$DATA_DIR/raw/metagenomics" \
        "$DATA_DIR/results/metagenomics/test_$(date +%Y%m%d)")
    if [[ -n "$jobid" ]]; then
        SUBMITTED_JOBS+=("$jobid:metagenomics")
        echo -e "${GREEN}✓${NC} metagenomics (Job $jobid)"
    fi
fi

# Methylation
if [[ " ${AVAILABLE_CONTAINERS[@]} " =~ " methylation " ]]; then
    jobid=$(submit_test "methylation" \
        "$DATA_DIR/raw/methylation" \
        "$DATA_DIR/results/methylation/test_$(date +%Y%m%d)")
    if [[ -n "$jobid" ]]; then
        SUBMITTED_JOBS+=("$jobid:methylation")
        echo -e "${GREEN}✓${NC} methylation (Job $jobid)"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "Submitted ${#SUBMITTED_JOBS[@]} test jobs"
echo "════════════════════════════════════════════════════"
echo ""

if [[ ${#SUBMITTED_JOBS[@]} -gt 0 ]]; then
    echo "Monitor jobs:"
    echo "  squeue --me"
    echo ""
    echo "Check logs:"
    echo "  ls -lht $LOG_DIR | head"
    echo ""
    echo "Job details:"
    for job_info in "${SUBMITTED_JOBS[@]}"; do
        jobid="${job_info%%:*}"
        pipeline="${job_info##*:}"
        echo "  Job $jobid: $pipeline"
    done
else
    echo -e "${YELLOW}No jobs submitted (no test data or containers not ready)${NC}"
fi

echo ""
