#!/bin/bash
# Pre-flight Check for Container-Based Pipeline Submission
# Validates all requirements before submitting jobs

set -euo pipefail

CONTAINER_DIR="/home/sdodl001_odu_edu/BioPipelines/containers/images"
DATA_DIR="/scratch/sdodl001/BioPipelines/data"
WORKFLOW_ENGINE="${CONTAINER_DIR}/workflow-engine_1.0.0.sif"
SNAKEMAKE_PROFILE="$HOME/BioPipelines/config/snakemake_profiles/containerized"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════"
echo "  BioPipelines Container Pre-Flight Check"
echo "════════════════════════════════════════════════════"
echo ""

# Check counter
ERRORS=0
WARNINGS=0

# Function to check item
check_item() {
    local name="$1"
    local condition="$2"
    local error_msg="$3"
    local is_critical="${4:-true}"
    
    if eval "$condition"; then
        echo -e "${GREEN}✓${NC} $name"
        return 0
    else
        if [[ "$is_critical" == "true" ]]; then
            echo -e "${RED}✗${NC} $name"
            echo -e "  ${RED}ERROR:${NC} $error_msg"
            ((ERRORS++))
        else
            echo -e "${YELLOW}⚠${NC} $name"
            echo -e "  ${YELLOW}WARNING:${NC} $error_msg"
            ((WARNINGS++))
        fi
        return 1
    fi
}

echo -e "${BLUE}[1/5]${NC} Checking Containers"
echo "────────────────────────────────────────────────────"

# Workflow engine
check_item "Workflow Engine" \
    "[[ -f '$WORKFLOW_ENGINE' ]]" \
    "Workflow engine container not found at $WORKFLOW_ENGINE" \
    true

# Pipeline containers
declare -a PIPELINES=(
    "rna-seq" "dna-seq" "chip-seq" "atac-seq" 
    "hic" "long-read" "scrna-seq" "metagenomics" 
    "methylation" "structural-variants"
)

MISSING_CONTAINERS=()
for pipeline in "${PIPELINES[@]}"; do
    container="${CONTAINER_DIR}/${pipeline}_1.0.0.sif"
    if [[ ! -f "$container" ]]; then
        MISSING_CONTAINERS+=("$pipeline")
    fi
done

if [[ ${#MISSING_CONTAINERS[@]} -eq 0 ]]; then
    echo -e "${GREEN}✓${NC} All 10 pipeline containers present"
else
    echo -e "${YELLOW}⚠${NC} Missing ${#MISSING_CONTAINERS[@]} containers: ${MISSING_CONTAINERS[*]}"
    ((WARNINGS++))
fi

echo ""
echo -e "${BLUE}[2/5]${NC} Checking Snakefiles"
echo "────────────────────────────────────────────────────"

# Check Snakefiles exist and have container directives
for pipeline in "${PIPELINES[@]}"; do
    pipeline_underscore="${pipeline//-/_}"
    snakefile="$HOME/BioPipelines/pipelines/${pipeline_underscore}/Snakefile"
    
    if [[ ! -f "$snakefile" ]]; then
        echo -e "${RED}✗${NC} $pipeline: Snakefile missing"
        ((ERRORS++))
        continue
    fi
    
    # Check for container directives
    container_count=$(grep -c "container:" "$snakefile" 2>/dev/null || echo "0")
    
    # Check for leftover conda directives
    leftover_envs=$(grep -c '^[[:space:]]*"envs/.*\.yaml"[[:space:]]*$' "$snakefile" 2>/dev/null || echo "0")
    
    if [[ "$container_count" -gt 0 ]]; then
        if [[ "$leftover_envs" -gt 0 ]]; then
            echo -e "${YELLOW}⚠${NC} $pipeline: Has $container_count container directives but $leftover_envs leftover env lines"
            ((WARNINGS++))
        else
            echo -e "${GREEN}✓${NC} $pipeline: $container_count container directives"
        fi
    else
        echo -e "${RED}✗${NC} $pipeline: No container directives found"
        ((ERRORS++))
    fi
done

echo ""
echo -e "${BLUE}[3/5]${NC} Checking Data Availability"
echo "────────────────────────────────────────────────────"

# Check for test data
PIPELINES_WITH_DATA=()
PIPELINES_WITHOUT_DATA=()

for pipeline in "${PIPELINES[@]}"; do
    pipeline_underscore="${pipeline//-/_}"
    data_dir="${DATA_DIR}/raw/${pipeline_underscore}"
    
    if [[ -d "$data_dir" ]] && [[ -n "$(ls -A "$data_dir" 2>/dev/null)" ]]; then
        file_count=$(ls -1 "$data_dir" 2>/dev/null | wc -l)
        PIPELINES_WITH_DATA+=("$pipeline")
        echo -e "${GREEN}✓${NC} $pipeline: $file_count files"
    else
        PIPELINES_WITHOUT_DATA+=("$pipeline")
        echo -e "${YELLOW}⊘${NC} $pipeline: No test data"
    fi
done

echo ""
echo "  Pipelines with data: ${#PIPELINES_WITH_DATA[@]}"
echo "  Pipelines without data: ${#PIPELINES_WITHOUT_DATA[@]} (will complete immediately)"

echo ""
echo -e "${BLUE}[4/5]${NC} Checking Environment"
echo "────────────────────────────────────────────────────"

# Check singularity on login node
check_item "Singularity on login node" \
    "which singularity &>/dev/null" \
    "Singularity not found in PATH on login node" \
    true

# Check singularity on compute nodes
COMPUTE_SINGULARITY=$(srun -p cpuspot --time=00:01:00 --mem=1G bash -c "which singularity" 2>/dev/null || echo "")
if [[ -n "$COMPUTE_SINGULARITY" ]]; then
    echo -e "${GREEN}✓${NC} Singularity on compute nodes: $COMPUTE_SINGULARITY"
else
    echo -e "${RED}✗${NC} Singularity not accessible on compute nodes"
    ((ERRORS++))
fi

# Check Snakemake profile
check_item "Snakemake profile" \
    "[[ -f '$SNAKEMAKE_PROFILE/config.yaml' ]]" \
    "Snakemake profile not found at $SNAKEMAKE_PROFILE" \
    true

# Check submission script
check_item "Submission script" \
    "[[ -f '$HOME/BioPipelines/scripts/submit_pipeline_with_container.sh' ]]" \
    "Submission script not found" \
    true

# Verify PATH fix in submission script
if grep -q 'export PATH="/cm/shared/applications/slurm/wrapper:\\$PATH"' "$HOME/BioPipelines/scripts/submit_pipeline_with_container.sh" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} PATH fix applied in submission script (escaped \$PATH)"
elif grep -q 'export PATH="/cm/shared/applications/slurm/wrapper:' "$HOME/BioPipelines/scripts/submit_pipeline_with_container.sh" 2>/dev/null; then
    # Found some PATH export, check if properly escaped
    if grep 'export PATH="/cm/shared/applications/slurm/wrapper:' "$HOME/BioPipelines/scripts/submit_pipeline_with_container.sh" | grep -q '\\$PATH'; then
        echo -e "${GREEN}✓${NC} PATH fix applied in submission script (escaped \$PATH)"
    else
        echo -e "${RED}✗${NC} PATH fix NOT escaped properly (using \$PATH instead of \\\$PATH)"
        ((ERRORS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} PATH export not found in submission script"
    ((WARNINGS++))
fi

echo ""
echo -e "${BLUE}[5/5]${NC} Checking Resources"
echo "────────────────────────────────────────────────────"

# Check disk space
CONTAINER_SIZE=$(du -sh "$CONTAINER_DIR" 2>/dev/null | awk '{print $1}')
echo -e "${GREEN}✓${NC} Container directory size: $CONTAINER_SIZE"

# Check scratch space
SCRATCH_AVAIL=$(df -h /scratch/sdodl001 2>/dev/null | tail -1 | awk '{print $4}')
echo -e "${GREEN}✓${NC} Scratch space available: $SCRATCH_AVAIL"

# Check SLURM availability
QUEUE_STATUS=$(squeue --me 2>&1 | head -1)
if [[ "$QUEUE_STATUS" == *"JOBID"* ]]; then
    RUNNING_JOBS=$(squeue --me 2>/dev/null | tail -n +2 | wc -l)
    echo -e "${GREEN}✓${NC} SLURM accessible (${RUNNING_JOBS} jobs currently running)"
else
    echo -e "${RED}✗${NC} SLURM not accessible"
    ((ERRORS++))
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════"
echo ""

if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "Ready to submit pipelines with:"
    echo "  cd ~/BioPipelines"
    echo "  bash scripts/submit_pipeline_with_container.sh <pipeline-name>"
    echo ""
    echo "Pipelines ready to run (with data):"
    for p in "${PIPELINES_WITH_DATA[@]}"; do
        echo "  - $p"
    done
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}⚠ PASSED WITH $WARNINGS WARNINGS${NC}"
    echo ""
    echo "System is functional but has non-critical issues."
    echo "You can proceed with caution."
    exit 0
else
    echo -e "${RED}✗ FAILED: $ERRORS critical errors, $WARNINGS warnings${NC}"
    echo ""
    echo "Fix critical errors before submitting pipelines."
    exit 1
fi
