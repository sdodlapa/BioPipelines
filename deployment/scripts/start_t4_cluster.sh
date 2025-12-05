#!/bin/bash
# =============================================================================
# BioPipelines: Start Multi-Node T4 vLLM Deployment
# =============================================================================
# Usage: ./start_t4_cluster.sh [OPTIONS]
#
# Options:
#   --nodes N       Number of T4 nodes to request (default: 10)
#   --dry-run       Show what would be done without executing
#   --router-only   Only start the router (backends must be running)
#   --help          Show this help message
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${DEPLOYMENT_DIR}/logs"
CONFIG_DIR="${DEPLOYMENT_DIR}/configs"
REGISTRY_FILE="${CONFIG_DIR}/active_servers.json"
ROUTER_PORT=8080

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
NUM_NODES=10
DRY_RUN=false
ROUTER_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --router-only)
            ROUTER_ONLY=true
            shift
            ;;
        --help)
            head -20 "$0" | grep -E "^#" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}BioPipelines T4 Multi-Node Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Ensure directories exist
mkdir -p "$LOG_DIR" "$CONFIG_DIR"

# Function to check SLURM availability
check_slurm() {
    if ! command -v sbatch &> /dev/null; then
        echo -e "${RED}Error: SLURM not available (sbatch not found)${NC}"
        exit 1
    fi
    
    if ! command -v squeue &> /dev/null; then
        echo -e "${RED}Error: SLURM not available (squeue not found)${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ SLURM available${NC}"
}

# Function to check T4 partition availability
check_t4_partition() {
    echo -e "${YELLOW}Checking t4flex partition...${NC}"
    
    T4_INFO=$(sinfo -p t4flex --noheader -o "%P %a %F" 2>/dev/null || echo "")
    if [ -z "$T4_INFO" ]; then
        echo -e "${RED}Error: t4flex partition not found${NC}"
        exit 1
    fi
    
    # Parse available/total nodes
    AVAIL_NODES=$(echo "$T4_INFO" | awk '{print $3}' | cut -d'/' -f2)
    TOTAL_NODES=$(echo "$T4_INFO" | awk '{print $3}' | cut -d'/' -f4)
    
    echo -e "${GREEN}✓ t4flex partition: ${AVAIL_NODES}/${TOTAL_NODES} nodes available${NC}"
    
    if [ "$AVAIL_NODES" -lt "$NUM_NODES" ]; then
        echo -e "${YELLOW}Warning: Requested ${NUM_NODES} nodes but only ${AVAIL_NODES} available${NC}"
        echo -e "${YELLOW}Proceeding with ${AVAIL_NODES} nodes...${NC}"
        NUM_NODES=$AVAIL_NODES
    fi
}

# Function to cancel existing jobs
cancel_existing_jobs() {
    echo -e "${YELLOW}Checking for existing vLLM jobs...${NC}"
    
    EXISTING_JOBS=$(squeue -u "$USER" -n vllm-t4-array --noheader -o "%i" 2>/dev/null || echo "")
    
    if [ -n "$EXISTING_JOBS" ]; then
        echo -e "${YELLOW}Found existing jobs: $EXISTING_JOBS${NC}"
        read -p "Cancel existing jobs? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            for job in $EXISTING_JOBS; do
                scancel "$job"
                echo -e "${GREEN}Cancelled job $job${NC}"
            done
        fi
    else
        echo -e "${GREEN}✓ No existing vLLM jobs${NC}"
    fi
}

# Function to initialize registry
init_registry() {
    echo -e "${YELLOW}Initializing server registry...${NC}"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] Would create: $REGISTRY_FILE${NC}"
        return
    fi
    
    echo '{"servers":{}, "created_at":"'$(date -Iseconds)'"}' > "$REGISTRY_FILE"
    echo -e "${GREEN}✓ Registry initialized: $REGISTRY_FILE${NC}"
}

# Function to submit job array
submit_job_array() {
    echo -e "${YELLOW}Submitting T4 job array...${NC}"
    
    # Adjust array range based on requested nodes
    ARRAY_RANGE="0-$((NUM_NODES - 1))"
    
    JOB_SCRIPT="${SCRIPT_DIR}/t4_job_array.slurm"
    
    if [ ! -f "$JOB_SCRIPT" ]; then
        echo -e "${RED}Error: Job script not found: $JOB_SCRIPT${NC}"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] Would submit:${NC}"
        echo "  sbatch --array=$ARRAY_RANGE $JOB_SCRIPT"
        return
    fi
    
    # Submit the job array
    JOB_OUTPUT=$(sbatch --array="$ARRAY_RANGE" "$JOB_SCRIPT" 2>&1)
    
    if [[ $JOB_OUTPUT =~ "Submitted batch job" ]]; then
        JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
        echo -e "${GREEN}✓ Job array submitted: $JOB_ID (array: $ARRAY_RANGE)${NC}"
        echo "$JOB_ID" > "${CONFIG_DIR}/current_job_id"
    else
        echo -e "${RED}Error submitting job: $JOB_OUTPUT${NC}"
        exit 1
    fi
}

# Function to wait for servers to start
wait_for_servers() {
    echo -e "${YELLOW}Waiting for servers to start (this may take 2-5 minutes)...${NC}"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] Would wait for servers${NC}"
        return
    fi
    
    MAX_WAIT=300  # 5 minutes
    WAIT_INTERVAL=10
    ELAPSED=0
    
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        sleep $WAIT_INTERVAL
        ELAPSED=$((ELAPSED + WAIT_INTERVAL))
        
        # Check how many servers are registered
        if [ -f "$REGISTRY_FILE" ]; then
            REGISTERED=$(jq '.servers | length' "$REGISTRY_FILE" 2>/dev/null || echo 0)
            HEALTHY=$(jq '[.servers[] | select(.status == "healthy" or .status == "starting")] | length' "$REGISTRY_FILE" 2>/dev/null || echo 0)
            
            echo -e "  [${ELAPSED}s] Servers registered: ${REGISTERED}/${NUM_NODES}, Starting/Healthy: ${HEALTHY}"
            
            if [ "$REGISTERED" -ge "$NUM_NODES" ]; then
                echo -e "${GREEN}✓ All servers registered${NC}"
                break
            fi
        fi
    done
    
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo -e "${YELLOW}Warning: Timeout waiting for all servers${NC}"
        echo -e "${YELLOW}Some servers may still be starting...${NC}"
    fi
}

# Function to start the router
start_router() {
    echo -e "${YELLOW}Starting model router...${NC}"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] Would start router on port $ROUTER_PORT${NC}"
        return
    fi
    
    # Check if router is already running
    if pgrep -f "model_router.py" > /dev/null; then
        echo -e "${YELLOW}Router already running. Restarting...${NC}"
        pkill -f "model_router.py" || true
        sleep 2
    fi
    
    # Activate environment
    source /home/sdodl001_odu_edu/miniforge3/etc/profile.d/conda.sh
    conda activate biopipelines
    
    # Ensure dependencies
    pip show fastapi httpx uvicorn > /dev/null 2>&1 || \
        pip install fastapi httpx uvicorn pydantic
    
    # Start router in background
    nohup python "${SCRIPT_DIR}/model_router.py" \
        --port $ROUTER_PORT \
        --registry "$REGISTRY_FILE" \
        > "${LOG_DIR}/router.log" 2>&1 &
    
    ROUTER_PID=$!
    echo "$ROUTER_PID" > "${CONFIG_DIR}/router.pid"
    
    sleep 3
    
    # Check if router started
    if kill -0 $ROUTER_PID 2>/dev/null; then
        echo -e "${GREEN}✓ Router started (PID: $ROUTER_PID, Port: $ROUTER_PORT)${NC}"
    else
        echo -e "${RED}Error: Router failed to start${NC}"
        echo -e "${RED}Check logs: ${LOG_DIR}/router.log${NC}"
        exit 1
    fi
}

# Function to display status
display_status() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Deployment Status${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN] No actual deployment${NC}"
        return
    fi
    
    echo ""
    echo -e "${YELLOW}SLURM Jobs:${NC}"
    squeue -u "$USER" -n vllm-t4-array --format="%.10i %.9P %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "  No jobs found"
    
    echo ""
    echo -e "${YELLOW}Registered Servers:${NC}"
    if [ -f "$REGISTRY_FILE" ]; then
        jq -r '.servers | to_entries[] | "  [\(.key)] \(.value.model | split("/")[-1]) @ \(.value.host):\(.value.port) [\(.value.status)]"' "$REGISTRY_FILE" 2>/dev/null || echo "  No servers registered"
    else
        echo "  Registry not found"
    fi
    
    echo ""
    echo -e "${YELLOW}Router:${NC}"
    if [ -f "${CONFIG_DIR}/router.pid" ]; then
        PID=$(cat "${CONFIG_DIR}/router.pid")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "  ${GREEN}Running (PID: $PID, Port: $ROUTER_PORT)${NC}"
        else
            echo -e "  ${RED}Not running${NC}"
        fi
    else
        echo -e "  ${RED}Not started${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Access the API:${NC}"
    echo -e "  curl http://localhost:${ROUTER_PORT}/health"
    echo -e "  curl http://localhost:${ROUTER_PORT}/servers"
    echo ""
    echo -e "${GREEN}Test a completion:${NC}"
    echo -e '  curl http://localhost:'${ROUTER_PORT}'/v1/chat/completions \'
    echo -e '    -H "Content-Type: application/json" \'
    echo -e '    -d '"'"'{"model": "intent", "messages": [{"role": "user", "content": "Hello"}]}'"'"
    echo -e "${BLUE}========================================${NC}"
}

# Main execution
main() {
    check_slurm
    check_t4_partition
    
    if [ "$ROUTER_ONLY" = true ]; then
        start_router
        display_status
        exit 0
    fi
    
    cancel_existing_jobs
    init_registry
    submit_job_array
    wait_for_servers
    start_router
    display_status
}

main
