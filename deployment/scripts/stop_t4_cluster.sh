#!/bin/bash
# =============================================================================
# BioPipelines: Stop Multi-Node T4 vLLM Deployment
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${DEPLOYMENT_DIR}/configs"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping BioPipelines T4 Deployment...${NC}"

# Stop router
if [ -f "${CONFIG_DIR}/router.pid" ]; then
    PID=$(cat "${CONFIG_DIR}/router.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo -e "${GREEN}✓ Router stopped (PID: $PID)${NC}"
    fi
    rm -f "${CONFIG_DIR}/router.pid"
fi

# Also kill any stray router processes
pkill -f "model_router.py" 2>/dev/null && echo -e "${GREEN}✓ Killed stray router processes${NC}" || true

# Cancel SLURM jobs
if [ -f "${CONFIG_DIR}/current_job_id" ]; then
    JOB_ID=$(cat "${CONFIG_DIR}/current_job_id")
    scancel "$JOB_ID" 2>/dev/null && echo -e "${GREEN}✓ Cancelled job array $JOB_ID${NC}" || true
    rm -f "${CONFIG_DIR}/current_job_id"
fi

# Also cancel by name
scancel -u "$USER" -n vllm-t4-array 2>/dev/null && echo -e "${GREEN}✓ Cancelled all vllm-t4-array jobs${NC}" || true

# Clear registry
if [ -f "${CONFIG_DIR}/active_servers.json" ]; then
    echo '{"servers":{}, "cleared_at":"'$(date -Iseconds)'"}' > "${CONFIG_DIR}/active_servers.json"
    echo -e "${GREEN}✓ Registry cleared${NC}"
fi

echo -e "${GREEN}Deployment stopped${NC}"
