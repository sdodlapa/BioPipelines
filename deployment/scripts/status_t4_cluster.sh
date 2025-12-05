#!/bin/bash
# =============================================================================
# BioPipelines: Check Cluster Status
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${DEPLOYMENT_DIR}/configs"
ROUTER_PORT=8080

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}BioPipelines T4 Cluster Status${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# SLURM Jobs
echo -e "${YELLOW}SLURM Jobs:${NC}"
squeue -u "$USER" -n vllm-t4-array --format="%.10i %.9P %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "  No jobs found"
echo ""

# Router Status
echo -e "${YELLOW}Router Status:${NC}"
if [ -f "${CONFIG_DIR}/router.pid" ]; then
    PID=$(cat "${CONFIG_DIR}/router.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "  ${GREEN}● Running${NC} (PID: $PID, Port: $ROUTER_PORT)"
        
        # Check router health
        HEALTH=$(curl -s "http://localhost:${ROUTER_PORT}/health" 2>/dev/null | jq -r '.status' 2>/dev/null || echo "unreachable")
        echo -e "  Health: ${HEALTH}"
    else
        echo -e "  ${RED}● Not running${NC} (stale PID file)"
    fi
else
    echo -e "  ${RED}● Not started${NC}"
fi
echo ""

# Server Registry
echo -e "${YELLOW}Registered Servers:${NC}"
REGISTRY="${CONFIG_DIR}/active_servers.json"
if [ -f "$REGISTRY" ]; then
    SERVER_COUNT=$(jq '.servers | length' "$REGISTRY" 2>/dev/null || echo 0)
    HEALTHY_COUNT=$(jq '[.servers[] | select(.status == "healthy")] | length' "$REGISTRY" 2>/dev/null || echo 0)
    
    echo -e "  Total: $SERVER_COUNT, Healthy: $HEALTHY_COUNT"
    echo ""
    
    jq -r '.servers | to_entries[] | "  [\(.key)] \(.value.status | if . == "healthy" then "● " else "○ " end)\(.value.model | split("/")[-1]) @ \(.value.host):\(.value.port)"' "$REGISTRY" 2>/dev/null
else
    echo "  Registry not found"
fi
echo ""

# Quick health check on backends
if [ -f "$REGISTRY" ]; then
    echo -e "${YELLOW}Backend Health Check:${NC}"
    jq -r '.servers | to_entries[] | "\(.value.host):\(.value.port)"' "$REGISTRY" 2>/dev/null | while read endpoint; do
        RESULT=$(curl -s -o /dev/null -w "%{http_code}" "http://${endpoint}/health" --connect-timeout 2 2>/dev/null || echo "timeout")
        if [ "$RESULT" = "200" ]; then
            echo -e "  ${GREEN}✓${NC} $endpoint"
        else
            echo -e "  ${RED}✗${NC} $endpoint ($RESULT)"
        fi
    done
fi

echo ""
echo -e "${BLUE}========================================${NC}"
