#!/bin/bash
# ============================================================================
# BioPipelines Ensemble Service Manager
# ============================================================================
#
# Manages the multi-model ensemble for intent parsing:
# - BioMistral-7B (GPU): L4 or H100
# - BERT models (CPU): Run locally, no GPU needed
#
# Usage:
#   ./manage_ensemble.sh start [l4|h100]  - Start BioMistral server
#   ./manage_ensemble.sh stop             - Stop BioMistral server
#   ./manage_ensemble.sh status           - Check service status
#   ./manage_ensemble.sh test             - Test ensemble parser
#   ./manage_ensemble.sh url              - Get vLLM server URL
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
CONNECTION_FILE="$PROJECT_DIR/.biomistral_server"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}        🧬 BioPipelines - Ensemble Service Manager               ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

start_server() {
    local gpu_type="${1:-l4}"
    
    print_header
    echo -e "${GREEN}Starting BioMistral-7B server on $gpu_type GPU...${NC}"
    echo ""
    
    # Check if already running
    if [ -f "$CONNECTION_FILE" ]; then
        source "$CONNECTION_FILE"
        if squeue -j "$SLURM_JOB_ID" &>/dev/null 2>&1; then
            echo -e "${YELLOW}Server already running!${NC}"
            echo "  Job ID: $SLURM_JOB_ID"
            echo "  URL: $BIOMISTRAL_URL"
            return 0
        fi
    fi
    
    # Submit job based on GPU type
    if [ "$gpu_type" = "h100" ]; then
        echo "Submitting to H100 partition..."
        JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/llm/serve_biomistral.sbatch")
    else
        echo "Submitting to L4 partition (recommended for 7B models)..."
        JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/llm/serve_biomistral_l4.sbatch")
    fi
    
    echo -e "${GREEN}✓ Submitted job: $JOB_ID${NC}"
    echo ""
    echo "Waiting for server to start..."
    
    # Wait for connection file or timeout
    for i in {1..60}; do
        if [ -f "$CONNECTION_FILE" ]; then
            source "$CONNECTION_FILE"
            echo -e "${GREEN}✓ Server is ready!${NC}"
            echo ""
            echo "  Host: $BIOMISTRAL_HOST"
            echo "  Port: $BIOMISTRAL_PORT"
            echo "  URL:  $BIOMISTRAL_URL"
            echo ""
            echo "To use in BioPipelines:"
            echo "  export VLLM_API_BASE=$BIOMISTRAL_URL"
            return 0
        fi
        sleep 5
        echo -n "."
    done
    
    echo ""
    echo -e "${YELLOW}Server is starting (check logs for progress)${NC}"
    echo "  Log: $LOG_DIR/biomistral_${gpu_type}_${JOB_ID}.out"
}

stop_server() {
    print_header
    echo -e "${YELLOW}Stopping BioMistral server...${NC}"
    
    if [ -f "$CONNECTION_FILE" ]; then
        source "$CONNECTION_FILE"
        if [ -n "$SLURM_JOB_ID" ]; then
            scancel "$SLURM_JOB_ID" 2>/dev/null && echo -e "${GREEN}✓ Cancelled job $SLURM_JOB_ID${NC}" || echo "Job not found"
        fi
        rm -f "$CONNECTION_FILE"
    else
        echo "No server connection file found"
        # Try to find and cancel any running biomistral jobs
        JOBS=$(squeue -u "$USER" -n "biomistral,biomistral_l4" -h -o "%i" 2>/dev/null)
        if [ -n "$JOBS" ]; then
            for job in $JOBS; do
                scancel "$job" && echo -e "${GREEN}✓ Cancelled job $job${NC}"
            done
        fi
    fi
}

show_status() {
    print_header
    echo -e "${BLUE}=== Service Status ===${NC}"
    echo ""
    
    # Check BioMistral server
    echo -e "${YELLOW}BioMistral-7B (GPU):${NC}"
    if [ -f "$CONNECTION_FILE" ]; then
        source "$CONNECTION_FILE"
        if squeue -j "$SLURM_JOB_ID" &>/dev/null 2>&1; then
            JOB_STATUS=$(squeue -j "$SLURM_JOB_ID" -h -o "%T")
            echo -e "  Status: ${GREEN}$JOB_STATUS${NC}"
            echo "  Job ID: $SLURM_JOB_ID"
            echo "  URL: $BIOMISTRAL_URL"
            
            # Test connection
            if curl -s "$BIOMISTRAL_URL/models" &>/dev/null; then
                echo -e "  Health: ${GREEN}✓ Responding${NC}"
            else
                echo -e "  Health: ${YELLOW}Starting...${NC}"
            fi
        else
            echo -e "  Status: ${RED}Not running${NC}"
        fi
    else
        echo -e "  Status: ${RED}Not running${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}BERT Models (CPU):${NC}"
    echo -e "  BiomedBERT: ${GREEN}✓ Always available (on-demand loading)${NC}"
    echo -e "  SciBERT: ${GREEN}✓ Always available (on-demand loading)${NC}"
    
    echo ""
    echo -e "${YELLOW}SLURM Queue:${NC}"
    squeue -u "$USER" --format="  %-12i %-15j %-8T %-10M %-10P" 2>/dev/null | head -10
}

get_url() {
    if [ -f "$CONNECTION_FILE" ]; then
        source "$CONNECTION_FILE"
        echo "$BIOMISTRAL_URL"
    else
        echo "http://localhost:8000/v1"
    fi
}

test_ensemble() {
    print_header
    echo -e "${BLUE}Testing Ensemble Parser...${NC}"
    echo ""
    
    # Activate environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate biopipelines
    
    # Get vLLM URL
    if [ -f "$CONNECTION_FILE" ]; then
        source "$CONNECTION_FILE"
        export VLLM_API_BASE="$BIOMISTRAL_URL"
        echo "Using BioMistral at: $VLLM_API_BASE"
    else
        echo -e "${YELLOW}BioMistral not running - testing BERT models only${NC}"
    fi
    
    echo ""
    
    # Test queries
    cd "$PROJECT_DIR"
    python -c "
from src.workflow_composer.core.ensemble_parser import EnsembleIntentParser

parser = EnsembleIntentParser()

test_queries = [
    'Build a long-read nanopore pipeline for mouse genome assembly',
    'RNA-seq differential expression for human cancer vs normal',
    'ChIP-seq peak calling for H3K27ac in mouse ES cells',
]

for query in test_queries:
    print(f'\\n{\"=\"*60}')
    print(f'Query: {query}')
    print('='*60)
    result = parser.parse(query)
    print(f'Type: {result.analysis_type}')
    print(f'Confidence: {result.confidence:.1%}')
    print(f'Organism: {result.organism or \"Not specified\"}')
    print(f'Tools: {\", \".join(result.tools_detected) or \"None\"}')
    print(f'Latency: {result.latency_ms:.0f}ms')
    print(f'Reasoning: {result.reasoning}')
"
}

# Main
case "${1:-status}" in
    start)
        start_server "${2:-l4}"
        ;;
    stop)
        stop_server
        ;;
    status)
        show_status
        ;;
    url)
        get_url
        ;;
    test)
        test_ensemble
        ;;
    *)
        echo "Usage: $0 {start [l4|h100]|stop|status|url|test}"
        echo ""
        echo "Commands:"
        echo "  start [l4|h100]  - Start BioMistral server on GPU"
        echo "  stop             - Stop BioMistral server"
        echo "  status           - Show service status"
        echo "  url              - Get vLLM server URL"
        echo "  test             - Test ensemble parser"
        exit 1
        ;;
esac
