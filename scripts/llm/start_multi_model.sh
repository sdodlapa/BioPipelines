#!/bin/bash
# =============================================================================
# Multi-Model vLLM Server Script for H100 GPUs
# =============================================================================
#
# Starts multiple vLLM servers for the multi-agent system.
# Each model runs on separate GPU(s) with its own port.
#
# Usage:
#   ./start_multi_model.sh [2gpu|4gpu|4gpu-deepseek|supervisor-only|coder-only]
#
# Ports:
#   8000 - Supervisor model (Llama-3.3-70B or main model)
#   8001 - Coding model (Qwen2.5-Coder-32B)
#
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="${PROJECT_ROOT}/logs/vllm"

# Model defaults
SUPERVISOR_MODEL="${SUPERVISOR_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
CODER_MODEL="${CODER_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"

# Ports
SUPERVISOR_PORT="${SUPERVISOR_PORT:-8000}"
CODER_PORT="${CODER_PORT:-8001}"

# Create log directory
mkdir -p "$LOG_DIR"

# Parse mode
MODE="${1:-2gpu}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Check if vLLM is installed
check_vllm() {
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please activate conda environment."
        exit 1
    fi
    
    if ! python -c "import vllm" 2>/dev/null; then
        log_error "vLLM not installed. Install with: pip install vllm"
        exit 1
    fi
    
    log_info "vLLM is available"
}

# Check GPU availability
check_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Are you on a GPU node?"
        exit 1
    fi
    
    local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log_info "Found $gpu_count GPU(s)"
    
    # Show GPU info
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
    
    echo "$gpu_count"
}

# Start supervisor model (Llama-3.3-70B)
start_supervisor() {
    local gpus="$1"
    local tp_size="$2"
    
    log_header "Starting Supervisor Model"
    log_info "Model: $SUPERVISOR_MODEL"
    log_info "GPUs: $gpus (TP=$tp_size)"
    log_info "Port: $SUPERVISOR_PORT"
    
    CUDA_VISIBLE_DEVICES="$gpus" python -m vllm.entrypoints.openai.api_server \
        --model "$SUPERVISOR_MODEL" \
        --port "$SUPERVISOR_PORT" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$tp_size" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --dtype bfloat16 \
        --trust-remote-code \
        2>&1 | tee "$LOG_DIR/supervisor.log" &
    
    SUPERVISOR_PID=$!
    echo "$SUPERVISOR_PID" > "$LOG_DIR/supervisor.pid"
    log_info "Supervisor PID: $SUPERVISOR_PID"
}

# Start coding model (Qwen2.5-Coder-32B)
start_coder() {
    local gpus="$1"
    local tp_size="$2"
    
    log_header "Starting Coding Model"
    log_info "Model: $CODER_MODEL"
    log_info "GPUs: $gpus (TP=$tp_size)"
    log_info "Port: $CODER_PORT"
    
    CUDA_VISIBLE_DEVICES="$gpus" python -m vllm.entrypoints.openai.api_server \
        --model "$CODER_MODEL" \
        --port "$CODER_PORT" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$tp_size" \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16 \
        --trust-remote-code \
        2>&1 | tee "$LOG_DIR/coder.log" &
    
    CODER_PID=$!
    echo "$CODER_PID" > "$LOG_DIR/coder.pid"
    log_info "Coder PID: $CODER_PID"
}

# Wait for server to be ready
wait_for_server() {
    local port="$1"
    local name="$2"
    local max_wait=300  # 5 minutes
    local waited=0
    
    log_info "Waiting for $name to be ready on port $port..."
    
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
            log_info "$name is ready!"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo -n "."
    done
    
    log_error "$name failed to start within $max_wait seconds"
    return 1
}

# Stop all servers
stop_servers() {
    log_header "Stopping vLLM Servers"
    
    if [ -f "$LOG_DIR/supervisor.pid" ]; then
        local pid=$(cat "$LOG_DIR/supervisor.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping supervisor (PID: $pid)"
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$LOG_DIR/supervisor.pid"
    fi
    
    if [ -f "$LOG_DIR/coder.pid" ]; then
        local pid=$(cat "$LOG_DIR/coder.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping coder (PID: $pid)"
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$LOG_DIR/coder.pid"
    fi
    
    log_info "All servers stopped"
}

# Status check
status() {
    log_header "vLLM Server Status"
    
    echo "Supervisor (port $SUPERVISOR_PORT):"
    if curl -s "http://localhost:$SUPERVISOR_PORT/health" >/dev/null 2>&1; then
        echo -e "  ${GREEN}●${NC} Running"
        curl -s "http://localhost:$SUPERVISOR_PORT/v1/models" | python -c "import sys,json; data=json.load(sys.stdin); print('  Model:', data['data'][0]['id'] if data.get('data') else 'Unknown')" 2>/dev/null || true
    else
        echo -e "  ${RED}●${NC} Not running"
    fi
    
    echo ""
    echo "Coder (port $CODER_PORT):"
    if curl -s "http://localhost:$CODER_PORT/health" >/dev/null 2>&1; then
        echo -e "  ${GREEN}●${NC} Running"
        curl -s "http://localhost:$CODER_PORT/v1/models" | python -c "import sys,json; data=json.load(sys.stdin); print('  Model:', data['data'][0]['id'] if data.get('data') else 'Unknown')" 2>/dev/null || true
    else
        echo -e "  ${RED}●${NC} Not running"
    fi
}

# Main execution
main() {
    case "$MODE" in
        "2gpu")
            log_header "Configuration: 2× H100 80GB"
            log_info "Running single model (Llama-3.3-70B) on both GPUs"
            log_info "Use 4gpu mode for separate coding model"
            
            check_vllm
            check_gpus
            
            # 2 GPU setup - single model spanning both
            start_supervisor "0,1" 2
            wait_for_server "$SUPERVISOR_PORT" "Supervisor"
            
            log_header "Setup Complete"
            log_info "Supervisor: http://localhost:$SUPERVISOR_PORT"
            log_info "For coding tasks, the supervisor will handle with prompt specialization"
            ;;
            
        "4gpu")
            log_header "Configuration: 4× H100 80GB (Dual Model)"
            log_info "GPU 0-1: Llama-3.3-70B (Supervisor)"
            log_info "GPU 2-3: Qwen2.5-Coder-32B (Coding)"
            
            check_vllm
            local gpu_count=$(check_gpus)
            
            if [ "$gpu_count" -lt 4 ]; then
                log_error "Need 4 GPUs for this mode. Found: $gpu_count"
                exit 1
            fi
            
            # Start both models
            start_supervisor "0,1" 2
            sleep 10  # Wait a bit before starting second server
            start_coder "2,3" 2
            
            # Wait for both
            wait_for_server "$SUPERVISOR_PORT" "Supervisor"
            wait_for_server "$CODER_PORT" "Coder"
            
            log_header "Setup Complete"
            log_info "Supervisor: http://localhost:$SUPERVISOR_PORT"
            log_info "Coder: http://localhost:$CODER_PORT"
            ;;
            
        "4gpu-deepseek")
            log_header "Configuration: 4× H100 80GB (DeepSeek-Coder-V2)"
            log_info "GPU 0-3: DeepSeek-Coder-V2-Instruct (All-in-one)"
            
            check_vllm
            local gpu_count=$(check_gpus)
            
            if [ "$gpu_count" -lt 4 ]; then
                log_error "Need 4 GPUs for this mode. Found: $gpu_count"
                exit 1
            fi
            
            # DeepSeek-Coder-V2 is a 236B MoE - needs all 4 GPUs
            SUPERVISOR_MODEL="deepseek-ai/DeepSeek-Coder-V2-Instruct"
            start_supervisor "0,1,2,3" 4
            wait_for_server "$SUPERVISOR_PORT" "DeepSeek-Coder-V2"
            
            log_header "Setup Complete"
            log_info "DeepSeek-Coder-V2: http://localhost:$SUPERVISOR_PORT"
            log_info "This model handles both orchestration and coding tasks"
            ;;
            
        "supervisor-only")
            log_header "Starting Supervisor Only"
            check_vllm
            check_gpus
            
            start_supervisor "0,1" 2
            wait_for_server "$SUPERVISOR_PORT" "Supervisor"
            
            log_info "Supervisor: http://localhost:$SUPERVISOR_PORT"
            ;;
            
        "coder-only")
            log_header "Starting Coder Only"
            check_vllm
            check_gpus
            
            start_coder "0,1" 2
            wait_for_server "$CODER_PORT" "Coder"
            
            log_info "Coder: http://localhost:$CODER_PORT"
            ;;
            
        "stop")
            stop_servers
            ;;
            
        "status")
            status
            ;;
            
        "help"|"-h"|"--help")
            echo "Usage: $0 [mode]"
            echo ""
            echo "Modes:"
            echo "  2gpu           Start Llama-3.3-70B on 2 GPUs (default)"
            echo "  4gpu           Start Llama + Qwen-Coder on 4 GPUs"
            echo "  4gpu-deepseek  Start DeepSeek-Coder-V2 on 4 GPUs"
            echo "  supervisor-only Start only the supervisor model"
            echo "  coder-only     Start only the coding model"
            echo "  stop           Stop all vLLM servers"
            echo "  status         Check server status"
            echo ""
            echo "Environment Variables:"
            echo "  SUPERVISOR_MODEL  Override supervisor model"
            echo "  CODER_MODEL       Override coding model"
            echo "  SUPERVISOR_PORT   Override supervisor port (default: 8000)"
            echo "  CODER_PORT        Override coder port (default: 8001)"
            ;;
            
        *)
            log_error "Unknown mode: $MODE"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Handle interrupt
trap 'log_warn "Interrupted. Use \"$0 stop\" to stop servers."' INT

# Run main
main
