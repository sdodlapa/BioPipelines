#!/bin/bash
# ============================================================================
# BioPipelines Core Model Deployment Script (Simplified)
# ============================================================================
#
# Deploys 4 core models on T4 GPUs following the refined strategy:
#   1. Generalist (Qwen2.5-7B-Instruct-AWQ) - intent, docs, general
#   2. Coder (Qwen2.5-Coder-7B-Instruct-AWQ) - code gen/validation  
#   3. Math (Qwen2.5-Math-7B-Instruct-AWQ) - math, statistics
#   4. Embeddings (BGE-M3) - RAG, semantic search
#
# Usage:
#   ./deploy_core_models.sh start           # Start all 4 core models
#   ./deploy_core_models.sh start generalist # Start specific model
#   ./deploy_core_models.sh status          # Check server status
#   ./deploy_core_models.sh stop            # Stop all servers
#   ./deploy_core_models.sh health          # Health check all endpoints
#
# Author: BioPipelines Team
# Version: 2.0 (Simplified 4-model architecture)
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/logs/vllm"
CONNECTION_DIR="$PROJECT_DIR/.model_connections"
SBATCH_DIR="$SCRIPT_DIR/generated_sbatch"

# SLURM configuration
PARTITION="${VLLM_PARTITION:-t4flex}"
TIME_LIMIT="${VLLM_TIME_LIMIT:-24:00:00}"
MEM="32G"
CPUS=8

# Create directories
mkdir -p "$LOG_DIR" "$CONNECTION_DIR" "$SBATCH_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Core Model Definitions (4 models only)
# ============================================================================
# Format: MODEL_ID|PORT|QUANTIZATION|MAX_MODEL_LEN|DESCRIPTION

declare -A CORE_MODELS=(
    ["generalist"]="Qwen/Qwen2.5-7B-Instruct-AWQ|8001|awq|8192|General purpose - intent, docs, analysis"
    ["coder"]="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ|8002|awq|16384|Code generation and validation"
    ["math"]="Qwen/Qwen2.5-Math-7B-Instruct-AWQ|8003|awq|4096|Math and statistics"
    ["embeddings"]="BAAI/bge-m3|8004|none|8192|Embeddings for RAG"
)

# Task to model mapping (for documentation)
declare -A TASK_ROUTING=(
    ["intent_parsing"]="generalist"
    ["code_generation"]="coder"
    ["code_validation"]="coder"
    ["data_analysis"]="generalist"
    ["math_statistics"]="math"
    ["documentation"]="generalist"
    ["biomedical"]="generalist"
    ["safety"]="generalist"
    ["embeddings"]="embeddings"
    ["orchestration"]="cloud"
)

# ============================================================================
# Functions
# ============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║          BioPipelines Core Model Deployment v2.0              ║"
    echo "║              4-Model Simplified Architecture                   ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

generate_sbatch_file() {
    local model_key=$1
    local model_info=${CORE_MODELS[$model_key]}
    
    IFS='|' read -r model_id port quant max_len desc <<< "$model_info"
    
    local sbatch_file="$SBATCH_DIR/serve_${model_key}.sbatch"
    
    # Determine if this is an embedding model
    local is_embedding=false
    [[ "$model_key" == "embeddings" ]] && is_embedding=true
    
    cat > "$sbatch_file" << 'SBATCH_HEADER'
#!/bin/bash
SBATCH_HEADER

    cat >> "$sbatch_file" << EOF
#SBATCH --job-name=vllm_${model_key}
#SBATCH --output=${LOG_DIR}/${model_key}_%j.out
#SBATCH --error=${LOG_DIR}/${model_key}_%j.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}

# ============================================================================
# ${desc}
# Model: ${model_id}
# Port: ${port}
# Quantization: ${quant}
# ============================================================================

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ Starting ${model_key} Server"
echo "║ Model: ${model_id}"
echo "║ Port: ${port}"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$(hostname)"
echo "Date: \$(date)"
echo ""

# Environment setup
export HOME=/home/sdodl001_odu_edu
export HF_HOME=\$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=\$HOME/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=\$HOME/.cache/huggingface

# Activate environment
source \$HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env 2>/dev/null || conda activate biopipelines 2>/dev/null || {
    echo "ERROR: Could not activate conda environment"
    exit 1
}

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Save connection info for other components
CONNECTION_FILE="${CONNECTION_DIR}/${model_key}.env"
cat > "\$CONNECTION_FILE" << CONN
MODEL_KEY=${model_key}
MODEL_ID=${model_id}
HOST=\$(hostname)
PORT=${port}
URL=http://\$(hostname):${port}
HEALTH_URL=http://\$(hostname):${port}/health
SLURM_JOB_ID=\$SLURM_JOB_ID
STARTED_AT=\$(date -Iseconds)
CONN

echo "Connection info: \$CONNECTION_FILE"
echo "API Endpoint: http://\$(hostname):${port}/v1"
echo ""
EOF

    # Add model-specific vLLM command
    if [[ "$is_embedding" == "true" ]]; then
        # Embedding model uses different entry point
        cat >> "$sbatch_file" << EOF
# Start embedding server
echo "Starting embedding server for ${model_id}..."
python -m vllm.entrypoints.openai.api_server \\
    --model ${model_id} \\
    --host 0.0.0.0 \\
    --port ${port} \\
    --task embedding \\
    --gpu-memory-utilization 0.85 \\
    --max-model-len ${max_len} \\
    --trust-remote-code \\
    --dtype float16
EOF
    else
        # Regular chat model
        cat >> "$sbatch_file" << EOF
# Start vLLM server
echo "Starting vLLM server for ${model_id}..."

VLLM_CMD="python -m vllm.entrypoints.openai.api_server \\
    --model ${model_id} \\
    --host 0.0.0.0 \\
    --port ${port} \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len ${max_len} \\
    --trust-remote-code \\
    --dtype float16"

EOF
        # Add quantization
        if [[ "$quant" == "awq" ]]; then
            cat >> "$sbatch_file" << 'EOF'
VLLM_CMD="$VLLM_CMD --quantization awq"
EOF
        elif [[ "$quant" == "int8" ]]; then
            cat >> "$sbatch_file" << 'EOF'
VLLM_CMD="$VLLM_CMD --quantization int8"
EOF
        fi

        cat >> "$sbatch_file" << 'EOF'

echo "Command: $VLLM_CMD"
echo ""
eval $VLLM_CMD
EOF
    fi

    # Cleanup on exit
    cat >> "$sbatch_file" << EOF

# Cleanup
echo ""
echo "Server stopped at \$(date)"
rm -f "\$CONNECTION_FILE"
EOF

    chmod +x "$sbatch_file"
    echo "$sbatch_file"
}

start_model() {
    local model_key=$1
    
    if [[ -z "${CORE_MODELS[$model_key]}" ]]; then
        log_error "Unknown model: $model_key"
        log_info "Available models: ${!CORE_MODELS[*]}"
        return 1
    fi
    
    local model_info=${CORE_MODELS[$model_key]}
    IFS='|' read -r model_id port quant max_len desc <<< "$model_info"
    
    # Check if already running
    local job_file="$CONNECTION_DIR/${model_key}.job"
    if [[ -f "$job_file" ]]; then
        local existing_job=$(cat "$job_file")
        local state=$(squeue -j "$existing_job" -h -o "%T" 2>/dev/null)
        if [[ "$state" == "RUNNING" || "$state" == "PENDING" ]]; then
            log_warn "$model_key already running/pending (job $existing_job)"
            return 0
        fi
    fi
    
    log_info "Starting $model_key ($model_id) on port $port..."
    
    # Generate and submit
    local sbatch_file=$(generate_sbatch_file "$model_key")
    local job_id=$(sbatch --parsable "$sbatch_file" 2>/dev/null)
    
    if [[ -n "$job_id" && "$job_id" =~ ^[0-9]+$ ]]; then
        echo "$job_id" > "$CONNECTION_DIR/${model_key}.job"
        log_success "Submitted job $job_id for $model_key (port $port)"
    else
        log_error "Failed to submit job for $model_key"
        return 1
    fi
}

stop_model() {
    local model_key=$1
    local job_file="$CONNECTION_DIR/${model_key}.job"
    
    if [[ -f "$job_file" ]]; then
        local job_id=$(cat "$job_file")
        log_info "Stopping $model_key (job $job_id)..."
        scancel "$job_id" 2>/dev/null && log_success "Cancelled job $job_id" || log_warn "Job may already be stopped"
        rm -f "$job_file" "$CONNECTION_DIR/${model_key}.env"
    else
        log_warn "No running job found for $model_key"
    fi
}

status_model() {
    local model_key=$1
    local model_info=${CORE_MODELS[$model_key]}
    local job_file="$CONNECTION_DIR/${model_key}.job"
    local env_file="$CONNECTION_DIR/${model_key}.env"
    
    IFS='|' read -r model_id port quant max_len desc <<< "$model_info"
    
    printf "%-12s " "$model_key"
    
    if [[ -f "$job_file" ]]; then
        local job_id=$(cat "$job_file")
        local state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
        
        case "$state" in
            RUNNING)
                echo -e "${GREEN}RUNNING${NC} (job $job_id, port $port)"
                if [[ -f "$env_file" ]]; then
                    source "$env_file"
                    echo "             └─ $URL/v1"
                fi
                ;;
            PENDING)
                echo -e "${YELLOW}PENDING${NC} (job $job_id)"
                ;;
            *)
                echo -e "${RED}STOPPED${NC} (last job: $job_id)"
                rm -f "$job_file"  # Clean up stale job file
                ;;
        esac
    else
        echo -e "NOT STARTED"
    fi
}

health_check() {
    local model_key=$1
    local env_file="$CONNECTION_DIR/${model_key}.env"
    
    if [[ ! -f "$env_file" ]]; then
        printf "%-12s ${RED}NO CONNECTION INFO${NC}\n" "$model_key"
        return 1
    fi
    
    source "$env_file"
    
    printf "%-12s " "$model_key"
    
    # Try health endpoint
    local response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$URL/health" 2>/dev/null)
    
    if [[ "$response" == "200" ]]; then
        echo -e "${GREEN}HEALTHY${NC} ($URL)"
    else
        echo -e "${RED}UNHEALTHY${NC} (HTTP $response)"
        return 1
    fi
}

# ============================================================================
# Main Commands
# ============================================================================

cmd_start() {
    local target="${1:-all}"
    
    if [[ "$target" == "all" ]]; then
        log_info "Starting all core models..."
        for model_key in "${!CORE_MODELS[@]}"; do
            start_model "$model_key"
        done
    else
        start_model "$target"
    fi
}

cmd_stop() {
    local target="${1:-all}"
    
    if [[ "$target" == "all" ]]; then
        log_info "Stopping all core models..."
        for model_key in "${!CORE_MODELS[@]}"; do
            stop_model "$model_key"
        done
    else
        stop_model "$target"
    fi
}

cmd_status() {
    echo ""
    echo "Core Model Status:"
    echo "─────────────────────────────────────────────────────"
    for model_key in generalist coder math embeddings; do
        status_model "$model_key"
    done
    echo ""
}

cmd_health() {
    echo ""
    echo "Health Check:"
    echo "─────────────────────────────────────────────────────"
    local all_healthy=true
    for model_key in generalist coder math embeddings; do
        health_check "$model_key" || all_healthy=false
    done
    echo ""
    
    if [[ "$all_healthy" == "true" ]]; then
        log_success "All models healthy"
    else
        log_warn "Some models unhealthy - check status"
    fi
}

cmd_restart() {
    local target="${1:-all}"
    cmd_stop "$target"
    sleep 2
    cmd_start "$target"
}

cmd_logs() {
    local model_key="${1:-generalist}"
    local job_file="$CONNECTION_DIR/${model_key}.job"
    
    if [[ -f "$job_file" ]]; then
        local job_id=$(cat "$job_file")
        local log_file="$LOG_DIR/${model_key}_${job_id}.out"
        if [[ -f "$log_file" ]]; then
            tail -f "$log_file"
        else
            log_warn "Log file not found: $log_file"
            log_info "Available logs:"
            ls -la "$LOG_DIR/${model_key}"* 2>/dev/null || echo "  (none)"
        fi
    else
        log_error "No running job for $model_key"
    fi
}

cmd_help() {
    echo "Usage: $0 <command> [model]"
    echo ""
    echo "Commands:"
    echo "  start [model]    Start model(s) - default: all"
    echo "  stop [model]     Stop model(s) - default: all"
    echo "  restart [model]  Restart model(s)"
    echo "  status           Show status of all models"
    echo "  health           Health check all endpoints"
    echo "  logs <model>     Tail logs for a model"
    echo ""
    echo "Models:"
    echo "  generalist   Qwen2.5-7B for general tasks"
    echo "  coder        Qwen2.5-Coder-7B for code"
    echo "  math         Qwen2.5-Math-7B for math/stats"
    echo "  embeddings   BGE-M3 for embeddings"
    echo ""
    echo "Environment Variables:"
    echo "  VLLM_PARTITION   SLURM partition (default: t4flex)"
    echo "  VLLM_TIME_LIMIT  Job time limit (default: 24:00:00)"
    echo ""
    echo "Examples:"
    echo "  $0 start                # Start all 4 models"
    echo "  $0 start coder          # Start only coder model"
    echo "  $0 status               # Check status"
    echo "  $0 health               # Health check endpoints"
}

# ============================================================================
# Entry Point
# ============================================================================

print_banner

case "${1:-help}" in
    start)   cmd_start "${2:-all}" ;;
    stop)    cmd_stop "${2:-all}" ;;
    restart) cmd_restart "${2:-all}" ;;
    status)  cmd_status ;;
    health)  cmd_health ;;
    logs)    cmd_logs "${2:-generalist}" ;;
    help|--help|-h) cmd_help ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
