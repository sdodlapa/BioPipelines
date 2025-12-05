#!/bin/bash
# ============================================================================
# BioPipelines T4 Fleet Deployment Script
# ============================================================================
# 
# Deploys specialized models across 10 T4 GPUs for the multi-model strategy.
# Each T4 has 16GB VRAM - models are selected to fit with room for KV cache.
#
# Architecture:
#   - Each node runs independent vLLM server(s)
#   - Router/load balancer distributes requests
#   - Some GPUs can host multiple small models
#
# Usage: ./deploy_t4_fleet.sh [start|stop|status|restart] [category]
#
# Examples:
#   ./deploy_t4_fleet.sh start          # Start all models
#   ./deploy_t4_fleet.sh start intent   # Start only intent parsing model
#   ./deploy_t4_fleet.sh status         # Check all servers
#   ./deploy_t4_fleet.sh stop           # Stop all servers
#
# Author: BioPipelines Team
# Last Updated: 2025-12-05
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/logs/vllm"
CONFIG_DIR="$PROJECT_DIR/config"
CONNECTION_DIR="$PROJECT_DIR/.model_connections"

# SLURM configuration
PARTITION="t4flex"
TIME_LIMIT="24:00:00"
MEM="32G"
CPUS=8

# Create directories
mkdir -p "$LOG_DIR" "$CONNECTION_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================================
# Model Definitions (T4-optimized)
# ============================================================================

# Format: NAME|MODEL_ID|PORT|QUANTIZATION|VRAM|CONTEXT_LEN|DESCRIPTION
declare -A MODELS=(
    # High-frequency models (might want multiple replicas)
    ["intent"]="llama-3.2-3b|meta-llama/Llama-3.2-3B-Instruct|8001|none|7GB|128000|Intent parsing and classification"
    ["intent2"]="llama-3.2-3b-r2|meta-llama/Llama-3.2-3B-Instruct|8002|none|7GB|128000|Intent parsing replica 2"
    
    # Code models
    ["codegen"]="qwen-coder-7b|Qwen/Qwen2.5-Coder-7B-Instruct|8010|awq|8GB|32000|Code generation"
    ["codegen2"]="qwen-coder-7b-r2|Qwen/Qwen2.5-Coder-7B-Instruct|8011|awq|8GB|32000|Code generation replica 2"
    ["validation"]="qwen-coder-1.5b|Qwen/Qwen2.5-Coder-1.5B-Instruct|8012|none|3GB|32000|Quick code validation"
    
    # Specialized models
    ["analysis"]="phi-3.5-mini|microsoft/Phi-3.5-mini-instruct|8020|none|8GB|128000|Data analysis"
    ["math"]="qwen-math-7b|Qwen/Qwen2.5-Math-7B-Instruct|8021|awq|7.5GB|8000|Math and statistics"
    ["biomedical"]="biomistral-7b|BioMistral/BioMistral-7B|8030|awq|7.5GB|8000|Biomedical reasoning"
    ["docs"]="gemma-2-9b|google/gemma-2-9b-it|8040|int8|9.5GB|8000|Documentation"
    
    # Infrastructure models
    ["embeddings"]="bge-m3|BAAI/bge-m3|8050|none|1.5GB|8192|Embeddings"
    ["safety"]="llama-guard|meta-llama/Llama-Guard-3-1B|8051|none|2.5GB|4096|Safety guardrails"
)

# GPU allocation plan
declare -A GPU_ALLOCATION=(
    # T4-1: Intent + Safety (7GB + 2.5GB = 9.5GB)
    ["t4_1"]="intent,safety"
    # T4-2: Intent replica + Embeddings (7GB + 1.5GB = 8.5GB)
    ["t4_2"]="intent2,embeddings"
    # T4-3: Code gen + Validation (8GB + 3GB = 11GB) - tight but works
    ["t4_3"]="codegen,validation"
    # T4-4: Code gen replica (8GB)
    ["t4_4"]="codegen2"
    # T4-5: Data analysis (8GB)
    ["t4_5"]="analysis"
    # T4-6: Math (7.5GB)
    ["t4_6"]="math"
    # T4-7: Biomedical (7.5GB)
    ["t4_7"]="biomedical"
    # T4-8: Documentation (9.5GB)
    ["t4_8"]="docs"
    # T4-9: Reserved/hot spare
    ["t4_9"]=""
    # T4-10: Reserved/hot spare
    ["t4_10"]=""
)

# ============================================================================
# Functions
# ============================================================================

generate_sbatch() {
    local model_key=$1
    local model_info=${MODELS[$model_key]}
    
    IFS='|' read -r name model_id port quant vram context desc <<< "$model_info"
    
    local sbatch_file="$SCRIPT_DIR/serve_${model_key}.sbatch"
    
    cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --job-name=vllm_${name}
#SBATCH --output=${LOG_DIR}/${name}_%j.out
#SBATCH --error=${LOG_DIR}/${name}_%j.err
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
# VRAM: ${vram}
# ============================================================================

echo "============================================"
echo "Starting ${name} Server"
echo "Date: \$(date)"
echo "Node: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
echo "============================================"

# Environment setup
export HOME=/home/sdodl001_odu_edu
export HF_HOME=\$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=\$HOME/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=\$HOME/.cache/huggingface

# Activate environment
source \$HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env 2>/dev/null || conda activate biopipelines

echo ""
echo "Environment:"
echo "  Python: \$(which python)"
echo "  vLLM version: \$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'checking...')"
echo "  Model: ${model_id}"
echo "  Port: ${port}"
echo ""

# GPU info
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Save connection info
CONNECTION_FILE="${CONNECTION_DIR}/${model_key}.env"
echo "MODEL_NAME=${name}" > "\$CONNECTION_FILE"
echo "MODEL_ID=${model_id}" >> "\$CONNECTION_FILE"
echo "HOST=\$(hostname)" >> "\$CONNECTION_FILE"
echo "PORT=${port}" >> "\$CONNECTION_FILE"
echo "URL=http://\$(hostname):${port}/v1" >> "\$CONNECTION_FILE"
echo "SLURM_JOB_ID=\$SLURM_JOB_ID" >> "\$CONNECTION_FILE"

echo "Connection info saved to: \$CONNECTION_FILE"
echo "Connect at: http://\$(hostname):${port}/v1"
echo ""

# Build vLLM command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \\
    --model ${model_id} \\
    --host 0.0.0.0 \\
    --port ${port} \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len ${context} \\
    --trust-remote-code \\
    --dtype float16"

# Add quantization if needed
if [ "${quant}" = "awq" ]; then
    VLLM_CMD="\$VLLM_CMD --quantization awq"
elif [ "${quant}" = "int8" ]; then
    VLLM_CMD="\$VLLM_CMD --quantization int8"
elif [ "${quant}" = "gptq" ]; then
    VLLM_CMD="\$VLLM_CMD --quantization gptq"
fi

echo "Starting vLLM server..."
echo "Command: \$VLLM_CMD"
echo ""

eval \$VLLM_CMD

echo ""
echo "Server stopped at \$(date)"

# Clean up
rm -f "\$CONNECTION_FILE"
EOF

    chmod +x "$sbatch_file"
    echo "$sbatch_file"
}

start_model() {
    local model_key=$1
    
    if [[ -z "${MODELS[$model_key]}" ]]; then
        print_error "Unknown model: $model_key"
        return 1
    fi
    
    local model_info=${MODELS[$model_key]}
    IFS='|' read -r name model_id port quant vram context desc <<< "$model_info"
    
    echo "Starting $name ($model_id)..."
    
    # Generate sbatch file
    local sbatch_file=$(generate_sbatch "$model_key")
    
    # Submit job
    local job_id=$(sbatch --parsable "$sbatch_file")
    
    if [[ -n "$job_id" ]]; then
        print_success "Submitted job $job_id for $name (port $port)"
        echo "$job_id" > "$CONNECTION_DIR/${model_key}.job"
    else
        print_error "Failed to submit job for $name"
        return 1
    fi
}

stop_model() {
    local model_key=$1
    local job_file="$CONNECTION_DIR/${model_key}.job"
    
    if [[ -f "$job_file" ]]; then
        local job_id=$(cat "$job_file")
        echo "Stopping $model_key (job $job_id)..."
        scancel "$job_id" 2>/dev/null && print_success "Cancelled job $job_id" || print_warning "Job may already be stopped"
        rm -f "$job_file" "$CONNECTION_DIR/${model_key}.env"
    else
        print_warning "No running job found for $model_key"
    fi
}

status_model() {
    local model_key=$1
    local model_info=${MODELS[$model_key]}
    local job_file="$CONNECTION_DIR/${model_key}.job"
    local env_file="$CONNECTION_DIR/${model_key}.env"
    
    IFS='|' read -r name model_id port quant vram context desc <<< "$model_info"
    
    echo -n "$model_key ($name, port $port): "
    
    if [[ -f "$job_file" ]]; then
        local job_id=$(cat "$job_file")
        local state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
        
        if [[ "$state" == "RUNNING" ]]; then
            print_success "RUNNING (job $job_id)"
            if [[ -f "$env_file" ]]; then
                source "$env_file"
                echo "  URL: $URL"
            fi
        elif [[ "$state" == "PENDING" ]]; then
            print_warning "PENDING (job $job_id)"
        else
            print_error "STOPPED (last job: $job_id)"
        fi
    else
        echo "NOT STARTED"
    fi
}

start_all() {
    print_header "Starting All T4 Models"
    
    for model_key in "${!MODELS[@]}"; do
        start_model "$model_key"
        sleep 2  # Small delay to avoid overwhelming scheduler
    done
    
    echo ""
    print_header "Submitted Jobs"
    squeue -u "$USER" --partition="$PARTITION" -o "%.10i %.20j %.8T %.10M %.6D %R"
}

stop_all() {
    print_header "Stopping All T4 Models"
    
    for model_key in "${!MODELS[@]}"; do
        stop_model "$model_key"
    done
}

status_all() {
    print_header "T4 Fleet Status"
    
    for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
        status_model "$model_key"
    done
    
    echo ""
    echo "SLURM Queue:"
    squeue -u "$USER" --partition="$PARTITION" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "No jobs running"
}

show_help() {
    cat << EOF
BioPipelines T4 Fleet Deployment

Usage: $0 <command> [model]

Commands:
  start [model]    Start all models or a specific model
  stop [model]     Stop all models or a specific model
  status [model]   Show status of all models or a specific model
  restart [model]  Restart all models or a specific model
  list             List available models
  help             Show this help message

Available Models:
EOF
    
    for model_key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
        local model_info=${MODELS[$model_key]}
        IFS='|' read -r name model_id port quant vram context desc <<< "$model_info"
        printf "  %-15s %s\n" "$model_key" "$desc ($vram)"
    done
}

# ============================================================================
# Main
# ============================================================================

case "${1:-help}" in
    start)
        if [[ -n "$2" ]]; then
            start_model "$2"
        else
            start_all
        fi
        ;;
    stop)
        if [[ -n "$2" ]]; then
            stop_model "$2"
        else
            stop_all
        fi
        ;;
    status)
        if [[ -n "$2" ]]; then
            status_model "$2"
        else
            status_all
        fi
        ;;
    restart)
        if [[ -n "$2" ]]; then
            stop_model "$2"
            sleep 2
            start_model "$2"
        else
            stop_all
            sleep 5
            start_all
        fi
        ;;
    list)
        show_help
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
