#!/bin/bash
#SBATCH --job-name=vllm-t4-array
#SBATCH --partition=t4flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/deployment/logs/vllm-t4-%A_%a.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/deployment/logs/vllm-t4-%A_%a.err

# =============================================================================
# BioPipelines: vLLM T4 Job Array
# Deploys 10 different vLLM model servers across 10 T4 GPU nodes
# =============================================================================

set -e

# Configuration
export DEPLOYMENT_DIR="/home/sdodl001_odu_edu/BioPipelines/deployment"
export MODEL_CACHE="${HOME}/.cache/huggingface"
export VLLM_PORT_BASE=8000
export HF_HOME="${MODEL_CACHE}"

# Ensure model cache directory exists
mkdir -p "${MODEL_CACHE}"

# Define model assignments for each array task
# Format: "model_id|quantization|gpu_memory_utilization|max_model_len"
declare -a MODEL_CONFIGS=(
    # Task 0: Intent Parsing - Llama-3.2-3B-Instruct
    "meta-llama/Llama-3.2-3B-Instruct|none|0.50|4096"
    
    # Task 1: Code Generation - Qwen2.5-Coder-7B-Instruct (INT8)
    "Qwen/Qwen2.5-Coder-7B-Instruct|int8|0.85|8192"
    
    # Task 2: Code Validation - Qwen2.5-Coder-1.5B-Instruct
    "Qwen/Qwen2.5-Coder-1.5B-Instruct|none|0.45|4096"
    
    # Task 3: Data Analysis - Phi-3.5-mini-instruct (INT8)
    "microsoft/Phi-3.5-mini-instruct|int8|0.85|4096"
    
    # Task 4: Orchestration - Qwen2.5-3B-Instruct (or fallback)
    "Qwen/Qwen2.5-3B-Instruct|none|0.60|4096"
    
    # Task 5: Documentation - Phi-3-mini-4k-instruct
    "microsoft/Phi-3-mini-4k-instruct|none|0.50|4096"
    
    # Task 6: Math/Statistics - Qwen2.5-Math-1.5B-Instruct
    "Qwen/Qwen2.5-Math-1.5B-Instruct|none|0.35|4096"
    
    # Task 7: Bio/Medical - BioMistral-7B (INT8)
    "BioMistral/BioMistral-7B|int8|0.85|4096"
    
    # Task 8: General Reasoning - Gemma-2-2b-it
    "google/gemma-2-2b-it|none|0.45|4096"
    
    # Task 9: Backup/Overflow - Llama-3.2-1B-Instruct
    "meta-llama/Llama-3.2-1B-Instruct|none|0.25|4096"
)

# Parse configuration for this task
CONFIG="${MODEL_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r MODEL_ID QUANTIZATION GPU_UTIL MAX_LEN <<< "$CONFIG"

# Calculate port for this instance
VLLM_PORT=$((VLLM_PORT_BASE + SLURM_ARRAY_TASK_ID))

# Get node hostname for service registration
NODE_HOSTNAME=$(hostname)
NODE_IP=$(hostname -i | awk '{print $1}')

# Log startup info
echo "=========================================="
echo "BioPipelines vLLM T4 Server Starting"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${NODE_HOSTNAME} (${NODE_IP})"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model: ${MODEL_ID}"
echo "Quantization: ${QUANTIZATION}"
echo "GPU Utilization: ${GPU_UTIL}"
echo "Max Length: ${MAX_LEN}"
echo "Port: ${VLLM_PORT}"
echo "=========================================="

# Activate conda environment
source /home/sdodl001_odu_edu/miniforge3/etc/profile.d/conda.sh
conda activate biopipelines

# Ensure vLLM is installed
pip show vllm > /dev/null 2>&1 || pip install vllm

# Build vLLM command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_ID} \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --max-model-len ${MAX_LEN} \
    --trust-remote-code \
    --download-dir ${MODEL_CACHE}"

# Add quantization if specified
if [ "$QUANTIZATION" = "int8" ]; then
    VLLM_CMD="${VLLM_CMD} --quantization squeezellm"
elif [ "$QUANTIZATION" = "awq" ]; then
    VLLM_CMD="${VLLM_CMD} --quantization awq"
fi

# Register this server with the central registry
REGISTRY_FILE="${DEPLOYMENT_DIR}/configs/active_servers.json"
mkdir -p "$(dirname ${REGISTRY_FILE})"

# Create/update server registration (atomic write)
TEMP_REG=$(mktemp)
if [ -f "$REGISTRY_FILE" ]; then
    jq --arg task_id "${SLURM_ARRAY_TASK_ID}" \
       --arg model "${MODEL_ID}" \
       --arg host "${NODE_IP}" \
       --arg port "${VLLM_PORT}" \
       --arg job_id "${SLURM_JOB_ID}" \
       --arg status "starting" \
       --arg timestamp "$(date -Iseconds)" \
       '.servers[$task_id] = {model: $model, host: $host, port: ($port|tonumber), job_id: $job_id, status: $status, timestamp: $timestamp}' \
       "$REGISTRY_FILE" > "$TEMP_REG" && mv "$TEMP_REG" "$REGISTRY_FILE"
else
    echo '{"servers":{}}' | jq --arg task_id "${SLURM_ARRAY_TASK_ID}" \
       --arg model "${MODEL_ID}" \
       --arg host "${NODE_IP}" \
       --arg port "${VLLM_PORT}" \
       --arg job_id "${SLURM_JOB_ID}" \
       --arg status "starting" \
       --arg timestamp "$(date -Iseconds)" \
       '.servers[$task_id] = {model: $model, host: $host, port: ($port|tonumber), job_id: $job_id, status: $status, timestamp: $timestamp}' \
       > "$REGISTRY_FILE"
fi

# Health check function
health_check() {
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VLLM_PORT}/health" 2>/dev/null
}

# Background health reporter
(
    sleep 60  # Wait for server to start
    while true; do
        STATUS=$(health_check)
        if [ "$STATUS" = "200" ]; then
            # Update registry with healthy status
            if [ -f "$REGISTRY_FILE" ]; then
                TEMP_REG=$(mktemp)
                jq --arg task_id "${SLURM_ARRAY_TASK_ID}" \
                   --arg status "healthy" \
                   --arg timestamp "$(date -Iseconds)" \
                   '.servers[$task_id].status = $status | .servers[$task_id].last_health = $timestamp' \
                   "$REGISTRY_FILE" > "$TEMP_REG" && mv "$TEMP_REG" "$REGISTRY_FILE"
            fi
        fi
        sleep 30
    done
) &
HEALTH_PID=$!

# Cleanup on exit
cleanup() {
    echo "Shutting down vLLM server..."
    kill $HEALTH_PID 2>/dev/null || true
    # Update registry to mark as stopped
    if [ -f "$REGISTRY_FILE" ]; then
        TEMP_REG=$(mktemp)
        jq --arg task_id "${SLURM_ARRAY_TASK_ID}" \
           --arg status "stopped" \
           --arg timestamp "$(date -Iseconds)" \
           '.servers[$task_id].status = $status | .servers[$task_id].stopped_at = $timestamp' \
           "$REGISTRY_FILE" > "$TEMP_REG" && mv "$TEMP_REG" "$REGISTRY_FILE"
    fi
}
trap cleanup EXIT

# Start vLLM server
echo "Starting vLLM server..."
echo "Command: ${VLLM_CMD}"
exec ${VLLM_CMD}
