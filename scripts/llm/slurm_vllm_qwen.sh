#!/bin/bash
#SBATCH --job-name=vllm-qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_qwen_%j.out
#SBATCH --error=logs/vllm_qwen_%j.err

# ============================================================
# SLURM Job: vLLM Server with Qwen2.5-Coder-32B
# ============================================================
# This is the recommended model for most coding tasks.
# Requires 1x H100 80GB GPU.
#
# Submit with: sbatch slurm_vllm_qwen.sh
# ============================================================

echo "============================================================"
echo "SLURM Job: vLLM with Qwen2.5-Coder-32B"
echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Started:    $(date)"
echo "============================================================"

# Load modules
module load cuda/12.1 2>/dev/null || true

# Activate environment
if [[ -f ~/envs/biopipelines/bin/activate ]]; then
    source ~/envs/biopipelines/bin/activate
elif [[ -f ~/.venv/bin/activate ]]; then
    source ~/.venv/bin/activate
fi

# Set environment
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_PORT="${VLLM_PORT:-8000}"

# Model settings
MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
TP_SIZE=1
MAX_LEN=32768
DTYPE="float16"

echo ""
echo "Model Configuration:"
echo "  Model:    $MODEL"
echo "  TP Size:  $TP_SIZE"
echo "  Max Len:  $MAX_LEN"
echo "  Dtype:    $DTYPE"
echo "  Port:     $VLLM_PORT"
echo ""

# Write connection info for client
NODE_IP=$(hostname -i | awk '{print $1}')
echo "VLLM_HOST=$NODE_IP" > /tmp/vllm_connection_${SLURM_JOB_ID}.env
echo "VLLM_PORT=$VLLM_PORT" >> /tmp/vllm_connection_${SLURM_JOB_ID}.env
echo "VLLM_MODEL=$MODEL" >> /tmp/vllm_connection_${SLURM_JOB_ID}.env

echo "Connection info saved to: /tmp/vllm_connection_${SLURM_JOB_ID}.env"
echo "Connect from client: export VLLM_HOST=$NODE_IP"
echo ""

# Start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_LEN \
    --dtype $DTYPE \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --trust-remote-code

echo ""
echo "============================================================"
echo "vLLM server stopped at $(date)"
echo "============================================================"
