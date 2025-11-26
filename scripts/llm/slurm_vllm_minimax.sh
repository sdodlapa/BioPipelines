#!/bin/bash
#SBATCH --job-name=vllm-minimax
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_minimax_%j.out
#SBATCH --error=logs/vllm_minimax_%j.err

# ============================================================
# SLURM Job: vLLM Server with MiniMax-M2
# ============================================================
# Large MoE model optimized for agentic workflows.
# Requires 4x H100 80GB GPUs.
#
# Submit with: sbatch slurm_vllm_minimax.sh
# ============================================================

echo "============================================================"
echo "SLURM Job: vLLM with MiniMax-M2"
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
MODEL="MiniMaxAI/MiniMax-M2"
TP_SIZE=4
MAX_LEN=32768
DTYPE="float8"

echo ""
echo "Model Configuration:"
echo "  Model:    $MODEL"
echo "  TP Size:  $TP_SIZE"
echo "  Max Len:  $MAX_LEN"
echo "  Dtype:    $DTYPE"
echo "  Port:     $VLLM_PORT"
echo ""

# Write connection info
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
    --gpu-memory-utilization 0.95 \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --trust-remote-code

echo ""
echo "============================================================"
echo "vLLM server stopped at $(date)"
echo "============================================================"
