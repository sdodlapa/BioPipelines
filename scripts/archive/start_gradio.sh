#!/bin/bash
# ============================================================================
# BioPipelines Web UI Launcher - Unified Smart Deployment
# ============================================================================
#
# Auto-detects available resources and chooses optimal deployment:
#   - H100/A100 GPUs available → Local vLLM with best model
#   - T4 GPUs available → Local vLLM with smaller model
#   - No GPUs → Cloud API fallback (Lightning.ai, OpenAI)
#
# Usage:
#   ./scripts/start_gradio.sh              # Auto-detect and run
#   ./scripts/start_gradio.sh --gpu        # Force GPU mode (submit SLURM job)
#   ./scripts/start_gradio.sh --cloud      # Force cloud mode (no GPU)
#   ./scripts/start_gradio.sh --multi      # 4 GPU multi-model mode
#
# ============================================================================

set -e

# Default settings
PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
SHARE=""
MODE="auto"  # auto, gpu, cloud, multi
GPU_MODEL="${GPU_MODEL:-auto}"
GPU_PARTITION="${GPU_PARTITION:-auto}"
GPU_TIME="${GPU_TIME:-8:00:00}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_CODER_PORT="${VLLM_CODER_PORT:-8001}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --share) SHARE="--share"; shift ;;
        --gpu|-g) MODE="gpu"; shift ;;
        --cloud|-c) MODE="cloud"; shift ;;
        --multi|-M) MODE="multi"; shift ;;  # 4 GPU: supervisor + coder
        --model|-m) GPU_MODEL="$2"; shift 2 ;;
        --partition) GPU_PARTITION="$2"; shift 2 ;;
        --time) GPU_TIME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  (default)       Auto-detect resources"
            echo "  --gpu           Force GPU mode (submit SLURM job)"
            echo "  --cloud         Force cloud mode (Lightning.ai/OpenAI)"
            echo "  --multi         Multi-model mode (4 GPU: Llama + Coder)"
            echo ""
            echo "Options:"
            echo "  --model MODEL   LLM model (default: auto-select)"
            echo "  --partition P   SLURM partition (default: auto-detect)"
            echo "  --time TIME     Job time (default: 8:00:00)"
            echo "  --port PORT     Gradio port (default: 7860)"
            echo "  --share         Create public Gradio link"
            echo ""
            echo "Auto-detection:"
            echo "  - Checks SLURM partitions for available GPUs"
            echo "  - H100 (80GB): Llama-3.3-70B or multi-model"
            echo "  - T4 (16GB): Qwen-3B or smaller models"
            echo "  - No GPU: Lightning.ai with DeepSeek-V3"
            echo ""
            echo "Models: qwen-1.5b, qwen-3b, phi-3, llama-70b, qwen-72b, qwen-coder-32b"
            exit 0
            ;;
        *) shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Model mappings
declare -A MODELS
MODELS["qwen-1.5b"]="Qwen/Qwen2.5-1.5B-Instruct"
MODELS["qwen-3b"]="Qwen/Qwen2.5-3B-Instruct"
MODELS["phi-3"]="microsoft/Phi-3-mini-4k-instruct"
MODELS["biomistral"]="BioMistral/BioMistral-7B-AWQ"
MODELS["llama-70b"]="meta-llama/Llama-3.3-70B-Instruct"
MODELS["qwen-72b"]="Qwen/Qwen2.5-72B-Instruct-AWQ"
MODELS["qwen-coder-32b"]="Qwen/Qwen2.5-Coder-32B-Instruct"
MODELS["deepseek-coder"]="deepseek-ai/deepseek-coder-33b-instruct"

# Partition configs: GPUs, CPUs, Memory, GPU Utilization, GPU Type
declare -A PARTITION_GPUS PARTITION_CPUS PARTITION_MEM PARTITION_UTIL PARTITION_TYPE
PARTITION_GPUS["t4flex"]=1;       PARTITION_CPUS["t4flex"]=8;   PARTITION_MEM["t4flex"]="56G";  PARTITION_UTIL["t4flex"]=0.9;  PARTITION_TYPE["t4flex"]="T4"
PARTITION_GPUS["h100flex"]=1;     PARTITION_CPUS["h100flex"]=26; PARTITION_MEM["h100flex"]="200G"; PARTITION_UTIL["h100flex"]=0.95; PARTITION_TYPE["h100flex"]="H100"
PARTITION_GPUS["h100dualflex"]=2; PARTITION_CPUS["h100dualflex"]=26; PARTITION_MEM["h100dualflex"]="450G"; PARTITION_UTIL["h100dualflex"]=0.95; PARTITION_TYPE["h100dualflex"]="H100"
PARTITION_GPUS["h100quadflex"]=4; PARTITION_CPUS["h100quadflex"]=52; PARTITION_MEM["h100quadflex"]="800G"; PARTITION_UTIL["h100quadflex"]=0.95; PARTITION_TYPE["h100quadflex"]="H100"

# ============================================================================
# Auto-detect available resources
# ============================================================================
detect_resources() {
    echo "🔍 Detecting available resources..."
    
    # Check if we're on a SLURM cluster
    if ! command -v sinfo &> /dev/null; then
        echo "   No SLURM detected - using cloud mode"
        DETECTED_MODE="cloud"
        return
    fi
    
    # Check available partitions
    AVAILABLE_PARTITIONS=$(sinfo -h -o "%P %a %D" 2>/dev/null | grep -E "up|idle" || echo "")
    
    if echo "$AVAILABLE_PARTITIONS" | grep -q "h100quadflex"; then
        DETECTED_PARTITION="h100quadflex"
        DETECTED_MODE="multi"  # 4 GPUs = can run multi-model
        echo "   Found h100quadflex (4× H100 80GB) - multi-model capable"
    elif echo "$AVAILABLE_PARTITIONS" | grep -q "h100dualflex"; then
        DETECTED_PARTITION="h100dualflex"
        DETECTED_MODE="gpu"
        echo "   Found h100dualflex (2× H100 80GB)"
    elif echo "$AVAILABLE_PARTITIONS" | grep -q "h100flex"; then
        DETECTED_PARTITION="h100flex"
        DETECTED_MODE="gpu"
        echo "   Found h100flex (1× H100 80GB)"
    elif echo "$AVAILABLE_PARTITIONS" | grep -q "t4flex"; then
        DETECTED_PARTITION="t4flex"
        DETECTED_MODE="gpu"
        echo "   Found t4flex (1× T4 16GB)"
    else
        DETECTED_MODE="cloud"
        echo "   No GPU partitions available - using cloud mode"
    fi
    
    # Auto-select model based on GPU type
    if [ "$GPU_MODEL" = "auto" ]; then
        case "$DETECTED_PARTITION" in
            h100quadflex|h100dualflex)
                GPU_MODEL="llama-70b"
                echo "   Auto-selected model: llama-70b (best for H100)"
                ;;
            h100flex)
                GPU_MODEL="qwen-coder-32b"
                echo "   Auto-selected model: qwen-coder-32b (fits 1× H100)"
                ;;
            t4flex)
                GPU_MODEL="qwen-3b"
                echo "   Auto-selected model: qwen-3b (fits T4)"
                ;;
        esac
    fi
    
    # Use detected partition if not specified
    if [ "$GPU_PARTITION" = "auto" ]; then
        GPU_PARTITION="${DETECTED_PARTITION:-h100dualflex}"
    fi
}

# Run auto-detection for auto mode
if [ "$MODE" = "auto" ]; then
    detect_resources
    MODE="${DETECTED_MODE:-cloud}"
    echo "   Selected mode: $MODE"
    echo ""
fi

# Validate model
if [ "$GPU_MODEL" != "auto" ] && [[ ! ${MODELS[$GPU_MODEL]+_} ]]; then
    echo "❌ Unknown model: $GPU_MODEL"
    echo "Available: ${!MODELS[@]}"
    exit 1
fi

# ============================================================================
# GPU MODE: Submit SLURM job with local vLLM
# ============================================================================
if [ "$MODE" = "gpu" ] || [ "$MODE" = "multi" ]; then
    
    MODEL_HF="${MODELS[$GPU_MODEL]}"
    NUM_GPUS="${PARTITION_GPUS[$GPU_PARTITION]:-2}"
    GPU_CPUS="${PARTITION_CPUS[$GPU_PARTITION]:-26}"
    GPU_MEM="${PARTITION_MEM[$GPU_PARTITION]:-450G}"
    GPU_UTIL="${PARTITION_UTIL[$GPU_PARTITION]:-0.95}"
    GPU_TYPE="${PARTITION_TYPE[$GPU_PARTITION]:-H100}"
    
    # Multi-model mode: run supervisor + coding model on separate GPU pairs
    MULTI_MODEL=false
    CODER_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
    if [ "$MODE" = "multi" ] && [ "$NUM_GPUS" -ge 4 ]; then
        MULTI_MODEL=true
        echo "🔧 Multi-model mode: Llama (GPU 0-1) + Coder (GPU 2-3)"
    fi
    
    mkdir -p "${PROJECT_DIR}/logs"
    
    echo "📦 Submitting SLURM job..."
    echo "   Partition: $GPU_PARTITION ($NUM_GPUS × $GPU_TYPE)"
    echo "   Model: $GPU_MODEL → $MODEL_HF"
    [ "$MULTI_MODEL" = true ] && echo "   Coder: $CODER_MODEL"
    echo ""
    
    # Submit job directly
    JOB_ID=$(sbatch --parsable << SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=biopipelines-gpu
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --cpus-per-task=${GPU_CPUS}
#SBATCH --mem=${GPU_MEM}
#SBATCH --time=${GPU_TIME}
#SBATCH --output=${PROJECT_DIR}/logs/gpu_server_%j.out
#SBATCH --error=${PROJECT_DIR}/logs/gpu_server_%j.err

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  🧬 BioPipelines GPU Server                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:     \$SLURM_JOB_ID"
echo "Node:       \$SLURM_NODELIST"
echo "Partition:  ${GPU_PARTITION} (${NUM_GPUS} × ${GPU_TYPE})"
echo "Model:      ${GPU_MODEL}"
echo "Multi:      ${MULTI_MODEL}"
echo "Started:    \$(date)"
echo ""

# Save connection info
cat > ${PROJECT_DIR}/logs/gpu_server_info.txt << EOF
SLURM_JOB_ID=\$SLURM_JOB_ID
SLURM_NODELIST=\$SLURM_NODELIST
VLLM_URL=http://\${SLURM_NODELIST}:${VLLM_PORT}/v1
VLLM_CODER_URL=http://\${SLURM_NODELIST}:${VLLM_CODER_PORT}/v1
GRADIO_URL=http://\${SLURM_NODELIST}:${PORT}
MULTI_MODEL=${MULTI_MODEL}
EOF

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines 2>/dev/null || conda activate base
cd ${PROJECT_DIR}

echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "=== Checking dependencies ==="
python -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || { pip install vllm -q && echo "vLLM installed"; }
python -c "import workflow_composer" 2>/dev/null || pip install -e . -q
python -c "import gradio" 2>/dev/null || pip install gradio -q
echo "Ready"
echo ""

echo "=== Loading API keys ==="
[ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=\$(cat .secrets/openai_key)
[ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=\$(cat .secrets/lightning_key)
[ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=\$(cat .secrets/google_api_key)
[ -f ".secrets/github_token" ] && export GITHUB_TOKEN=\$(cat .secrets/github_token)
[ -f ".secrets/hf_token" ] && export HF_TOKEN=\$(cat .secrets/hf_token)
echo "Done"
echo ""

# ========================================
# Start vLLM server(s)
# ========================================

if [ "${MULTI_MODEL}" = "true" ]; then
    # Multi-model: Split GPUs between supervisor and coder
    echo "=== Starting Multi-Model vLLM (4 GPUs) ==="
    echo "  GPU 0-1: ${MODEL_HF} (Supervisor)"
    echo "  GPU 2-3: ${CODER_MODEL} (Coding Agent)"
    echo ""
    
    # Start supervisor model on GPU 0-1
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model "${MODEL_HF}" --host 0.0.0.0 --port ${VLLM_PORT} --tensor-parallel-size 2 --gpu-memory-utilization ${GPU_UTIL} --max-model-len 32768 --trust-remote-code --enable-auto-tool-choice --tool-call-parser hermes 2>&1 | tee ${PROJECT_DIR}/logs/vllm_supervisor_\${SLURM_JOB_ID}.log &
    SUPERVISOR_PID=\$!
    
    # Wait a bit then start coder model on GPU 2-3
    sleep 30
    
    CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server --model "${CODER_MODEL}" --host 0.0.0.0 --port ${VLLM_CODER_PORT} --tensor-parallel-size 2 --gpu-memory-utilization ${GPU_UTIL} --max-model-len 65536 --trust-remote-code 2>&1 | tee ${PROJECT_DIR}/logs/vllm_coder_\${SLURM_JOB_ID}.log &
    CODER_PID=\$!
    
    # Wait for both servers
    echo "Waiting for vLLM servers..."
    for i in {1..180}; do
        SUP_OK=false
        COD_OK=false
        curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1 && SUP_OK=true
        curl -s "http://localhost:${VLLM_CODER_PORT}/health" > /dev/null 2>&1 && COD_OK=true
        
        if [ "\$SUP_OK" = true ] && [ "\$COD_OK" = true ]; then
            echo "✅ Both vLLM servers ready"
            break
        fi
        sleep 5
        echo "  [\$((i*5))s] Supervisor: \$SUP_OK, Coder: \$COD_OK"
    done
    
    export VLLM_CODER_URL="http://localhost:${VLLM_CODER_PORT}/v1"
    
else
    # Single model mode
    echo "=== Starting vLLM (${NUM_GPUS} GPUs) ==="
    python -m vllm.entrypoints.openai.api_server --model "${MODEL_HF}" --host 0.0.0.0 --port ${VLLM_PORT} --tensor-parallel-size ${NUM_GPUS} --gpu-memory-utilization ${GPU_UTIL} --max-model-len 32768 --trust-remote-code --enable-auto-tool-choice --tool-call-parser hermes 2>&1 | tee ${PROJECT_DIR}/logs/vllm_\${SLURM_JOB_ID}.log &
    VLLM_PID=\$!

    echo "Waiting for vLLM..."
    for i in {1..180}; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "✅ vLLM ready"
            break
        fi
        sleep 5
        echo "  [\$((i*5))s] Loading model..."
    done
fi

export VLLM_URL="http://localhost:${VLLM_PORT}/v1"
export USE_LOCAL_LLM="true"

echo ""
echo "=== Starting Gradio ==="
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  🎉 Server Ready!                                                ║"
echo "║  Gradio:     http://\${SLURM_NODELIST}:${PORT}                    "
echo "║  Supervisor: http://\${SLURM_NODELIST}:${VLLM_PORT}/v1            "
if [ "${MULTI_MODEL}" = "true" ]; then
echo "║  Coder:      http://\${SLURM_NODELIST}:${VLLM_CODER_PORT}/v1      "
fi
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

python -m workflow_composer.web.gradio_app --host 0.0.0.0 --port ${PORT} ${SHARE}

# Cleanup
if [ "${MULTI_MODEL}" = "true" ]; then
    kill \$SUPERVISOR_PID \$CODER_PID 2>/dev/null || true
else
    kill \$VLLM_PID 2>/dev/null || true
fi
SLURM_SCRIPT
)

    echo "✅ Submitted job: $JOB_ID"
    echo ""
    echo "Monitor:  tail -f ${PROJECT_DIR}/logs/gpu_server_${JOB_ID}.out"
    echo "Status:   squeue -j $JOB_ID"
    echo "Cancel:   scancel $JOB_ID"
    exit 0
fi

# ============================================================================
# CLOUD MODE: Run on login node with cloud APIs
# ============================================================================
echo "🧬 BioPipelines - Cloud Mode (Lightning.ai / OpenAI / GitHub Copilot)"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines 2>/dev/null || conda activate base
cd "$PROJECT_DIR"

# Load API keys
echo "Loading API keys..."
[ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key) && echo "  ✓ OpenAI"
[ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=$(cat .secrets/lightning_key) && echo "  ✓ Lightning.ai"
[ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key) && echo "  ✓ Google"
[ -f ".secrets/github_token" ] && export GITHUB_TOKEN=$(cat .secrets/github_token) && echo "  ✓ GitHub"
echo ""

# Check which providers are available
echo "Available providers:"
[ -n "$LIGHTNING_API_KEY" ] && echo "  ✓ Lightning.ai (30M free tokens - recommended!)"
[ -n "$OPENAI_API_KEY" ] && echo "  ✓ OpenAI (GPT-4o)"
[ -n "$GITHUB_TOKEN" ] && echo "  ✓ GitHub Copilot"
[ -n "$GOOGLE_API_KEY" ] && echo "  ✓ Google Gemini"
echo ""

python -c "import workflow_composer" 2>/dev/null || pip install -e . -q

echo "Starting Gradio on port $PORT..."
echo "No GPU - using cloud LLM providers"
echo ""
python -m workflow_composer.web.gradio_app --host "$HOST" --port "$PORT" $SHARE
