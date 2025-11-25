#!/bin/bash
# ============================================================================
# BioPipelines LLM Setup Script
# ============================================================================
# This script helps set up LLM backends for the Workflow Composer:
# - OpenAI API configuration
# - vLLM server for GPU inference with HuggingFace models
# - Verify LLM connectivity
#
# Usage:
#   ./scripts/llm/setup_llm.sh [command]
#
# Commands:
#   install-vllm    Install vLLM and dependencies
#   start-vllm      Start vLLM server with default model
#   stop-vllm       Stop running vLLM server
#   test-openai     Test OpenAI API connectivity
#   test-vllm       Test vLLM server connectivity
#   test-all        Test all configured LLM providers
#   env-setup       Set up environment variables
#   help            Show this help message
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}

# Default models
DEFAULT_VLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OPENAI_MODEL="gpt-4o"

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        return 0
    else
        log_warn "No NVIDIA GPU detected. vLLM requires GPU for optimal performance."
        return 1
    fi
}

# ============================================================================
# vLLM Installation
# ============================================================================

install_vllm() {
    log_info "Installing vLLM and dependencies..."
    
    # Check for GPU
    check_gpu || log_warn "Continuing without GPU verification..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python version: $python_version"
    
    # Install vLLM
    log_info "Installing vLLM..."
    pip install vllm --upgrade
    
    # Install HuggingFace Hub for model downloading
    log_info "Installing huggingface_hub..."
    pip install huggingface_hub --upgrade
    
    # Check installation
    if python3 -c "import vllm" 2>/dev/null; then
        log_success "vLLM installed successfully!"
        python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
    else
        log_error "vLLM installation failed"
        exit 1
    fi
}

# ============================================================================
# vLLM Server Management
# ============================================================================

start_vllm() {
    local model="${1:-$DEFAULT_VLLM_MODEL}"
    local gpu_util="${2:-0.9}"
    local tensor_parallel="${3:-1}"
    
    log_info "Starting vLLM server..."
    log_info "Model: $model"
    log_info "Port: $VLLM_PORT"
    log_info "GPU Memory Utilization: $gpu_util"
    log_info "Tensor Parallel Size: $tensor_parallel"
    
    # Check if server is already running
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        log_warn "vLLM server already running on port $VLLM_PORT"
        return 0
    fi
    
    # Check GPU memory
    if check_gpu; then
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        log_info "Available GPU memory: ${free_mem}MB"
    fi
    
    # Start server in background
    log_info "Starting vLLM server (this may take a few minutes to load the model)..."
    
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --gpu-memory-utilization "$gpu_util" \
        --tensor-parallel-size "$tensor_parallel" \
        > "$PROJECT_ROOT/logs/vllm_server.log" 2>&1 &
    
    vllm_pid=$!
    echo "$vllm_pid" > "$PROJECT_ROOT/logs/vllm_server.pid"
    
    log_info "vLLM server starting with PID $vllm_pid"
    log_info "Log file: $PROJECT_ROOT/logs/vllm_server.log"
    
    # Wait for server to be ready
    log_info "Waiting for server to be ready..."
    for i in {1..60}; do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            log_success "vLLM server is ready!"
            log_info "API endpoint: http://localhost:$VLLM_PORT/v1"
            return 0
        fi
        sleep 5
        echo -n "."
    done
    
    log_error "Server failed to start within 5 minutes. Check logs: $PROJECT_ROOT/logs/vllm_server.log"
    exit 1
}

stop_vllm() {
    log_info "Stopping vLLM server..."
    
    pid_file="$PROJECT_ROOT/logs/vllm_server.pid"
    
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            rm "$pid_file"
            log_success "vLLM server stopped (PID: $pid)"
        else
            log_warn "vLLM server process not found"
            rm "$pid_file"
        fi
    else
        # Try to find and kill by port
        pid=$(lsof -ti:$VLLM_PORT 2>/dev/null || true)
        if [[ -n "$pid" ]]; then
            kill "$pid"
            log_success "vLLM server stopped (PID: $pid)"
        else
            log_warn "No vLLM server running on port $VLLM_PORT"
        fi
    fi
}

# ============================================================================
# Test Functions
# ============================================================================

test_openai() {
    log_info "Testing OpenAI API connectivity..."
    
    if [[ -z "$OPENAI_API_KEY" ]]; then
        log_error "OPENAI_API_KEY environment variable not set"
        log_info "Set it with: export OPENAI_API_KEY='your-api-key'"
        return 1
    fi
    
    log_info "API key found (${OPENAI_API_KEY:0:8}...)"
    
    # Test API with a simple request
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        "https://api.openai.com/v1/models" 2>/dev/null || echo "000")
    
    if [[ "$response" == "200" ]]; then
        log_success "OpenAI API is accessible"
        
        # Test a simple completion
        log_info "Testing chat completion..."
        python3 << EOF
import os
import sys
sys.path.insert(0, '$PROJECT_ROOT/src')
from workflow_composer.llm import get_llm

try:
    llm = get_llm("openai", model="gpt-4o")
    response = llm.complete("Say 'Hello, BioPipelines!' in one line.")
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.tokens_used}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
        log_success "OpenAI test completed successfully!"
    else
        log_error "OpenAI API returned status code: $response"
        return 1
    fi
}

test_vllm() {
    log_info "Testing vLLM server connectivity..."
    
    vllm_url="${VLLM_BASE_URL:-http://localhost:$VLLM_PORT}"
    
    # Check health endpoint
    if ! curl -s "$vllm_url/health" > /dev/null 2>&1; then
        log_error "vLLM server not accessible at $vllm_url"
        log_info "Start it with: ./scripts/llm/setup_llm.sh start-vllm"
        return 1
    fi
    
    log_success "vLLM server is accessible"
    
    # Get available models
    log_info "Available models:"
    curl -s "$vllm_url/v1/models" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for model in data.get('data', []):
    print(f\"  - {model.get('id', 'unknown')}\")"
    
    # Test a simple completion
    log_info "Testing chat completion..."
    python3 << EOF
import os
import sys
sys.path.insert(0, '$PROJECT_ROOT/src')
from workflow_composer.llm import get_llm

try:
    llm = get_llm("vllm")
    response = llm.complete("Say 'Hello, BioPipelines!' in one line.")
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.tokens_used}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
    log_success "vLLM test completed successfully!"
}

test_all() {
    log_info "Testing all LLM providers..."
    
    python3 << EOF
import os
import sys
sys.path.insert(0, '$PROJECT_ROOT/src')
from workflow_composer.llm import check_providers

print("\nLLM Provider Status:")
print("-" * 40)
status = check_providers()
for provider, available in status.items():
    status_str = "✓ Available" if available else "✗ Not available"
    print(f"  {provider:15} {status_str}")
print("-" * 40)
EOF
}

# ============================================================================
# Environment Setup
# ============================================================================

env_setup() {
    log_info "Setting up environment variables..."
    
    env_file="$PROJECT_ROOT/.env.llm"
    
    cat > "$env_file" << 'EOF'
# BioPipelines LLM Environment Variables
# Source this file: source .env.llm

# OpenAI API
# Get your API key at: https://platform.openai.com/api-keys
export OPENAI_API_KEY=""

# HuggingFace Token (optional, for gated models)
# Get your token at: https://huggingface.co/settings/tokens
export HF_TOKEN=""

# vLLM Server URL
export VLLM_BASE_URL="http://localhost:8000"

# Optional: vLLM API key (for secured deployments)
# export VLLM_API_KEY=""
EOF
    
    log_success "Environment template created: $env_file"
    log_info "Edit the file and add your API keys, then run: source $env_file"
    
    # Also show instructions for adding to shell profile
    echo ""
    log_info "To persist settings, add to your ~/.bashrc or ~/.zshrc:"
    echo "  source $PROJECT_ROOT/.env.llm"
}

# ============================================================================
# Help
# ============================================================================

show_help() {
    cat << EOF
BioPipelines LLM Setup Script
=============================

Usage: $0 [command] [options]

Commands:
  install-vllm              Install vLLM and dependencies
  start-vllm [model]        Start vLLM server (default: $DEFAULT_VLLM_MODEL)
  stop-vllm                 Stop running vLLM server
  test-openai               Test OpenAI API connectivity
  test-vllm                 Test vLLM server connectivity
  test-all                  Test all configured LLM providers
  env-setup                 Create environment variable template
  help                      Show this help message

Environment Variables:
  OPENAI_API_KEY           OpenAI API key
  HF_TOKEN                 HuggingFace token (for gated models)
  VLLM_BASE_URL            vLLM server URL (default: http://localhost:8000)
  VLLM_PORT                vLLM server port (default: 8000)

Examples:
  # Install vLLM
  $0 install-vllm

  # Start vLLM with default model
  $0 start-vllm

  # Start vLLM with specific model
  $0 start-vllm meta-llama/Llama-3.1-70B-Instruct

  # Test OpenAI
  export OPENAI_API_KEY="sk-..."
  $0 test-openai

  # Test vLLM
  $0 test-vllm

Recommended Models for Bioinformatics:
  General Purpose:
    - meta-llama/Llama-3.1-8B-Instruct (8B params, fast)
    - meta-llama/Llama-3.1-70B-Instruct (70B params, high quality)
    - mistralai/Mistral-7B-Instruct-v0.3 (7B params, balanced)
    - Qwen/Qwen2.5-7B-Instruct (7B params, good multilingual)

  Code Generation:
    - codellama/CodeLlama-34b-Instruct-hf (34B params, code focused)
    - deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (code focused)
    - Qwen/Qwen2.5-Coder-7B-Instruct (7B params, code focused)

EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Create logs directory if needed
    mkdir -p "$PROJECT_ROOT/logs"
    
    case "${1:-help}" in
        install-vllm)
            install_vllm
            ;;
        start-vllm)
            start_vllm "${2:-}" "${3:-}" "${4:-}"
            ;;
        stop-vllm)
            stop_vllm
            ;;
        test-openai)
            test_openai
            ;;
        test-vllm)
            test_vllm
            ;;
        test-all)
            test_all
            ;;
        env-setup)
            env_setup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
