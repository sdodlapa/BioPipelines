#!/usr/bin/env python3
"""
BioPipelines vLLM Health Monitor

Monitors health of vLLM servers and optionally restarts unhealthy ones.
Designed to run as a cron job or systemd timer.

Usage:
    python monitor_vllm_health.py              # Check and report
    python monitor_vllm_health.py --restart    # Auto-restart unhealthy
    python monitor_vllm_health.py --json       # JSON output for scripts

Cron example (every 5 minutes):
    */5 * * * * /path/to/monitor_vllm_health.py --restart >> /var/log/vllm_health.log 2>&1
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    sys.exit(1)


# Configuration
PROJECT_DIR = Path(__file__).parent.parent.parent
CONNECTION_DIR = PROJECT_DIR / ".model_connections"
LOG_DIR = PROJECT_DIR / "logs" / "health"
DEPLOY_SCRIPT = PROJECT_DIR / "scripts" / "llm" / "deploy_core_models.sh"

# Core models to monitor
CORE_MODELS = {
    "generalist": {"port": 8001, "description": "General purpose"},
    "coder": {"port": 8002, "description": "Code generation"},
    "math": {"port": 8003, "description": "Math/statistics"},
    "embeddings": {"port": 8004, "description": "Embeddings"},
}

HEALTH_TIMEOUT = 5  # seconds
INFERENCE_TIMEOUT = 30  # seconds


@dataclass
class HealthStatus:
    """Health status for a single model."""
    model_key: str
    healthy: bool
    endpoint: Optional[str]
    response_time_ms: Optional[float]
    error: Optional[str]
    slurm_job_id: Optional[str]
    slurm_state: Optional[str]
    checked_at: str


@dataclass
class HealthReport:
    """Aggregated health report."""
    timestamp: str
    all_healthy: bool
    models: List[HealthStatus]
    summary: Dict[str, int]


def load_connection_info(model_key: str) -> Optional[Dict[str, str]]:
    """Load connection info from .env file."""
    env_file = CONNECTION_DIR / f"{model_key}.env"
    if not env_file.exists():
        return None
    
    info = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                info[key] = value
    return info


def get_slurm_job_state(job_id: str) -> Optional[str]:
    """Get SLURM job state."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def check_health_endpoint(url: str) -> tuple[bool, float, Optional[str]]:
    """Check /health endpoint."""
    try:
        import time
        start = time.time()
        response = requests.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
        elapsed_ms = (time.time() - start) * 1000
        
        if response.status_code == 200:
            return True, elapsed_ms, None
        else:
            return False, elapsed_ms, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectTimeout:
        return False, 0, "Connection timeout"
    except requests.exceptions.ConnectionError as e:
        return False, 0, f"Connection error: {str(e)[:50]}"
    except Exception as e:
        return False, 0, str(e)[:100]


def check_model_health(model_key: str) -> HealthStatus:
    """Check health of a single model."""
    now = datetime.now().isoformat()
    
    # Load connection info
    conn_info = load_connection_info(model_key)
    if not conn_info:
        return HealthStatus(
            model_key=model_key,
            healthy=False,
            endpoint=None,
            response_time_ms=None,
            error="No connection info - not deployed",
            slurm_job_id=None,
            slurm_state=None,
            checked_at=now,
        )
    
    url = conn_info.get("URL", "")
    job_id = conn_info.get("SLURM_JOB_ID", "")
    
    # Check SLURM state
    slurm_state = get_slurm_job_state(job_id) if job_id else None
    
    if slurm_state and slurm_state != "RUNNING":
        return HealthStatus(
            model_key=model_key,
            healthy=False,
            endpoint=url,
            response_time_ms=None,
            error=f"SLURM job {slurm_state}",
            slurm_job_id=job_id,
            slurm_state=slurm_state,
            checked_at=now,
        )
    
    # Check health endpoint
    healthy, response_ms, error = check_health_endpoint(url)
    
    return HealthStatus(
        model_key=model_key,
        healthy=healthy,
        endpoint=url,
        response_time_ms=response_ms if healthy else None,
        error=error,
        slurm_job_id=job_id,
        slurm_state=slurm_state,
        checked_at=now,
    )


def restart_model(model_key: str) -> bool:
    """Restart a model using deploy script."""
    if not DEPLOY_SCRIPT.exists():
        print(f"  ERROR: Deploy script not found: {DEPLOY_SCRIPT}")
        return False
    
    try:
        # Stop
        subprocess.run(
            [str(DEPLOY_SCRIPT), "stop", model_key],
            capture_output=True, timeout=30
        )
        
        # Start
        result = subprocess.run(
            [str(DEPLOY_SCRIPT), "start", model_key],
            capture_output=True, text=True, timeout=30
        )
        
        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR restarting {model_key}: {e}")
        return False


def generate_report(statuses: List[HealthStatus]) -> HealthReport:
    """Generate aggregated health report."""
    healthy_count = sum(1 for s in statuses if s.healthy)
    unhealthy_count = len(statuses) - healthy_count
    
    return HealthReport(
        timestamp=datetime.now().isoformat(),
        all_healthy=(unhealthy_count == 0),
        models=statuses,
        summary={
            "total": len(statuses),
            "healthy": healthy_count,
            "unhealthy": unhealthy_count,
        }
    )


def print_report(report: HealthReport, json_output: bool = False):
    """Print health report."""
    if json_output:
        print(json.dumps({
            "timestamp": report.timestamp,
            "all_healthy": report.all_healthy,
            "summary": report.summary,
            "models": [asdict(m) for m in report.models],
        }, indent=2))
        return
    
    print(f"\n{'='*60}")
    print(f"vLLM Health Report - {report.timestamp}")
    print(f"{'='*60}\n")
    
    for status in report.models:
        icon = "✓" if status.healthy else "✗"
        color_start = "\033[92m" if status.healthy else "\033[91m"
        color_end = "\033[0m"
        
        print(f"{color_start}{icon}{color_end} {status.model_key:12s}", end="")
        
        if status.healthy:
            print(f" HEALTHY ({status.response_time_ms:.0f}ms)")
            if status.endpoint:
                print(f"    └─ {status.endpoint}")
        else:
            print(f" UNHEALTHY - {status.error}")
            if status.slurm_job_id:
                print(f"    └─ Job {status.slurm_job_id} ({status.slurm_state or 'unknown'})")
    
    print(f"\n{'─'*60}")
    print(f"Summary: {report.summary['healthy']}/{report.summary['total']} healthy")
    
    if not report.all_healthy:
        print("\n⚠️  Some models are unhealthy. Run with --restart to auto-restart.")


def main():
    parser = argparse.ArgumentParser(description="Monitor vLLM server health")
    parser.add_argument("--restart", action="store_true", help="Auto-restart unhealthy models")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--model", type=str, help="Check specific model only")
    args = parser.parse_args()
    
    # Determine which models to check
    if args.model:
        if args.model not in CORE_MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(CORE_MODELS.keys())}")
            sys.exit(1)
        models_to_check = [args.model]
    else:
        models_to_check = list(CORE_MODELS.keys())
    
    # Check health
    statuses = [check_model_health(m) for m in models_to_check]
    report = generate_report(statuses)
    
    # Print report
    print_report(report, json_output=args.json)
    
    # Restart if requested
    if args.restart:
        unhealthy = [s for s in statuses if not s.healthy]
        if unhealthy:
            print(f"\nRestarting {len(unhealthy)} unhealthy model(s)...")
            for status in unhealthy:
                print(f"  Restarting {status.model_key}...", end=" ")
                if restart_model(status.model_key):
                    print("submitted")
                else:
                    print("FAILED")
    
    # Exit code
    sys.exit(0 if report.all_healthy else 1)


if __name__ == "__main__":
    main()
