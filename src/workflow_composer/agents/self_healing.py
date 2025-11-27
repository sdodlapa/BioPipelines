"""
Self-Healing System
====================

Automatically detects failed jobs and attempts to fix them using the CodingAgent.

Flow:
1. JobMonitor detects failed job
2. SelfHealer analyzes error logs
3. CodingAgent diagnoses root cause
4. System applies fix (config change, code patch, etc.)
5. Job retried with fix
6. Result logged to AgentMemory for learning

This creates a closed-loop system that learns from failures.
"""

import asyncio
import logging
import subprocess
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class HealingAction(Enum):
    """Type of healing action to take."""
    MODIFY_PARAMS = "modify_params"       # Change tool parameters
    INCREASE_RESOURCES = "increase_resources"  # Request more memory/time
    FIX_INPUT = "fix_input"               # Fix input file issues
    PATCH_CONFIG = "patch_config"         # Modify config file
    SKIP_STEP = "skip_step"               # Skip problematic step
    MANUAL_INTERVENTION = "manual"        # Needs human help
    RETRY_AS_IS = "retry"                 # Transient error, just retry


class HealingStatus(Enum):
    """Status of healing attempt."""
    ANALYZING = "analyzing"
    FIXING = "fixing"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class HealingAttempt:
    """Record of a healing attempt."""
    job_id: str
    timestamp: str
    error_type: str
    action: HealingAction
    fix_description: str
    status: HealingStatus
    retry_job_id: Optional[str] = None
    success: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobInfo:
    """Information about a job."""
    job_id: str
    name: str
    state: str
    exit_code: Optional[int] = None
    error_log: Optional[str] = None
    config_path: Optional[str] = None
    work_dir: Optional[str] = None
    submit_time: Optional[str] = None
    end_time: Optional[str] = None


# =============================================================================
# Self Healer
# =============================================================================

class SelfHealer:
    """
    Automatic job failure recovery system.
    
    Monitors jobs and automatically attempts to fix and retry failed ones.
    Uses CodingAgent for intelligent error diagnosis.
    
    Example:
        healer = SelfHealer(coding_agent=agent, memory=memory)
        result = await healer.heal_job("12345")
    """
    
    # Maximum retries per job
    MAX_RETRIES = 3
    
    # Common transient errors that just need retry
    TRANSIENT_ERRORS = [
        "Connection timed out",
        "Network is unreachable",
        "Temporary failure",
        "Resource temporarily unavailable",
        "Killed",  # OOM killer, try with more memory
    ]
    
    # Errors that need resource increase
    RESOURCE_ERRORS = [
        "out of memory",
        "memory allocation",
        "cannot allocate",
        "oom-kill",
        "exceeded memory",
        "DUE TO TIME LIMIT",
        "walltime limit",
    ]
    
    def __init__(
        self,
        coding_agent: Optional[Any] = None,  # CodingAgent instance
        memory: Optional[Any] = None,  # AgentMemory instance
        log_dir: str = "logs",
        on_heal_complete: Optional[Callable] = None,
    ):
        """
        Initialize self-healer.
        
        Args:
            coding_agent: CodingAgent for error diagnosis
            memory: AgentMemory for learning from fixes
            log_dir: Directory for job logs
            on_heal_complete: Callback when healing completes
        """
        self.coding_agent = coding_agent
        self.memory = memory
        self.log_dir = Path(log_dir)
        self.on_heal_complete = on_heal_complete
        
        # Track healing attempts per job
        self.attempts: Dict[str, List[HealingAttempt]] = {}
    
    async def heal_job(self, job_id: str) -> HealingAttempt:
        """
        Attempt to heal a failed job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            HealingAttempt with result
        """
        logger.info(f"Starting self-healing for job {job_id}")
        
        # Check retry count
        previous_attempts = self.attempts.get(job_id, [])
        if len(previous_attempts) >= self.MAX_RETRIES:
            return HealingAttempt(
                job_id=job_id,
                timestamp=datetime.now().isoformat(),
                error_type="max_retries",
                action=HealingAction.MANUAL_INTERVENTION,
                fix_description=f"Max retries ({self.MAX_RETRIES}) exceeded for job {job_id}",
                status=HealingStatus.FAILED,
            )
        
        # Get job info
        job_info = await self._get_job_info(job_id)
        if not job_info:
            return HealingAttempt(
                job_id=job_id,
                timestamp=datetime.now().isoformat(),
                error_type="job_not_found",
                action=HealingAction.MANUAL_INTERVENTION,
                fix_description=f"Could not find job {job_id}",
                status=HealingStatus.FAILED,
            )
        
        # Read error logs
        error_log = await self._read_error_log(job_id, job_info)
        
        # Analyze error
        attempt = await self._analyze_and_fix(job_id, job_info, error_log)
        
        # Track attempt
        if job_id not in self.attempts:
            self.attempts[job_id] = []
        self.attempts[job_id].append(attempt)
        
        # Store in memory for learning
        if self.memory and attempt.status in [HealingStatus.SUCCEEDED, HealingStatus.FAILED]:
            await self._store_learning(attempt, error_log)
        
        # Callback
        if self.on_heal_complete:
            await self.on_heal_complete(attempt)
        
        return attempt
    
    async def _get_job_info(self, job_id: str) -> Optional[JobInfo]:
        """Get information about a job from SLURM."""
        try:
            # Get job info using sacct
            result = subprocess.run(
                [
                    "sacct", "-j", job_id,
                    "--format=JobID,JobName,State,ExitCode,Start,End,WorkDir",
                    "--parsable2", "--noheader"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning(f"sacct failed: {result.stderr}")
                return None
            
            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                return None
            
            # Parse first line (main job, not steps)
            parts = lines[0].split('|')
            if len(parts) >= 4:
                exit_parts = parts[3].split(':')
                exit_code = int(exit_parts[0]) if exit_parts[0].isdigit() else None
                
                return JobInfo(
                    job_id=job_id,
                    name=parts[1] if len(parts) > 1 else "unknown",
                    state=parts[2] if len(parts) > 2 else "unknown",
                    exit_code=exit_code,
                    submit_time=parts[4] if len(parts) > 4 else None,
                    end_time=parts[5] if len(parts) > 5 else None,
                    work_dir=parts[6] if len(parts) > 6 else None,
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting job info: {e}")
            return None
    
    async def _read_error_log(
        self,
        job_id: str,
        job_info: JobInfo,
    ) -> str:
        """Read error log for a job."""
        # Try common log patterns
        patterns = [
            self.log_dir / f"server_{job_id}.err",
            self.log_dir / f"slurm-{job_id}.out",
            self.log_dir / f"{job_info.name}_{job_id}.err",
            Path(f"slurm-{job_id}.out"),
        ]
        
        for pattern in patterns:
            if pattern.exists():
                try:
                    content = pattern.read_text()
                    # Return last 5000 chars (most relevant)
                    return content[-5000:] if len(content) > 5000 else content
                except Exception as e:
                    logger.warning(f"Error reading {pattern}: {e}")
        
        # If work_dir available, check there
        if job_info.work_dir:
            work_path = Path(job_info.work_dir)
            for log_file in work_path.glob("*.err"):
                try:
                    content = log_file.read_text()
                    return content[-5000:] if len(content) > 5000 else content
                except:
                    continue
        
        return "No error log found"
    
    async def _analyze_and_fix(
        self,
        job_id: str,
        job_info: JobInfo,
        error_log: str,
    ) -> HealingAttempt:
        """Analyze error and determine fix."""
        timestamp = datetime.now().isoformat()
        error_lower = error_log.lower()
        
        # Check for transient errors (just retry)
        for pattern in self.TRANSIENT_ERRORS:
            if pattern.lower() in error_lower:
                return HealingAttempt(
                    job_id=job_id,
                    timestamp=timestamp,
                    error_type="transient",
                    action=HealingAction.RETRY_AS_IS,
                    fix_description=f"Transient error detected ({pattern}), retrying",
                    status=HealingStatus.RETRYING,
                )
        
        # Check for resource errors (increase resources)
        for pattern in self.RESOURCE_ERRORS:
            if pattern.lower() in error_lower:
                action = await self._apply_resource_increase(job_id, job_info, pattern)
                return HealingAttempt(
                    job_id=job_id,
                    timestamp=timestamp,
                    error_type="resource_limit",
                    action=HealingAction.INCREASE_RESOURCES,
                    fix_description=f"Resource limit hit ({pattern}), increasing allocation",
                    status=HealingStatus.RETRYING,
                    details=action,
                )
        
        # Use CodingAgent for complex diagnosis
        if self.coding_agent:
            return await self._diagnose_with_agent(job_id, job_info, error_log, timestamp)
        
        # Fallback: manual intervention
        return HealingAttempt(
            job_id=job_id,
            timestamp=timestamp,
            error_type="unknown",
            action=HealingAction.MANUAL_INTERVENTION,
            fix_description="Could not automatically diagnose error. Manual intervention required.",
            status=HealingStatus.FAILED,
            details={"error_preview": error_log[:500]},
        )
    
    async def _apply_resource_increase(
        self,
        job_id: str,
        job_info: JobInfo,
        pattern: str,
    ) -> Dict[str, Any]:
        """Generate new job with increased resources."""
        # Memory increase for OOM
        if "memory" in pattern.lower() or "oom" in pattern.lower():
            return {
                "change": "memory",
                "old_value": "default",
                "new_value": "doubled",
                "sbatch_arg": "--mem=200G",
            }
        
        # Time increase for timeout
        if "time" in pattern.lower() or "walltime" in pattern.lower():
            return {
                "change": "time",
                "old_value": "default",
                "new_value": "doubled",
                "sbatch_arg": "--time=48:00:00",
            }
        
        return {"change": "general", "note": "Increase resources"}
    
    async def _diagnose_with_agent(
        self,
        job_id: str,
        job_info: JobInfo,
        error_log: str,
        timestamp: str,
    ) -> HealingAttempt:
        """Use CodingAgent for intelligent diagnosis."""
        try:
            # Get diagnosis from CodingAgent
            diagnosis = await self.coding_agent.diagnose_error(
                error_log=error_log,
                context={
                    "job_id": job_id,
                    "job_name": job_info.name,
                    "exit_code": job_info.exit_code,
                }
            )
            
            # Map diagnosis to healing action
            action = self._map_diagnosis_to_action(diagnosis)
            
            # Apply fix if possible
            fix_applied = await self._apply_fix(diagnosis, job_info)
            
            return HealingAttempt(
                job_id=job_id,
                timestamp=timestamp,
                error_type=diagnosis.error_type if hasattr(diagnosis, 'error_type') else "diagnosed",
                action=action,
                fix_description=diagnosis.explanation if hasattr(diagnosis, 'explanation') else str(diagnosis),
                status=HealingStatus.RETRYING if fix_applied else HealingStatus.FAILED,
                details={
                    "diagnosis": str(diagnosis),
                    "fix_applied": fix_applied,
                },
            )
            
        except Exception as e:
            logger.error(f"CodingAgent diagnosis failed: {e}")
            return HealingAttempt(
                job_id=job_id,
                timestamp=timestamp,
                error_type="diagnosis_error",
                action=HealingAction.MANUAL_INTERVENTION,
                fix_description=f"Diagnosis failed: {e}",
                status=HealingStatus.FAILED,
            )
    
    def _map_diagnosis_to_action(self, diagnosis: Any) -> HealingAction:
        """Map CodingAgent diagnosis to healing action."""
        if not diagnosis:
            return HealingAction.MANUAL_INTERVENTION
        
        # Check diagnosis type
        diagnosis_str = str(diagnosis).lower()
        
        if any(x in diagnosis_str for x in ["memory", "resource"]):
            return HealingAction.INCREASE_RESOURCES
        elif any(x in diagnosis_str for x in ["parameter", "argument", "flag"]):
            return HealingAction.MODIFY_PARAMS
        elif any(x in diagnosis_str for x in ["file", "input", "path", "missing"]):
            return HealingAction.FIX_INPUT
        elif any(x in diagnosis_str for x in ["config", "setting"]):
            return HealingAction.PATCH_CONFIG
        elif any(x in diagnosis_str for x in ["skip", "optional"]):
            return HealingAction.SKIP_STEP
        else:
            return HealingAction.MANUAL_INTERVENTION
    
    async def _apply_fix(self, diagnosis: Any, job_info: JobInfo) -> bool:
        """Apply the suggested fix."""
        # Check if diagnosis has code fixes
        if hasattr(diagnosis, 'fixes') and diagnosis.fixes:
            for fix in diagnosis.fixes:
                if hasattr(fix, 'file_path') and hasattr(fix, 'new_content'):
                    try:
                        Path(fix.file_path).write_text(fix.new_content)
                        logger.info(f"Applied fix to {fix.file_path}")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to apply fix: {e}")
        
        # For now, log what would be done
        logger.info(f"Would apply fix: {diagnosis}")
        return False
    
    async def _store_learning(
        self,
        attempt: HealingAttempt,
        error_log: str,
    ):
        """Store healing result in memory for learning."""
        if not self.memory:
            return
        
        # Create memory entry
        memory_content = f"""
Error Type: {attempt.error_type}
Action Taken: {attempt.action.value}
Fix: {attempt.fix_description}
Result: {"SUCCESS" if attempt.success else "FAILED"}
Error Sample: {error_log[:500]}
"""
        
        try:
            await self.memory.add(
                content=memory_content,
                memory_type="error",
                metadata={
                    "job_id": attempt.job_id,
                    "action": attempt.action.value,
                    "success": attempt.success,
                    "timestamp": attempt.timestamp,
                }
            )
            logger.info(f"Stored healing result in memory")
        except Exception as e:
            logger.warning(f"Failed to store in memory: {e}")


# =============================================================================
# Job Monitor (Background Task)
# =============================================================================

class JobMonitor:
    """
    Background monitor that watches for failed jobs.
    
    Periodically checks SLURM queue and triggers SelfHealer for failures.
    """
    
    def __init__(
        self,
        healer: SelfHealer,
        check_interval: int = 60,
        auto_heal: bool = True,
    ):
        """
        Initialize job monitor.
        
        Args:
            healer: SelfHealer instance
            check_interval: Seconds between checks
            auto_heal: Automatically trigger healing
        """
        self.healer = healer
        self.check_interval = check_interval
        self.auto_heal = auto_heal
        
        self._running = False
        self._seen_jobs: set = set()
    
    async def start(self):
        """Start monitoring loop."""
        self._running = True
        logger.info(f"Job monitor started (interval={self.check_interval}s)")
        
        while self._running:
            try:
                await self._check_jobs()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop monitoring loop."""
        self._running = False
        logger.info("Job monitor stopped")
    
    async def _check_jobs(self):
        """Check for failed jobs."""
        try:
            # Get recent failed jobs
            result = subprocess.run(
                [
                    "sacct", "-u", "$USER",
                    "--state=FAILED,TIMEOUT,OUT_OF_MEMORY",
                    "--starttime=now-1hour",
                    "--format=JobID",
                    "--parsable2", "--noheader"
                ],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            
            if result.returncode != 0:
                return
            
            for line in result.stdout.strip().split('\n'):
                job_id = line.strip()
                if not job_id or '.' in job_id:  # Skip job steps
                    continue
                
                if job_id not in self._seen_jobs:
                    self._seen_jobs.add(job_id)
                    logger.info(f"Detected failed job: {job_id}")
                    
                    if self.auto_heal:
                        await self.healer.heal_job(job_id)
                        
        except Exception as e:
            logger.error(f"Error checking jobs: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

_healer_instance: Optional[SelfHealer] = None
_monitor_instance: Optional[JobMonitor] = None


def get_self_healer(
    coding_agent: Optional[Any] = None,
    memory: Optional[Any] = None,
    log_dir: str = "logs",
) -> SelfHealer:
    """Get or create global SelfHealer instance."""
    global _healer_instance
    if _healer_instance is None:
        _healer_instance = SelfHealer(
            coding_agent=coding_agent,
            memory=memory,
            log_dir=log_dir,
        )
    return _healer_instance


async def start_job_monitor(
    healer: Optional[SelfHealer] = None,
    check_interval: int = 60,
) -> JobMonitor:
    """Start the job monitoring system."""
    global _monitor_instance
    
    if healer is None:
        healer = get_self_healer()
    
    _monitor_instance = JobMonitor(healer, check_interval)
    
    # Start in background
    asyncio.create_task(_monitor_instance.start())
    
    return _monitor_instance


def stop_job_monitor():
    """Stop the job monitor."""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop()
