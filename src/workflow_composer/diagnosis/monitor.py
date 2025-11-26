"""
Real-time job monitoring with automatic error diagnosis.

Provides callbacks and hooks that trigger automatic diagnosis
when pipeline jobs fail.
"""

import logging
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class MonitorEvent(Enum):
    """Types of monitoring events."""
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    DIAGNOSIS_STARTED = "diagnosis_started"
    DIAGNOSIS_COMPLETED = "diagnosis_completed"
    FIX_SUGGESTED = "fix_suggested"
    FIX_APPLIED = "fix_applied"


@dataclass
class MonitorCallback:
    """A registered callback for monitoring events."""
    event: MonitorEvent
    callback: Callable
    priority: int = 0  # Higher priority runs first
    name: str = ""


@dataclass
class DiagnosisAlert:
    """An alert from automatic diagnosis."""
    timestamp: datetime
    job_id: str
    workflow_name: str
    category: str
    confidence: float
    root_cause: str
    suggested_fixes: List[str]
    auto_fixed: bool = False
    user_notified: bool = False


class RealTimeMonitor:
    """
    Real-time monitoring with automatic diagnosis.
    
    Monitors pipeline jobs and automatically triggers diagnosis
    when failures are detected.
    
    Features:
    - Callback hooks for job events
    - Automatic diagnosis on failure
    - Alert queue for notifications
    - Fix suggestion tracking
    
    Example:
        monitor = RealTimeMonitor()
        
        # Register callback for failures
        @monitor.on_failure
        def handle_failure(job, diagnosis):
            print(f"Job {job.job_id} failed: {diagnosis.root_cause}")
            
        # Start monitoring
        monitor.start()
    """
    
    def __init__(
        self,
        auto_diagnose: bool = True,
        auto_fix_safe: bool = False,
        poll_interval: float = 10.0,
        max_alerts: int = 100,
    ):
        """
        Initialize the monitor.
        
        Args:
            auto_diagnose: Automatically diagnose failed jobs
            auto_fix_safe: Automatically apply safe fixes
            poll_interval: Seconds between status checks
            max_alerts: Maximum number of alerts to keep
        """
        self.auto_diagnose = auto_diagnose
        self.auto_fix_safe = auto_fix_safe
        self.poll_interval = poll_interval
        
        self._callbacks: Dict[MonitorEvent, List[MonitorCallback]] = {
            event: [] for event in MonitorEvent
        }
        self._alerts: deque = deque(maxlen=max_alerts)
        self._watched_jobs: Dict[str, Any] = {}  # job_id -> job object
        self._processed_failures: set = set()  # Already diagnosed job_ids
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Diagnosis agent (lazy load)
        self._diagnosis_agent = None
        self._auto_fix_engine = None
    
    def _get_diagnosis_agent(self):
        """Lazy load diagnosis agent."""
        if self._diagnosis_agent is None:
            try:
                from .agent import ErrorDiagnosisAgent
                self._diagnosis_agent = ErrorDiagnosisAgent(
                    enable_history=True,
                    history_boost_factor=0.15,
                )
            except Exception as e:
                logger.error(f"Failed to load diagnosis agent: {e}")
        return self._diagnosis_agent
    
    def _get_auto_fix_engine(self):
        """Lazy load auto-fix engine."""
        if self._auto_fix_engine is None:
            try:
                from .auto_fix import AutoFixEngine
                self._auto_fix_engine = AutoFixEngine(
                    dry_run=not self.auto_fix_safe,
                    track_history=True,
                )
            except Exception as e:
                logger.error(f"Failed to load auto-fix engine: {e}")
        return self._auto_fix_engine
    
    # -------------------------------------------------------------------------
    # Callback Registration
    # -------------------------------------------------------------------------
    
    def register_callback(
        self,
        event: MonitorEvent,
        callback: Callable,
        priority: int = 0,
        name: str = "",
    ) -> None:
        """
        Register a callback for a monitoring event.
        
        Args:
            event: Event type to listen for
            callback: Function to call when event occurs
            priority: Higher priority runs first
            name: Optional name for the callback
        """
        cb = MonitorCallback(
            event=event,
            callback=callback,
            priority=priority,
            name=name or callback.__name__,
        )
        with self._lock:
            self._callbacks[event].append(cb)
            # Sort by priority (descending)
            self._callbacks[event].sort(key=lambda x: -x.priority)
        logger.debug(f"Registered callback '{cb.name}' for {event.value}")
    
    def on_failure(self, func: Callable) -> Callable:
        """
        Decorator to register a failure callback.
        
        Usage:
            @monitor.on_failure
            def handle_failure(job, diagnosis):
                print(f"Job failed: {job.job_id}")
        """
        self.register_callback(MonitorEvent.JOB_FAILED, func)
        return func
    
    def on_diagnosis(self, func: Callable) -> Callable:
        """
        Decorator to register a diagnosis completion callback.
        
        Usage:
            @monitor.on_diagnosis
            def handle_diagnosis(job, diagnosis):
                send_notification(diagnosis)
        """
        self.register_callback(MonitorEvent.DIAGNOSIS_COMPLETED, func)
        return func
    
    def on_fix_applied(self, func: Callable) -> Callable:
        """Decorator to register a fix applied callback."""
        self.register_callback(MonitorEvent.FIX_APPLIED, func)
        return func
    
    def _fire_callbacks(self, event: MonitorEvent, *args, **kwargs) -> None:
        """Fire all callbacks for an event."""
        callbacks = self._callbacks.get(event, [])
        for cb in callbacks:
            try:
                cb.callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback '{cb.name}' failed: {e}")
    
    # -------------------------------------------------------------------------
    # Job Watching
    # -------------------------------------------------------------------------
    
    def watch_job(self, job) -> None:
        """
        Add a job to the watch list.
        
        Args:
            job: PipelineJob object to monitor
        """
        with self._lock:
            self._watched_jobs[job.job_id] = job
        logger.info(f"Watching job: {job.job_id}")
        self._fire_callbacks(MonitorEvent.JOB_STARTED, job)
    
    def unwatch_job(self, job_id: str) -> None:
        """Remove a job from the watch list."""
        with self._lock:
            self._watched_jobs.pop(job_id, None)
    
    def get_watched_jobs(self) -> List[Any]:
        """Get list of currently watched jobs."""
        with self._lock:
            return list(self._watched_jobs.values())
    
    # -------------------------------------------------------------------------
    # Automatic Diagnosis
    # -------------------------------------------------------------------------
    
    def diagnose_job(self, job) -> Optional[Any]:
        """
        Run diagnosis on a failed job.
        
        Args:
            job: PipelineJob object
            
        Returns:
            ErrorDiagnosis or None
        """
        agent = self._get_diagnosis_agent()
        if not agent:
            return None
        
        try:
            self._fire_callbacks(MonitorEvent.DIAGNOSIS_STARTED, job)
            
            diagnosis = agent.diagnose_sync(job)
            
            # Create alert
            alert = DiagnosisAlert(
                timestamp=datetime.now(),
                job_id=job.job_id,
                workflow_name=getattr(job, 'workflow_name', 'Unknown'),
                category=diagnosis.category.value,
                confidence=diagnosis.confidence,
                root_cause=diagnosis.root_cause,
                suggested_fixes=[f.description for f in diagnosis.suggested_fixes[:5]],
            )
            self._alerts.append(alert)
            
            self._fire_callbacks(MonitorEvent.DIAGNOSIS_COMPLETED, job, diagnosis)
            self._fire_callbacks(MonitorEvent.FIX_SUGGESTED, job, diagnosis.suggested_fixes)
            
            # Auto-fix if enabled
            if self.auto_fix_safe and diagnosis.has_auto_fixes:
                self._try_auto_fix(job, diagnosis, alert)
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"Diagnosis failed for job {job.job_id}: {e}")
            return None
    
    def _try_auto_fix(self, job, diagnosis, alert: DiagnosisAlert) -> None:
        """Attempt to auto-fix a diagnosed issue."""
        engine = self._get_auto_fix_engine()
        if not engine:
            return
        
        try:
            import asyncio
            
            context = {
                "job_id": job.job_id,
                "workflow_dir": getattr(job, 'workflow_dir', '.'),
                "slurm_job_id": getattr(job, 'slurm_job_id', ''),
            }
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                engine.execute_all_safe(diagnosis, context)
            )
            loop.close()
            
            success = any(r.success for r in results)
            if success:
                alert.auto_fixed = True
                self._fire_callbacks(MonitorEvent.FIX_APPLIED, job, results)
                logger.info(f"Auto-fix applied for job {job.job_id}")
                
        except Exception as e:
            logger.error(f"Auto-fix failed: {e}")
    
    # -------------------------------------------------------------------------
    # Monitoring Loop
    # -------------------------------------------------------------------------
    
    def start(self) -> None:
        """Start the monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="DiagnosisMonitor",
        )
        self._monitor_thread.start()
        logger.info("Real-time monitoring started")
    
    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Real-time monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_jobs()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            time.sleep(self.poll_interval)
    
    def _check_jobs(self) -> None:
        """Check all watched jobs for status changes."""
        with self._lock:
            jobs = list(self._watched_jobs.values())
        
        for job in jobs:
            # Check if job has failed
            job_status = getattr(job, 'status', None)
            
            # Handle different status representations
            if hasattr(job_status, 'value'):
                status_str = job_status.value
            else:
                status_str = str(job_status) if job_status else ""
            
            if status_str.lower() == 'failed' and job.job_id not in self._processed_failures:
                self._processed_failures.add(job.job_id)
                logger.info(f"Detected failure for job: {job.job_id}")
                
                self._fire_callbacks(MonitorEvent.JOB_FAILED, job)
                
                if self.auto_diagnose:
                    self.diagnose_job(job)
            
            elif status_str.lower() == 'completed':
                self._fire_callbacks(MonitorEvent.JOB_COMPLETED, job)
                self.unwatch_job(job.job_id)
            
            elif status_str.lower() == 'cancelled':
                self._fire_callbacks(MonitorEvent.JOB_CANCELLED, job)
                self.unwatch_job(job.job_id)
    
    # -------------------------------------------------------------------------
    # Alerts
    # -------------------------------------------------------------------------
    
    def get_alerts(self, limit: int = 20) -> List[DiagnosisAlert]:
        """Get recent alerts."""
        return list(self._alerts)[-limit:]
    
    def get_unnotified_alerts(self) -> List[DiagnosisAlert]:
        """Get alerts that haven't been notified."""
        return [a for a in self._alerts if not a.user_notified]
    
    def mark_notified(self, alert: DiagnosisAlert) -> None:
        """Mark an alert as notified."""
        alert.user_notified = True
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        alerts = list(self._alerts)
        
        return {
            "total_alerts": len(alerts),
            "auto_fixed_count": sum(1 for a in alerts if a.auto_fixed),
            "watched_jobs": len(self._watched_jobs),
            "processed_failures": len(self._processed_failures),
            "is_running": self._running,
            "categories": self._count_categories(alerts),
        }
    
    def _count_categories(self, alerts: List[DiagnosisAlert]) -> Dict[str, int]:
        """Count alerts by category."""
        counts = {}
        for alert in alerts:
            counts[alert.category] = counts.get(alert.category, 0) + 1
        return counts


# Singleton instance
_monitor: Optional[RealTimeMonitor] = None


def get_realtime_monitor(
    auto_diagnose: bool = True,
    auto_fix_safe: bool = False,
) -> RealTimeMonitor:
    """
    Get or create the real-time monitor singleton.
    
    Args:
        auto_diagnose: Auto-diagnose on failure
        auto_fix_safe: Auto-apply safe fixes
        
    Returns:
        RealTimeMonitor instance
    """
    global _monitor
    
    if _monitor is None:
        _monitor = RealTimeMonitor(
            auto_diagnose=auto_diagnose,
            auto_fix_safe=auto_fix_safe,
        )
    
    return _monitor


def start_monitoring(**kwargs) -> RealTimeMonitor:
    """Start real-time monitoring and return the monitor."""
    monitor = get_realtime_monitor(**kwargs)
    monitor.start()
    return monitor


def stop_monitoring() -> None:
    """Stop real-time monitoring."""
    global _monitor
    if _monitor:
        _monitor.stop()
