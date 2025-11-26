"""
Error Diagnosis Agent for BioPipelines.

This package provides AI-powered error diagnosis capabilities for bioinformatics
workflow failures. It uses a tiered approach:

1. Pattern matching (fast, offline)
2. Historical learning (boost confidence from past diagnoses)
3. LLM analysis (comprehensive, contextual)
4. Auto-fix execution (safe remediation)
5. Real-time monitoring (automatic diagnosis on failure)

Usage:
    from workflow_composer.diagnosis import ErrorDiagnosisAgent
    
    agent = ErrorDiagnosisAgent()
    diagnosis = await agent.diagnose(failed_job)
    
    print(f"Error: {diagnosis.category}")
    print(f"Root Cause: {diagnosis.root_cause}")
    for fix in diagnosis.suggested_fixes:
        print(f"  - {fix.description}")
        
Real-time monitoring:
    from workflow_composer.diagnosis import start_monitoring
    
    monitor = start_monitoring(auto_diagnose=True)
    
    @monitor.on_failure
    def handle_failure(job, diagnosis):
        print(f"Job {job.job_id} failed: {diagnosis.root_cause}")
"""

from .categories import (
    ErrorCategory,
    FixRiskLevel,
    ErrorPattern,
    FixSuggestion,
    ErrorDiagnosis,
)
from .patterns import ERROR_PATTERNS, get_pattern, get_all_patterns
from .log_collector import LogCollector, CollectedLogs
from .agent import ErrorDiagnosisAgent, diagnose_job, diagnose_log
from .auto_fix import AutoFixEngine, FixResult, get_auto_fix_engine
from .gemini_adapter import GeminiAdapter, get_gemini, check_gemini_available
from .lightning_adapter import LightningDiagnosisAdapter, get_lightning_adapter
from .github_agent import GitHubCopilotAgent, get_github_copilot_agent
from .history import DiagnosisHistory, DiagnosisRecord, get_diagnosis_history, record_diagnosis
from .monitor import (
    RealTimeMonitor,
    MonitorEvent,
    DiagnosisAlert,
    get_realtime_monitor,
    start_monitoring,
    stop_monitoring,
)

__all__ = [
    # Categories
    "ErrorCategory",
    "FixRiskLevel",
    "ErrorPattern",
    "FixSuggestion",
    "ErrorDiagnosis",
    # Patterns
    "ERROR_PATTERNS",
    "get_pattern",
    "get_all_patterns",
    # Log Collection
    "LogCollector",
    "CollectedLogs",
    # Main Agent
    "ErrorDiagnosisAgent",
    "diagnose_job",
    "diagnose_log",
    # Auto Fix
    "AutoFixEngine",
    "FixResult",
    "get_auto_fix_engine",
    # LLM Adapters
    "GeminiAdapter",
    "get_gemini",
    "check_gemini_available",
    "LightningDiagnosisAdapter",
    "get_lightning_adapter",
    # GitHub Integration
    "GitHubCopilotAgent",
    "get_github_copilot_agent",
    # History Tracking
    "DiagnosisHistory",
    "DiagnosisRecord",
    "get_diagnosis_history",
    "record_diagnosis",
    # Real-time Monitoring
    "RealTimeMonitor",
    "MonitorEvent",
    "DiagnosisAlert",
    "get_realtime_monitor",
    "start_monitoring",
    "stop_monitoring",
]
