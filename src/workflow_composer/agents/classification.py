"""
Unified Task Classification
===========================

Single source of truth for classifying user queries into task types.

This module consolidates the multiple classification implementations
that existed across the codebase:
- unified_agent.py: classify_task() with TASK_KEYWORDS
- autonomous/agent.py: _classify_task() with simple/coding/complex
- router.py: LLM-based routing
- core/query_parser.py: IntentParser

Usage:
    from workflow_composer.agents.classification import classify_task, TaskType
    
    task_type = classify_task("scan /data for FASTQ files")
    # Returns: TaskType.DATA
    
    # For AutonomousAgent compatibility:
    from workflow_composer.agents.classification import classify_simple
    
    simple_type = classify_simple("fix this error")
    # Returns: "coding"
"""

import re
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from functools import lru_cache

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Classification of user queries."""
    WORKFLOW = "workflow"          # Generate/manage workflows
    DIAGNOSIS = "diagnosis"        # Error diagnosis and recovery
    DATA = "data"                  # Data discovery and management
    JOB = "job"                    # Job submission and monitoring
    ANALYSIS = "analysis"          # Result analysis
    EDUCATION = "education"        # Explain concepts
    CODING = "coding"              # Generate code
    SYSTEM = "system"              # System health, vLLM restart
    GENERAL = "general"            # General questions


# Fallback keywords if YAML config not available
_FALLBACK_KEYWORDS = {
    TaskType.WORKFLOW: [
        "workflow", "pipeline", "generate", "create workflow", "run pipeline",
        "nextflow", "snakemake", "rna-seq", "chip-seq", "atac-seq", "wgs",
        "variant calling", "alignment", "rnaseq", "chipseq", "atacseq",
    ],
    TaskType.DIAGNOSIS: [
        "error", "fail", "diagnose", "debug", "fix", "recover",
        "crash", "problem", "issue", "wrong", "broken", "not working",
    ],
    TaskType.DATA: [
        "scan", "find", "search", "data", "download", "dataset",
        "fastq", "bam", "vcf", "reference", "genome", "index",
        "tcga", "geo", "sra", "encode", "files", "samples",
    ],
    TaskType.JOB: [
        "job", "submit", "status", "running", "queue", "slurm",
        "cancel", "resubmit", "watch", "monitor", "pending",
    ],
    TaskType.ANALYSIS: [
        "analyze", "results", "compare", "visualize", "plot",
        "statistics", "quality", "metrics", "report",
    ],
    TaskType.EDUCATION: [
        "explain", "what is", "how does", "help", "tutorial",
        "understand", "learn", "concept", "definition",
    ],
    TaskType.SYSTEM: [
        "system health", "vllm", "restart", "server", "service",
        "gpu", "memory", "disk", "health check",
    ],
    TaskType.CODING: [
        "code", "script", "function", "implement", "write code",
        "python", "bash", "nextflow config", "snakemake rule",
    ],
}


@lru_cache(maxsize=1)
def _load_config() -> Optional[Dict]:
    """
    Load classification config from YAML.
    
    Returns:
        Config dict or None if file not found/invalid.
    """
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not available, using fallback keywords")
        return None
    
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "config" / "task_classification.yaml",
        Path(__file__).parent.parent.parent / "config" / "task_classification.yaml",
        Path.cwd() / "config" / "task_classification.yaml",
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    logger.debug(f"Loaded task classification config from {config_path}")
                    return config
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
    
    logger.debug("No task classification config found, using fallback keywords")
    return None


def _classify_with_config(query_lower: str, config: Dict) -> Tuple[Optional[TaskType], int]:
    """
    Classify query using YAML config.
    
    Returns:
        (TaskType or None, score)
    """
    task_types = config.get("task_types", {})
    if not task_types:
        return None, 0
    
    # Score each task type
    scores: Dict[str, Tuple[int, int]] = {}  # task_name -> (priority, score)
    
    for task_name, task_config in task_types.items():
        score = 0
        priority = task_config.get("priority", 2)
        
        # Keyword matching
        for kw in task_config.get("keywords", []):
            if kw in query_lower:
                score += 1
        
        # Pattern matching (regex)
        for pattern in task_config.get("patterns", []):
            try:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 2  # Patterns worth more
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
        
        if score > 0:
            scores[task_name] = (priority, score)
    
    if not scores:
        return None, 0
    
    # Sort by priority (higher first), then score (higher first)
    best_name, (priority, score) = max(
        scores.items(), 
        key=lambda x: (x[1][0], x[1][1])
    )
    
    # Convert to TaskType
    try:
        return TaskType[best_name.upper()], score
    except KeyError:
        logger.warning(f"Unknown task type in config: {best_name}")
        return None, 0


def _classify_with_fallback(query_lower: str) -> TaskType:
    """
    Classify query using hardcoded fallback keywords.
    
    This is the original logic from unified_agent.py.
    """
    # Priority keywords that override other classifications
    priority_education_patterns = ["explain", "what is", "how does", "understand", "learn"]
    for pattern in priority_education_patterns:
        if pattern in query_lower:
            return TaskType.EDUCATION
    
    # Count keyword matches for each type
    scores = {}
    for task_type, keywords in _FALLBACK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[task_type] = score
    
    if not scores:
        return TaskType.GENERAL
    
    # Return type with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


def classify_task(query: str) -> TaskType:
    """
    Classify a user query into a task type.
    
    Uses config/task_classification.yaml for keywords and patterns.
    Falls back to hardcoded defaults if config not found.
    
    Args:
        query: User's natural language query
        
    Returns:
        TaskType enum value
        
    Example:
        >>> classify_task("scan /data for FASTQ files")
        TaskType.DATA
        >>> classify_task("explain what RNA-seq is")
        TaskType.EDUCATION
    """
    query_lower = query.lower()
    
    # Use hardcoded fallback logic to maintain backward compatibility
    # The config-based approach can be enabled once all edge cases are handled
    return _classify_with_fallback(query_lower)


def classify_simple(query: str, context: Optional[Dict] = None) -> str:
    """
    Classify query into simple categories for AutonomousAgent compatibility.
    
    Returns:
        - "simple": Direct tool execution
        - "coding": Code analysis/modification
        - "complex": Multi-step reasoning needed
        
    This provides backward compatibility with AutonomousAgent._classify_task().
    
    Args:
        query: User's natural language query
        context: Optional context dict with 'has_error', 'traceback' keys
    """
    # Check context first (original AutonomousAgent logic)
    if context:
        if context.get("has_error") or context.get("traceback"):
            return "coding"
    
    # Use unified classification
    task_type = classify_task(query)
    
    # Map to simple/coding/complex
    if task_type == TaskType.DIAGNOSIS:
        return "coding"
    elif task_type in (TaskType.ANALYSIS, TaskType.WORKFLOW):
        return "complex"
    else:
        return "simple"


# Convenience exports
__all__ = [
    "TaskType",
    "classify_task",
    "classify_simple",
]
