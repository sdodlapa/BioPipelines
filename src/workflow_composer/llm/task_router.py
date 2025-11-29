"""
Task Router
===========

Intelligent routing of tasks to appropriate models based on task type.

The TaskRouter analyzes prompts/tasks and routes them to the best
model based on:
- Task complexity
- Required capabilities (coding, reasoning, analysis)
- Cost constraints
- Latency requirements

Usage:
    from workflow_composer.llm import TaskRouter, TaskType
    
    router = TaskRouter()
    
    # Classify a task
    task_type = router.classify("Generate a Snakemake workflow for RNA-seq")
    # -> TaskType.WORKFLOW_GENERATION
    
    # Get recommended model
    model = router.recommend_model(task_type)
    # -> "local" (uses local for heavy generation)
    
    # Integrated with orchestrator
    orch = get_orchestrator()
    response = await orch.complete("...", task_type=TaskType.CODE_REVIEW)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .providers import ProviderType, ModelCapability

logger = logging.getLogger(__name__)


# =============================================================================
# Task Types
# =============================================================================

class TaskType(Enum):
    """Types of tasks for routing decisions."""
    
    # Workflow-related
    WORKFLOW_GENERATION = "workflow_generation"
    """Generate complete workflows (Snakemake, Nextflow)."""
    
    WORKFLOW_MODIFICATION = "workflow_modification"
    """Modify or extend existing workflows."""
    
    WORKFLOW_VALIDATION = "workflow_validation"
    """Validate workflow correctness."""
    
    # Code-related
    CODE_GENERATION = "code_generation"
    """Generate code snippets or scripts."""
    
    CODE_REVIEW = "code_review"
    """Review and analyze code."""
    
    CODE_DEBUG = "code_debug"
    """Debug and fix code issues."""
    
    CODE_EXPLAIN = "code_explain"
    """Explain code functionality."""
    
    # Analysis-related
    DATA_ANALYSIS = "data_analysis"
    """Analyze data or results."""
    
    ERROR_DIAGNOSIS = "error_diagnosis"
    """Diagnose errors and issues."""
    
    TOOL_RECOMMENDATION = "tool_recommendation"
    """Recommend tools or methods."""
    
    # Information-related
    QUESTION_ANSWER = "question_answer"
    """Answer questions about bioinformatics."""
    
    DOCUMENTATION = "documentation"
    """Generate or improve documentation."""
    
    SUMMARIZATION = "summarization"
    """Summarize content or results."""
    
    # Meta
    GENERAL = "general"
    """General purpose task."""
    
    UNKNOWN = "unknown"
    """Could not classify task."""


class TaskComplexity(Enum):
    """Complexity level of a task."""
    TRIVIAL = "trivial"      # Simple lookup or formatting
    SIMPLE = "simple"        # Basic Q&A or small edits
    MODERATE = "moderate"    # Standard generation or analysis
    COMPLEX = "complex"      # Multi-step reasoning
    EXPERT = "expert"        # Requires deep domain knowledge


# =============================================================================
# Task Analysis Result
# =============================================================================

@dataclass
class TaskAnalysis:
    """Result of task analysis."""
    task_type: TaskType
    complexity: TaskComplexity
    required_capabilities: List[ModelCapability]
    recommended_provider: ProviderType
    recommended_model: Optional[str]
    confidence: float  # 0.0 to 1.0
    
    # Routing hints
    prefer_local: bool = False
    requires_large_context: bool = False
    is_critical: bool = False  # Suggest ensemble
    
    def __str__(self) -> str:
        return f"{self.task_type.value} ({self.complexity.value}, conf={self.confidence:.2f})"


# =============================================================================
# Classification Patterns
# =============================================================================

# Keywords and patterns for task classification
TASK_PATTERNS: Dict[TaskType, List[str]] = {
    TaskType.WORKFLOW_GENERATION: [
        r"generat\w* (?:a )?(?:snakemake|nextflow|workflow|pipeline)",
        r"create (?:a )?(?:snakemake|nextflow|workflow|pipeline)",
        r"write (?:a )?(?:snakemake|nextflow|workflow|pipeline)",
        r"build (?:a )?workflow",
        r"(?:rna-?seq|chip-?seq|atac-?seq|wgs|wes|variant) (?:workflow|pipeline|analysis)",
    ],
    TaskType.WORKFLOW_MODIFICATION: [
        r"modif\w* (?:the )?workflow",
        r"add (?:a )?step to",
        r"extend (?:the )?pipeline",
        r"update (?:the )?workflow",
        r"change (?:the )?(?:workflow|pipeline)",
    ],
    TaskType.WORKFLOW_VALIDATION: [
        r"valid\w* (?:this |the )?(?:.*)?workflow",
        r"valid\w* (?:this |the )?(?:.*)?pipeline",
        r"check (?:this |the )?(?:.*)?(?:workflow|pipeline) (?:for )?(?:correct|error)",
        r"is (?:this |the )?(?:.*)?workflow correct",
        r"review (?:this |the )?(?:.*)?pipeline",
        r"^validate\b",  # "validate" at start
    ],
    TaskType.CODE_GENERATION: [
        r"generat\w* (?:a )?(?:\w+ )?(?:script|code|function|class)",
        r"write (?:a )?(?:\w+ )?(?:script|code|function|class)",
        r"write (?:a )?python\b",  # "write a python script"
        r"create (?:a )?(?:\w+ )?(?:script|function)",
        r"implement\b",
    ],
    TaskType.CODE_REVIEW: [
        r"review (?:this )?code",
        r"check (?:this )?code",
        r"analyze (?:this )?code",
        r"what do you think of (?:this )?code",
    ],
    TaskType.CODE_DEBUG: [
        r"debug",
        r"fix (?:this )?(?:bug|error|issue)",
        r"why (?:is|does) (?:this|my) (?:code|script)",
        r"(?:code|script) (?:not working|failing|broken)",
        r"troubleshoot",
    ],
    TaskType.CODE_EXPLAIN: [
        r"explain (?:this )?(?:code|function|script)",
        r"what does (?:this )?(?:code|function) do",
        r"how does (?:this )?(?:code|function) work",
        r"walk me through",
    ],
    TaskType.DATA_ANALYSIS: [
        r"analyz\w* (?:the )?(?:data|results|output)",
        r"interpret (?:the )?results",
        r"what do (?:these )?results",
        r"statistic",
    ],
    TaskType.ERROR_DIAGNOSIS: [
        r"error\b",  # Match standalone "error"
        r"exception\b",
        r"fail\w*",  # Match fail, failed, failing, failure
        r"diagnos\w*",
        r"what went wrong",
        r"why (?:is|did) (?:it|this|my)\b.+(?:fail|error)",  # "why is my workflow failing"
        r"what does (?:this )?error",  # "what does this error mean"
        r"exit code\s+\d+",  # "exit code 137"
        r"FileNotFoundError|KeyError|ValueError|TypeError|ImportError|AttributeError",
    ],
    TaskType.TOOL_RECOMMENDATION: [
        r"recommend (?:a )?tool",
        r"suggest (?:a )?(?:tool|method|approach)",
        r"which tool (?:should|can) I use",
        r"best (?:tool|method) for",
        r"what (?:tool|aligner|caller) should",
    ],
    TaskType.QUESTION_ANSWER: [
        r"what is",
        r"how do(?:es)? (?:I|you)",
        r"can you (?:tell|explain)",
        r"why (?:do|does|is)",
        r"when should",
    ],
    TaskType.DOCUMENTATION: [
        r"document(?:ation)?",
        r"write (?:a )?(?:readme|docstring|comment)",
        r"add comments",
    ],
    TaskType.SUMMARIZATION: [
        r"summariz\w*",
        r"give me (?:a )?(?:summary|overview)",
        r"tldr",
        r"in short",
        r"briefly",
    ],
}

# Task to capability mapping
TASK_CAPABILITIES: Dict[TaskType, List[ModelCapability]] = {
    TaskType.WORKFLOW_GENERATION: [ModelCapability.CODING],
    TaskType.WORKFLOW_MODIFICATION: [ModelCapability.CODING],
    TaskType.WORKFLOW_VALIDATION: [ModelCapability.CODING, ModelCapability.REASONING],
    TaskType.CODE_GENERATION: [ModelCapability.CODING],
    TaskType.CODE_REVIEW: [ModelCapability.CODING, ModelCapability.REASONING],
    TaskType.CODE_DEBUG: [ModelCapability.CODING, ModelCapability.REASONING],
    TaskType.CODE_EXPLAIN: [ModelCapability.GENERAL],
    TaskType.DATA_ANALYSIS: [ModelCapability.ANALYSIS, ModelCapability.REASONING],
    TaskType.ERROR_DIAGNOSIS: [ModelCapability.CODING, ModelCapability.REASONING],
    TaskType.TOOL_RECOMMENDATION: [ModelCapability.GENERAL],
    TaskType.QUESTION_ANSWER: [ModelCapability.GENERAL],
    TaskType.DOCUMENTATION: [ModelCapability.GENERAL],
    TaskType.SUMMARIZATION: [ModelCapability.FAST],
    TaskType.GENERAL: [ModelCapability.GENERAL],
    TaskType.UNKNOWN: [ModelCapability.GENERAL],
}

# Task to complexity mapping (default)
TASK_COMPLEXITY: Dict[TaskType, TaskComplexity] = {
    TaskType.WORKFLOW_GENERATION: TaskComplexity.COMPLEX,
    TaskType.WORKFLOW_MODIFICATION: TaskComplexity.MODERATE,
    TaskType.WORKFLOW_VALIDATION: TaskComplexity.MODERATE,
    TaskType.CODE_GENERATION: TaskComplexity.MODERATE,
    TaskType.CODE_REVIEW: TaskComplexity.MODERATE,
    TaskType.CODE_DEBUG: TaskComplexity.COMPLEX,
    TaskType.CODE_EXPLAIN: TaskComplexity.SIMPLE,
    TaskType.DATA_ANALYSIS: TaskComplexity.MODERATE,
    TaskType.ERROR_DIAGNOSIS: TaskComplexity.COMPLEX,
    TaskType.TOOL_RECOMMENDATION: TaskComplexity.SIMPLE,
    TaskType.QUESTION_ANSWER: TaskComplexity.SIMPLE,
    TaskType.DOCUMENTATION: TaskComplexity.SIMPLE,
    TaskType.SUMMARIZATION: TaskComplexity.TRIVIAL,
    TaskType.GENERAL: TaskComplexity.SIMPLE,
    TaskType.UNKNOWN: TaskComplexity.SIMPLE,
}


# =============================================================================
# Task Router
# =============================================================================

@dataclass
class RouterConfig:
    """Configuration for task router."""
    prefer_local_for_generation: bool = True
    """Use local models for heavy generation tasks."""
    
    use_cloud_for_critical: bool = True
    """Use cloud models for critical/validation tasks."""
    
    ensemble_for_validation: bool = True
    """Suggest ensemble for validation tasks."""
    
    max_local_complexity: TaskComplexity = TaskComplexity.COMPLEX
    """Maximum complexity to handle locally."""
    
    cost_threshold: float = 0.10
    """Cost threshold above which to warn."""


class TaskRouter:
    """
    Routes tasks to appropriate models based on analysis.
    
    Analyzes prompts to determine:
    - Task type (generation, validation, Q&A, etc.)
    - Complexity level
    - Required capabilities
    - Best provider/model
    
    Example:
        router = TaskRouter()
        
        # Classify a task
        analysis = router.analyze("Generate a Snakemake workflow for ChIP-seq")
        print(f"Type: {analysis.task_type}")
        print(f"Complexity: {analysis.complexity}")
        print(f"Provider: {analysis.recommended_provider}")
        
        # Get routing decision
        provider, model = router.route("Explain RNA-seq analysis")
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        """Initialize router with optional config."""
        self.config = config or RouterConfig()
        
        # Compile patterns for efficiency
        self._compiled_patterns: Dict[TaskType, List[re.Pattern]] = {}
        for task_type, patterns in TASK_PATTERNS.items():
            self._compiled_patterns[task_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def classify(self, prompt: str) -> TaskType:
        """
        Classify a prompt into a task type.
        
        Args:
            prompt: The user prompt or task description
            
        Returns:
            TaskType classification
        """
        prompt_lower = prompt.lower()
        
        # Check each task type's patterns
        best_match: Optional[TaskType] = None
        best_score = 0
        
        for task_type, patterns in self._compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(prompt_lower):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = task_type
        
        if best_match:
            return best_match
        
        # Fallback heuristics
        # Check for validation keywords FIRST (high priority)
        if any(word in prompt_lower for word in ["validate", "verify", "correct"]):
            if any(word in prompt_lower for word in ["workflow", "pipeline"]):
                return TaskType.WORKFLOW_VALIDATION
        if any(word in prompt_lower for word in ["workflow", "pipeline", "snakemake", "nextflow"]):
            return TaskType.WORKFLOW_GENERATION
        if any(word in prompt_lower for word in ["error", "failed", "exception"]):
            return TaskType.ERROR_DIAGNOSIS
        if "?" in prompt:
            return TaskType.QUESTION_ANSWER
        
        return TaskType.GENERAL
    
    def analyze(self, prompt: str) -> TaskAnalysis:
        """
        Perform full analysis of a task.
        
        Args:
            prompt: The user prompt or task description
            
        Returns:
            TaskAnalysis with type, complexity, and recommendations
        """
        task_type = self.classify(prompt)
        complexity = self._estimate_complexity(prompt, task_type)
        capabilities = TASK_CAPABILITIES.get(task_type, [ModelCapability.GENERAL])
        
        # Determine provider preference
        recommended_provider, recommended_model = self._recommend_provider(
            task_type, complexity, prompt
        )
        
        # Calculate confidence based on pattern matches
        confidence = self._calculate_confidence(prompt, task_type)
        
        # Routing hints
        prefer_local = (
            self.config.prefer_local_for_generation and
            task_type in [TaskType.WORKFLOW_GENERATION, TaskType.CODE_GENERATION]
        )
        
        requires_large_context = len(prompt) > 8000 or task_type in [
            TaskType.DATA_ANALYSIS, TaskType.WORKFLOW_VALIDATION
        ]
        
        is_critical = task_type in [
            TaskType.WORKFLOW_VALIDATION,
            TaskType.CODE_REVIEW,
        ] and self.config.ensemble_for_validation
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            required_capabilities=capabilities,
            recommended_provider=recommended_provider,
            recommended_model=recommended_model,
            confidence=confidence,
            prefer_local=prefer_local,
            requires_large_context=requires_large_context,
            is_critical=is_critical,
        )
    
    def route(
        self,
        prompt: str,
        local_available: bool = True,
        cloud_available: bool = True,
    ) -> Tuple[ProviderType, Optional[str]]:
        """
        Get routing decision for a prompt.
        
        Args:
            prompt: The user prompt
            local_available: Is local provider available?
            cloud_available: Is cloud provider available?
            
        Returns:
            Tuple of (ProviderType, Optional model name)
        """
        analysis = self.analyze(prompt)
        
        # Adjust for availability
        if analysis.recommended_provider == ProviderType.LOCAL and not local_available:
            if cloud_available:
                return ProviderType.CLOUD, None
            raise ValueError("No provider available")
        
        if analysis.recommended_provider == ProviderType.CLOUD and not cloud_available:
            if local_available:
                return ProviderType.LOCAL, None
            raise ValueError("No provider available")
        
        return analysis.recommended_provider, analysis.recommended_model
    
    def _estimate_complexity(self, prompt: str, task_type: TaskType) -> TaskComplexity:
        """Estimate task complexity."""
        # Start with default for task type
        base_complexity = TASK_COMPLEXITY.get(task_type, TaskComplexity.SIMPLE)
        
        # Adjust based on prompt characteristics
        prompt_len = len(prompt)
        
        # Long prompts are typically more complex
        if prompt_len > 5000:
            if base_complexity == TaskComplexity.SIMPLE:
                return TaskComplexity.MODERATE
            elif base_complexity == TaskComplexity.MODERATE:
                return TaskComplexity.COMPLEX
        
        # Multiple requirements increase complexity
        requirement_words = ["and", "also", "additionally", "furthermore", "then"]
        requirement_count = sum(1 for word in requirement_words if word in prompt.lower())
        if requirement_count >= 3:
            if base_complexity.value < TaskComplexity.COMPLEX.value:
                return TaskComplexity.COMPLEX
        
        return base_complexity
    
    def _recommend_provider(
        self,
        task_type: TaskType,
        complexity: TaskComplexity,
        prompt: str,
    ) -> Tuple[ProviderType, Optional[str]]:
        """Recommend provider and model for a task."""
        # Heavy generation -> local (free, GPU accelerated)
        if task_type in [TaskType.WORKFLOW_GENERATION, TaskType.CODE_GENERATION]:
            if self.config.prefer_local_for_generation:
                return ProviderType.LOCAL, None
        
        # Critical validation -> cloud (for ensemble capability)
        if task_type == TaskType.WORKFLOW_VALIDATION and self.config.use_cloud_for_critical:
            return ProviderType.CLOUD, "gpt-4o"
        
        # Simple tasks -> local (fast and free)
        if complexity in [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE]:
            return ProviderType.LOCAL, None
        
        # Complex reasoning -> cloud (better models)
        if complexity == TaskComplexity.EXPERT:
            return ProviderType.CLOUD, "claude-3-5-sonnet"
        
        # Default to local
        return ProviderType.LOCAL, None
    
    def _calculate_confidence(self, prompt: str, task_type: TaskType) -> float:
        """Calculate classification confidence."""
        if task_type == TaskType.UNKNOWN:
            return 0.0
        
        # Count pattern matches
        patterns = self._compiled_patterns.get(task_type, [])
        matches = sum(1 for p in patterns if p.search(prompt.lower()))
        
        if matches == 0:
            return 0.3  # Fallback heuristic matched
        elif matches == 1:
            return 0.6
        elif matches == 2:
            return 0.8
        else:
            return 0.95


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TaskType",
    "TaskComplexity",
    "TaskAnalysis",
    "TaskRouter",
    "RouterConfig",
    "TASK_PATTERNS",
    "TASK_CAPABILITIES",
    "TASK_COMPLEXITY",
]
