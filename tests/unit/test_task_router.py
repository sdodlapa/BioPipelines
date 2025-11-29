"""
Tests for Task Router
=====================

Tests the TaskRouter and task classification.
"""

import pytest

from workflow_composer.llm import (
    TaskRouter,
    TaskType,
    TaskComplexity,
    TaskAnalysis,
    RouterConfig,
    ProviderType,
)


# =============================================================================
# TaskType Tests
# =============================================================================

class TestTaskType:
    """Test TaskType enum."""
    
    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.WORKFLOW_GENERATION.value == "workflow_generation"
        assert TaskType.WORKFLOW_VALIDATION.value == "workflow_validation"
        assert TaskType.CODE_GENERATION.value == "code_generation"
        assert TaskType.CODE_DEBUG.value == "code_debug"
        assert TaskType.ERROR_DIAGNOSIS.value == "error_diagnosis"
        assert TaskType.QUESTION_ANSWER.value == "question_answer"
        assert TaskType.GENERAL.value == "general"
    
    def test_all_task_types_have_values(self):
        """Test all task types have string values."""
        for task_type in TaskType:
            assert isinstance(task_type.value, str)
            assert len(task_type.value) > 0


# =============================================================================
# TaskComplexity Tests
# =============================================================================

class TestTaskComplexity:
    """Test TaskComplexity enum."""
    
    def test_complexity_values(self):
        """Test TaskComplexity enum values."""
        assert TaskComplexity.TRIVIAL.value == "trivial"
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MODERATE.value == "moderate"
        assert TaskComplexity.COMPLEX.value == "complex"
        assert TaskComplexity.EXPERT.value == "expert"


# =============================================================================
# RouterConfig Tests
# =============================================================================

class TestRouterConfig:
    """Test RouterConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = RouterConfig()
        assert config.prefer_local_for_generation is True
        assert config.use_cloud_for_critical is True
        assert config.ensemble_for_validation is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RouterConfig(
            prefer_local_for_generation=False,
            cost_threshold=0.50,
        )
        assert config.prefer_local_for_generation is False
        assert config.cost_threshold == 0.50


# =============================================================================
# TaskAnalysis Tests
# =============================================================================

class TestTaskAnalysis:
    """Test TaskAnalysis dataclass."""
    
    def test_analysis_creation(self):
        """Test creating a task analysis."""
        analysis = TaskAnalysis(
            task_type=TaskType.WORKFLOW_GENERATION,
            complexity=TaskComplexity.COMPLEX,
            required_capabilities=[],
            recommended_provider=ProviderType.LOCAL,
            recommended_model=None,
            confidence=0.9,
        )
        assert analysis.task_type == TaskType.WORKFLOW_GENERATION
        assert analysis.complexity == TaskComplexity.COMPLEX
        assert analysis.confidence == 0.9
    
    def test_analysis_str(self):
        """Test analysis string representation."""
        analysis = TaskAnalysis(
            task_type=TaskType.CODE_DEBUG,
            complexity=TaskComplexity.MODERATE,
            required_capabilities=[],
            recommended_provider=ProviderType.LOCAL,
            recommended_model=None,
            confidence=0.75,
        )
        assert "code_debug" in str(analysis)
        assert "moderate" in str(analysis)


# =============================================================================
# TaskRouter Classification Tests
# =============================================================================

class TestTaskRouterClassification:
    """Test TaskRouter classification functionality."""
    
    @pytest.fixture
    def router(self):
        return TaskRouter()
    
    # Workflow generation tests
    def test_classify_workflow_generation_snakemake(self, router):
        """Test classifying Snakemake workflow generation."""
        result = router.classify("Generate a Snakemake workflow for RNA-seq")
        assert result == TaskType.WORKFLOW_GENERATION
    
    def test_classify_workflow_generation_nextflow(self, router):
        """Test classifying Nextflow workflow generation."""
        result = router.classify("Create a Nextflow pipeline for variant calling")
        assert result == TaskType.WORKFLOW_GENERATION
    
    def test_classify_workflow_generation_generic(self, router):
        """Test classifying generic workflow generation."""
        result = router.classify("Build a workflow for ChIP-seq analysis")
        assert result == TaskType.WORKFLOW_GENERATION
    
    # Workflow validation tests
    def test_classify_workflow_validation(self, router):
        """Test classifying workflow validation."""
        result = router.classify("Validate this workflow for correctness")
        assert result == TaskType.WORKFLOW_VALIDATION
    
    def test_classify_workflow_validation_chipseq(self, router):
        """Test classifying ChIP-seq workflow validation."""
        result = router.classify("Validate this ChIP-seq workflow for correctness")
        assert result == TaskType.WORKFLOW_VALIDATION
    
    # Code generation tests
    def test_classify_code_generation(self, router):
        """Test classifying code generation."""
        result = router.classify("Generate a Python script to parse VCF files")
        assert result == TaskType.CODE_GENERATION
    
    def test_classify_code_generation_function(self, router):
        """Test classifying function generation."""
        result = router.classify("Write a function to calculate GC content")
        assert result == TaskType.CODE_GENERATION
    
    # Code debug tests
    def test_classify_code_debug(self, router):
        """Test classifying code debugging."""
        result = router.classify("Debug this Python script that crashes")
        assert result == TaskType.CODE_DEBUG
    
    def test_classify_code_debug_fix(self, router):
        """Test classifying bug fix."""
        result = router.classify("Fix this bug in my alignment script")
        assert result == TaskType.CODE_DEBUG
    
    # Error diagnosis tests
    def test_classify_error_diagnosis(self, router):
        """Test classifying error diagnosis."""
        result = router.classify("Why is my workflow failing with exit code 137?")
        assert result == TaskType.ERROR_DIAGNOSIS
    
    def test_classify_error_diagnosis_exception(self, router):
        """Test classifying exception diagnosis."""
        result = router.classify("Error: FileNotFoundError when running BWA")
        assert result == TaskType.ERROR_DIAGNOSIS
    
    # Question answer tests
    def test_classify_question_answer(self, router):
        """Test classifying question answer."""
        result = router.classify("What is the difference between BWA and Bowtie2?")
        assert result == TaskType.QUESTION_ANSWER
    
    def test_classify_question_answer_how(self, router):
        """Test classifying how-to question."""
        result = router.classify("How do I install STAR aligner?")
        assert result == TaskType.QUESTION_ANSWER
    
    # Summarization tests
    def test_classify_summarization(self, router):
        """Test classifying summarization."""
        result = router.classify("Summarize the QC results from FastQC")
        assert result == TaskType.SUMMARIZATION
    
    def test_classify_summarization_tldr(self, router):
        """Test classifying TLDR request."""
        result = router.classify("Give me a TLDR of this paper")
        assert result == TaskType.SUMMARIZATION


# =============================================================================
# TaskRouter Analysis Tests
# =============================================================================

class TestTaskRouterAnalysis:
    """Test TaskRouter full analysis functionality."""
    
    @pytest.fixture
    def router(self):
        return TaskRouter()
    
    def test_analyze_returns_task_analysis(self, router):
        """Test analyze returns TaskAnalysis."""
        analysis = router.analyze("Generate a workflow")
        assert isinstance(analysis, TaskAnalysis)
    
    def test_analyze_workflow_generation(self, router):
        """Test analyzing workflow generation task."""
        analysis = router.analyze("Generate a Snakemake workflow for RNA-seq")
        assert analysis.task_type == TaskType.WORKFLOW_GENERATION
        assert analysis.complexity == TaskComplexity.COMPLEX
        assert analysis.prefer_local is True
    
    def test_analyze_workflow_validation(self, router):
        """Test analyzing workflow validation task."""
        analysis = router.analyze("Validate this workflow for correctness")
        assert analysis.task_type == TaskType.WORKFLOW_VALIDATION
        assert analysis.is_critical is True
        assert analysis.recommended_provider == ProviderType.CLOUD
    
    def test_analyze_simple_question(self, router):
        """Test analyzing simple question."""
        analysis = router.analyze("What is RNA-seq?")
        assert analysis.task_type == TaskType.QUESTION_ANSWER
        assert analysis.complexity == TaskComplexity.SIMPLE
    
    def test_analyze_includes_confidence(self, router):
        """Test analysis includes confidence score."""
        analysis = router.analyze("Generate a Snakemake workflow")
        assert 0.0 <= analysis.confidence <= 1.0


# =============================================================================
# TaskRouter Routing Tests
# =============================================================================

class TestTaskRouterRouting:
    """Test TaskRouter routing decisions."""
    
    @pytest.fixture
    def router(self):
        return TaskRouter()
    
    def test_route_returns_tuple(self, router):
        """Test route returns provider tuple."""
        provider, model = router.route("Generate a workflow")
        assert isinstance(provider, ProviderType)
    
    def test_route_generation_to_local(self, router):
        """Test generation tasks route to local."""
        provider, _ = router.route("Generate a Snakemake workflow")
        assert provider == ProviderType.LOCAL
    
    def test_route_validation_to_cloud(self, router):
        """Test validation tasks route to cloud."""
        provider, model = router.route("Validate this workflow")
        assert provider == ProviderType.CLOUD
    
    def test_route_respects_availability(self, router):
        """Test routing respects provider availability."""
        # Force cloud only
        provider, _ = router.route(
            "Generate workflow",
            local_available=False,
            cloud_available=True,
        )
        assert provider == ProviderType.CLOUD
    
    def test_route_raises_when_none_available(self, router):
        """Test routing raises when no provider available."""
        with pytest.raises(ValueError):
            router.route(
                "Generate workflow",
                local_available=False,
                cloud_available=False,
            )
