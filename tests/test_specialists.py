"""
Tests for Phase 2.4: Multi-Agent Coordination
=============================================

Tests for:
- PlannerAgent workflow design
- CodeGenAgent Nextflow generation
- ValidatorAgent code validation
- DocAgent documentation generation
- QCAgent quality control
- SupervisorAgent coordination
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.workflow_composer.agents.specialists import (
    PlannerAgent,
    CodeGenAgent,
    ValidatorAgent,
    DocAgent,
    QCAgent,
    SupervisorAgent,
    WorkflowPlan,
    WorkflowStep,
    WorkflowResult,
    WorkflowState,
    ValidationResult,
)
from src.workflow_composer.agents.specialists.qc import QCMetric, QCReport


class TestPlannerAgent:
    """Tests for PlannerAgent workflow design."""
    
    def test_create_plan_sync_rnaseq(self):
        """Test synchronous RNA-seq plan creation."""
        planner = PlannerAgent()
        plan = planner.create_plan_sync("RNA-seq analysis for human samples")
        
        assert plan.name == "rnaseq_analysis"
        assert plan.analysis_type == "rna-seq"
        assert plan.organism == "human"
        assert plan.genome_build == "GRCh38"
        assert len(plan.steps) > 0
        
        # Check required steps are present
        step_names = [s.name for s in plan.steps]
        assert "fastqc" in step_names
        assert "alignment" in step_names
        assert "quantification" in step_names
    
    def test_create_plan_sync_chipseq(self):
        """Test ChIP-seq plan creation."""
        planner = PlannerAgent()
        plan = planner.create_plan_sync("ChIP-seq peak calling pipeline")
        
        assert plan.analysis_type == "chip-seq"
        assert any("macs2" in s.tool.lower() for s in plan.steps)
    
    def test_create_plan_sync_dnaseq(self):
        """Test DNA-seq variant calling plan."""
        planner = PlannerAgent()
        plan = planner.create_plan_sync("WGS variant calling for human DNA")
        
        assert plan.analysis_type == "dna-seq"
        assert any("gatk" in s.tool.lower() for s in plan.steps)
    
    def test_workflow_plan_serialization(self):
        """Test WorkflowPlan JSON serialization."""
        plan = WorkflowPlan(
            name="test_workflow",
            description="Test workflow",
            analysis_type="rna-seq",
            steps=[
                WorkflowStep(
                    name="step1",
                    tool="Tool1",
                    description="Step 1 description",
                    inputs=["input1"],
                    outputs=["output1"],
                )
            ],
        )
        
        # Serialize to JSON
        json_str = plan.to_json()
        data = json.loads(json_str)
        
        assert data["name"] == "test_workflow"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["tool"] == "Tool1"
        
        # Deserialize back
        plan2 = WorkflowPlan.from_json(json_str)
        assert plan2.name == plan.name
        assert len(plan2.steps) == len(plan.steps)
    
    def test_mouse_organism_detection(self):
        """Test organism detection from query."""
        planner = PlannerAgent()
        plan = planner.create_plan_sync("RNA-seq for mouse liver samples")
        
        assert plan.organism == "mouse"
        assert plan.genome_build == "GRCm39"


class TestCodeGenAgent:
    """Tests for CodeGenAgent Nextflow generation."""
    
    def test_generate_sync_basic(self):
        """Test basic code generation."""
        codegen = CodeGenAgent()
        plan = WorkflowPlan(
            name="test_rnaseq",
            description="Test RNA-seq pipeline",
            analysis_type="rna-seq",
            organism="human",
            genome_build="GRCh38",
            steps=[
                WorkflowStep(
                    name="fastqc",
                    tool="FastQC",
                    description="QC check",
                    inputs=["reads"],
                    outputs=["fastqc_reports"],
                    resources={"memory": "4 GB", "cpus": 2},
                ),
                WorkflowStep(
                    name="alignment",
                    tool="STAR",
                    description="Alignment",
                    inputs=["reads"],
                    outputs=["aligned_bam"],
                    resources={"memory": "32 GB", "cpus": 8},
                    dependencies=["fastqc"],
                ),
            ],
        )
        
        code = codegen.generate_sync(plan)
        
        # Check DSL2 header
        assert "nextflow.enable.dsl = 2" in code
        
        # Check processes exist
        assert "process FASTQC {" in code
        assert "process ALIGNMENT {" in code
        
        # Check workflow block
        assert "workflow {" in code
        
        # Check containers
        assert "container" in code
    
    def test_generate_config(self):
        """Test configuration file generation."""
        codegen = CodeGenAgent()
        plan = WorkflowPlan(
            name="test_pipeline",
            description="Test",
            analysis_type="rna-seq",
            recommended_cpus=8,
            recommended_memory_gb=32,
        )
        
        config = codegen.generate_config(plan)
        
        assert "process {" in config
        assert "cpus = 8" in config
        assert "memory = '32 GB'" in config
        assert "profiles {" in config
        assert "slurm" in config
        assert "docker" in config
        assert "singularity" in config
    
    def test_tool_specific_scripts(self):
        """Test tool-specific script generation."""
        codegen = CodeGenAgent()
        plan = WorkflowPlan(
            name="test",
            description="Test",
            analysis_type="rna-seq",
            steps=[
                WorkflowStep(
                    name="fastp",
                    tool="fastp",
                    description="Trimming",
                    inputs=["reads"],
                    outputs=["trimmed_reads"],
                ),
            ],
        )
        
        code = codegen.generate_sync(plan)
        
        # Check fastp-specific commands
        assert "fastp" in code
        assert ".trimmed.fq.gz" in code


class TestValidatorAgent:
    """Tests for ValidatorAgent code validation."""
    
    def test_validate_valid_code(self):
        """Test validation of valid Nextflow code."""
        validator = ValidatorAgent()
        
        valid_code = """
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

process FASTQC {
    container 'quay.io/biocontainers/fastqc:0.12.1'
    errorStrategy 'retry'
    
    input:
    tuple val(meta), path(reads)
    
    output:
    path('*.html'), emit: html
    path('versions.yml'), emit: versions
    
    script:
    \"\"\"
    fastqc $reads
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastqc: "0.12.1"
    END_VERSIONS
    \"\"\"
}

workflow {
    reads_ch = Channel.of([id: 'sample1'], file('test.fq.gz'))
    FASTQC(reads_ch)
}
"""
        result = validator.validate_sync(valid_code)
        
        # Should be valid (no errors, only info/warnings allowed)
        assert result.valid is True
    
    def test_validate_missing_dsl2(self):
        """Test detection of missing DSL2 declaration."""
        validator = ValidatorAgent()
        
        code = """
process FASTQC {
    input:
    path(reads)
}

workflow {
}
"""
        result = validator.validate_sync(code)
        
        assert result.valid is False
        assert any("DSL2" in issue for issue in result.issues)
    
    def test_validate_missing_workflow(self):
        """Test detection of missing workflow block."""
        validator = ValidatorAgent()
        
        code = """
nextflow.enable.dsl = 2

process FASTQC {
    input:
    path(reads)
    
    output:
    path('*.html')
    
    script:
    \"\"\"
    echo test
    \"\"\"
}
"""
        result = validator.validate_sync(code)
        
        assert result.valid is False
        assert any("Workflow" in issue or "workflow" in issue for issue in result.issues)
    
    def test_deprecated_pattern_detection(self):
        """Test detection of deprecated patterns."""
        validator = ValidatorAgent()
        
        code = """
nextflow.enable.dsl = 2

process TEST {
    input:
    file(reads)
    
    output:
    path('*')
    
    script:
    \"\"\"
    echo test
    \"\"\"
}

workflow {
    Channel.from(['a', 'b']).set { ch }
    TEST(ch)
}
"""
        result = validator.validate_sync(code)
        
        # Should have warnings about deprecated patterns
        assert len(result.warnings) > 0 or any("deprecated" in issue.lower() for issue in result.issues)


class TestDocAgent:
    """Tests for DocAgent documentation generation."""
    
    def test_generate_readme_sync(self):
        """Test README generation."""
        docs = DocAgent()
        plan = WorkflowPlan(
            name="rnaseq_analysis",
            description="RNA-seq analysis pipeline",
            analysis_type="rna-seq",
            organism="human",
            genome_build="GRCh38",
            steps=[
                WorkflowStep(
                    name="fastqc",
                    tool="FastQC",
                    description="Quality control",
                    inputs=["reads"],
                    outputs=["qc_reports"],
                ),
                WorkflowStep(
                    name="alignment",
                    tool="STAR",
                    description="Read alignment",
                    inputs=["reads"],
                    outputs=["bam"],
                ),
            ],
            recommended_cpus=8,
            recommended_memory_gb=32,
        )
        
        readme = docs.generate_readme_sync(plan)
        
        # Check essential sections
        assert "# rnaseq_analysis" in readme
        assert "Quick Start" in readme
        assert "--input" in readme
        assert "Requirements" in readme
        assert "Pipeline Steps" in readme
        assert "Parameters" in readme
    
    def test_generate_dag(self):
        """Test DAG diagram generation."""
        docs = DocAgent()
        plan = WorkflowPlan(
            name="test",
            description="Test",
            analysis_type="rna-seq",
            steps=[
                WorkflowStep(name="step1", tool="Tool1", description="Step 1"),
                WorkflowStep(name="step2", tool="Tool2", description="Step 2"),
            ],
        )
        
        dag = docs.generate_dag(plan)
        
        assert "```mermaid" in dag
        assert "graph TD" in dag
        assert "STEP1" in dag
        assert "STEP2" in dag
        assert "INPUT" in dag
        assert "OUTPUT" in dag
    
    def test_generate_parameters_doc(self):
        """Test parameter documentation generation."""
        docs = DocAgent()
        plan = WorkflowPlan(
            name="test",
            description="Test",
            analysis_type="rna-seq",
            organism="human",
            genome_build="GRCh38",
            steps=[],
            recommended_cpus=8,
            recommended_memory_gb=32,
        )
        
        params_doc = docs.generate_parameters_doc(plan)
        
        assert "# Pipeline Parameters" in params_doc
        assert "--input" in params_doc
        assert "--genome" in params_doc
        assert "GRCh38" in params_doc


class TestQCAgent:
    """Tests for QCAgent quality control."""
    
    def test_default_thresholds(self):
        """Test default QC thresholds by analysis type."""
        qc = QCAgent(analysis_type="rna-seq")
        
        assert "mapping_rate" in qc.thresholds
        assert qc.thresholds["mapping_rate"]["min"] == 70.0
        
        qc_wgs = QCAgent(analysis_type="wgs")
        assert qc_wgs.thresholds["mean_coverage"]["min"] == 30.0
    
    def test_qc_metric_evaluation(self):
        """Test QC metric threshold evaluation."""
        metric = QCMetric(
            name="mapping_rate",
            value=85.0,
            unit="%",
            threshold_min=70.0,
            threshold_max=100.0,
        )
        
        status = metric.evaluate()
        assert status == "pass"
        assert metric.status == "pass"
        
        # Test failing metric
        metric_fail = QCMetric(
            name="mapping_rate",
            value=50.0,
            unit="%",
            threshold_min=70.0,
        )
        status = metric_fail.evaluate()
        assert status in ["fail", "warn"]
    
    def test_qc_report_generation(self):
        """Test QC report creation."""
        report = QCReport(sample_id="sample1")
        
        report.add_metric(QCMetric(
            name="total_reads",
            value=20_000_000,
            threshold_min=10_000_000,
        ))
        
        report.add_metric(QCMetric(
            name="mapping_rate",
            value=85.0,
            unit="%",
            threshold_min=70.0,
        ))
        
        assert len(report.metrics) == 2
        assert report.passed is True
        assert len(report.errors) == 0
    
    def test_qc_summary_generation(self):
        """Test QC summary markdown generation."""
        qc = QCAgent(analysis_type="rna-seq")
        
        reports = [
            QCReport(sample_id="sample1"),
            QCReport(sample_id="sample2"),
        ]
        
        reports[0].add_metric(QCMetric("total_reads", 15_000_000, threshold_min=10_000_000))
        reports[1].add_metric(QCMetric("total_reads", 20_000_000, threshold_min=10_000_000))
        
        summary = qc.generate_qc_summary(reports)
        
        assert "# Quality Control Summary" in summary
        assert "sample1" in summary
        assert "sample2" in summary
        # Check for "Samples Analyzed" with flexible formatting
        assert "Samples Analyzed" in summary
        assert "2" in summary


class TestSupervisorAgent:
    """Tests for SupervisorAgent coordination."""
    
    def test_execute_sync_rnaseq(self):
        """Test synchronous workflow generation."""
        supervisor = SupervisorAgent()
        
        result = supervisor.execute_sync("RNA-seq analysis for human samples")
        
        assert result.success is True
        assert result.plan is not None
        assert result.plan.analysis_type == "rna-seq"
        assert result.code is not None
        assert "nextflow.enable.dsl = 2" in result.code
        assert result.config is not None
        assert result.documentation is not None
    
    def test_execute_sync_chipseq(self):
        """Test ChIP-seq workflow generation."""
        supervisor = SupervisorAgent()
        
        result = supervisor.execute_sync("ChIP-seq peak calling for H3K27ac")
        
        assert result.success is True
        assert result.plan.analysis_type == "chip-seq"
        assert "macs2" in result.code.lower() or "peak" in result.code.lower()
    
    def test_get_state(self):
        """Test state tracking."""
        supervisor = SupervisorAgent()
        
        # Before execution
        state = supervisor.get_state()
        assert state["state"] == "idle"
        
        # After execution
        supervisor.execute_sync("RNA-seq analysis")
        state = supervisor.get_state()
        assert state["has_plan"] is True
        assert state["has_code"] is True
    
    def test_write_outputs(self, tmp_path):
        """Test writing outputs to disk."""
        supervisor = SupervisorAgent()
        
        result = supervisor.execute_sync(
            "RNA-seq analysis",
            output_dir=str(tmp_path / "workflow_output")
        )
        
        assert result.success is True
        assert "main.nf" in result.output_files
        assert "nextflow.config" in result.output_files
        assert "README.md" in result.output_files
        
        # Check files exist
        assert (tmp_path / "workflow_output" / "main.nf").exists()
        assert (tmp_path / "workflow_output" / "nextflow.config").exists()
        assert (tmp_path / "workflow_output" / "README.md").exists()


@pytest.mark.asyncio
class TestAsyncOperations:
    """Tests for async agent operations."""
    
    async def test_planner_async(self):
        """Test async plan creation with mock router."""
        mock_router = AsyncMock()
        mock_router.route_async.return_value = json.dumps({
            "name": "test_workflow",
            "description": "Test",
            "analysis_type": "rna-seq",
            "organism": "human",
            "genome_build": "GRCh38",
            "input_type": "fastq",
            "read_type": "paired",
            "steps": [
                {
                    "name": "fastqc",
                    "tool": "FastQC",
                    "description": "QC",
                    "inputs": ["reads"],
                    "outputs": ["qc"],
                    "parameters": {},
                    "resources": {},
                    "dependencies": [],
                }
            ],
            "qc_checkpoints": ["final"],
            "outputs": [],
            "estimated_runtime_hours": 1.0,
            "recommended_memory_gb": 16,
            "recommended_cpus": 8,
        })
        
        planner = PlannerAgent(router=mock_router)
        plan = await planner.create_plan("RNA-seq analysis")
        
        assert plan.name == "test_workflow"
        assert plan.analysis_type == "rna-seq"
        mock_router.route_async.assert_called_once()
    
    async def test_supervisor_streaming(self):
        """Test streaming workflow generation."""
        supervisor = SupervisorAgent()
        
        updates = []
        async for update in supervisor.execute_streaming("RNA-seq analysis"):
            updates.append(update)
        
        # Check all phases were reported
        phases = [u.get("phase") for u in updates]
        assert "planning" in phases
        assert "codegen" in phases
        assert "validation" in phases
        assert "documentation" in phases
        assert "complete" in phases


class TestIntegration:
    """Integration tests for multi-agent coordination."""
    
    def test_end_to_end_rnaseq_workflow(self, tmp_path):
        """Test complete RNA-seq workflow generation."""
        supervisor = SupervisorAgent()
        output_dir = tmp_path / "rnaseq_workflow"
        
        result = supervisor.execute_sync(
            "Differential expression RNA-seq analysis for human cancer samples "
            "with paired-end reads",
            output_dir=str(output_dir)
        )
        
        # Verify success
        assert result.success is True
        
        # Verify plan
        assert result.plan.organism == "human"
        assert "differential" in result.plan.name.lower() or any(
            "deseq" in s.tool.lower() or "de" in s.name.lower()
            for s in result.plan.steps
        )
        
        # Verify code quality
        assert "nextflow.enable.dsl = 2" in result.code
        assert "process" in result.code
        assert "workflow" in result.code
        
        # Verify files were written
        assert (output_dir / "main.nf").exists()
        main_nf_content = (output_dir / "main.nf").read_text()
        assert len(main_nf_content) > 100
        
        # Verify README
        readme_content = (output_dir / "README.md").read_text()
        assert "Quick Start" in readme_content
    
    def test_multiple_workflow_types(self):
        """Test generating multiple workflow types."""
        supervisor = SupervisorAgent()
        
        queries = [
            ("RNA-seq analysis", "rna-seq"),
            ("ChIP-seq peak calling", "chip-seq"),
            ("Whole genome variant calling", "dna-seq"),
        ]
        
        for query, expected_type in queries:
            result = supervisor.execute_sync(query)
            assert result.success is True
            assert result.plan.analysis_type == expected_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
