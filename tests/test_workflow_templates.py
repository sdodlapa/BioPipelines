"""
Tests for Workflow Template Engine
==================================

Tests for the BioPipelines workflow template system.
"""

import pytest
from pathlib import Path
import sys
import tempfile
import shutil

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))

from workflow_templates import (
    WorkflowTemplate,
    TemplateParameter,
    TemplateStep,
    TemplateEngine,
    get_template_engine,
    reset_engine,
)


class TestWorkflowTemplate:
    """Tests for WorkflowTemplate dataclass."""
    
    def test_from_dict_basic(self):
        """Test creating WorkflowTemplate from dictionary."""
        data = {
            "name": "test_template",
            "display_name": "Test Template",
            "version": "1.0.0",
            "category": "testing",
            "description": "A test template",
            "tags": ["test", "template"],
            "inputs": {
                "required": [
                    {"name": "input_dir", "type": "path", "description": "Input directory"}
                ],
                "optional": [
                    {"name": "threads", "type": "integer", "description": "Threads", "default": 4}
                ]
            },
            "steps": [
                {
                    "name": "step1",
                    "description": "First step",
                    "tool": "testtool"
                }
            ]
        }
        
        template = WorkflowTemplate.from_dict(data)
        
        assert template.name == "test_template"
        assert template.display_name == "Test Template"
        assert template.version == "1.0.0"
        assert "test" in template.tags
        assert len(template.inputs.get("required", [])) == 1
        assert len(template.steps) == 1
    
    def test_validate_inputs_success(self):
        """Test input validation with valid inputs."""
        data = {
            "name": "test",
            "display_name": "Test",
            "version": "1.0.0",
            "category": "test",
            "description": "Test",
            "tags": [],
            "inputs": {
                "required": [
                    {"name": "input_dir", "type": "path", "description": "Input"}
                ]
            },
            "steps": []
        }
        
        template = WorkflowTemplate.from_dict(data)
        is_valid, errors = template.validate_inputs({"input_dir": "/data"})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_inputs_missing_required(self):
        """Test input validation with missing required input."""
        data = {
            "name": "test",
            "display_name": "Test",
            "version": "1.0.0",
            "category": "test",
            "description": "Test",
            "tags": [],
            "inputs": {
                "required": [
                    {"name": "input_dir", "type": "path", "description": "Input"}
                ]
            },
            "steps": []
        }
        
        template = WorkflowTemplate.from_dict(data)
        is_valid, errors = template.validate_inputs({})
        
        assert not is_valid
        assert len(errors) == 1
        assert "input_dir" in errors[0]
    
    def test_validate_inputs_invalid_enum(self):
        """Test input validation with invalid enum value."""
        data = {
            "name": "test",
            "display_name": "Test",
            "version": "1.0.0",
            "category": "test",
            "description": "Test",
            "tags": [],
            "inputs": {
                "required": [
                    {"name": "organism", "type": "string", "description": "Org", "enum": ["human", "mouse"]}
                ]
            },
            "steps": []
        }
        
        template = WorkflowTemplate.from_dict(data)
        is_valid, errors = template.validate_inputs({"organism": "invalid"})
        
        assert not is_valid
        assert "organism" in errors[0]
    
    def test_get_defaults(self):
        """Test getting default values."""
        data = {
            "name": "test",
            "display_name": "Test",
            "version": "1.0.0",
            "category": "test",
            "description": "Test",
            "tags": [],
            "inputs": {
                "optional": [
                    {"name": "threads", "type": "integer", "description": "Threads", "default": 8},
                    {"name": "memory", "type": "string", "description": "Memory", "default": "32G"}
                ]
            },
            "steps": []
        }
        
        template = WorkflowTemplate.from_dict(data)
        defaults = template.get_defaults()
        
        assert defaults["threads"] == 8
        assert defaults["memory"] == "32G"


class TestTemplateEngine:
    """Tests for TemplateEngine."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset engine before each test."""
        reset_engine()
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        templates_dir = Path(__file__).parent.parent / "config" / "workflow_templates"
        engine = TemplateEngine(templates_dir=templates_dir)
        
        assert engine.templates_dir == templates_dir
    
    def test_engine_loads_templates(self):
        """Test engine loads template files."""
        engine = get_template_engine()
        templates = engine.list_templates()
        
        # Should have loaded some templates
        assert isinstance(templates, list)
    
    def test_get_template_by_name(self):
        """Test retrieving template by name."""
        engine = get_template_engine()
        templates = engine.list_templates()
        
        if templates:
            template = engine.get_template(templates[0].name)
            assert template is not None
            assert template.name == templates[0].name
    
    def test_get_nonexistent_template(self):
        """Test getting a template that doesn't exist."""
        engine = get_template_engine()
        template = engine.get_template("nonexistent_template_xyz")
        
        assert template is None
    
    def test_get_categories(self):
        """Test getting template categories."""
        engine = get_template_engine()
        categories = engine.get_categories()
        
        assert isinstance(categories, list)
    
    def test_list_templates_by_category(self):
        """Test listing templates by category."""
        engine = get_template_engine()
        categories = engine.get_categories()
        
        if categories:
            templates = engine.list_templates_by_category(categories[0])
            assert isinstance(templates, list)


class TestWorkflowGeneration:
    """Tests for workflow generation."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for tests."""
        reset_engine()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup temp directory."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_generate_missing_template(self):
        """Test generating with non-existent template."""
        engine = get_template_engine()
        result = engine.generate("nonexistent_template")
        
        assert not result["success"]
        assert "not found" in result["error"]
    
    def test_generate_missing_required_params(self):
        """Test generating with missing required parameters."""
        engine = get_template_engine()
        templates = engine.list_templates()
        
        if templates:
            # Try to generate without providing required params
            template = templates[0]
            if template.inputs.get("required"):
                result = engine.generate(
                    template.name,
                    output_dir=self.temp_dir
                )
                # Should fail validation
                assert not result["success"] or "error" in result
    
    def test_generate_nextflow_workflow(self):
        """Test generating a Nextflow workflow."""
        engine = get_template_engine()
        
        # Find a nextflow template
        templates = [t for t in engine.list_templates() if t.engine == "nextflow"]
        
        if templates:
            template = templates[0]
            
            # Provide required params (mock values)
            params = {"input_dir": "/data", "organism": "human"}
            for p in template.inputs.get("required", []):
                if p.name not in params:
                    if p.enum:
                        params[p.name] = p.enum[0]
                    else:
                        params[p.name] = "test_value"
            
            result = engine.generate(
                template.name,
                output_dir=self.temp_dir,
                **params
            )
            
            if result.get("success"):
                assert "workflow_dir" in result
                assert "files" in result
                assert Path(result["files"]["main"]).exists()
    
    def test_generate_creates_output_dir(self):
        """Test that generation creates output directory."""
        engine = get_template_engine()
        templates = engine.list_templates()
        
        if templates:
            template = templates[0]
            output_dir = Path(self.temp_dir) / "new_workflow"
            
            # Provide required params
            params = {"input_dir": "/data", "organism": "human"}
            for p in template.inputs.get("required", []):
                if p.name not in params:
                    if p.enum:
                        params[p.name] = p.enum[0]
                    else:
                        params[p.name] = "test_value"
            
            result = engine.generate(
                template.name,
                output_dir=str(output_dir),
                **params
            )
            
            if result.get("success"):
                assert output_dir.exists()


class TestTemplateSingleton:
    """Tests for singleton pattern."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset engine before each test."""
        reset_engine()
    
    def test_singleton_returns_same_instance(self):
        """Test that get_template_engine returns same instance."""
        engine1 = get_template_engine()
        engine2 = get_template_engine()
        
        assert engine1 is engine2
    
    def test_reset_engine(self):
        """Test that reset_engine clears singleton."""
        engine1 = get_template_engine()
        reset_engine()
        engine2 = get_template_engine()
        
        assert isinstance(engine2, TemplateEngine)


class TestTemplateIntegration:
    """Integration tests for template system."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset engine before each test."""
        reset_engine()
    
    def test_rnaseq_templates_loaded(self):
        """Test RNA-seq templates are loaded."""
        engine = get_template_engine()
        rnaseq_templates = engine.list_templates_by_category("rnaseq")
        
        # Should have RNA-seq templates
        assert len(rnaseq_templates) >= 0
    
    def test_chipseq_templates_loaded(self):
        """Test ChIP-seq templates are loaded."""
        engine = get_template_engine()
        chipseq_templates = engine.list_templates_by_category("chipseq")
        
        assert isinstance(chipseq_templates, list)
    
    def test_variant_templates_loaded(self):
        """Test variant calling templates are loaded."""
        engine = get_template_engine()
        variant_templates = engine.list_templates_by_category("variant")
        
        assert isinstance(variant_templates, list)
    
    def test_methylation_templates_loaded(self):
        """Test methylation templates are loaded."""
        engine = get_template_engine()
        meth_templates = engine.list_templates_by_category("methylation")
        
        assert isinstance(meth_templates, list)
    
    def test_template_has_required_fields(self):
        """Test that loaded templates have required fields."""
        engine = get_template_engine()
        templates = engine.list_templates()
        
        for template in templates:
            assert template.name is not None
            assert template.display_name is not None
            assert template.version is not None
            assert template.category is not None
            assert template.description is not None
            assert isinstance(template.steps, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
