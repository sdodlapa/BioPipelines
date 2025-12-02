"""
Tests for Skill Loader Module
=============================

Tests for the skill documentation system inspired by Claude Scientific Skills.
"""

import pytest
from pathlib import Path
import tempfile
import yaml
import os

from workflow_composer.agents.intent.skill_loader import (
    SkillLoader,
    Skill,
    SkillMatch,
    get_skill_loader,
    reset_skill_loader,
    find_skills,
    get_skill_help_text,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_skill_data():
    """Sample skill data for testing."""
    return {
        "name": "test_tool",
        "display_name": "Test Tool",
        "version": "1.0.0",
        "category": "quality_control",
        "description": "A test tool for quality control analysis.",
        "capabilities": [
            "Perform quality analysis",
            "Generate reports",
            "Identify issues",
        ],
        "aliases": ["testtool", "quality_checker"],
        "trigger_phrases": [
            "check quality",
            "run quality control",
            "analyze data quality",
        ],
        "examples": [
            {
                "query": "Check the quality of my data",
                "expected_behavior": "Runs quality analysis",
            }
        ],
        "parameters": [
            {
                "name": "input_file",
                "type": "file_path",
                "required": True,
                "description": "Input file to analyze",
            },
            {
                "name": "output_dir",
                "type": "directory_path",
                "required": False,
                "default": "./output",
                "description": "Output directory",
            },
        ],
        "outputs": [
            {
                "name": "report",
                "type": "file",
                "format": "html",
                "description": "Quality report",
            }
        ],
        "related_skills": [
            {"name": "fastqc", "relationship": "precedes", "description": "Run before"},
            {"name": "multiqc", "relationship": "follows", "description": "Aggregate reports"},
        ],
        "limitations": ["Cannot process very large files"],
        "best_practices": ["Always run on raw data first"],
    }


@pytest.fixture
def temp_skills_dir(sample_skill_data):
    """Create temporary skills directory with test skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir)
        
        # Write test skill
        with open(skills_dir / "test_tool.yaml", "w") as f:
            yaml.dump(sample_skill_data, f)
        
        # Write a second skill for testing
        second_skill = {
            "name": "alignment_tool",
            "display_name": "Alignment Tool",
            "version": "1.0.0",
            "category": "alignment",
            "description": "Align sequences to reference genome.",
            "capabilities": ["Align reads", "Support paired-end data"],
            "aliases": ["aligner", "mapper"],
            "trigger_phrases": ["align my reads", "map sequences"],
        }
        with open(skills_dir / "alignment_tool.yaml", "w") as f:
            yaml.dump(second_skill, f)
        
        yield skills_dir


@pytest.fixture
def skill_loader(temp_skills_dir):
    """Create a skill loader with test skills."""
    return SkillLoader(skills_dir=temp_skills_dir)


@pytest.fixture(autouse=True)
def reset_global_loader():
    """Reset global loader before each test."""
    reset_skill_loader()
    yield
    reset_skill_loader()


# ============================================================================
# Skill Class Tests
# ============================================================================

class TestSkill:
    """Tests for the Skill dataclass."""
    
    def test_from_dict(self, sample_skill_data):
        """Test creating Skill from dictionary."""
        skill = Skill.from_dict(sample_skill_data)
        
        assert skill.name == "test_tool"
        assert skill.display_name == "Test Tool"
        assert skill.version == "1.0.0"
        assert skill.category == "quality_control"
        assert "quality control" in skill.description.lower()
        assert len(skill.capabilities) == 3
        assert len(skill.aliases) == 2
        assert len(skill.trigger_phrases) == 3
    
    def test_get_all_trigger_terms(self, sample_skill_data):
        """Test getting all trigger terms."""
        skill = Skill.from_dict(sample_skill_data)
        terms = skill.get_all_trigger_terms()
        
        assert "test_tool" in terms
        assert "test tool" in terms
        assert "testtool" in terms
        assert "quality_checker" in terms
        assert "check quality" in terms
        assert "run quality control" in terms
    
    def test_get_required_parameters(self, sample_skill_data):
        """Test getting required parameters."""
        skill = Skill.from_dict(sample_skill_data)
        required = skill.get_required_parameters()
        
        assert len(required) == 1
        assert required[0]["name"] == "input_file"
    
    def test_get_optional_parameters(self, sample_skill_data):
        """Test getting optional parameters."""
        skill = Skill.from_dict(sample_skill_data)
        optional = skill.get_optional_parameters()
        
        assert len(optional) == 1
        assert optional[0]["name"] == "output_dir"


# ============================================================================
# SkillLoader Tests
# ============================================================================

class TestSkillLoader:
    """Tests for the SkillLoader class."""
    
    def test_init_with_custom_dir(self, temp_skills_dir):
        """Test initialization with custom directory."""
        loader = SkillLoader(skills_dir=temp_skills_dir)
        
        assert loader.skills_dir == temp_skills_dir
        assert len(loader.get_all_skills()) == 2
    
    def test_get_skill(self, skill_loader):
        """Test getting a skill by name."""
        skill = skill_loader.get_skill("test_tool")
        
        assert skill is not None
        assert skill.name == "test_tool"
        assert skill.display_name == "Test Tool"
    
    def test_get_skill_not_found(self, skill_loader):
        """Test getting non-existent skill."""
        skill = skill_loader.get_skill("nonexistent")
        assert skill is None
    
    def test_get_all_skills(self, skill_loader):
        """Test getting all skills."""
        skills = skill_loader.get_all_skills()
        
        assert len(skills) == 2
        names = [s.name for s in skills]
        assert "test_tool" in names
        assert "alignment_tool" in names
    
    def test_get_skills_by_category(self, skill_loader):
        """Test getting skills by category."""
        qc_skills = skill_loader.get_skills_by_category("quality_control")
        
        assert len(qc_skills) == 1
        assert qc_skills[0].name == "test_tool"
    
    def test_get_categories(self, skill_loader):
        """Test getting all categories."""
        categories = skill_loader.get_categories()
        
        assert "quality_control" in categories
        assert "alignment" in categories
    
    def test_reload(self, skill_loader, temp_skills_dir):
        """Test reloading skills."""
        # Add a new skill
        new_skill = {
            "name": "new_tool",
            "display_name": "New Tool",
            "version": "1.0.0",
            "category": "variant_calling",
            "description": "A new tool.",
            "capabilities": ["Do something new"],
        }
        with open(temp_skills_dir / "new_tool.yaml", "w") as f:
            yaml.dump(new_skill, f)
        
        # Reload
        skill_loader.reload()
        
        # Check new skill is loaded
        assert skill_loader.get_skill("new_tool") is not None
        assert len(skill_loader.get_all_skills()) == 3


# ============================================================================
# Query Matching Tests
# ============================================================================

class TestSkillMatching:
    """Tests for skill matching functionality."""
    
    def test_find_skills_direct_name_match(self, skill_loader):
        """Test finding skills by direct name match."""
        matches = skill_loader.find_skills_for_query("run test_tool")
        
        assert len(matches) > 0
        assert matches[0].name == "test_tool"
        assert matches[0].score >= 0.8
        assert "Direct match" in str(matches[0].match_reasons)
    
    def test_find_skills_alias_match(self, skill_loader):
        """Test finding skills by alias."""
        matches = skill_loader.find_skills_for_query("use the quality_checker")
        
        assert len(matches) > 0
        top_match = matches[0]
        assert top_match.name == "test_tool"
        assert "alias" in str(top_match.match_reasons).lower()
    
    def test_find_skills_trigger_phrase_match(self, skill_loader):
        """Test finding skills by trigger phrase."""
        matches = skill_loader.find_skills_for_query("check quality of my data")
        
        assert len(matches) > 0
        assert matches[0].name == "test_tool"
    
    def test_find_skills_category_match(self, skill_loader):
        """Test finding skills by category mention."""
        matches = skill_loader.find_skills_for_query("I need quality_control tools")
        
        assert len(matches) > 0
        # Should match test_tool due to category
        names = [m.name for m in matches]
        assert "test_tool" in names
    
    def test_find_skills_multiple_matches(self, skill_loader):
        """Test finding multiple matching skills."""
        matches = skill_loader.find_skills_for_query("analyze my data", limit=5)
        
        # Both tools could match general analysis terms
        assert len(matches) >= 1
    
    def test_find_skills_no_match(self, skill_loader):
        """Test when no skills match."""
        matches = skill_loader.find_skills_for_query("make coffee")
        
        # Should return empty or low-scoring matches
        assert len(matches) == 0 or all(m.score < 0.3 for m in matches)
    
    def test_find_skills_limit(self, skill_loader):
        """Test limiting number of results."""
        matches = skill_loader.find_skills_for_query("tool", limit=1)
        assert len(matches) <= 1
    
    def test_find_skills_min_score(self, skill_loader):
        """Test minimum score filtering."""
        matches = skill_loader.find_skills_for_query("test_tool", min_score=0.8)
        
        assert all(m.score >= 0.8 for m in matches)


# ============================================================================
# Help Generation Tests
# ============================================================================

class TestSkillHelp:
    """Tests for help text generation."""
    
    def test_get_skill_help(self, skill_loader):
        """Test generating help text."""
        help_text = skill_loader.get_skill_help("test_tool")
        
        assert help_text is not None
        assert "Test Tool" in help_text
        assert "Capabilities" in help_text
        assert "quality analysis" in help_text.lower()
        assert "Required Parameters" in help_text
        assert "input_file" in help_text
    
    def test_get_skill_help_not_found(self, skill_loader):
        """Test help for non-existent skill."""
        help_text = skill_loader.get_skill_help("nonexistent")
        assert help_text is None
    
    def test_get_skill_parameters_for_prompt(self, skill_loader):
        """Test generating parameter info for prompts."""
        param_text = skill_loader.get_skill_parameters_for_prompt("test_tool")
        
        assert "Parameters for Test Tool" in param_text
        assert "input_file" in param_text
        assert "required" in param_text
        assert "output_dir" in param_text
        assert "optional" in param_text


# ============================================================================
# Workflow Suggestion Tests
# ============================================================================

class TestWorkflowSuggestion:
    """Tests for workflow suggestion functionality."""
    
    def test_suggest_workflow(self, skill_loader):
        """Test suggesting workflow from skill relationships."""
        workflow = skill_loader.suggest_workflow("test_tool")
        
        assert len(workflow) >= 1
        
        # Check main step exists
        main_steps = [s for s in workflow if s["step"] == "main"]
        assert len(main_steps) == 1
        assert main_steps[0]["skill"] == "test_tool"
    
    def test_suggest_workflow_not_found(self, skill_loader):
        """Test workflow suggestion for non-existent skill."""
        workflow = skill_loader.suggest_workflow("nonexistent")
        assert workflow == []


# ============================================================================
# Global Loader Tests
# ============================================================================

class TestGlobalLoader:
    """Tests for global loader singleton."""
    
    def test_get_skill_loader_singleton(self):
        """Test that get_skill_loader returns singleton."""
        loader1 = get_skill_loader()
        loader2 = get_skill_loader()
        
        assert loader1 is loader2
    
    def test_reset_skill_loader(self):
        """Test resetting global loader."""
        loader1 = get_skill_loader()
        reset_skill_loader()
        loader2 = get_skill_loader()
        
        # Should be different instances after reset
        assert loader1 is not loader2


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_find_skills(self, temp_skills_dir, monkeypatch):
        """Test find_skills convenience function."""
        # Patch the skills directory
        loader = SkillLoader(skills_dir=temp_skills_dir)
        monkeypatch.setattr(
            "workflow_composer.agents.intent.skill_loader._skill_loader", 
            loader
        )
        
        matches = find_skills("check quality")
        assert len(matches) > 0
    
    def test_get_skill_help_text(self, temp_skills_dir, monkeypatch):
        """Test get_skill_help_text convenience function."""
        loader = SkillLoader(skills_dir=temp_skills_dir)
        monkeypatch.setattr(
            "workflow_composer.agents.intent.skill_loader._skill_loader",
            loader
        )
        
        help_text = get_skill_help_text("test_tool")
        assert help_text is not None
        assert "Test Tool" in help_text


# ============================================================================
# Integration Tests with Real Skills
# ============================================================================

class TestRealSkills:
    """Tests using actual skill files if they exist."""
    
    def test_load_real_skills(self):
        """Test loading real skill files from config/skills."""
        # This test uses the actual skills directory
        # Find project root by looking for pyproject.toml
        current = Path(__file__).resolve()
        project_root = None
        for parent in current.parents:
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break
        
        if project_root is None:
            pytest.skip("Could not find project root")
        
        skills_dir = project_root / "config" / "skills"
        
        if not skills_dir.exists():
            pytest.skip("Skills directory not found")
        
        loader = SkillLoader(skills_dir=skills_dir)
        skills = loader.get_all_skills()
        
        # Should have loaded some skills
        assert len(skills) > 0
        
        # Check for expected skills
        skill_names = [s.name for s in skills]
        expected_skills = ["fastqc", "star", "bwa", "gatk", "deseq2"]
        found = [s for s in expected_skills if s in skill_names]
        
        if len(found) > 0:
            # Verify skill structure
            fastqc = loader.get_skill("fastqc")
            if fastqc:
                assert fastqc.display_name == "FastQC"
                assert fastqc.category == "quality_control"
                assert len(fastqc.capabilities) > 0
                assert len(fastqc.trigger_phrases) > 0
    
    def test_real_skill_matching(self):
        """Test matching against real skills."""
        # Find project root
        current = Path(__file__).resolve()
        project_root = None
        for parent in current.parents:
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break
        
        if project_root is None:
            pytest.skip("Could not find project root")
        
        skills_dir = project_root / "config" / "skills"
        
        if not skills_dir.exists():
            pytest.skip("Skills directory not found")
        
        loader = SkillLoader(skills_dir=skills_dir)
        
        # Test various queries
        test_queries = [
            ("check quality of my reads", ["fastqc"]),
            ("align RNA-seq data", ["star"]),
            ("call variants", ["gatk"]),
            ("differential expression analysis", ["deseq2"]),
            ("align DNA reads to genome", ["bwa"]),
            ("methylation analysis", ["bismark"]),
            ("classify metagenomics", ["kraken2"]),
            ("single cell clustering", ["scanpy"]),
        ]
        
        for query, expected_tools in test_queries:
            matches = loader.find_skills_for_query(query)
            matched_names = [m.name for m in matches]
            
            # At least one expected tool should be in top matches
            found = [t for t in expected_tools if t in matched_names]
            if loader.get_skill(expected_tools[0]):  # Only check if skill exists
                assert len(found) > 0, f"Query '{query}' should match {expected_tools}, got {matched_names}"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_skills_directory(self):
        """Test with empty skills directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SkillLoader(skills_dir=Path(tmpdir))
            
            assert len(loader.get_all_skills()) == 0
            assert loader.get_skill("anything") is None
    
    def test_nonexistent_skills_directory(self):
        """Test with non-existent skills directory."""
        loader = SkillLoader(skills_dir=Path("/nonexistent/path"))
        
        # Should not raise, just have no skills
        assert len(loader.get_all_skills()) == 0
    
    def test_malformed_yaml(self):
        """Test handling of malformed YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            
            # Write malformed YAML
            with open(skills_dir / "bad.yaml", "w") as f:
                f.write("name: test\n  invalid: yaml: structure")
            
            # Valid skill
            with open(skills_dir / "good.yaml", "w") as f:
                yaml.dump({
                    "name": "good_tool",
                    "display_name": "Good Tool",
                    "version": "1.0.0",
                    "category": "test",
                    "description": "A good tool",
                    "capabilities": ["Do stuff"],
                }, f)
            
            loader = SkillLoader(skills_dir=skills_dir)
            
            # Should load the valid skill
            assert loader.get_skill("good_tool") is not None
    
    def test_skill_without_required_fields(self):
        """Test handling skill files missing required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            
            # Skill missing name
            with open(skills_dir / "incomplete.yaml", "w") as f:
                yaml.dump({
                    "display_name": "Incomplete",
                    "category": "test",
                }, f)
            
            loader = SkillLoader(skills_dir=skills_dir)
            
            # Should not crash, just skip the incomplete skill
            assert len(loader.get_all_skills()) == 0
    
    def test_special_characters_in_query(self, skill_loader):
        """Test query with special characters."""
        matches = skill_loader.find_skills_for_query("test_tool (v1.0) @ #special!")
        
        # Should still find the tool
        assert len(matches) > 0
    
    def test_very_long_query(self, skill_loader):
        """Test with very long query."""
        long_query = "check quality " * 100
        matches = skill_loader.find_skills_for_query(long_query)
        
        # Should not crash
        assert isinstance(matches, list)
    
    def test_unicode_in_query(self, skill_loader):
        """Test query with unicode characters."""
        matches = skill_loader.find_skills_for_query("check quality 质量检查 品質チェック")
        
        # Should not crash
        assert isinstance(matches, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
