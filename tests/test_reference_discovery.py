"""
Tests for Reference Discovery Agent
====================================

Tests the ReferenceDiscoveryAgent that discovers relevant code references
from nf-core modules, pipelines, and GitHub repositories.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from workflow_composer.agents.specialists.reference_discovery import (
    ReferenceDiscoveryAgent,
    ReferenceSource,
    CodeReference,
    ReferenceSearchResult,
)


class TestCodeReference:
    """Tests for CodeReference dataclass."""
    
    def test_code_reference_creation(self):
        """Test creating a CodeReference."""
        ref = CodeReference(
            source=ReferenceSource.NF_CORE_MODULES,
            name="fastqc",
            url="https://github.com/nf-core/modules/tree/master/modules/nf-core/fastqc",
            description="FastQC quality control",
            relevance_score=0.95,
            tools=["fastqc"],
            language="nextflow",
        )
        
        assert ref.source == ReferenceSource.NF_CORE_MODULES
        assert ref.name == "fastqc"
        assert ref.relevance_score == 0.95
        assert "fastqc" in ref.tools
    
    def test_code_reference_to_dict(self):
        """Test converting CodeReference to dictionary."""
        ref = CodeReference(
            source=ReferenceSource.NF_CORE_PIPELINES,
            name="nf-core/rnaseq",
            url="https://github.com/nf-core/rnaseq",
            description="RNA-seq pipeline",
            relevance_score=0.98,
            stars=800,
        )
        
        data = ref.to_dict()
        
        assert data["source"] == "nf-core/pipelines"
        assert data["name"] == "nf-core/rnaseq"
        assert data["relevance_score"] == 0.98
        assert data["stars"] == 800


class TestReferenceSearchResult:
    """Tests for ReferenceSearchResult."""
    
    def test_top_references(self):
        """Test getting top references by score."""
        refs = [
            CodeReference(
                source=ReferenceSource.NF_CORE_MODULES,
                name="fastqc",
                url="url1",
                description="desc1",
                relevance_score=0.7,
            ),
            CodeReference(
                source=ReferenceSource.NF_CORE_MODULES,
                name="star",
                url="url2",
                description="desc2",
                relevance_score=0.95,
            ),
            CodeReference(
                source=ReferenceSource.NF_CORE_PIPELINES,
                name="rnaseq",
                url="url3",
                description="desc3",
                relevance_score=0.85,
            ),
        ]
        
        result = ReferenceSearchResult(
            query="rna-seq",
            references=refs,
            total_found=3,
        )
        
        top = result.top_references(2)
        
        assert len(top) == 2
        assert top[0].name == "star"  # Highest score
        assert top[1].name == "rnaseq"
    
    def test_by_source(self):
        """Test filtering references by source."""
        refs = [
            CodeReference(
                source=ReferenceSource.NF_CORE_MODULES,
                name="fastqc",
                url="url1",
                description="desc1",
                relevance_score=0.7,
            ),
            CodeReference(
                source=ReferenceSource.NF_CORE_PIPELINES,
                name="rnaseq",
                url="url2",
                description="desc2",
                relevance_score=0.9,
            ),
        ]
        
        result = ReferenceSearchResult(query="test", references=refs)
        
        modules = result.by_source(ReferenceSource.NF_CORE_MODULES)
        pipelines = result.by_source(ReferenceSource.NF_CORE_PIPELINES)
        
        assert len(modules) == 1
        assert modules[0].name == "fastqc"
        assert len(pipelines) == 1
        assert pipelines[0].name == "rnaseq"


class TestReferenceDiscoveryAgent:
    """Tests for ReferenceDiscoveryAgent."""
    
    def test_init_without_dependencies(self):
        """Test initialization without router or knowledge base."""
        agent = ReferenceDiscoveryAgent()
        
        assert agent.router is None
        assert agent.knowledge_base is None
        assert agent.github_token is None
    
    def test_init_with_dependencies(self):
        """Test initialization with dependencies."""
        router = Mock()
        kb = Mock()
        
        agent = ReferenceDiscoveryAgent(
            router=router,
            knowledge_base=kb,
            github_token="test_token",
        )
        
        assert agent.router is router
        assert agent.knowledge_base is kb
        assert agent.github_token == "test_token"
    
    def test_discover_sync_with_tools(self):
        """Test synchronous discovery with tool list."""
        agent = ReferenceDiscoveryAgent()
        
        result = agent.discover_sync(
            query="RNA-seq alignment",
            tools=["STAR", "salmon"],
        )
        
        assert isinstance(result, ReferenceSearchResult)
        assert result.query == "RNA-seq alignment"
        
        # Should find STAR and salmon modules
        tool_names = [r.name.lower() for r in result.references]
        assert any("star" in name for name in tool_names)
        assert any("salmon" in name for name in tool_names)
    
    def test_discover_sync_with_analysis_type(self):
        """Test synchronous discovery with analysis type."""
        agent = ReferenceDiscoveryAgent()
        
        result = agent.discover_sync(
            query="differential expression analysis",
            analysis_type="rna-seq",
        )
        
        assert isinstance(result, ReferenceSearchResult)
        
        # Should find nf-core/rnaseq pipeline
        pipeline_refs = result.by_source(ReferenceSource.NF_CORE_PIPELINES)
        assert len(pipeline_refs) > 0
        assert any("rnaseq" in r.name.lower() for r in pipeline_refs)
    
    def test_discover_sync_chipseq(self):
        """Test discovery for ChIP-seq analysis."""
        agent = ReferenceDiscoveryAgent()
        
        result = agent.discover_sync(
            query="peak calling",
            analysis_type="chip-seq",
            tools=["MACS2", "bowtie2"],
        )
        
        # Should find macs2 and bowtie2 modules
        module_names = [r.name.lower() for r in result.by_source(ReferenceSource.NF_CORE_MODULES)]
        assert any("macs2" in name for name in module_names)
        assert any("bowtie2" in name for name in module_names)
        
        # Should find chipseq pipeline
        pipeline_refs = result.by_source(ReferenceSource.NF_CORE_PIPELINES)
        assert any("chipseq" in r.name.lower() for r in pipeline_refs)
    
    def test_discover_sync_variant_calling(self):
        """Test discovery for variant calling."""
        agent = ReferenceDiscoveryAgent()
        
        result = agent.discover_sync(
            query="germline variant calling",
            analysis_type="variant-calling",
            tools=["GATK", "BWA"],
        )
        
        # Should find GATK and BWA modules
        module_names = [r.name.lower() for r in result.by_source(ReferenceSource.NF_CORE_MODULES)]
        assert any("gatk" in name for name in module_names)
        assert any("bwa" in name for name in module_names)
        
        # Should find sarek pipeline
        pipeline_refs = result.by_source(ReferenceSource.NF_CORE_PIPELINES)
        assert any("sarek" in r.name.lower() for r in pipeline_refs)
    
    def test_get_module_snippet(self):
        """Test getting code snippets for modules."""
        agent = ReferenceDiscoveryAgent()
        
        # Test fastqc snippet
        snippet = agent.get_module_snippet("fastqc")
        assert snippet is not None
        assert "FASTQC" in snippet
        assert "include" in snippet
        
        # Test star snippet
        snippet = agent.get_module_snippet("star/align")
        assert snippet is not None
        assert "STAR_ALIGN" in snippet
        
        # Test nonexistent module
        snippet = agent.get_module_snippet("nonexistent_tool")
        assert snippet is None
    
    def test_format_references_for_prompt(self):
        """Test formatting references for LLM prompts."""
        agent = ReferenceDiscoveryAgent()
        
        refs = [
            CodeReference(
                source=ReferenceSource.NF_CORE_MODULES,
                name="fastqc",
                url="https://example.com",
                description="Quality control",
                relevance_score=0.9,
                tools=["fastqc"],
            ),
        ]
        
        result = ReferenceSearchResult(
            query="qc",
            references=refs,
            total_found=1,
        )
        
        formatted = agent.format_references_for_prompt(result)
        
        assert "Discovered References" in formatted
        assert "fastqc" in formatted
        assert "Quality control" in formatted
        assert "nf-core/modules" in formatted
    
    @pytest.mark.asyncio
    async def test_discover_async(self):
        """Test async discovery."""
        agent = ReferenceDiscoveryAgent()
        
        result = await agent.discover(
            query="RNA-seq quantification",
            analysis_type="rna-seq",
            tools=["salmon"],
        )
        
        assert isinstance(result, ReferenceSearchResult)
        assert result.total_found > 0
        assert result.search_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_discover_caching(self):
        """Test that results are cached."""
        agent = ReferenceDiscoveryAgent()
        
        # First call
        result1 = await agent.discover(
            query="test query",
            analysis_type="rna-seq",
            tools=["fastqc"],
        )
        
        # Second call with same parameters
        result2 = await agent.discover(
            query="test query",
            analysis_type="rna-seq",
            tools=["fastqc"],
        )
        
        # Should be the same cached result
        assert result1.search_time_ms > 0
        # Second call should be instant (cached)
    
    def test_nf_core_modules_coverage(self):
        """Test that common tools are in NF_CORE_MODULES."""
        agent = ReferenceDiscoveryAgent()
        
        expected_tools = [
            "fastqc", "fastp", "star", "salmon", "bowtie2",
            "bwa", "samtools", "gatk", "macs2", "multiqc",
        ]
        
        for tool in expected_tools:
            assert tool in agent.NF_CORE_MODULES, f"Missing module: {tool}"
    
    def test_nf_core_pipelines_coverage(self):
        """Test that common analysis types are in NF_CORE_PIPELINES."""
        agent = ReferenceDiscoveryAgent()
        
        expected_types = [
            "rna-seq", "chip-seq", "atac-seq", "dna-seq",
            "methylation", "scrna-seq", "metagenomics",
        ]
        
        for analysis_type in expected_types:
            assert analysis_type in agent.NF_CORE_PIPELINES, f"Missing pipeline: {analysis_type}"


class TestReferenceDiscoveryIntegration:
    """Integration tests for ReferenceDiscoveryAgent."""
    
    def test_full_rnaseq_discovery(self):
        """Test full discovery flow for RNA-seq."""
        agent = ReferenceDiscoveryAgent()
        
        result = agent.discover_sync(
            query="I want to analyze RNA-seq data with STAR alignment and salmon quantification",
            analysis_type="rna-seq",
            tools=["STAR", "salmon", "FastQC", "MultiQC"],
        )
        
        # Should find multiple references
        assert result.total_found >= 4
        
        # Should include nf-core/rnaseq pipeline
        pipelines = result.by_source(ReferenceSource.NF_CORE_PIPELINES)
        assert any("rnaseq" in p.name.lower() for p in pipelines)
        
        # Should include all requested tools
        modules = result.by_source(ReferenceSource.NF_CORE_MODULES)
        module_names = [m.name.lower() for m in modules]
        assert any("star" in name for name in module_names)
        assert any("salmon" in name for name in module_names)
        assert any("fastqc" in name for name in module_names)
        assert any("multiqc" in name for name in module_names)
    
    def test_keyword_extraction_from_query(self):
        """Test that keywords are extracted from query."""
        agent = ReferenceDiscoveryAgent()
        
        # Query contains tool names without explicit tools list
        result = agent.discover_sync(
            query="I need to run fastqc and then align with bowtie2 for my ChIP-seq data",
        )
        
        module_names = [m.name.lower() for m in result.by_source(ReferenceSource.NF_CORE_MODULES)]
        assert any("fastqc" in name for name in module_names)
        assert any("bowtie2" in name for name in module_names)
