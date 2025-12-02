"""
Tests for Codebase Indexer
==========================

Tests the CodebaseIndexer that indexes Nextflow codebases for
intelligent reference during code generation.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from workflow_composer.agents.specialists.codebase_indexer import (
    CodebaseIndexer,
    CodebaseIndex,
    CodeElement,
    CodeElementType,
)


# Sample Nextflow code for testing
SAMPLE_PROCESS = '''
process FASTQC {
    tag "$meta.id"
    label 'process_medium'
    container 'quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0'

    cpus 4
    memory '8 GB'

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path('*.html'), emit: html
    tuple val(meta), path('*.zip'), emit: zip
    path('versions.yml'), emit: versions

    script:
    """
    fastqc -t $task.cpus -o . $reads

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastqc: \\$(fastqc --version | sed "s/FastQC v//")
    END_VERSIONS
    """
}
'''

SAMPLE_WORKFLOW = '''
nextflow.enable.dsl = 2

include { FASTQC } from './modules/fastqc'
include { STAR_ALIGN } from './modules/star'

params.input = null
params.outdir = './results'

workflow RNASEQ {
    take:
    reads

    main:
    FASTQC(reads)
    STAR_ALIGN(reads, params.genome_index)

    emit:
    fastqc_results = FASTQC.out.html
    aligned_bam = STAR_ALIGN.out.bam
}

workflow {
    Channel
        .fromFilePairs(params.input)
        .set { reads_ch }
    
    RNASEQ(reads_ch)
}
'''

SAMPLE_CONFIG = '''
params {
    input = null
    outdir = './results'
    genome = 'GRCh38'
    max_memory = '128.GB'
    max_cpus = 16
    max_time = '240.h'
}

process {
    cpus = 4
    memory = '8 GB'
    time = '4h'
}
'''


class TestCodeElement:
    """Tests for CodeElement dataclass."""
    
    def test_code_element_creation(self):
        """Test creating a CodeElement."""
        element = CodeElement(
            element_type=CodeElementType.PROCESS,
            name="FASTQC",
            file_path="/path/to/main.nf",
            line_number=10,
            content="process FASTQC { ... }",
            description="FastQC quality control",
            inputs=["tuple val(meta), path(reads)"],
            outputs=["tuple val(meta), path('*.html'), emit: html"],
            container="quay.io/biocontainers/fastqc:0.12.1",
            tools=["fastqc"],
        )
        
        assert element.element_type == CodeElementType.PROCESS
        assert element.name == "FASTQC"
        assert "fastqc" in element.tools
    
    def test_code_element_to_dict(self):
        """Test converting CodeElement to dictionary."""
        element = CodeElement(
            element_type=CodeElementType.WORKFLOW,
            name="main",
            file_path="/path/to/main.nf",
            line_number=50,
            content="workflow { ... }",
            dependencies=["FASTQC", "STAR_ALIGN"],
        )
        
        data = element.to_dict()
        
        assert data["element_type"] == "workflow"
        assert data["name"] == "main"
        assert "FASTQC" in data["dependencies"]


class TestCodebaseIndex:
    """Tests for CodebaseIndex."""
    
    def test_get_processes(self):
        """Test getting all processes."""
        elements = [
            CodeElement(
                element_type=CodeElementType.PROCESS,
                name="FASTQC",
                file_path="main.nf",
                line_number=1,
                content="process FASTQC {}",
            ),
            CodeElement(
                element_type=CodeElementType.WORKFLOW,
                name="main",
                file_path="main.nf",
                line_number=50,
                content="workflow {}",
            ),
            CodeElement(
                element_type=CodeElementType.PROCESS,
                name="STAR",
                file_path="main.nf",
                line_number=30,
                content="process STAR {}",
            ),
        ]
        
        index = CodebaseIndex(
            name="test",
            root_path="/test",
            elements=elements,
        )
        
        processes = index.get_processes()
        
        assert len(processes) == 2
        assert all(p.element_type == CodeElementType.PROCESS for p in processes)
    
    def test_get_by_name(self):
        """Test getting element by name."""
        elements = [
            CodeElement(
                element_type=CodeElementType.PROCESS,
                name="FASTQC",
                file_path="main.nf",
                line_number=1,
                content="process FASTQC {}",
            ),
        ]
        
        index = CodebaseIndex(name="test", root_path="/test", elements=elements)
        
        # Exact match
        element = index.get_by_name("FASTQC")
        assert element is not None
        assert element.name == "FASTQC"
        
        # Case insensitive
        element = index.get_by_name("fastqc")
        assert element is not None
        
        # Not found
        element = index.get_by_name("NONEXISTENT")
        assert element is None
    
    def test_get_by_tool(self):
        """Test getting elements by tool."""
        elements = [
            CodeElement(
                element_type=CodeElementType.PROCESS,
                name="FASTQC",
                file_path="main.nf",
                line_number=1,
                content="process FASTQC {}",
                tools=["fastqc"],
            ),
            CodeElement(
                element_type=CodeElementType.PROCESS,
                name="ALIGNMENT",
                file_path="main.nf",
                line_number=30,
                content="process ALIGNMENT {}",
                tools=["star", "samtools"],
            ),
        ]
        
        index = CodebaseIndex(name="test", root_path="/test", elements=elements)
        
        fastqc_elements = index.get_by_tool("fastqc")
        assert len(fastqc_elements) == 1
        assert fastqc_elements[0].name == "FASTQC"
        
        star_elements = index.get_by_tool("star")
        assert len(star_elements) == 1
        assert star_elements[0].name == "ALIGNMENT"
    
    def test_search(self):
        """Test searching elements."""
        elements = [
            CodeElement(
                element_type=CodeElementType.PROCESS,
                name="FASTQC",
                file_path="main.nf",
                line_number=1,
                content="process FASTQC { fastqc command }",
                description="Quality control using FastQC",
                tools=["fastqc"],
            ),
            CodeElement(
                element_type=CodeElementType.PROCESS,
                name="ALIGNMENT",
                file_path="main.nf",
                line_number=30,
                content="process ALIGNMENT { star aligner }",
                description="Alignment with STAR",
                tools=["star"],
            ),
        ]
        
        index = CodebaseIndex(name="test", root_path="/test", elements=elements)
        
        # Search by name
        results = index.search("fastqc")
        assert len(results) > 0
        assert results[0].name == "FASTQC"
        
        # Search by description
        results = index.search("quality")
        assert len(results) > 0
        assert results[0].name == "FASTQC"
        
        # Search by tool
        results = index.search("star")
        assert len(results) > 0
        assert results[0].name == "ALIGNMENT"
    
    def test_save_and_load(self):
        """Test saving and loading index."""
        elements = [
            CodeElement(
                element_type=CodeElementType.PROCESS,
                name="FASTQC",
                file_path="main.nf",
                line_number=1,
                content="process FASTQC {}",
                tools=["fastqc"],
            ),
        ]
        
        original = CodebaseIndex(
            name="test",
            root_path="/test",
            elements=elements,
            statistics={"processes": 1},
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            original.save(f.name)
            
            loaded = CodebaseIndex.load(f.name)
            
            assert loaded.name == original.name
            assert loaded.root_path == original.root_path
            assert len(loaded.elements) == len(original.elements)
            assert loaded.elements[0].name == "FASTQC"
            assert loaded.statistics["processes"] == 1


class TestCodebaseIndexer:
    """Tests for CodebaseIndexer."""
    
    def test_init(self):
        """Test initialization."""
        indexer = CodebaseIndexer()
        
        assert indexer.router is None
        assert len(indexer._indices) == 0
    
    def test_index_file_with_process(self):
        """Test indexing a file with a process."""
        indexer = CodebaseIndexer()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nf", delete=False) as f:
            f.write(SAMPLE_PROCESS)
            f.flush()
            
            elements = indexer.index_file(f.name)
            
            assert len(elements) >= 1
            
            # Find the FASTQC process
            process = next((e for e in elements if e.name == "FASTQC"), None)
            assert process is not None
            assert process.element_type == CodeElementType.PROCESS
            assert "fastqc" in process.tools
            assert process.container == "quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0"
    
    def test_index_file_with_workflow(self):
        """Test indexing a file with workflow."""
        indexer = CodebaseIndexer()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nf", delete=False) as f:
            f.write(SAMPLE_WORKFLOW)
            f.flush()
            
            elements = indexer.index_file(f.name)
            
            # Should find workflows
            workflows = [e for e in elements if e.element_type == CodeElementType.WORKFLOW]
            assert len(workflows) >= 1
            
            # Should find includes
            imports = [e for e in elements if e.element_type == CodeElementType.IMPORT]
            assert len(imports) >= 1
            
            # Should find parameters
            params = [e for e in elements if e.element_type == CodeElementType.PARAMETER]
            assert len(params) >= 1
    
    def test_index_directory(self):
        """Test indexing a directory."""
        indexer = CodebaseIndexer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create main.nf
            main_nf = Path(tmpdir) / "main.nf"
            main_nf.write_text(SAMPLE_WORKFLOW)
            
            # Create modules directory
            modules_dir = Path(tmpdir) / "modules"
            modules_dir.mkdir()
            
            fastqc_nf = modules_dir / "fastqc.nf"
            fastqc_nf.write_text(SAMPLE_PROCESS)
            
            # Create config
            config = Path(tmpdir) / "nextflow.config"
            config.write_text(SAMPLE_CONFIG)
            
            # Index directory
            index = indexer.index_directory(tmpdir, name="test_pipeline")
            
            assert index.name == "test_pipeline"
            assert len(index.elements) > 0
            assert index.statistics["processes"] >= 1
            assert index.statistics["workflows"] >= 1
    
    def test_get_cached_index(self):
        """Test getting cached index."""
        indexer = CodebaseIndexer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            main_nf = Path(tmpdir) / "main.nf"
            main_nf.write_text(SAMPLE_PROCESS)
            
            # Index should be cached
            index1 = indexer.index_directory(tmpdir, name="cached_test")
            index2 = indexer.get_index("cached_test")
            
            assert index2 is not None
            assert index1 is index2
    
    def test_list_indices(self):
        """Test listing cached indices."""
        indexer = CodebaseIndexer()
        
        assert indexer.list_indices() == []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            main_nf = Path(tmpdir) / "main.nf"
            main_nf.write_text(SAMPLE_PROCESS)
            
            indexer.index_directory(tmpdir, name="index1")
            indexer.index_directory(tmpdir, name="index2")
            
            indices = indexer.list_indices()
            
            assert "index1" in indices
            assert "index2" in indices
    
    def test_format_for_prompt(self):
        """Test formatting index for LLM prompts."""
        indexer = CodebaseIndexer()
        
        elements = [
            CodeElement(
                element_type=CodeElementType.PROCESS,
                name="FASTQC",
                file_path="main.nf",
                line_number=1,
                content="process FASTQC { ... }",
                tools=["fastqc"],
                container="quay.io/biocontainers/fastqc",
            ),
        ]
        
        index = CodebaseIndex(
            name="test",
            root_path="/test",
            elements=elements,
            statistics={"processes": 1, "workflows": 0},
        )
        
        formatted = indexer.format_for_prompt(index)
        
        assert "Codebase Index: test" in formatted
        assert "FASTQC" in formatted
        assert "fastqc" in formatted
    
    def test_find_similar_processes(self):
        """Test finding similar processes."""
        indexer = CodebaseIndexer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            main_nf = Path(tmpdir) / "main.nf"
            main_nf.write_text(SAMPLE_PROCESS)
            
            indexer.index_directory(tmpdir, name="search_test")
            
            # Search for FastQC
            results = indexer.find_similar_processes("fastqc")
            
            assert len(results) >= 1
            assert results[0].name == "FASTQC"
    
    def test_get_tool_usage_examples(self):
        """Test getting tool usage examples."""
        indexer = CodebaseIndexer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            main_nf = Path(tmpdir) / "main.nf"
            main_nf.write_text(SAMPLE_PROCESS)
            
            indexer.index_directory(tmpdir, name="tool_test")
            
            # Get fastqc examples
            examples = indexer.get_tool_usage_examples("fastqc")
            
            assert len(examples) >= 1
            assert "fastqc" in examples[0].tools


class TestCodebaseIndexerPatterns:
    """Tests for Nextflow parsing patterns."""
    
    def test_process_pattern(self):
        """Test process regex pattern."""
        indexer = CodebaseIndexer()
        
        code = '''
process FASTQC {
    input:
    tuple val(meta), path(reads)
    
    output:
    path('*.html')
    
    script:
    """
    fastqc $reads
    """
}
'''
        
        matches = indexer.PATTERNS["process"].findall(code)
        assert len(matches) == 1
        assert matches[0][0] == "FASTQC"
    
    def test_container_pattern(self):
        """Test container regex pattern."""
        indexer = CodebaseIndexer()
        
        code = "container 'quay.io/biocontainers/fastqc:0.12.1'"
        
        match = indexer.PATTERNS["container"].search(code)
        assert match is not None
        assert match.group(1) == "quay.io/biocontainers/fastqc:0.12.1"
    
    def test_include_pattern(self):
        """Test include regex pattern."""
        indexer = CodebaseIndexer()
        
        code = "include { FASTQC } from './modules/fastqc'"
        
        match = indexer.PATTERNS["include"].search(code)
        assert match is not None
        assert match.group(1) == "FASTQC"
        assert match.group(2) == "./modules/fastqc"
    
    def test_tool_detection_pattern(self):
        """Test tool detection in scripts."""
        indexer = CodebaseIndexer()
        
        scripts = [
            ("fastqc $reads", ["fastqc"]),
            ("STAR --runMode alignReads", ["star"]),
            ("salmon quant -i index", ["salmon"]),
            ("samtools sort -o out.bam", ["samtools"]),
            ("macs2 callpeak -t treatment.bam", ["macs2"]),
        ]
        
        for script, expected_tools in scripts:
            matches = indexer.PATTERNS["tool_in_script"].findall(script)
            found_tools = [m.lower() for m in matches]
            for tool in expected_tools:
                assert tool in found_tools, f"Expected {tool} in {script}"


class TestCodebaseIndexerIntegration:
    """Integration tests for CodebaseIndexer."""
    
    def test_full_pipeline_indexing(self):
        """Test indexing a complete pipeline structure."""
        indexer = CodebaseIndexer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            # Create main.nf
            (base / "main.nf").write_text('''
nextflow.enable.dsl = 2

include { FASTQC } from './modules/fastqc/main'
include { STAR_ALIGN } from './modules/star/align/main'
include { SALMON_QUANT } from './modules/salmon/quant/main'

params.input = null
params.outdir = './results'
params.genome = 'GRCh38'

workflow RNASEQ {
    take:
    reads

    main:
    FASTQC(reads)
    STAR_ALIGN(reads, params.index)
    SALMON_QUANT(reads, params.salmon_index)

    emit:
    qc = FASTQC.out.html
    bam = STAR_ALIGN.out.bam
    counts = SALMON_QUANT.out.quant
}

workflow {
    reads_ch = Channel.fromFilePairs(params.input)
    RNASEQ(reads_ch)
}
''')
            
            # Create modules
            modules = base / "modules"
            
            # FastQC module
            (modules / "fastqc").mkdir(parents=True)
            (modules / "fastqc" / "main.nf").write_text('''
process FASTQC {
    container 'quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0'
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path('*.html'), emit: html
    path('versions.yml'), emit: versions
    
    script:
    """
    fastqc -t $task.cpus $reads
    """
}
''')
            
            # STAR module
            (modules / "star" / "align").mkdir(parents=True)
            (modules / "star" / "align" / "main.nf").write_text('''
process STAR_ALIGN {
    container 'quay.io/biocontainers/star:2.7.11a'
    
    input:
    tuple val(meta), path(reads)
    path(index)
    
    output:
    tuple val(meta), path('*.bam'), emit: bam
    path('versions.yml'), emit: versions
    
    script:
    """
    STAR --runMode alignReads --genomeDir $index --readFilesIn $reads
    samtools sort -o aligned.bam Aligned.out.bam
    """
}
''')
            
            # Salmon module
            (modules / "salmon" / "quant").mkdir(parents=True)
            (modules / "salmon" / "quant" / "main.nf").write_text('''
process SALMON_QUANT {
    container 'quay.io/biocontainers/salmon:1.10.2'
    
    input:
    tuple val(meta), path(reads)
    path(index)
    
    output:
    tuple val(meta), path('quant'), emit: quant
    path('versions.yml'), emit: versions
    
    script:
    """
    salmon quant -i $index -l A -1 ${reads[0]} -2 ${reads[1]} -o quant
    """
}
''')
            
            # Create config
            (base / "nextflow.config").write_text('''
params {
    input = null
    outdir = './results'
}

process {
    cpus = 4
    memory = '8 GB'
}
''')
            
            # Index the pipeline
            index = indexer.index_directory(tmpdir, name="rnaseq_pipeline")
            
            # Verify statistics
            assert index.statistics["processes"] == 3
            assert index.statistics["workflows"] >= 2
            assert index.statistics["imports"] >= 3
            assert index.statistics["unique_tools"] >= 3
            
            # Verify tools detected
            tools_found = set()
            for e in index.get_processes():
                tools_found.update(e.tools)
            
            assert "fastqc" in tools_found
            assert "star" in tools_found
            assert "salmon" in tools_found
            assert "samtools" in tools_found
            
            # Verify containers extracted
            containers = [e.container for e in index.get_processes() if e.container]
            assert len(containers) == 3
            assert any("fastqc" in c for c in containers)
            assert any("star" in c for c in containers)
            assert any("salmon" in c for c in containers)
