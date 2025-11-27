"""
Data Discovery Tab Component for BioPipelines Gradio UI.

This module provides the UI components for the data-first workflow:
1. Local sample discovery and scanning
2. Remote database search (ENCODE, GEO, Ensembl)
3. Reference genome management
4. Data summary and manifest visualization

The data tab is designed to be the FIRST step in workflow creation,
ensuring users have their data properly configured before tool selection.
"""

import gradio as gr
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
from datetime import datetime

# Import data infrastructure
try:
    from ...data.manifest import DataManifest, SampleInfo, ReferenceInfo
    from ...data.scanner import LocalSampleScanner
    from ...data.reference_manager import ReferenceManager
except ImportError:
    DataManifest = None
    LocalSampleScanner = None
    ReferenceManager = None

# Import discovery infrastructure
try:
    from ...data.discovery.orchestrator import DataDiscovery
    from ...data.discovery.models import SearchQuery, DataSource
except ImportError:
    DataDiscovery = None
    SearchQuery = None
    DataSource = None


# ============================================================================
# State Management
# ============================================================================

class DataTabState:
    """Manages state for the data tab across Gradio callbacks."""
    
    def __init__(self):
        self.manifest: Optional[DataManifest] = None
        self.scanner: Optional[LocalSampleScanner] = None
        self.reference_manager: Optional[ReferenceManager] = None
        self.discovery: Optional[DataDiscovery] = None
        self.last_scan_results: List[SampleInfo] = []
        self.last_search_results: List[Dict] = []
        
    def initialize(self, reference_base_path: str = "data/references"):
        """Initialize all data infrastructure."""
        if LocalSampleScanner:
            self.scanner = LocalSampleScanner()
        if ReferenceManager:
            self.reference_manager = ReferenceManager(base_dir=reference_base_path)
        if DataDiscovery:
            self.discovery = DataDiscovery()
        self.manifest = DataManifest() if DataManifest else None
        
    def get_manifest_dict(self) -> Dict[str, Any]:
        """Get manifest as dictionary for display."""
        if self.manifest:
            return self.manifest.to_dict()
        return {}


# Global state instance
_data_state = DataTabState()


# ============================================================================
# Local Scanner UI
# ============================================================================

def create_local_scanner_ui() -> Tuple[gr.Column, Dict[str, gr.Component]]:
    """
    Create the local sample scanner UI component.
    
    Returns:
        Tuple of (column, component_dict) for integration
    """
    components = {}
    
    with gr.Column() as scanner_col:
        gr.Markdown("""
        ## 📁 Local Sample Discovery
        
        Scan a directory containing your sequencing data (FASTQ files).
        The scanner will automatically:
        - Detect paired-end reads (R1/R2)
        - Infer sample names from filenames
        - Identify file formats and compression
        - Group samples by condition if naming patterns detected
        """)
        
        with gr.Row():
            components["scan_path"] = gr.Textbox(
                label="📂 Directory Path",
                placeholder="/path/to/your/fastq/files",
                info="Enter the full path to the directory containing your FASTQ files",
                scale=4
            )
            components["scan_btn"] = gr.Button(
                "🔍 Scan Directory",
                variant="primary",
                scale=1
            )
        
        with gr.Row():
            components["recursive"] = gr.Checkbox(
                label="Scan subdirectories",
                value=True,
                info="Include files in nested folders"
            )
            components["pattern_filter"] = gr.Textbox(
                label="Pattern filter (optional)",
                placeholder="*.fastq.gz",
                info="Filter files by pattern"
            )
        
        with gr.Accordion("🔧 Advanced Options", open=False):
            with gr.Row():
                components["min_file_size"] = gr.Number(
                    label="Min file size (MB)",
                    value=1,
                    minimum=0,
                    info="Ignore files smaller than this"
                )
                components["max_samples"] = gr.Number(
                    label="Max samples",
                    value=1000,
                    minimum=1,
                    info="Maximum samples to display"
                )
        
        # Results section
        gr.Markdown("### 📊 Scan Results")
        
        components["scan_status"] = gr.Textbox(
            label="Status",
            interactive=False,
            visible=True
        )
        
        components["sample_table"] = gr.Dataframe(
            headers=["Sample", "Read 1", "Read 2", "Organism", "Paired", "Size (MB)"],
            label="Discovered Samples",
            interactive=False,
            wrap=True
        )
        
        with gr.Row():
            components["add_selected_btn"] = gr.Button(
                "✅ Add to Manifest",
                variant="secondary"
            )
            components["clear_scan_btn"] = gr.Button(
                "🗑️ Clear Results"
            )
    
    return scanner_col, components


def scan_directory_handler(
    path: str,
    recursive: bool,
    pattern: str,
    min_size: float,
    max_samples: int
) -> Tuple[str, List[List[Any]]]:
    """Handle directory scan request."""
    if not path or not Path(path).exists():
        return "❌ Invalid or non-existent path", []
    
    if not _data_state.scanner:
        _data_state.initialize()
    
    if not _data_state.scanner:
        return "❌ Scanner not available. Check installation.", []
    
    try:
        # Perform scan - returns ScanResult object
        result = _data_state.scanner.scan_directory(
            path=Path(path),
            recursive=recursive,
            pattern=pattern if pattern else None
        )
        
        samples = result.samples
        
        # Filter by size (size_bytes to MB conversion)
        if min_size > 0:
            min_size_bytes = min_size * 1024 * 1024
            samples = [s for s in samples if s.size_bytes >= min_size_bytes]
        
        # Limit results
        samples = samples[:int(max_samples)]
        
        # Store results
        _data_state.last_scan_results = samples
        
        # Format for display
        table_data = []
        for sample in samples:
            size_mb = sample.size_bytes / (1024 * 1024) if sample.size_bytes else 0
            table_data.append([
                sample.sample_id,
                sample.fastq_1.name if sample.fastq_1 else "-",
                sample.fastq_2.name if sample.fastq_2 else "-",
                sample.metadata.get("organism", "Unknown"),
                "✓" if sample.is_paired else "✗",
                f"{size_mb:.1f}" if size_mb else "-"
            ])
        
        # Build status message
        status = f"✅ Found {len(samples)} samples in {path}"
        if result.unpaired_files:
            status += f" ({len(result.unpaired_files)} unpaired files)"
        if result.warnings:
            status += f" ⚠️ {len(result.warnings)} warnings"
        
        return status, table_data
        
    except Exception as e:
        return f"❌ Scan failed: {str(e)}", []


def add_samples_to_manifest() -> str:
    """Add scanned samples to the manifest."""
    if not _data_state.last_scan_results:
        return "⚠️ No samples to add. Scan a directory first."
    
    if not _data_state.manifest:
        _data_state.manifest = DataManifest() if DataManifest else None
        
    if not _data_state.manifest:
        return "❌ Manifest not available."
    
    added = 0
    for sample in _data_state.last_scan_results:
        try:
            _data_state.manifest.add_sample(sample)
            added += 1
        except Exception as e:
            pass  # Skip duplicates or invalid samples
    
    return f"✅ Added {added} samples to manifest. Total: {len(_data_state.manifest.samples)}"


# ============================================================================
# Remote Search UI
# ============================================================================

def create_remote_search_ui() -> Tuple[gr.Column, Dict[str, gr.Component]]:
    """
    Create the remote database search UI component.
    
    Returns:
        Tuple of (column, component_dict) for integration
    """
    components = {}
    
    with gr.Column() as search_col:
        gr.Markdown("""
        ## 🌐 Remote Database Search
        
        Search public databases for sequencing datasets:
        - **ENCODE**: Functional genomics data
        - **GEO**: Gene expression datasets
        - **Ensembl**: Reference genomes and annotations
        """)
        
        with gr.Row():
            components["search_query"] = gr.Textbox(
                label="🔍 Search Query",
                placeholder="e.g., 'H3K27ac ChIP-seq human GM12878'",
                info="Describe the data you're looking for",
                scale=4
            )
            components["search_btn"] = gr.Button(
                "Search",
                variant="primary",
                scale=1
            )
        
        with gr.Row():
            components["source_encode"] = gr.Checkbox(
                label="ENCODE",
                value=True
            )
            components["source_geo"] = gr.Checkbox(
                label="GEO",
                value=True
            )
            components["source_ensembl"] = gr.Checkbox(
                label="Ensembl",
                value=False
            )
        
        with gr.Accordion("🎯 Filter Options", open=False):
            with gr.Row():
                components["organism_filter"] = gr.Dropdown(
                    label="Organism",
                    choices=["Any", "Homo sapiens", "Mus musculus", "Drosophila melanogaster",
                             "Caenorhabditis elegans", "Danio rerio", "Arabidopsis thaliana"],
                    value="Any"
                )
                components["assay_filter"] = gr.Dropdown(
                    label="Assay Type",
                    choices=["Any", "RNA-seq", "ChIP-seq", "ATAC-seq", "DNase-seq",
                             "Hi-C", "WGBS", "RRBS", "WGS", "WES"],
                    value="Any"
                )
            with gr.Row():
                components["max_results"] = gr.Slider(
                    label="Max Results",
                    minimum=10,
                    maximum=100,
                    value=25,
                    step=5
                )
        
        # Results section
        gr.Markdown("### 📋 Search Results")
        
        components["search_status"] = gr.Textbox(
            label="Status",
            interactive=False
        )
        
        components["result_table"] = gr.Dataframe(
            headers=["Source", "Accession", "Title", "Organism", "Assay", "Files"],
            label="Available Datasets",
            interactive=False,
            wrap=True
        )
        
        with gr.Row():
            components["download_selected_btn"] = gr.Button(
                "📥 Download Selected",
                variant="secondary"
            )
            components["preview_btn"] = gr.Button(
                "👁️ Preview Metadata"
            )
    
    return search_col, components


def search_databases_handler(
    query: str,
    use_encode: bool,
    use_geo: bool,
    use_ensembl: bool,
    organism: str,
    assay: str,
    max_results: int
) -> Tuple[str, List[List[Any]]]:
    """Handle database search request."""
    if not query:
        return "⚠️ Please enter a search query", []
    
    if not _data_state.discovery:
        _data_state.initialize()
    
    if not _data_state.discovery:
        return "❌ Data discovery not available. Check installation.", []
    
    # Build source list
    sources = []
    if use_encode:
        sources.append("encode")
    if use_geo:
        sources.append("geo")
    if use_ensembl:
        sources.append("ensembl")
    
    if not sources:
        return "⚠️ Select at least one data source", []
    
    try:
        # Build enhanced query with filters
        enhanced_query = query
        if organism and organism != "Any":
            enhanced_query += f" {organism}"
        if assay and assay != "Any":
            enhanced_query += f" {assay}"
        
        # Perform search - returns SearchResults object
        search_results = _data_state.discovery.search(
            query=enhanced_query,
            sources=sources,
            max_results=int(max_results)
        )
        
        # Store results (the datasets list)
        _data_state.last_search_results = search_results.datasets
        
        # Format for display
        table_data = []
        for dataset in search_results.datasets:
            # DatasetInfo attributes: id, source, title, description, organism, experiment_type, etc.
            source_str = dataset.source.value if hasattr(dataset.source, 'value') else str(dataset.source)
            title_str = dataset.title or dataset.description or "-"
            if len(title_str) > 60:
                title_str = title_str[:60] + "..."
            
            table_data.append([
                source_str,
                dataset.id or "-",
                title_str,
                dataset.organism or "-",
                dataset.assay_type or "-",
                str(dataset.file_count) if dataset.file_count else str(len(dataset.download_urls)) if dataset.download_urls else "-"
            ])
        
        status = f"✅ Found {search_results.total_count} datasets matching '{query}'"
        if search_results.errors:
            status += f" ⚠️ {len(search_results.errors)} errors"
        
        return status, table_data
        
    except Exception as e:
        return f"❌ Search failed: {str(e)}", []


# ============================================================================
# Reference Manager UI
# ============================================================================

def create_reference_manager_ui() -> Tuple[gr.Column, Dict[str, gr.Component]]:
    """
    Create the reference genome manager UI component.
    
    Returns:
        Tuple of (column, component_dict) for integration
    """
    components = {}
    
    with gr.Column() as ref_col:
        gr.Markdown("""
        ## 🧬 Reference Genome Manager
        
        Manage reference genomes, annotations, and alignment indexes.
        Required for alignment-based workflows.
        """)
        
        with gr.Row():
            components["organism_select"] = gr.Dropdown(
                label="🧫 Organism",
                choices=[
                    "Homo sapiens (human)",
                    "Mus musculus (mouse)",
                    "Drosophila melanogaster (fruit fly)",
                    "Caenorhabditis elegans (worm)",
                    "Danio rerio (zebrafish)",
                    "Saccharomyces cerevisiae (yeast)",
                    "Arabidopsis thaliana (plant)",
                ],
                value="Homo sapiens (human)",
                scale=2
            )
            components["genome_build"] = gr.Dropdown(
                label="📋 Genome Build",
                choices=["GRCh38", "GRCh37/hg19", "T2T-CHM13"],
                value="GRCh38",
                scale=1
            )
            components["check_ref_btn"] = gr.Button(
                "🔍 Check Status",
                scale=1
            )
        
        # Reference status display
        gr.Markdown("### 📊 Reference Status")
        
        with gr.Row():
            with gr.Column(scale=1):
                components["genome_status"] = gr.Textbox(
                    label="Genome FASTA",
                    value="Not checked",
                    interactive=False
                )
            with gr.Column(scale=1):
                components["annotation_status"] = gr.Textbox(
                    label="GTF Annotation",
                    value="Not checked",
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                components["bwa_status"] = gr.Textbox(
                    label="BWA Index",
                    value="Not checked",
                    interactive=False
                )
            with gr.Column(scale=1):
                components["star_status"] = gr.Textbox(
                    label="STAR Index",
                    value="Not checked",
                    interactive=False
                )
        
        gr.Markdown("### 📥 Download & Build")
        
        with gr.Row():
            components["component_select"] = gr.CheckboxGroup(
                label="Components to Download/Build",
                choices=["Genome FASTA", "GTF Annotation", "BWA Index", "STAR Index",
                         "Bowtie2 Index", "HISAT2 Index"],
                value=["Genome FASTA", "GTF Annotation"]
            )
        
        with gr.Row():
            components["download_btn"] = gr.Button(
                "📥 Download Selected",
                variant="primary"
            )
            components["build_index_btn"] = gr.Button(
                "🔨 Build Indexes",
                variant="secondary"
            )
        
        components["download_progress"] = gr.Textbox(
            label="Progress",
            interactive=False,
            lines=3
        )
        
        with gr.Accordion("🔧 Advanced: Custom Reference", open=False):
            with gr.Row():
                components["custom_genome"] = gr.Textbox(
                    label="Custom Genome Path",
                    placeholder="/path/to/genome.fa"
                )
                components["custom_annotation"] = gr.Textbox(
                    label="Custom Annotation Path",
                    placeholder="/path/to/genes.gtf"
                )
            components["register_custom_btn"] = gr.Button(
                "Register Custom Reference"
            )
    
    return ref_col, components


def check_reference_status_handler(
    organism: str,
    genome_build: str
) -> Tuple[str, str, str, str]:
    """Check the status of reference files for the selected organism/build."""
    if not _data_state.reference_manager:
        _data_state.initialize()
    
    if not _data_state.reference_manager:
        return "❌ Not available", "❌ Not available", "❌ Not available", "❌ Not available"
    
    # Parse organism name (e.g., "Homo sapiens (human)" -> "human")
    org_key = organism.split("(")[0].strip().lower().replace(" ", "_")
    
    try:
        # check_references returns a ReferenceInfo object
        ref_info = _data_state.reference_manager.check_references(
            organism=org_key,
            assembly=genome_build
        )
        
        # ReferenceInfo has attributes for paths (None if not found)
        genome_st = "✅ Available" if ref_info.genome_fasta else "❌ Not found"
        annot_st = "✅ Available" if ref_info.annotation_gtf else "❌ Not found"
        bwa_st = "✅ Available" if ref_info.bwa_index else "⚪ Not built"
        star_st = "✅ Available" if ref_info.star_index else "⚪ Not built"
        
        return genome_st, annot_st, bwa_st, star_st
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "❌ Error", "❌ Error", "❌ Error"


# ============================================================================
# Data Summary Panel
# ============================================================================

def create_data_summary_panel() -> Tuple[gr.Column, Dict[str, gr.Component]]:
    """
    Create a summary panel showing current data manifest status.
    
    Returns:
        Tuple of (column, component_dict) for integration
    """
    components = {}
    
    with gr.Column() as summary_col:
        gr.Markdown("""
        ## 📋 Data Manifest Summary
        
        Overview of your configured samples and references.
        This data will be used for workflow generation.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                components["sample_count"] = gr.Number(
                    label="Total Samples",
                    value=0,
                    interactive=False
                )
            with gr.Column(scale=1):
                components["paired_count"] = gr.Number(
                    label="Paired-End",
                    value=0,
                    interactive=False
                )
            with gr.Column(scale=1):
                components["organisms"] = gr.Textbox(
                    label="Organisms",
                    value="-",
                    interactive=False
                )
        
        with gr.Row():
            components["reference_info"] = gr.Textbox(
                label="Reference Genome",
                value="Not configured",
                interactive=False
            )
        
        components["manifest_json"] = gr.JSON(
            label="Manifest Details",
            visible=False
        )
        
        with gr.Row():
            components["refresh_btn"] = gr.Button(
                "🔄 Refresh",
                size="sm"
            )
            components["export_btn"] = gr.Button(
                "📤 Export Manifest",
                size="sm"
            )
            components["clear_manifest_btn"] = gr.Button(
                "🗑️ Clear All",
                size="sm",
                variant="stop"
            )
    
    return summary_col, components


def refresh_manifest_summary() -> Tuple[int, int, str, str, Dict]:
    """Refresh the manifest summary display."""
    if not _data_state.manifest:
        return 0, 0, "-", "Not configured", {}
    
    m = _data_state.manifest
    sample_count = len(m.samples)
    paired_count = sum(1 for s in m.samples if s.is_paired)
    
    # Get unique organisms from sample metadata
    organisms = set()
    for s in m.samples:
        if hasattr(s, 'metadata') and s.metadata.get('organism'):
            organisms.add(s.metadata['organism'])
    organisms_str = ", ".join(organisms) if organisms else "-"
    
    # Reference info
    if m.reference:
        ref_str = f"{m.reference.organism} ({m.reference.assembly})"
    else:
        ref_str = "Not configured"
    
    return sample_count, paired_count, organisms_str, ref_str, m.to_dict()


def export_manifest() -> str:
    """Export manifest to JSON file."""
    if not _data_state.manifest:
        return "⚠️ No manifest to export"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_manifest_{timestamp}.json"
    
    try:
        manifest_dict = _data_state.manifest.to_dict()
        with open(filename, "w") as f:
            json.dump(manifest_dict, f, indent=2)
        return f"✅ Exported to {filename}"
    except Exception as e:
        return f"❌ Export failed: {str(e)}"


def clear_manifest() -> Tuple[int, int, str, str, Dict]:
    """Clear all manifest data."""
    _data_state.manifest = DataManifest() if DataManifest else None
    _data_state.last_scan_results = []
    _data_state.last_search_results = []
    return 0, 0, "-", "Not configured", {}


# ============================================================================
# Main Data Tab Creator
# ============================================================================

def create_data_tab() -> Tuple[gr.Tab, Dict[str, gr.Component]]:
    """
    Create the complete Data Discovery tab for the Gradio UI.
    
    This is the main entry point for integrating the data tab into
    the BioPipelines application. It combines all sub-components:
    - Local sample scanner
    - Remote database search
    - Reference genome manager
    - Data summary panel
    
    Returns:
        Tuple of (Tab component, dictionary of all interactive components)
    """
    all_components = {}
    
    # Initialize state
    _data_state.initialize()
    
    with gr.Tab("📁 Data") as data_tab:
        gr.Markdown("""
        # 🧬 Data Discovery & Configuration
        
        **Start here!** Configure your input data before generating workflows.
        This ensures your workflow will have the correct input paths and reference files.
        
        ---
        """)
        
        # Summary panel at top
        summary_col, summary_components = create_data_summary_panel()
        all_components.update({f"summary_{k}": v for k, v in summary_components.items()})
        
        gr.Markdown("---")
        
        # Tabbed interface for different data sources
        with gr.Tabs():
            with gr.Tab("📁 Local Files"):
                scanner_col, scanner_components = create_local_scanner_ui()
                all_components.update({f"scanner_{k}": v for k, v in scanner_components.items()})
            
            with gr.Tab("🌐 Remote Databases"):
                search_col, search_components = create_remote_search_ui()
                all_components.update({f"search_{k}": v for k, v in search_components.items()})
            
            with gr.Tab("🧬 Reference Genomes"):
                ref_col, ref_components = create_reference_manager_ui()
                all_components.update({f"ref_{k}": v for k, v in ref_components.items()})
        
        # Wire up event handlers
        _wire_event_handlers(all_components)
    
    return data_tab, all_components


def _wire_event_handlers(components: Dict[str, gr.Component]):
    """Wire up all event handlers for the data tab components."""
    
    # Scanner handlers
    if "scanner_scan_btn" in components:
        components["scanner_scan_btn"].click(
            fn=scan_directory_handler,
            inputs=[
                components["scanner_scan_path"],
                components["scanner_recursive"],
                components["scanner_pattern_filter"],
                components["scanner_min_file_size"],
                components["scanner_max_samples"]
            ],
            outputs=[
                components["scanner_scan_status"],
                components["scanner_sample_table"]
            ]
        )
    
    if "scanner_add_selected_btn" in components:
        components["scanner_add_selected_btn"].click(
            fn=add_samples_to_manifest,
            inputs=[],
            outputs=[components["scanner_scan_status"]]
        ).then(
            fn=refresh_manifest_summary,
            inputs=[],
            outputs=[
                components["summary_sample_count"],
                components["summary_paired_count"],
                components["summary_organisms"],
                components["summary_reference_info"],
                components["summary_manifest_json"]
            ]
        )
    
    if "scanner_clear_scan_btn" in components:
        components["scanner_clear_scan_btn"].click(
            fn=lambda: ("Cleared", []),
            inputs=[],
            outputs=[
                components["scanner_scan_status"],
                components["scanner_sample_table"]
            ]
        )
    
    # Search handlers
    if "search_search_btn" in components:
        components["search_search_btn"].click(
            fn=search_databases_handler,
            inputs=[
                components["search_search_query"],
                components["search_source_encode"],
                components["search_source_geo"],
                components["search_source_ensembl"],
                components["search_organism_filter"],
                components["search_assay_filter"],
                components["search_max_results"]
            ],
            outputs=[
                components["search_search_status"],
                components["search_result_table"]
            ]
        )
    
    # Reference handlers
    if "ref_check_ref_btn" in components:
        components["ref_check_ref_btn"].click(
            fn=check_reference_status_handler,
            inputs=[
                components["ref_organism_select"],
                components["ref_genome_build"]
            ],
            outputs=[
                components["ref_genome_status"],
                components["ref_annotation_status"],
                components["ref_bwa_status"],
                components["ref_star_status"]
            ]
        )
    
    # Summary handlers
    if "summary_refresh_btn" in components:
        components["summary_refresh_btn"].click(
            fn=refresh_manifest_summary,
            inputs=[],
            outputs=[
                components["summary_sample_count"],
                components["summary_paired_count"],
                components["summary_organisms"],
                components["summary_reference_info"],
                components["summary_manifest_json"]
            ]
        )
    
    if "summary_export_btn" in components:
        components["summary_export_btn"].click(
            fn=export_manifest,
            inputs=[],
            outputs=[components["summary_reference_info"]]  # Reuse for status
        )
    
    if "summary_clear_manifest_btn" in components:
        components["summary_clear_manifest_btn"].click(
            fn=clear_manifest,
            inputs=[],
            outputs=[
                components["summary_sample_count"],
                components["summary_paired_count"],
                components["summary_organisms"],
                components["summary_reference_info"],
                components["summary_manifest_json"]
            ]
        )


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    """Run standalone for testing."""
    with gr.Blocks(title="Data Discovery Tab Test") as demo:
        data_tab, components = create_data_tab()
    
    demo.launch(share=False)
