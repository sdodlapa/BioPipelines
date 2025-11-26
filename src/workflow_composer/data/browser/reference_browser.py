"""
Reference Browser UI Component
==============================

Gradio UI component for browsing and downloading genomics data.

Features:
- Natural language search
- Multi-source browsing (ENCODE, GEO, Ensembl)
- Download queue management
- Local reference validation

Usage in Gradio app:
    from workflow_composer.data.browser import create_reference_browser_tab
    
    with gr.Blocks() as app:
        with gr.Tabs():
            with gr.Tab("Reference Browser"):
                create_reference_browser_tab()
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from ..discovery import (
    DataDiscovery, SearchResults, DatasetInfo,
    SearchQuery, DataSource, LocalReference
)

logger = logging.getLogger(__name__)

# Initialize discovery instance
_discovery = None


def get_discovery() -> DataDiscovery:
    """Get or create the discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = DataDiscovery()
    return _discovery


def search_datasets(
    query: str,
    source_filter: str,
    max_results: int
) -> Tuple[str, List[List[str]]]:
    """
    Search for datasets across sources.
    
    Returns:
        Tuple of (status message, results table data)
    """
    if not query.strip():
        return "Please enter a search query", []
    
    discovery = get_discovery()
    
    # Determine sources
    sources = None
    if source_filter and source_filter != "All":
        sources = [source_filter.lower()]
    
    try:
        results = discovery.search(
            query,
            sources=sources,
            max_results=max_results
        )
        
        # Format results as table rows
        table_data = []
        for dataset in results.datasets:
            table_data.append([
                dataset.source.value.upper(),
                dataset.id,
                dataset.title[:60] + "..." if len(dataset.title) > 60 else dataset.title,
                dataset.organism or "-",
                dataset.assay_type or "-",
                dataset.tissue or dataset.cell_line or "-",
                f"{len(dataset.download_urls)} files" if dataset.download_urls else "-",
                dataset.web_url or "-",
            ])
        
        status = (
            f"Found {results.total_count} datasets in {results.search_time_ms:.0f}ms "
            f"(showing {len(results.datasets)})"
        )
        
        if results.errors:
            status += f"\n⚠️ Errors: {', '.join(results.errors)}"
        
        return status, table_data
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {e}", []


def get_dataset_details(dataset_id: str, source: str) -> str:
    """Get detailed information about a dataset."""
    if not dataset_id:
        return "Select a dataset to view details"
    
    discovery = get_discovery()
    
    try:
        source_enum = DataSource(source.lower()) if source else None
        dataset = discovery.get_dataset(dataset_id, source_enum)
        
        if not dataset:
            return f"Dataset {dataset_id} not found"
        
        # Format details
        details = f"""
## {dataset.title}

**ID:** {dataset.id}  
**Source:** {dataset.source.value.upper()}  
**Organism:** {dataset.organism or 'Not specified'}  
**Assembly:** {dataset.assembly or 'Not specified'}  
**Assay Type:** {dataset.assay_type or 'Not specified'}  
**Target:** {dataset.target or 'Not specified'}  
**Tissue:** {dataset.tissue or 'Not specified'}  
**Cell Line:** {dataset.cell_line or 'Not specified'}  

### Description
{dataset.description or 'No description available'}

### Files
"""
        
        if dataset.download_urls:
            for url in dataset.download_urls[:10]:
                details += f"- `{url.filename}` ({url.file_type.value}) - {url.size_human}\n"
            if len(dataset.download_urls) > 10:
                details += f"\n... and {len(dataset.download_urls) - 10} more files\n"
        else:
            details += "No files available\n"
        
        details += f"\n**Web URL:** [{dataset.web_url}]({dataset.web_url})\n"
        
        return details
        
    except Exception as e:
        logger.error(f"Failed to get details for {dataset_id}: {e}")
        return f"Error loading details: {e}"


def download_dataset(
    dataset_id: str,
    source: str,
    output_dir: str,
    file_types: List[str]
) -> str:
    """Start downloading a dataset."""
    if not dataset_id:
        return "No dataset selected"
    
    if not output_dir:
        return "Please specify an output directory"
    
    discovery = get_discovery()
    
    try:
        source_enum = DataSource(source.lower()) if source else None
        dataset = discovery.get_dataset(dataset_id, source_enum)
        
        if not dataset:
            return f"Dataset {dataset_id} not found"
        
        job = discovery.download(
            dataset,
            output_dir,
            file_types=file_types if file_types else None
        )
        
        return f"""
## Download Started

**Job ID:** {job.id}
**Dataset:** {dataset.id}
**Output:** {output_dir}
**Status:** {job.status}
**Files:** {len(job.downloaded_files)} downloaded, {len(job.failed_files)} failed

Check the output directory for downloaded files.
"""
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return f"Download failed: {e}"


def scan_local_references(ref_dir: str) -> Tuple[str, List[List[str]]]:
    """Scan local directory for reference files."""
    if not ref_dir:
        return "Please specify a reference directory", []
    
    ref_path = Path(ref_dir)
    if not ref_path.exists():
        return f"Directory does not exist: {ref_dir}", []
    
    try:
        results = []
        
        # Scan for reference files
        patterns = [
            ("*.fa", "genome"),
            ("*.fasta", "genome"),
            ("*.fa.gz", "genome"),
            ("*.gtf", "annotation"),
            ("*.gtf.gz", "annotation"),
            ("*.gff", "annotation"),
            ("*.gff3", "annotation"),
        ]
        
        for pattern, ref_type in patterns:
            for f in ref_path.rglob(pattern):
                size = f.stat().st_size
                size_str = format_size(size)
                
                results.append([
                    f.name,
                    ref_type,
                    str(f.relative_to(ref_path)),
                    size_str,
                    "✓" if size > 0 else "✗",
                ])
        
        # Scan for index directories
        index_indicators = {
            "STAR": ["Genome", "SA", "SAindex"],
            "bowtie2": ["*.bt2"],
            "bwa": ["*.bwt", "*.sa"],
            "salmon": ["info.json", "seq.bin"],
        }
        
        for aligner, indicators in index_indicators.items():
            for d in ref_path.rglob("*"):
                if d.is_dir():
                    # Check if directory looks like an index
                    if any(list(d.glob(ind)) for ind in indicators):
                        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                        results.append([
                            d.name,
                            f"index ({aligner})",
                            str(d.relative_to(ref_path)),
                            format_size(size),
                            "✓",
                        ])
        
        status = f"Found {len(results)} reference files/directories"
        return status, results
        
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        return f"Scan failed: {e}", []


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def create_reference_browser_tab():
    """
    Create the reference browser Gradio tab.
    
    Returns Gradio components that should be added to a Tab.
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required for the reference browser UI")
    
    with gr.Column():
        gr.Markdown("# 🔬 Data Discovery & Reference Browser")
        gr.Markdown(
            "Search for genomics data across ENCODE, GEO/SRA, and Ensembl. "
            "Use natural language queries like 'human liver ChIP-seq H3K27ac' "
            "or 'mouse RNA-seq brain tissue'."
        )
        
        with gr.Tabs():
            # Tab 1: Search
            with gr.Tab("🔍 Search Databases"):
                with gr.Row():
                    with gr.Column(scale=3):
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., human liver ChIP-seq H3K27ac, mouse scRNA-seq brain",
                            lines=1,
                        )
                    with gr.Column(scale=1):
                        source_dropdown = gr.Dropdown(
                            label="Source",
                            choices=["All", "ENCODE", "GEO", "Ensembl"],
                            value="All",
                        )
                    with gr.Column(scale=1):
                        max_results_slider = gr.Slider(
                            label="Max Results",
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=10,
                        )
                
                search_btn = gr.Button("🔍 Search", variant="primary")
                
                search_status = gr.Textbox(label="Status", interactive=False)
                
                results_table = gr.Dataframe(
                    headers=["Source", "ID", "Title", "Organism", "Assay", "Sample", "Files", "URL"],
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                    label="Search Results",
                    wrap=True,
                    interactive=False,
                )
                
                # Dataset details section
                gr.Markdown("### Dataset Details")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        selected_id = gr.Textbox(
                            label="Dataset ID",
                            placeholder="Enter dataset ID (e.g., ENCSR000ABC, GSE12345)",
                        )
                        selected_source = gr.Dropdown(
                            label="Source",
                            choices=["ENCODE", "GEO", "SRA", "Ensembl"],
                            value="ENCODE",
                        )
                        get_details_btn = gr.Button("Get Details")
                    
                    with gr.Column(scale=2):
                        details_output = gr.Markdown("Select a dataset to view details")
                
                # Search button handler
                search_btn.click(
                    fn=search_datasets,
                    inputs=[search_input, source_dropdown, max_results_slider],
                    outputs=[search_status, results_table],
                )
                
                # Details button handler
                get_details_btn.click(
                    fn=get_dataset_details,
                    inputs=[selected_id, selected_source],
                    outputs=[details_output],
                )
            
            # Tab 2: Download
            with gr.Tab("📥 Download"):
                gr.Markdown("### Download Dataset")
                
                with gr.Row():
                    download_id = gr.Textbox(
                        label="Dataset ID",
                        placeholder="e.g., ENCSR000ABC",
                    )
                    download_source = gr.Dropdown(
                        label="Source",
                        choices=["ENCODE", "GEO", "SRA", "Ensembl"],
                        value="ENCODE",
                    )
                
                output_dir = gr.Textbox(
                    label="Output Directory",
                    placeholder="/path/to/downloads",
                    value="./data/downloads",
                )
                
                file_type_filter = gr.CheckboxGroup(
                    label="File Types (leave empty for all)",
                    choices=["fastq", "fastq.gz", "bam", "bed", "bigwig", "gtf", "fasta"],
                )
                
                download_btn = gr.Button("📥 Download", variant="primary")
                
                download_output = gr.Markdown("Select a dataset and click Download")
                
                download_btn.click(
                    fn=download_dataset,
                    inputs=[download_id, download_source, output_dir, file_type_filter],
                    outputs=[download_output],
                )
            
            # Tab 3: Local References
            with gr.Tab("📁 Local References"):
                gr.Markdown("### Scan Local Reference Files")
                gr.Markdown(
                    "Scan a directory to find reference genomes, annotations, "
                    "and aligner indexes."
                )
                
                ref_dir_input = gr.Textbox(
                    label="Reference Directory",
                    placeholder="/path/to/references",
                    value="./data/references",
                )
                
                scan_btn = gr.Button("🔍 Scan Directory", variant="secondary")
                
                scan_status = gr.Textbox(label="Status", interactive=False)
                
                local_refs_table = gr.Dataframe(
                    headers=["Name", "Type", "Path", "Size", "Valid"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="Local References",
                    wrap=True,
                )
                
                scan_btn.click(
                    fn=scan_local_references,
                    inputs=[ref_dir_input],
                    outputs=[scan_status, local_refs_table],
                )
            
            # Tab 4: Quick Reference
            with gr.Tab("📚 Quick Reference"):
                gr.Markdown("""
### Common Data Sources

| Source | Best For | Example Query |
|--------|----------|---------------|
| **ENCODE** | ChIP-seq, ATAC-seq, Hi-C | `human H3K27ac ChIP-seq liver` |
| **GEO/SRA** | RNA-seq, scRNA-seq, all types | `mouse brain scRNA-seq` |
| **Ensembl** | Reference genomes, GTF annotations | `human genome GRCh38` |

### Query Tips

- **Organism**: human, mouse, rat, zebrafish, fly, worm, yeast
- **Assay types**: RNA-seq, ChIP-seq, ATAC-seq, scRNA-seq, Hi-C, WGBS, WGS
- **ChIP targets**: H3K27ac, H3K4me3, H3K27me3, CTCF, Pol2
- **Cell lines**: K562, HeLa, GM12878, HepG2, A549, MCF7

### Download Methods

| Data Type | Method | Notes |
|-----------|--------|-------|
| ENCODE | HTTPS | Direct download |
| GEO supplementary | FTP | TAR archives |
| SRA (SRR*) | SRA Toolkit | Use `fasterq-dump` |
| Ensembl | HTTPS/FTP | GZIP compressed |
                """)
    
    return {
        "search_input": search_input,
        "search_btn": search_btn,
        "results_table": results_table,
    }
