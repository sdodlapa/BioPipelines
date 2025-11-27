"""
Data Management Module
======================

Tools for downloading, managing, and discovering reference data.

Includes:
- DataDownloader: Download reference genomes, annotations, and indexes
- Discovery: LLM-powered data discovery across ENCODE, GEO, Ensembl
- Browser: Gradio UI components for data browsing
- Manifest: Data manifest for tracking samples and references
- Scanner: Local file system scanner for FASTQ/BAM files
- ReferenceManager: Unified reference genome management

Quick Start:
    # Download references
    from workflow_composer.data import DataDownloader
    downloader = DataDownloader()
    genome = downloader.get_genome("human")
    
    # Discover data
    from workflow_composer.data import DataDiscovery
    discovery = DataDiscovery()
    results = discovery.search("human liver ChIP-seq H3K27ac")
    
    # Scan local files
    from workflow_composer.data import LocalSampleScanner, DataManifest
    scanner = LocalSampleScanner()
    samples = scanner.scan_directory("/path/to/fastq")
    manifest = DataManifest.from_scan(samples)
"""

from .downloader import (
    DataDownloader,
    DownloadedFile,
    Reference,
    REFERENCE_SOURCES,
    INDEX_SOURCES,
    SAMPLE_DATASETS
)

# Import discovery module components
from .discovery import (
    # Main orchestrator
    DataDiscovery,
    
    # Query parsing
    QueryParser,
    parse_query,
    
    # Data models
    DataSource,
    SearchQuery,
    DatasetInfo,
    SearchResults,
    
    # Convenience functions
    quick_search,
    search_encode,
    search_geo,
    search_references,
)

# Import new data-first components
from .manifest import (
    DataManifest,
    SampleInfo,
    ReferenceInfo,
    DataSourceType,
    LibraryLayout,
)

from .scanner import (
    LocalSampleScanner,
)

from .reference_manager import (
    ReferenceManager,
)

# Import browser components (optional, requires gradio)
try:
    from .browser import create_reference_browser_tab
except ImportError:
    create_reference_browser_tab = None

__all__ = [
    # Downloader
    "DataDownloader",
    "DownloadedFile", 
    "Reference",
    "REFERENCE_SOURCES",
    "INDEX_SOURCES",
    "SAMPLE_DATASETS",
    
    # Discovery
    "DataDiscovery",
    "QueryParser",
    "parse_query",
    "DataSource",
    "SearchQuery",
    "DatasetInfo",
    "SearchResults",
    "quick_search",
    "search_encode",
    "search_geo",
    "search_references",
    
    # Data Manifest (data-first workflow)
    "DataManifest",
    "SampleInfo",
    "ReferenceInfo",
    "DataSourceType",
    "LibraryLayout",
    
    # Local Scanner
    "LocalSampleScanner",
    
    # Reference Manager
    "ReferenceManager",
    
    # Browser UI
    "create_reference_browser_tab",
]
