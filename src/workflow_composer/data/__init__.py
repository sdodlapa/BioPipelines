"""
Data Management Module
======================

Tools for downloading and managing reference data.

Includes:
- DataDownloader: Download reference genomes, annotations, and indexes
- Discovery: LLM-powered data discovery across ENCODE, GEO, Ensembl
- Browser: Gradio UI components for data browsing

Quick Start:
    # Download references
    from workflow_composer.data import DataDownloader
    downloader = DataDownloader()
    genome = downloader.get_genome("human")
    
    # Discover data
    from workflow_composer.data import DataDiscovery
    discovery = DataDiscovery()
    results = discovery.search("human liver ChIP-seq H3K27ac")
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
    
    # Browser UI
    "create_reference_browser_tab",
]
