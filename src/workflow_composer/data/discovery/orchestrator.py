"""
Data Discovery Orchestrator
===========================

Coordinates data discovery across multiple sources.

The orchestrator:
1. Parses natural language queries
2. Routes queries to appropriate adapters
3. Aggregates and ranks results
4. Manages downloads

Usage:
    from workflow_composer.data.discovery import DataDiscovery
    
    # Initialize
    discovery = DataDiscovery()
    
    # Search
    results = discovery.search("human liver ChIP-seq H3K27ac")
    
    # Print results
    for dataset in results.datasets:
        print(f"{dataset.id}: {dataset.title}")
    
    # Download
    discovery.download(results.datasets[0], output_dir="/data/downloads")
"""

import logging
import asyncio
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import (
    SearchQuery, SearchResults, DatasetInfo, DownloadJob,
    DataSource, DownloadURL
)
from .query_parser import QueryParser, ParseResult
from .adapters import (
    BaseAdapter, ENCODEAdapter, GEOAdapter, EnsemblAdapter,
    get_adapter, list_available_sources
)

logger = logging.getLogger(__name__)


class DataDiscovery:
    """
    Main data discovery orchestrator.
    
    Coordinates searches across multiple databases and provides
    a unified interface for discovering and downloading genomics data.
    """
    
    def __init__(
        self,
        llm_client=None,
        enable_caching: bool = True,
        max_workers: int = 4,
        timeout: int = 30
    ):
        """
        Initialize the data discovery system.
        
        Args:
            llm_client: Optional LLM client for query parsing
            enable_caching: Enable result caching
            max_workers: Max parallel searches
            timeout: Request timeout in seconds
        """
        self.llm_client = llm_client
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        self.timeout = timeout
        
        # Initialize query parser
        self.query_parser = QueryParser(llm_client)
        
        # Initialize adapters
        self.adapters: Dict[DataSource, BaseAdapter] = {
            DataSource.ENCODE: ENCODEAdapter(enable_caching, timeout),
            DataSource.GEO: GEOAdapter(cache_enabled=enable_caching, timeout=timeout),
            DataSource.SRA: GEOAdapter(cache_enabled=enable_caching, timeout=timeout),
            DataSource.ENSEMBL: EnsemblAdapter(enable_caching, timeout),
        }
        
        # Download queue
        self._download_queue: List[DownloadJob] = []
    
    def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: int = 50
    ) -> SearchResults:
        """
        Search for datasets across data sources.
        
        Args:
            query: Natural language or structured query
            sources: Specific sources to search (default: auto-select)
            max_results: Maximum results per source
            
        Returns:
            SearchResults with matching datasets
        """
        start_time = time.time()
        
        # Parse the query
        if isinstance(query, str):
            parse_result = self.query_parser.parse(query)
            search_query = parse_result.query
            suggested_sources = parse_result.suggested_sources
        else:
            search_query = query
            suggested_sources = []
        
        search_query.max_results = max_results
        
        # Determine which sources to search
        if sources:
            # Use specified sources
            target_sources = [
                DataSource(s.lower()) for s in sources
                if s.lower() in [ds.value for ds in DataSource]
            ]
        elif search_query.source:
            # Use source from query
            target_sources = [search_query.source]
        elif suggested_sources:
            # Use suggested sources
            target_sources = suggested_sources[:3]  # Limit to top 3
        else:
            # Default: search all major sources
            target_sources = [DataSource.ENCODE, DataSource.GEO]
        
        logger.info(f"Searching {len(target_sources)} sources: {target_sources}")
        
        # Search each source
        all_datasets = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for source in target_sources:
                if source in self.adapters:
                    adapter = self.adapters[source]
                    futures[executor.submit(adapter.search, search_query)] = source
            
            for future in as_completed(futures, timeout=self.timeout * 2):
                source = futures[future]
                try:
                    datasets = future.result()
                    all_datasets.extend(datasets)
                    logger.info(f"Got {len(datasets)} results from {source.value}")
                except Exception as e:
                    error_msg = f"Search failed for {source.value}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        # Rank and deduplicate results
        ranked_datasets = self._rank_results(all_datasets, search_query)
        
        # Create results object
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SearchResults(
            query=search_query,
            datasets=ranked_datasets[:max_results],
            total_count=len(all_datasets),
            sources_searched=target_sources,
            search_time_ms=elapsed_ms,
            errors=errors,
        )
    
    def get_dataset(self, dataset_id: str, source: Optional[str] = None) -> Optional[DatasetInfo]:
        """
        Get detailed information about a specific dataset.
        
        Args:
            dataset_id: Dataset identifier
            source: Data source (auto-detected if not provided)
            
        Returns:
            Dataset info or None
        """
        # Auto-detect source from ID format
        if not source:
            source = self._detect_source(dataset_id)
        
        if source and source in self.adapters:
            adapter = self.adapters[source]
            return adapter.get_dataset(dataset_id)
        
        # Try all adapters
        for adapter in self.adapters.values():
            dataset = adapter.get_dataset(dataset_id)
            if dataset:
                return dataset
        
        return None
    
    def get_download_urls(self, dataset_id: str, source: Optional[str] = None) -> List[DownloadURL]:
        """
        Get download URLs for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            source: Data source
            
        Returns:
            List of downloadable files
        """
        if not source:
            source = self._detect_source(dataset_id)
        
        if source and source in self.adapters:
            adapter = self.adapters[source]
            return adapter.get_download_urls(dataset_id)
        
        return []
    
    def download(
        self,
        dataset: DatasetInfo,
        output_dir: str,
        file_types: Optional[List[str]] = None
    ) -> DownloadJob:
        """
        Download a dataset.
        
        Args:
            dataset: Dataset to download
            output_dir: Output directory
            file_types: Filter by file types (e.g., ["fastq", "bam"])
            
        Returns:
            DownloadJob for tracking progress
        """
        import uuid
        
        # Create download job
        job = DownloadJob(
            id=str(uuid.uuid4())[:8],
            dataset=dataset,
            output_dir=output_dir,
        )
        
        self._download_queue.append(job)
        
        # Get download URLs if not already populated
        if not dataset.download_urls:
            dataset.download_urls = self.get_download_urls(
                dataset.id, dataset.source
            )
        
        # Filter by file type if specified
        urls_to_download = dataset.download_urls
        if file_types:
            urls_to_download = [
                u for u in urls_to_download
                if u.file_type.value in file_types
            ]
        
        # Start download (could be async/background)
        self._perform_download(job, urls_to_download)
        
        return job
    
    def _rank_results(
        self,
        datasets: List[DatasetInfo],
        query: SearchQuery
    ) -> List[DatasetInfo]:
        """Rank and score results by relevance."""
        for dataset in datasets:
            score = 0.0
            
            # Organism match
            if query.organism and dataset.organism:
                if query.organism.lower() in dataset.organism.lower():
                    score += 1.0
            
            # Assay type match
            if query.assay_type and dataset.assay_type:
                if query.assay_type.lower() == dataset.assay_type.lower():
                    score += 1.0
            
            # Target match (for ChIP-seq)
            if query.target and dataset.target:
                if query.target.lower() == dataset.target.lower():
                    score += 1.0
            
            # Tissue match
            if query.tissue and dataset.tissue:
                if query.tissue.lower() in dataset.tissue.lower():
                    score += 0.5
            
            # Cell line match
            if query.cell_line and dataset.cell_line:
                if query.cell_line.lower() in dataset.cell_line.lower():
                    score += 0.5
            
            # Keyword matches
            for kw in query.keywords:
                if kw.lower() in dataset.title.lower():
                    score += 0.2
                if kw.lower() in dataset.description.lower():
                    score += 0.1
            
            dataset.relevance_score = score
        
        # Sort by relevance score
        datasets.sort(key=lambda d: d.relevance_score, reverse=True)
        
        return datasets
    
    def _detect_source(self, dataset_id: str) -> Optional[DataSource]:
        """Detect data source from dataset ID format."""
        dataset_id = dataset_id.upper()
        
        if dataset_id.startswith("ENCSR") or dataset_id.startswith("ENCFF"):
            return DataSource.ENCODE
        elif dataset_id.startswith("GSE") or dataset_id.startswith("GSM"):
            return DataSource.GEO
        elif dataset_id.startswith("SRR") or dataset_id.startswith("SRX"):
            return DataSource.SRA
        
        return None
    
    def _perform_download(self, job: DownloadJob, urls: List[DownloadURL]) -> None:
        """Perform the actual download."""
        import subprocess
        from pathlib import Path
        
        job.status = "downloading"
        output_path = Path(job.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        job.total_bytes = sum(u.size_bytes or 0 for u in urls)
        
        for i, url_info in enumerate(urls):
            job.current_file = url_info.filename
            job.progress = (i / len(urls)) * 100
            
            try:
                output_file = output_path / url_info.filename
                
                # Download based on method
                if url_info.download_method.value == "sra":
                    # Use fasterq-dump for SRA data
                    sra_id = url_info.url.replace("sra://", "")
                    cmd = ["fasterq-dump", sra_id, "-O", str(output_path)]
                else:
                    # Use wget/curl for HTTP/FTP
                    cmd = ["wget", "-O", str(output_file), url_info.url]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    job.downloaded_files.append(str(output_file))
                    if url_info.size_bytes:
                        job.bytes_downloaded += url_info.size_bytes
                else:
                    job.failed_files.append(url_info.filename)
                    logger.error(f"Download failed: {result.stderr}")
                    
            except Exception as e:
                job.failed_files.append(url_info.filename)
                logger.error(f"Download error for {url_info.filename}: {e}")
        
        # Update final status
        if job.failed_files:
            job.status = "completed_with_errors"
        else:
            job.status = "completed"
        job.progress = 100.0
    
    def list_sources(self) -> List[str]:
        """List available data sources."""
        return [s.value for s in self.adapters.keys()]


# Convenience functions
def quick_search(query: str, **kwargs) -> SearchResults:
    """
    Quick search across all sources.
    
    Args:
        query: Natural language query
        **kwargs: Additional search parameters
        
    Returns:
        SearchResults
    """
    discovery = DataDiscovery()
    return discovery.search(query, **kwargs)


def search_encode(query: str, **kwargs) -> SearchResults:
    """Search ENCODE only."""
    discovery = DataDiscovery()
    return discovery.search(query, sources=["encode"], **kwargs)


def search_geo(query: str, **kwargs) -> SearchResults:
    """Search GEO only."""
    discovery = DataDiscovery()
    return discovery.search(query, sources=["geo"], **kwargs)


def search_references(organism: str = "human", **kwargs) -> SearchResults:
    """Search for reference data."""
    discovery = DataDiscovery()
    query = f"{organism} genome reference annotation"
    return discovery.search(query, sources=["ensembl"], **kwargs)
