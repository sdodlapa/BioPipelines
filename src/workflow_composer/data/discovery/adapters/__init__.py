"""
Data Discovery Adapters
=======================

Database-specific adapters for searching and downloading genomics data.

Supported databases:
- ENCODE Portal (ChIP-seq, ATAC-seq, RNA-seq, etc.)
- NCBI GEO/SRA (All experiment types)  
- NCI GDC/TCGA (Cancer genomics data)
- Ensembl (Reference genomes, annotations)

Usage:
    from workflow_composer.data.discovery.adapters import (
        ENCODEAdapter,
        GEOAdapter,
        GDCAdapter,
        EnsemblAdapter,
    )
    
    # Search ENCODE
    encode = ENCODEAdapter()
    results = encode.search(SearchQuery(organism="human", assay_type="ChIP-seq"))
    
    # Search GEO
    geo = GEOAdapter()
    results = geo.search(SearchQuery(organism="human", assay_type="RNA-seq"))
    
    # Search GDC/TCGA
    gdc = GDCAdapter()
    results = gdc.search(SearchQuery(keywords=["brain", "methylation"]))
    
    # Get reference data
    ensembl = EnsemblAdapter()
    refs = ensembl.search(SearchQuery(organism="human", keywords=["genome"]))
"""

from .base import BaseAdapter
from .encode import ENCODEAdapter, search_encode
from .encode_enhanced import EnhancedENCODEAdapter, search_encode_rich
from .geo import GEOAdapter, search_geo
from .gdc import GDCAdapter, search_gdc
from .ensembl import (
    EnsemblAdapter,
    get_human_genome_url,
    get_human_gtf_url,
    get_mouse_genome_url,
    get_mouse_gtf_url,
)

__all__ = [
    # Base class
    "BaseAdapter",
    
    # Adapters
    "ENCODEAdapter",
    "EnhancedENCODEAdapter",  # Rich metadata version
    "GEOAdapter",
    "GDCAdapter",
    "EnsemblAdapter",
    
    # Convenience functions
    "search_encode",
    "search_encode_rich",  # Rich metadata search
    "search_geo",
    "search_gdc",
    "get_human_genome_url",
    "get_human_gtf_url",
    "get_mouse_genome_url",
    "get_mouse_gtf_url",
]


# Registry of available adapters
ADAPTER_REGISTRY = {
    "encode": ENCODEAdapter,
    "geo": GEOAdapter,
    "sra": GEOAdapter,  # GEO adapter also handles SRA
    "gdc": GDCAdapter,
    "tcga": GDCAdapter,  # Alias for GDC
    "ensembl": EnsemblAdapter,
}


def get_adapter(source: str) -> BaseAdapter:
    """
    Get an adapter instance for a data source.
    
    Args:
        source: Data source name (encode, geo, sra, ensembl)
        
    Returns:
        Adapter instance
        
    Raises:
        ValueError: If source is not supported
    """
    source = source.lower()
    if source not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown data source: {source}. "
            f"Supported: {list(ADAPTER_REGISTRY.keys())}"
        )
    
    adapter_class = ADAPTER_REGISTRY[source]
    return adapter_class()


def list_available_sources() -> list:
    """Get list of available data sources."""
    return list(ADAPTER_REGISTRY.keys())
