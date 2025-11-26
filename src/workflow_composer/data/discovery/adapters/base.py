"""
Base Database Adapter
=====================

Abstract base class for all database adapters.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging

from ..models import SearchQuery, DatasetInfo, DownloadURL, DataSource

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """
    Abstract base class for database adapters.
    
    All adapters (ENCODE, GEO, Ensembl, etc.) should inherit from this class
    and implement the required methods.
    """
    
    # Subclasses should set this
    SOURCE: DataSource = DataSource.CUSTOM
    BASE_URL: str = ""
    
    def __init__(self, cache_enabled: bool = True, timeout: int = 30):
        """
        Initialize the adapter.
        
        Args:
            cache_enabled: Whether to cache search results
            timeout: Request timeout in seconds
        """
        self.cache_enabled = cache_enabled
        self.timeout = timeout
        self._cache: Dict[str, Any] = {}
    
    @abstractmethod
    def search(self, query: SearchQuery) -> List[DatasetInfo]:
        """
        Search the database with a structured query.
        
        Args:
            query: Structured search query
            
        Returns:
            List of matching datasets
        """
        pass
    
    @abstractmethod
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get detailed information about a specific dataset.
        
        Args:
            dataset_id: Dataset identifier (e.g., ENCSR000ABC, GSE12345)
            
        Returns:
            Dataset info or None if not found
        """
        pass
    
    @abstractmethod
    def get_download_urls(self, dataset_id: str) -> List[DownloadURL]:
        """
        Get download URLs for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            List of downloadable files
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if the data source is available/reachable.
        
        Returns:
            True if the source is available
        """
        try:
            import requests
            response = requests.head(self.BASE_URL, timeout=5)
            return response.status_code < 500
        except Exception:
            return False
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result."""
        if not self.cache_enabled:
            return None
        return self._cache.get(key)
    
    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached result."""
        if self.cache_enabled:
            self._cache[key] = value
    
    def _clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
    
    def _build_cache_key(self, query: SearchQuery) -> str:
        """Build a cache key from a query."""
        parts = [
            self.SOURCE.value,
            query.organism or "",
            query.assay_type or "",
            query.target or "",
            query.tissue or "",
            query.cell_line or "",
            "_".join(query.keywords),
        ]
        return ":".join(parts)
