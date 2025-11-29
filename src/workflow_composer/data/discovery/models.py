"""
Data Discovery Models
=====================

Core data classes for the data discovery and search system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


class DataSource(Enum):
    """Supported data sources."""
    ENCODE = "encode"
    GEO = "geo"
    SRA = "sra"
    GDC = "gdc"      # NCI Genomic Data Commons
    TCGA = "tcga"    # The Cancer Genome Atlas (via GDC)
    ENSEMBL = "ensembl"
    GENCODE = "gencode"
    UCSC = "ucsc"
    CUSTOM = "custom"
    LOCAL = "local"


class AssayType(Enum):
    """Common assay types."""
    RNA_SEQ = "RNA-seq"
    CHIP_SEQ = "ChIP-seq"
    ATAC_SEQ = "ATAC-seq"
    CUT_AND_RUN = "CUT&RUN"
    CUT_AND_TAG = "CUT&Tag"
    DNASE_SEQ = "DNase-seq"
    WGBS = "WGBS"  # Whole genome bisulfite sequencing
    RRBS = "RRBS"  # Reduced representation bisulfite sequencing
    HIC = "Hi-C"
    WGS = "WGS"  # Whole genome sequencing
    WES = "WES"  # Whole exome sequencing
    SCRNA_SEQ = "scRNA-seq"
    SCATAC_SEQ = "scATAC-seq"
    METAGENOMICS = "Metagenomics"
    LONG_READ = "Long-read"
    OTHER = "Other"


class FileType(Enum):
    """Data file types."""
    FASTQ = "fastq"
    FASTQ_GZ = "fastq.gz"
    BAM = "bam"
    CRAM = "cram"
    BED = "bed"
    BIGWIG = "bigwig"
    BIGBED = "bigbed"
    VCF = "vcf"
    FASTA = "fasta"
    GTF = "gtf"
    GFF = "gff"
    H5AD = "h5ad"
    TSV = "tsv"
    CSV = "csv"
    PEAKS = "peaks"
    OTHER = "other"


class DownloadMethod(Enum):
    """Download protocols/methods."""
    HTTP = "http"
    HTTPS = "https"
    FTP = "ftp"
    S3 = "s3"
    GCS = "gs"
    SRA = "sra"      # SRA toolkit (prefetch/fasterq-dump)
    ASPERA = "aspera"  # High-speed for NCBI
    RSYNC = "rsync"


@dataclass
class SearchQuery:
    """
    Structured search query parsed from natural language.
    
    Example:
        query = SearchQuery(
            organism="human",
            assembly="GRCh38",
            assay_type="ChIP-seq",
            target="H3K27ac",
            tissue="liver",
            source=DataSource.ENCODE
        )
    """
    # Required context
    raw_query: str = ""  # Original natural language query
    
    # Biological context
    organism: Optional[str] = None
    assembly: Optional[str] = None
    
    # Experiment type
    assay_type: Optional[str] = None
    target: Optional[str] = None  # ChIP/CUT&RUN target (H3K27ac, CTCF, etc.)
    
    # Sample information
    tissue: Optional[str] = None
    cell_line: Optional[str] = None
    cell_type: Optional[str] = None
    treatment: Optional[str] = None
    
    # Source preference
    source: Optional[DataSource] = None
    
    # Additional filters
    keywords: List[str] = field(default_factory=list)
    file_types: List[FileType] = field(default_factory=list)
    min_replicates: int = 0
    only_released: bool = True
    
    # Pagination
    max_results: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "organism": self.organism,
            "assembly": self.assembly,
            "assay_type": self.assay_type,
            "target": self.target,
            "tissue": self.tissue,
            "cell_line": self.cell_line,
            "cell_type": self.cell_type,
            "treatment": self.treatment,
            "source": self.source.value if self.source else None,
            "keywords": self.keywords,
            "max_results": self.max_results,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchQuery":
        """Create from dictionary."""
        source = data.get("source")
        if source and isinstance(source, str):
            source = DataSource(source)
        
        return cls(
            raw_query=data.get("raw_query", ""),
            organism=data.get("organism"),
            assembly=data.get("assembly"),
            assay_type=data.get("assay_type"),
            target=data.get("target"),
            tissue=data.get("tissue"),
            cell_line=data.get("cell_line"),
            cell_type=data.get("cell_type"),
            treatment=data.get("treatment"),
            source=source,
            keywords=data.get("keywords", []),
            max_results=data.get("max_results", 50),
        )


@dataclass
class DownloadURL:
    """
    A downloadable file with metadata.
    """
    url: str
    filename: str
    file_type: FileType = FileType.OTHER
    size_bytes: Optional[int] = None
    md5: Optional[str] = None
    download_method: DownloadMethod = DownloadMethod.HTTPS
    
    # Additional metadata
    replicate: Optional[str] = None  # e.g., "rep1", "rep2"
    read_type: Optional[str] = None  # "R1", "R2", "single"
    
    @property
    def size_human(self) -> str:
        """Human-readable file size."""
        if not self.size_bytes:
            return "Unknown"
        size = float(self.size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "filename": self.filename,
            "file_type": self.file_type.value,
            "size_bytes": self.size_bytes,
            "size_human": self.size_human,
            "md5": self.md5,
            "download_method": self.download_method.value,
            "replicate": self.replicate,
            "read_type": self.read_type,
        }


@dataclass
class DatasetInfo:
    """
    Information about a discovered dataset.
    
    Represents a single experiment or dataset from a database.
    """
    # Identifiers
    id: str  # e.g., ENCSR000ABC, GSE12345, SRR12345
    source: DataSource
    
    # Basic info
    title: str
    description: str = ""
    
    # Biological context
    organism: str = ""
    assembly: Optional[str] = None
    assay_type: Optional[str] = None
    target: Optional[str] = None
    tissue: Optional[str] = None
    cell_line: Optional[str] = None
    
    # Files
    download_urls: List[DownloadURL] = field(default_factory=list)
    file_count: int = 0
    total_size_bytes: Optional[int] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    publication: Optional[str] = None  # DOI or PMID
    date_released: Optional[datetime] = None
    
    # Quality/ranking
    quality_score: float = 0.0
    relevance_score: float = 0.0
    
    # URLs
    web_url: Optional[str] = None  # Link to view on source website
    
    @property
    def total_size_human(self) -> str:
        """Human-readable total size."""
        if not self.total_size_bytes:
            # Calculate from files
            total = sum(f.size_bytes or 0 for f in self.download_urls)
            if total == 0:
                return "Unknown"
            self.total_size_bytes = total
        
        size = float(self.total_size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source.value,
            "title": self.title,
            "description": self.description,
            "organism": self.organism,
            "assembly": self.assembly,
            "assay_type": self.assay_type,
            "target": self.target,
            "tissue": self.tissue,
            "cell_line": self.cell_line,
            "file_count": len(self.download_urls),
            "total_size": self.total_size_human,
            "web_url": self.web_url,
            "download_urls": [f.to_dict() for f in self.download_urls],
        }


@dataclass
class SearchResults:
    """
    Results from a data search.
    """
    query: SearchQuery
    datasets: List[DatasetInfo] = field(default_factory=list)
    total_count: int = 0
    sources_searched: List[DataSource] = field(default_factory=list)
    search_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def has_results(self) -> bool:
        """Check if there are any results."""
        return len(self.datasets) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query.to_dict(),
            "total_count": self.total_count,
            "sources_searched": [s.value for s in self.sources_searched],
            "search_time_ms": self.search_time_ms,
            "datasets": [d.to_dict() for d in self.datasets],
            "errors": self.errors,
        }


@dataclass
class LocalReference:
    """
    Information about a local reference file or directory.
    """
    path: str
    name: str
    ref_type: str  # genome, annotation, index, etc.
    
    # Validation
    is_valid: bool = False
    validation_message: str = ""
    
    # Metadata
    organism: Optional[str] = None
    assembly: Optional[str] = None
    size_bytes: int = 0
    last_modified: Optional[datetime] = None
    
    # For indexes
    aligner: Optional[str] = None  # star, bwa, bowtie2, etc.
    is_complete: bool = False
    missing_files: List[str] = field(default_factory=list)
    
    @property
    def size_human(self) -> str:
        """Human-readable size."""
        size = float(self.size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "ref_type": self.ref_type,
            "is_valid": self.is_valid,
            "validation_message": self.validation_message,
            "organism": self.organism,
            "assembly": self.assembly,
            "size": self.size_human,
            "aligner": self.aligner,
            "is_complete": self.is_complete,
            "missing_files": self.missing_files,
        }


@dataclass
class DownloadJob:
    """
    A download job in the queue.
    """
    id: str
    dataset: DatasetInfo
    output_dir: str
    
    # Status
    status: str = "pending"  # pending, downloading, completed, failed, cancelled
    progress: float = 0.0
    current_file: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    downloaded_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # Stats
    bytes_downloaded: int = 0
    total_bytes: int = 0
    
    @property
    def progress_percent(self) -> float:
        """Progress as percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.bytes_downloaded / self.total_bytes) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "dataset_id": self.dataset.id,
            "dataset_title": self.dataset.title,
            "output_dir": self.output_dir,
            "status": self.status,
            "progress": self.progress_percent,
            "current_file": self.current_file,
            "downloaded_files": self.downloaded_files,
            "failed_files": self.failed_files,
            "error_message": self.error_message,
        }
