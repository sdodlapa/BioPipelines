"""
Rich Dataset Models
===================

Enhanced data models that capture ALL relevant metadata upfront,
eliminating the need for follow-up "get details" calls.

Design Philosophy:
- Collect everything we might need during initial search
- Structure data for immediate use in workflows and UI
- Include quality/validation information
- Support progressive disclosure (summary → details)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DataQuality(Enum):
    """Data quality assessment levels."""
    EXCELLENT = "excellent"  # All QC passed, high replicates
    GOOD = "good"            # Most QC passed
    ACCEPTABLE = "acceptable" # Some QC issues but usable
    POOR = "poor"            # Significant QC issues
    UNKNOWN = "unknown"      # No QC information available


@dataclass
class ReplicateInfo:
    """Information about a biological/technical replicate."""
    replicate_id: str
    replicate_type: str  # "biological", "technical", "isogenic"
    biosample_id: Optional[str] = None
    treatment: Optional[str] = None
    time_point: Optional[str] = None
    file_count: int = 0
    total_reads: Optional[int] = None
    mapped_reads: Optional[int] = None
    quality_score: Optional[float] = None


@dataclass
class QualityMetrics:
    """Quality control metrics for a dataset."""
    # Sequencing quality
    total_reads: Optional[int] = None
    read_length: Optional[int] = None
    gc_content: Optional[float] = None
    duplication_rate: Optional[float] = None
    
    # Alignment quality
    mapping_rate: Optional[float] = None
    unique_mapping_rate: Optional[float] = None
    
    # ChIP-seq specific
    nrf: Optional[float] = None  # Non-redundant fraction
    pbc1: Optional[float] = None  # PCR bottleneck coefficient
    nsc: Optional[float] = None  # Normalized strand cross-correlation
    rsc: Optional[float] = None  # Relative strand cross-correlation
    
    # RNA-seq specific
    rin: Optional[float] = None  # RNA integrity number
    genes_detected: Optional[int] = None
    exonic_rate: Optional[float] = None
    
    # Overall assessment
    quality_level: DataQuality = DataQuality.UNKNOWN
    audit_warnings: List[str] = field(default_factory=list)
    audit_errors: List[str] = field(default_factory=list)
    
    def to_summary(self) -> str:
        """Generate a brief quality summary."""
        parts = []
        if self.quality_level != DataQuality.UNKNOWN:
            parts.append(f"Quality: {self.quality_level.value}")
        if self.total_reads:
            parts.append(f"{self.total_reads / 1e6:.1f}M reads")
        if self.mapping_rate:
            parts.append(f"{self.mapping_rate:.1%} mapped")
        if self.audit_warnings:
            parts.append(f"⚠️ {len(self.audit_warnings)} warnings")
        if self.audit_errors:
            parts.append(f"❌ {len(self.audit_errors)} errors")
        return " | ".join(parts) if parts else "No QC data"


@dataclass
class FileInfo:
    """Detailed information about a downloadable file."""
    accession: str
    filename: str
    file_format: str
    file_type: str  # "raw data", "alignments", "signal", "peaks", etc.
    output_type: str  # More specific: "reads", "alignments", "fold change over control"
    size_bytes: Optional[int] = None
    md5sum: Optional[str] = None
    
    # Download info
    download_url: str = ""
    s3_uri: Optional[str] = None
    cloud_metadata: Optional[Dict[str, Any]] = None
    
    # Replicate mapping
    biological_replicate: Optional[str] = None
    technical_replicate: Optional[str] = None
    
    # Processing info
    assembly: Optional[str] = None
    derived_from: List[str] = field(default_factory=list)
    analysis_step: Optional[str] = None
    
    # Quality
    quality_metrics: Optional[Dict[str, Any]] = None
    audit_status: str = "ok"
    
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
    
    def matches_filter(self, file_filter: Optional[str]) -> bool:
        """Check if this file matches a filter expression."""
        if not file_filter:
            return True
        
        filter_lower = file_filter.lower()
        
        # Exclusion filters
        if "without" in filter_lower or "exclude" in filter_lower or "no " in filter_lower:
            # Parse what to exclude
            if "fastq" in filter_lower:
                return self.file_format.lower() != "fastq"
            if "bam" in filter_lower:
                return self.file_format.lower() != "bam"
            if "raw" in filter_lower:
                return self.file_type.lower() != "raw data"
        
        # Inclusion filters
        if "only" in filter_lower:
            if "fastq" in filter_lower:
                return self.file_format.lower() == "fastq"
            if "bam" in filter_lower:
                return self.file_format.lower() == "bam"
            if "processed" in filter_lower:
                return self.file_type.lower() != "raw data"
        
        return True


@dataclass
class ExperimentCondition:
    """Experimental condition/treatment information."""
    name: str
    description: str = ""
    treatment: Optional[str] = None
    treatment_amount: Optional[str] = None
    treatment_duration: Optional[str] = None
    genetic_modification: Optional[str] = None
    is_control: bool = False


@dataclass
class RichDatasetInfo:
    """
    Comprehensive dataset information collected upfront.
    
    This model captures ALL relevant metadata during search,
    so we don't need follow-up API calls for common use cases.
    """
    # Core identifiers
    id: str
    source: str  # "ENCODE", "GEO", "TCGA", etc.
    accession: str
    
    # Basic metadata
    title: str
    description: str = ""
    web_url: str = ""
    
    # Biological context
    organism: str = ""
    organism_scientific: str = ""
    assembly: str = ""
    assay_type: str = ""
    assay_title: str = ""  # More specific assay name
    target: Optional[str] = None  # ChIP target, etc.
    target_label: Optional[str] = None  # Human-readable target name
    
    # Sample information
    biosample_type: str = ""  # "tissue", "cell line", "primary cell", etc.
    tissue: str = ""
    cell_line: Optional[str] = None
    cell_type: Optional[str] = None
    organ: Optional[str] = None
    life_stage: Optional[str] = None  # "adult", "embryonic", etc.
    age: Optional[str] = None
    sex: Optional[str] = None
    ethnicity: Optional[str] = None  # For human samples
    disease_state: Optional[str] = None
    
    # Experimental design
    conditions: List[ExperimentCondition] = field(default_factory=list)
    replicates: List[ReplicateInfo] = field(default_factory=list)
    biological_replicate_count: int = 0
    technical_replicate_count: int = 0
    has_controls: bool = False
    control_type: Optional[str] = None  # "input DNA", "IgG", "isotype", etc.
    
    # Files summary
    files: List[FileInfo] = field(default_factory=list)
    file_count: int = 0
    total_size_bytes: int = 0
    
    # File type breakdown
    file_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # e.g., {"FASTQ": {"count": 4, "size_bytes": 10GB}, "BAM": {"count": 2, ...}}
    
    # Quality information
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    status: str = "released"  # "released", "in progress", "archived"
    
    # Lab/Project info
    lab: Optional[str] = None
    lab_pi: Optional[str] = None
    project: Optional[str] = None
    award: Optional[str] = None
    
    # Publication/Citation
    publication: Optional[str] = None
    publication_doi: Optional[str] = None
    publication_pmid: Optional[str] = None
    
    # Timestamps
    date_created: Optional[datetime] = None
    date_released: Optional[datetime] = None
    date_submitted: Optional[datetime] = None
    
    # Raw API response (for edge cases)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Search relevance
    relevance_score: float = 0.0
    match_reasons: List[str] = field(default_factory=list)  # Why this matched the query
    
    @property
    def total_size_human(self) -> str:
        """Human-readable total size."""
        if not self.total_size_bytes:
            return "Unknown"
        size = float(self.total_size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    @property
    def sample_summary(self) -> str:
        """One-line sample description."""
        parts = []
        if self.organism:
            parts.append(self.organism)
        if self.tissue:
            parts.append(self.tissue)
        elif self.cell_line:
            parts.append(self.cell_line)
        elif self.cell_type:
            parts.append(self.cell_type)
        if self.disease_state and self.disease_state != "normal":
            parts.append(self.disease_state)
        if self.life_stage:
            parts.append(self.life_stage)
        return ", ".join(parts) if parts else "Unknown sample"
    
    @property
    def experiment_summary(self) -> str:
        """One-line experiment description."""
        parts = [self.assay_type or self.assay_title or "Unknown assay"]
        if self.target_label:
            parts.append(f"({self.target_label})")
        if self.biological_replicate_count > 0:
            parts.append(f"• {self.biological_replicate_count} replicates")
        if self.has_controls:
            parts.append(f"• {self.control_type or 'control'}")
        return " ".join(parts)
    
    def get_files_by_type(self, file_format: str) -> List[FileInfo]:
        """Get all files of a specific format."""
        format_lower = file_format.lower()
        return [f for f in self.files if f.file_format.lower() == format_lower]
    
    def get_raw_files(self) -> List[FileInfo]:
        """Get raw data files (FASTQ, etc.)."""
        return [f for f in self.files if f.file_type.lower() == "raw data"]
    
    def get_processed_files(self) -> List[FileInfo]:
        """Get processed files (BAM, bigWig, peaks, etc.)."""
        return [f for f in self.files if f.file_type.lower() != "raw data"]
    
    def to_compact_dict(self) -> Dict[str, Any]:
        """Convert to compact dictionary for API responses."""
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title[:80] + "..." if len(self.title) > 80 else self.title,
            "assay": self.assay_type,
            "target": self.target_label,
            "sample": self.sample_summary,
            "organism": self.organism,
            "assembly": self.assembly,
            "files": self.file_count,
            "size": self.total_size_human,
            "replicates": self.biological_replicate_count,
            "quality": self.quality.quality_level.value,
            "url": self.web_url,
        }
    
    def to_full_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary with all details."""
        return {
            "id": self.id,
            "source": self.source,
            "accession": self.accession,
            "title": self.title,
            "description": self.description,
            "web_url": self.web_url,
            "organism": self.organism,
            "organism_scientific": self.organism_scientific,
            "assembly": self.assembly,
            "assay_type": self.assay_type,
            "assay_title": self.assay_title,
            "target": self.target,
            "target_label": self.target_label,
            "biosample_type": self.biosample_type,
            "tissue": self.tissue,
            "cell_line": self.cell_line,
            "cell_type": self.cell_type,
            "organ": self.organ,
            "life_stage": self.life_stage,
            "disease_state": self.disease_state,
            "conditions": [
                {
                    "name": c.name,
                    "treatment": c.treatment,
                    "is_control": c.is_control,
                }
                for c in self.conditions
            ],
            "replicates": [
                {
                    "id": r.replicate_id,
                    "type": r.replicate_type,
                    "files": r.file_count,
                }
                for r in self.replicates
            ],
            "biological_replicate_count": self.biological_replicate_count,
            "has_controls": self.has_controls,
            "control_type": self.control_type,
            "files": [
                {
                    "accession": f.accession,
                    "format": f.file_format,
                    "type": f.output_type,
                    "size": f.size_human,
                    "replicate": f.biological_replicate,
                }
                for f in self.files
            ],
            "file_count": self.file_count,
            "total_size": self.total_size_human,
            "file_types": self.file_types,
            "quality": {
                "level": self.quality.quality_level.value,
                "summary": self.quality.to_summary(),
                "warnings": len(self.quality.audit_warnings),
                "errors": len(self.quality.audit_errors),
            },
            "status": self.status,
            "lab": self.lab,
            "project": self.project,
            "publication": self.publication_doi or self.publication_pmid,
            "date_released": self.date_released.isoformat() if self.date_released else None,
        }
    
    def format_markdown_summary(self) -> str:
        """Format as Markdown for display."""
        lines = [
            f"## 📋 {self.title[:60]}{'...' if len(self.title) > 60 else ''}",
            "",
            f"**ID:** `{self.id}` | **Source:** {self.source} | **Status:** {self.status}",
            "",
            "### Sample Information",
            f"- **Organism:** {self.organism} ({self.assembly or 'N/A'})",
            f"- **Sample:** {self.sample_summary}",
            f"- **Assay:** {self.experiment_summary}",
        ]
        
        if self.disease_state:
            lines.append(f"- **Disease:** {self.disease_state}")
        
        lines.extend([
            "",
            "### Files",
            f"**Total:** {self.file_count} files ({self.total_size_human})",
            "",
        ])
        
        # File type breakdown
        if self.file_types:
            lines.append("| Format | Count | Size |")
            lines.append("|--------|-------|------|")
            for fmt, info in sorted(self.file_types.items(), key=lambda x: -x[1].get('size_bytes', 0)):
                size = info.get('size_human', 'Unknown')
                lines.append(f"| {fmt} | {info['count']} | {size} |")
        
        # Quality info
        lines.extend([
            "",
            "### Quality",
            f"**Level:** {self.quality.quality_level.value.upper()}",
        ])
        
        if self.quality.audit_warnings:
            lines.append(f"⚠️ **Warnings:** {', '.join(self.quality.audit_warnings[:3])}")
        if self.quality.audit_errors:
            lines.append(f"❌ **Errors:** {', '.join(self.quality.audit_errors[:3])}")
        
        # Actions
        lines.extend([
            "",
            "---",
            f"🔗 [View on {self.source}]({self.web_url})",
            "",
            "**Actions:**",
            f"- `download {self.id}` - Download all files",
            f"- `download {self.id} without fastq` - Download processed files only",
        ])
        
        return "\n".join(lines)


@dataclass
class RichSearchResults:
    """Enhanced search results with comprehensive metadata."""
    query: str
    datasets: List[RichDatasetInfo] = field(default_factory=list)
    total_count: int = 0
    sources_searched: List[str] = field(default_factory=list)
    search_time_ms: float = 0.0
    
    # Aggregated statistics
    organisms_found: Dict[str, int] = field(default_factory=dict)
    assay_types_found: Dict[str, int] = field(default_factory=dict)
    tissues_found: Dict[str, int] = field(default_factory=dict)
    
    # Query understanding
    query_interpretation: Dict[str, Any] = field(default_factory=dict)
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def compute_statistics(self):
        """Compute aggregated statistics from datasets."""
        for ds in self.datasets:
            if ds.organism:
                self.organisms_found[ds.organism] = self.organisms_found.get(ds.organism, 0) + 1
            if ds.assay_type:
                self.assay_types_found[ds.assay_type] = self.assay_types_found.get(ds.assay_type, 0) + 1
            if ds.tissue:
                self.tissues_found[ds.tissue] = self.tissues_found.get(ds.tissue, 0) + 1
