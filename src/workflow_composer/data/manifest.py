"""
Data Manifest
=============

Core data structures for the data-first workflow generation.

The DataManifest is the central data structure that holds all information
about samples, references, and experimental design needed to generate
a complete, ready-to-run workflow.

Usage:
    from workflow_composer.data.manifest import DataManifest, SampleInfo, ReferenceInfo
    
    # Create samples
    samples = [
        SampleInfo(
            sample_id="ctrl_1",
            fastq_1=Path("/data/ctrl1_R1.fq.gz"),
            fastq_2=Path("/data/ctrl1_R2.fq.gz"),
            condition="control"
        ),
        SampleInfo(
            sample_id="treat_1",
            fastq_1=Path("/data/treat1_R1.fq.gz"),
            fastq_2=Path("/data/treat1_R2.fq.gz"),
            condition="treated"
        ),
    ]
    
    # Create reference
    reference = ReferenceInfo(
        organism="human",
        assembly="GRCh38",
        genome_fasta=Path("/refs/GRCh38.fa"),
        annotation_gtf=Path("/refs/gencode.v44.gtf")
    )
    
    # Create manifest
    manifest = DataManifest(samples=samples, reference=reference)
    manifest.validate()
    
    # Generate samplesheet
    csv_content = manifest.to_samplesheet()
"""

import os
import gzip
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Source of the data."""
    LOCAL = "local"
    SRA = "sra"
    GEO = "geo"
    ENCODE = "encode"
    MANUAL = "manual"


class LibraryLayout(Enum):
    """Sequencing library layout."""
    SINGLE = "single"
    PAIRED = "paired"
    UNKNOWN = "unknown"


@dataclass
class SampleInfo:
    """
    Information about a single sequencing sample.
    
    Represents one biological sample with associated FASTQ files
    and metadata about experimental conditions.
    """
    # Required
    sample_id: str
    fastq_1: Path  # R1 or single-end file
    
    # Optional for paired-end
    fastq_2: Optional[Path] = None
    
    # Experimental design
    condition: Optional[str] = None
    replicate: Optional[int] = None
    batch: Optional[str] = None
    
    # Detected properties (filled by scanner)
    is_paired: bool = False
    read_length: Optional[int] = None
    read_count: Optional[int] = None
    instrument: Optional[str] = None
    
    # File info
    size_bytes: int = 0
    
    # Source tracking
    source: DataSourceType = DataSourceType.LOCAL
    accession: Optional[str] = None  # SRR/GSM/ENCFF ID
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Convert paths to Path objects if strings
        if isinstance(self.fastq_1, str):
            self.fastq_1 = Path(self.fastq_1)
        if isinstance(self.fastq_2, str):
            self.fastq_2 = Path(self.fastq_2)
        
        # Set paired-end flag
        self.is_paired = self.fastq_2 is not None
        
        # Calculate size if not set
        if self.size_bytes == 0:
            self._calculate_size()
    
    def _calculate_size(self):
        """Calculate total file size."""
        try:
            if self.fastq_1.exists():
                self.size_bytes = self.fastq_1.stat().st_size
            if self.fastq_2 and self.fastq_2.exists():
                self.size_bytes += self.fastq_2.stat().st_size
        except Exception:
            pass
    
    @property
    def size_human(self) -> str:
        """Human-readable file size."""
        size = float(self.size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    @property
    def layout(self) -> LibraryLayout:
        """Get library layout."""
        if self.is_paired:
            return LibraryLayout.PAIRED
        return LibraryLayout.SINGLE
    
    def exists(self) -> bool:
        """Check if all files exist."""
        if not self.fastq_1.exists():
            return False
        if self.fastq_2 and not self.fastq_2.exists():
            return False
        return True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the sample.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not self.sample_id:
            errors.append("Sample ID is required")
        
        if not self.fastq_1:
            errors.append(f"Sample {self.sample_id}: FASTQ file is required")
        elif not self.fastq_1.exists():
            errors.append(f"Sample {self.sample_id}: R1 file not found: {self.fastq_1}")
        
        if self.fastq_2 and not self.fastq_2.exists():
            errors.append(f"Sample {self.sample_id}: R2 file not found: {self.fastq_2}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "fastq_1": str(self.fastq_1),
            "fastq_2": str(self.fastq_2) if self.fastq_2 else None,
            "condition": self.condition,
            "replicate": self.replicate,
            "batch": self.batch,
            "is_paired": self.is_paired,
            "read_length": self.read_length,
            "read_count": self.read_count,
            "size": self.size_human,
            "source": self.source.value,
            "accession": self.accession,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleInfo":
        """Create from dictionary."""
        source = data.get("source", "local")
        if isinstance(source, str):
            source = DataSourceType(source)
        
        return cls(
            sample_id=data["sample_id"],
            fastq_1=Path(data["fastq_1"]),
            fastq_2=Path(data["fastq_2"]) if data.get("fastq_2") else None,
            condition=data.get("condition"),
            replicate=data.get("replicate"),
            batch=data.get("batch"),
            read_length=data.get("read_length"),
            read_count=data.get("read_count"),
            source=source,
            accession=data.get("accession"),
        )


@dataclass
class ReferenceInfo:
    """
    Information about reference data (genome, annotations, indexes).
    
    Tracks what reference files are available and what needs to be
    downloaded or built.
    """
    # Required
    organism: str
    assembly: str
    
    # Reference files (None if not available)
    genome_fasta: Optional[Path] = None
    annotation_gtf: Optional[Path] = None
    transcriptome_fasta: Optional[Path] = None
    
    # Aligner indexes (directory paths)
    star_index: Optional[Path] = None
    hisat2_index: Optional[Path] = None
    bwa_index: Optional[Path] = None
    salmon_index: Optional[Path] = None
    kallisto_index: Optional[Path] = None
    bowtie2_index: Optional[Path] = None
    
    # What's missing
    missing: List[str] = field(default_factory=list)
    
    # Download URLs for missing items
    download_urls: Dict[str, str] = field(default_factory=dict)
    
    # Validation status
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Convert paths
        for attr in ['genome_fasta', 'annotation_gtf', 'transcriptome_fasta',
                     'star_index', 'hisat2_index', 'bwa_index', 
                     'salmon_index', 'kallisto_index', 'bowtie2_index']:
            value = getattr(self, attr)
            if isinstance(value, str):
                setattr(self, attr, Path(value))
    
    def check_availability(self) -> None:
        """Check what's available and what's missing."""
        self.missing = []
        
        # Check genome
        if not self.genome_fasta or not self.genome_fasta.exists():
            self.missing.append("genome")
        
        # Check annotation
        if not self.annotation_gtf or not self.annotation_gtf.exists():
            self.missing.append("annotation")
        
        # Check indexes
        index_checks = [
            ('star_index', 'STAR index'),
            ('salmon_index', 'Salmon index'),
            ('hisat2_index', 'HISAT2 index'),
            ('bwa_index', 'BWA index'),
            ('kallisto_index', 'Kallisto index'),
        ]
        
        for attr, name in index_checks:
            path = getattr(self, attr)
            if path and not path.exists():
                self.missing.append(name)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate reference configuration.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not self.organism:
            errors.append("Organism is required")
        
        if not self.assembly:
            errors.append("Assembly is required")
        
        # Check that at least genome is available
        if not self.genome_fasta:
            errors.append("Genome FASTA path is required")
        elif not self.genome_fasta.exists():
            errors.append(f"Genome FASTA not found: {self.genome_fasta}")
        
        self.validation_errors = errors
        self.is_valid = len(errors) == 0
        
        return self.is_valid, errors
    
    def get_available_indexes(self) -> List[str]:
        """Get list of available aligner indexes."""
        available = []
        
        if self.star_index and self.star_index.exists():
            available.append("star")
        if self.salmon_index and self.salmon_index.exists():
            available.append("salmon")
        if self.hisat2_index and self.hisat2_index.exists():
            available.append("hisat2")
        if self.bwa_index and self.bwa_index.exists():
            available.append("bwa")
        if self.kallisto_index and self.kallisto_index.exists():
            available.append("kallisto")
        if self.bowtie2_index and self.bowtie2_index.exists():
            available.append("bowtie2")
        
        return available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "organism": self.organism,
            "assembly": self.assembly,
            "genome_fasta": str(self.genome_fasta) if self.genome_fasta else None,
            "annotation_gtf": str(self.annotation_gtf) if self.annotation_gtf else None,
            "transcriptome_fasta": str(self.transcriptome_fasta) if self.transcriptome_fasta else None,
            "star_index": str(self.star_index) if self.star_index else None,
            "salmon_index": str(self.salmon_index) if self.salmon_index else None,
            "hisat2_index": str(self.hisat2_index) if self.hisat2_index else None,
            "bwa_index": str(self.bwa_index) if self.bwa_index else None,
            "kallisto_index": str(self.kallisto_index) if self.kallisto_index else None,
            "available_indexes": self.get_available_indexes(),
            "missing": self.missing,
            "is_valid": self.is_valid,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceInfo":
        """Create from dictionary."""
        return cls(
            organism=data["organism"],
            assembly=data["assembly"],
            genome_fasta=Path(data["genome_fasta"]) if data.get("genome_fasta") else None,
            annotation_gtf=Path(data["annotation_gtf"]) if data.get("annotation_gtf") else None,
            transcriptome_fasta=Path(data["transcriptome_fasta"]) if data.get("transcriptome_fasta") else None,
            star_index=Path(data["star_index"]) if data.get("star_index") else None,
            salmon_index=Path(data["salmon_index"]) if data.get("salmon_index") else None,
            hisat2_index=Path(data["hisat2_index"]) if data.get("hisat2_index") else None,
            bwa_index=Path(data["bwa_index"]) if data.get("bwa_index") else None,
            kallisto_index=Path(data["kallisto_index"]) if data.get("kallisto_index") else None,
        )


@dataclass
class DataManifest:
    """
    Complete data manifest for workflow generation.
    
    This is the central data structure that connects samples, references,
    and experimental design. It's passed to the Composer to generate
    workflows with real file paths.
    
    Example:
        manifest = DataManifest(
            samples=[sample1, sample2, ...],
            reference=reference_info
        )
        manifest.validate()
        
        # Generate samplesheet
        csv = manifest.to_samplesheet()
        
        # Pass to composer
        workflow = composer.generate(query, data_manifest=manifest)
    """
    # Samples
    samples: List[SampleInfo] = field(default_factory=list)
    
    # Reference data
    reference: Optional[ReferenceInfo] = None
    
    # Detected characteristics (populated by validate())
    is_paired_end: bool = False
    avg_read_length: int = 0
    total_size_bytes: int = 0
    
    # Experimental design
    conditions: List[str] = field(default_factory=list)
    comparisons: List[Tuple[str, str]] = field(default_factory=list)
    
    # Validation status
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Source tracking
    created_from: str = ""  # "local_scan", "sra_download", "manual"
    created_at: datetime = field(default_factory=datetime.now)
    
    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("./results"))
    
    @property
    def sample_count(self) -> int:
        """Get number of samples."""
        return len(self.samples)
    
    @property
    def total_size_human(self) -> str:
        """Human-readable total size."""
        size = float(self.total_size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def add_sample(self, sample: SampleInfo) -> None:
        """Add a sample to the manifest."""
        self.samples.append(sample)
        self._update_stats()
    
    def _update_stats(self) -> None:
        """Update computed statistics."""
        if not self.samples:
            return
        
        # Check if paired-end (all samples should be same)
        self.is_paired_end = all(s.is_paired for s in self.samples)
        
        # Calculate average read length
        lengths = [s.read_length for s in self.samples if s.read_length]
        self.avg_read_length = sum(lengths) // len(lengths) if lengths else 0
        
        # Calculate total size
        self.total_size_bytes = sum(s.size_bytes for s in self.samples)
        
        # Extract unique conditions
        self.conditions = sorted(set(
            s.condition for s in self.samples if s.condition
        ))
    
    def validate(self) -> bool:
        """
        Validate the entire manifest.
        
        Checks:
        - All samples exist and are valid
        - Reference is valid (if provided)
        - Experimental design is consistent
        
        Returns:
            True if valid, False otherwise
        """
        self.validation_errors = []
        self.warnings = []
        
        # Check we have samples
        if not self.samples:
            self.validation_errors.append("No samples in manifest")
            self.is_valid = False
            return False
        
        # Validate each sample
        for sample in self.samples:
            is_valid, errors = sample.validate()
            self.validation_errors.extend(errors)
        
        # Check paired-end consistency
        paired_statuses = set(s.is_paired for s in self.samples)
        if len(paired_statuses) > 1:
            self.warnings.append(
                "Mixed single-end and paired-end samples detected"
            )
        
        # Validate reference if provided
        if self.reference:
            is_valid, errors = self.reference.validate()
            self.validation_errors.extend(errors)
            
            # Check for missing indexes
            self.reference.check_availability()
            if self.reference.missing:
                self.warnings.append(
                    f"Missing reference data: {', '.join(self.reference.missing)}"
                )
        
        # Check experimental design
        if len(self.conditions) == 0:
            self.warnings.append(
                "No experimental conditions defined. "
                "Consider assigning conditions for differential analysis."
            )
        elif len(self.conditions) == 1:
            self.warnings.append(
                "Only one condition defined. "
                "Differential analysis requires at least two conditions."
            )
        
        # Update stats
        self._update_stats()
        
        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid
    
    def to_samplesheet(self, format: str = "csv") -> str:
        """
        Generate a sample sheet in the specified format.
        
        Args:
            format: Output format ("csv" or "tsv")
        
        Returns:
            Sample sheet content as string
        """
        sep = "," if format == "csv" else "\t"
        
        # Determine columns based on data
        if self.is_paired_end:
            header = ["sample", "fastq_1", "fastq_2"]
        else:
            header = ["sample", "fastq_1"]
        
        # Add condition if present
        if self.conditions:
            header.append("condition")
        
        # Add replicate if present
        if any(s.replicate for s in self.samples):
            header.append("replicate")
        
        # Add batch if present
        if any(s.batch for s in self.samples):
            header.append("batch")
        
        lines = [sep.join(header)]
        
        for sample in self.samples:
            row = [sample.sample_id, str(sample.fastq_1)]
            
            if self.is_paired_end:
                row.append(str(sample.fastq_2) if sample.fastq_2 else "")
            
            if self.conditions:
                row.append(sample.condition or "")
            
            if "replicate" in header:
                row.append(str(sample.replicate) if sample.replicate else "")
            
            if "batch" in header:
                row.append(sample.batch or "")
            
            lines.append(sep.join(row))
        
        return "\n".join(lines)
    
    def to_nextflow_params(self) -> Dict[str, Any]:
        """
        Generate Nextflow parameters from manifest.
        
        Returns:
            Dictionary suitable for params.yaml
        """
        params = {
            "input": "./samplesheet.csv",
            "outdir": str(self.output_dir),
        }
        
        if self.reference:
            ref = self.reference
            if ref.genome_fasta:
                params["fasta"] = str(ref.genome_fasta)
            if ref.annotation_gtf:
                params["gtf"] = str(ref.annotation_gtf)
            if ref.transcriptome_fasta:
                params["transcript_fasta"] = str(ref.transcriptome_fasta)
            if ref.star_index:
                params["star_index"] = str(ref.star_index)
            if ref.salmon_index:
                params["salmon_index"] = str(ref.salmon_index)
            if ref.hisat2_index:
                params["hisat2_index"] = str(ref.hisat2_index)
        
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "samples": [s.to_dict() for s in self.samples],
            "reference": self.reference.to_dict() if self.reference else None,
            "sample_count": self.sample_count,
            "is_paired_end": self.is_paired_end,
            "avg_read_length": self.avg_read_length,
            "total_size": self.total_size_human,
            "conditions": self.conditions,
            "comparisons": self.comparisons,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "warnings": self.warnings,
            "created_from": self.created_from,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataManifest":
        """Create manifest from dictionary."""
        samples = [SampleInfo.from_dict(s) for s in data.get("samples", [])]
        reference = None
        if data.get("reference"):
            reference = ReferenceInfo.from_dict(data["reference"])
        
        manifest = cls(
            samples=samples,
            reference=reference,
            created_from=data.get("created_from", "manual"),
        )
        
        # Set comparisons if provided
        if data.get("comparisons"):
            manifest.comparisons = [tuple(c) for c in data["comparisons"]]
        
        return manifest
    
    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "DataManifest":
        """Load manifest from JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "## Data Manifest Summary",
            "",
            f"**Samples:** {self.sample_count}",
            f"**Layout:** {'Paired-end' if self.is_paired_end else 'Single-end'}",
            f"**Total Size:** {self.total_size_human}",
        ]
        
        if self.avg_read_length:
            lines.append(f"**Avg Read Length:** {self.avg_read_length} bp")
        
        if self.conditions:
            lines.append(f"**Conditions:** {', '.join(self.conditions)}")
        
        if self.reference:
            lines.extend([
                "",
                "### Reference",
                f"**Organism:** {self.reference.organism}",
                f"**Assembly:** {self.reference.assembly}",
            ])
            
            available = self.reference.get_available_indexes()
            if available:
                lines.append(f"**Available Indexes:** {', '.join(available)}")
            
            if self.reference.missing:
                lines.append(f"**Missing:** {', '.join(self.reference.missing)}")
        
        if self.validation_errors:
            lines.extend([
                "",
                "### ❌ Errors",
                *[f"- {e}" for e in self.validation_errors]
            ])
        
        if self.warnings:
            lines.extend([
                "",
                "### ⚠️ Warnings", 
                *[f"- {w}" for w in self.warnings]
            ])
        
        return "\n".join(lines)
