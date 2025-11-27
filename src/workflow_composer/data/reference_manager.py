"""
Reference Manager
=================

Manages reference data (genomes, annotations, indexes) for bioinformatics workflows.

Features:
- Check local reference availability
- Download references from Ensembl, GENCODE, UCSC
- Build aligner indexes (STAR, Salmon, BWA, etc.)
- Validate reference file integrity

Usage:
    from workflow_composer.data.reference_manager import ReferenceManager
    
    manager = ReferenceManager(base_dir=Path("/data/references"))
    
    # Check what's available for human/GRCh38
    ref_info = manager.check_references("human", "GRCh38")
    print(f"Missing: {ref_info.missing}")
    
    # Download missing genome
    manager.download_reference("human", "GRCh38", "genome")
    
    # Build STAR index
    manager.build_index("star", ref_info.genome_fasta, ref_info.annotation_gtf)
"""

import os
import subprocess
import logging
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import gzip

from .manifest import ReferenceInfo

logger = logging.getLogger(__name__)


# Reference data sources
REFERENCE_SOURCES = {
    "human": {
        "GRCh38": {
            "genome": {
                "url": "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
                "filename": "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
                "decompressed": "GRCh38.fa",
            },
            "gtf": {
                "url": "https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz",
                "filename": "Homo_sapiens.GRCh38.110.gtf.gz",
                "decompressed": "GRCh38.gtf",
            },
            "transcriptome": {
                "url": "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
                "filename": "Homo_sapiens.GRCh38.cdna.all.fa.gz",
                "decompressed": "GRCh38.cdna.fa",
            },
        },
        "GRCh37": {
            "genome": {
                "url": "https://ftp.ensembl.org/pub/grch37/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz",
                "filename": "Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz",
                "decompressed": "GRCh37.fa",
            },
            "gtf": {
                "url": "https://ftp.ensembl.org/pub/grch37/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.gtf.gz",
                "filename": "Homo_sapiens.GRCh37.87.gtf.gz",
                "decompressed": "GRCh37.gtf",
            },
        },
    },
    "mouse": {
        "GRCm39": {
            "genome": {
                "url": "https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz",
                "filename": "Mus_musculus.GRCm39.dna.primary_assembly.fa.gz",
                "decompressed": "GRCm39.fa",
            },
            "gtf": {
                "url": "https://ftp.ensembl.org/pub/release-110/gtf/mus_musculus/Mus_musculus.GRCm39.110.gtf.gz",
                "filename": "Mus_musculus.GRCm39.110.gtf.gz",
                "decompressed": "GRCm39.gtf",
            },
            "transcriptome": {
                "url": "https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/cdna/Mus_musculus.GRCm39.cdna.all.fa.gz",
                "filename": "Mus_musculus.GRCm39.cdna.all.fa.gz",
                "decompressed": "GRCm39.cdna.fa",
            },
        },
        "GRCm38": {
            "genome": {
                "url": "https://ftp.ensembl.org/pub/release-102/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz",
                "filename": "Mus_musculus.GRCm38.dna.primary_assembly.fa.gz",
                "decompressed": "GRCm38.fa",
            },
            "gtf": {
                "url": "https://ftp.ensembl.org/pub/release-102/gtf/mus_musculus/Mus_musculus.GRCm38.102.gtf.gz",
                "filename": "Mus_musculus.GRCm38.102.gtf.gz",
                "decompressed": "GRCm38.gtf",
            },
        },
    },
    "rat": {
        "mRatBN7.2": {
            "genome": {
                "url": "https://ftp.ensembl.org/pub/release-110/fasta/rattus_norvegicus/dna/Rattus_norvegicus.mRatBN7.2.dna.toplevel.fa.gz",
                "filename": "Rattus_norvegicus.mRatBN7.2.dna.toplevel.fa.gz",
                "decompressed": "mRatBN7.2.fa",
            },
            "gtf": {
                "url": "https://ftp.ensembl.org/pub/release-110/gtf/rattus_norvegicus/Rattus_norvegicus.mRatBN7.2.110.gtf.gz",
                "filename": "Rattus_norvegicus.mRatBN7.2.110.gtf.gz",
                "decompressed": "mRatBN7.2.gtf",
            },
        },
    },
    "zebrafish": {
        "GRCz11": {
            "genome": {
                "url": "https://ftp.ensembl.org/pub/release-110/fasta/danio_rerio/dna/Danio_rerio.GRCz11.dna.primary_assembly.fa.gz",
                "filename": "Danio_rerio.GRCz11.dna.primary_assembly.fa.gz",
                "decompressed": "GRCz11.fa",
            },
            "gtf": {
                "url": "https://ftp.ensembl.org/pub/release-110/gtf/danio_rerio/Danio_rerio.GRCz11.110.gtf.gz",
                "filename": "Danio_rerio.GRCz11.110.gtf.gz",
                "decompressed": "GRCz11.gtf",
            },
        },
    },
}

# Index file patterns for validation
INDEX_FILE_PATTERNS = {
    "star": ["Genome", "SA", "SAindex", "chrLength.txt", "chrName.txt"],
    "salmon": ["info.json", "seq.bin", "pos.bin"],
    "kallisto": ["kallisto.idx"],
    "hisat2": [".1.ht2", ".2.ht2", ".3.ht2", ".4.ht2"],
    "bwa": [".bwt", ".pac", ".ann", ".amb", ".sa"],
    "bowtie2": [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2"],
}


@dataclass
class DownloadProgress:
    """Track download progress."""
    url: str
    filename: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    status: str = "pending"  # pending, downloading, completed, failed
    error: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100


@dataclass
class IndexBuildJob:
    """Track index building job."""
    aligner: str
    genome_path: Path
    gtf_path: Optional[Path]
    output_dir: Path
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    log_file: Optional[Path] = None


class ReferenceManager:
    """
    Manages reference data for bioinformatics workflows.
    
    Handles:
    - Local reference discovery
    - Reference downloads from Ensembl/GENCODE
    - Aligner index building
    - Reference validation
    """
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        decompress: bool = True,
    ):
        """
        Initialize the reference manager.
        
        Args:
            base_dir: Base directory for reference data
            decompress: Whether to decompress downloaded files
        """
        self.base_dir = Path(base_dir) if base_dir else Path("data/references")
        self.decompress = decompress
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_organism_dir(self, organism: str) -> Path:
        """Get directory for a specific organism."""
        org_dir = self.base_dir / organism.lower()
        org_dir.mkdir(parents=True, exist_ok=True)
        return org_dir
    
    def check_references(
        self,
        organism: str,
        assembly: str,
    ) -> ReferenceInfo:
        """
        Check what reference data is available locally.
        
        Args:
            organism: Organism name (human, mouse, etc.)
            assembly: Genome assembly (GRCh38, GRCm39, etc.)
        
        Returns:
            ReferenceInfo with paths to available files and list of missing
        """
        org_dir = self.get_organism_dir(organism)
        
        ref_info = ReferenceInfo(
            organism=organism,
            assembly=assembly,
        )
        
        # Look for genome FASTA
        for pattern in [f"{assembly}.fa", f"{assembly}.fasta", f"*{assembly}*.fa"]:
            matches = list(org_dir.glob(pattern))
            if matches:
                ref_info.genome_fasta = matches[0]
                break
        
        # Look for GTF
        for pattern in [f"{assembly}.gtf", f"*{assembly}*.gtf"]:
            matches = list(org_dir.glob(pattern))
            if matches:
                ref_info.annotation_gtf = matches[0]
                break
        
        # Look for transcriptome
        for pattern in [f"{assembly}.cdna.fa", f"*{assembly}*cdna*.fa"]:
            matches = list(org_dir.glob(pattern))
            if matches:
                ref_info.transcriptome_fasta = matches[0]
                break
        
        # Check for indexes
        index_dir = org_dir / "indexes"
        if index_dir.exists():
            # STAR index
            star_dir = index_dir / f"star_{assembly}"
            if star_dir.exists() and self._validate_index("star", star_dir):
                ref_info.star_index = star_dir
            
            # Salmon index
            salmon_dir = index_dir / f"salmon_{assembly}"
            if salmon_dir.exists() and self._validate_index("salmon", salmon_dir):
                ref_info.salmon_index = salmon_dir
            
            # HISAT2 index
            hisat2_prefix = index_dir / f"hisat2_{assembly}"
            if self._validate_index("hisat2", hisat2_prefix):
                ref_info.hisat2_index = hisat2_prefix
            
            # BWA index
            bwa_prefix = index_dir / f"bwa_{assembly}"
            if self._validate_index("bwa", bwa_prefix):
                ref_info.bwa_index = bwa_prefix
        
        # Populate download URLs for missing items
        ref_info.download_urls = self._get_download_urls(organism, assembly)
        
        # Check what's missing
        ref_info.check_availability()
        
        return ref_info
    
    def _get_download_urls(
        self,
        organism: str,
        assembly: str,
    ) -> Dict[str, str]:
        """Get download URLs for reference files."""
        urls = {}
        
        org_lower = organism.lower()
        if org_lower in REFERENCE_SOURCES:
            if assembly in REFERENCE_SOURCES[org_lower]:
                for resource, info in REFERENCE_SOURCES[org_lower][assembly].items():
                    urls[resource] = info["url"]
        
        return urls
    
    def _validate_index(self, aligner: str, path: Path) -> bool:
        """Validate that an index is complete."""
        if aligner not in INDEX_FILE_PATTERNS:
            return path.exists()
        
        patterns = INDEX_FILE_PATTERNS[aligner]
        
        if aligner in ["star", "salmon"]:
            # Directory-based indexes
            if not path.is_dir():
                return False
            for pattern in patterns:
                if not list(path.glob(f"*{pattern}*")) and not (path / pattern).exists():
                    return False
            return True
        else:
            # Prefix-based indexes (hisat2, bwa, bowtie2)
            for pattern in patterns:
                expected = Path(str(path) + pattern)
                if not expected.exists():
                    return False
            return True
    
    def download_reference(
        self,
        organism: str,
        assembly: str,
        resource: str,  # "genome", "gtf", "transcriptome"
        progress_callback=None,
    ) -> Optional[Path]:
        """
        Download a reference file.
        
        Args:
            organism: Organism name
            assembly: Genome assembly
            resource: Resource type to download
            progress_callback: Optional callback for progress updates
        
        Returns:
            Path to downloaded (and decompressed) file, or None if failed
        """
        org_lower = organism.lower()
        
        # Get download info
        if org_lower not in REFERENCE_SOURCES:
            logger.error(f"Unknown organism: {organism}")
            return None
        
        if assembly not in REFERENCE_SOURCES[org_lower]:
            logger.error(f"Unknown assembly: {assembly} for {organism}")
            return None
        
        if resource not in REFERENCE_SOURCES[org_lower][assembly]:
            logger.error(f"Unknown resource: {resource}")
            return None
        
        info = REFERENCE_SOURCES[org_lower][assembly][resource]
        url = info["url"]
        filename = info["filename"]
        decompressed_name = info.get("decompressed", filename.replace(".gz", ""))
        
        # Setup paths
        org_dir = self.get_organism_dir(organism)
        download_path = org_dir / filename
        final_path = org_dir / decompressed_name
        
        # Skip if already exists
        if final_path.exists():
            logger.info(f"Reference already exists: {final_path}")
            return final_path
        
        try:
            logger.info(f"Downloading {url}")
            
            # Download with progress tracking
            def reporthook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if progress_callback:
                    progress_callback(downloaded, total_size)
            
            urllib.request.urlretrieve(url, download_path, reporthook)
            
            # Decompress if needed
            if self.decompress and str(download_path).endswith('.gz'):
                logger.info(f"Decompressing to {final_path}")
                with gzip.open(download_path, 'rb') as f_in:
                    with open(final_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove compressed file
                download_path.unlink()
            else:
                final_path = download_path
            
            logger.info(f"Successfully downloaded: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # Cleanup partial download
            if download_path.exists():
                download_path.unlink()
            return None
    
    def build_index(
        self,
        aligner: str,
        genome_path: Path,
        gtf_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        threads: int = 8,
        memory_gb: int = 32,
    ) -> Optional[Path]:
        """
        Build an aligner index.
        
        Args:
            aligner: Aligner name (star, salmon, bwa, hisat2, etc.)
            genome_path: Path to genome FASTA
            gtf_path: Path to GTF (required for STAR, optional for others)
            output_dir: Output directory for index
            threads: Number of threads to use
            memory_gb: Memory limit in GB
        
        Returns:
            Path to built index, or None if failed
        """
        if not genome_path.exists():
            logger.error(f"Genome not found: {genome_path}")
            return None
        
        # Determine output directory
        if output_dir is None:
            org_dir = genome_path.parent
            index_dir = org_dir / "indexes"
            index_dir.mkdir(exist_ok=True)
            assembly = genome_path.stem.split('.')[0]
            output_dir = index_dir / f"{aligner}_{assembly}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build index based on aligner
        try:
            if aligner.lower() == "star":
                return self._build_star_index(
                    genome_path, gtf_path, output_dir, threads, memory_gb
                )
            elif aligner.lower() == "salmon":
                return self._build_salmon_index(
                    genome_path, output_dir, threads
                )
            elif aligner.lower() == "bwa":
                return self._build_bwa_index(genome_path, output_dir)
            elif aligner.lower() == "hisat2":
                return self._build_hisat2_index(
                    genome_path, output_dir, threads
                )
            elif aligner.lower() == "kallisto":
                return self._build_kallisto_index(genome_path, output_dir)
            else:
                logger.error(f"Unknown aligner: {aligner}")
                return None
                
        except Exception as e:
            logger.error(f"Index build failed: {e}")
            return None
    
    def _build_star_index(
        self,
        genome_path: Path,
        gtf_path: Optional[Path],
        output_dir: Path,
        threads: int,
        memory_gb: int,
    ) -> Optional[Path]:
        """Build STAR index."""
        cmd = [
            "STAR",
            "--runMode", "genomeGenerate",
            "--runThreadN", str(threads),
            "--genomeDir", str(output_dir),
            "--genomeFastaFiles", str(genome_path),
            "--limitGenomeGenerateRAM", str(memory_gb * 1024**3),
        ]
        
        if gtf_path and gtf_path.exists():
            cmd.extend(["--sjdbGTFfile", str(gtf_path)])
        
        logger.info(f"Building STAR index: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"STAR index failed: {result.stderr}")
            return None
        
        return output_dir
    
    def _build_salmon_index(
        self,
        transcriptome_path: Path,
        output_dir: Path,
        threads: int,
    ) -> Optional[Path]:
        """Build Salmon index from transcriptome."""
        cmd = [
            "salmon", "index",
            "-t", str(transcriptome_path),
            "-i", str(output_dir),
            "-p", str(threads),
        ]
        
        logger.info(f"Building Salmon index: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"Salmon index failed: {result.stderr}")
            return None
        
        return output_dir
    
    def _build_bwa_index(
        self,
        genome_path: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Build BWA index."""
        prefix = output_dir / genome_path.stem
        
        cmd = [
            "bwa", "index",
            "-p", str(prefix),
            str(genome_path),
        ]
        
        logger.info(f"Building BWA index: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"BWA index failed: {result.stderr}")
            return None
        
        return prefix
    
    def _build_hisat2_index(
        self,
        genome_path: Path,
        output_dir: Path,
        threads: int,
    ) -> Optional[Path]:
        """Build HISAT2 index."""
        prefix = output_dir / genome_path.stem
        
        cmd = [
            "hisat2-build",
            "-p", str(threads),
            str(genome_path),
            str(prefix),
        ]
        
        logger.info(f"Building HISAT2 index: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"HISAT2 index failed: {result.stderr}")
            return None
        
        return prefix
    
    def _build_kallisto_index(
        self,
        transcriptome_path: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Build Kallisto index."""
        index_path = output_dir / "kallisto.idx"
        
        cmd = [
            "kallisto", "index",
            "-i", str(index_path),
            str(transcriptome_path),
        ]
        
        logger.info(f"Building Kallisto index: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"Kallisto index failed: {result.stderr}")
            return None
        
        return index_path
    
    def list_available_organisms(self) -> List[str]:
        """List organisms with known reference sources."""
        return list(REFERENCE_SOURCES.keys())
    
    def list_available_assemblies(self, organism: str) -> List[str]:
        """List available assemblies for an organism."""
        org_lower = organism.lower()
        if org_lower in REFERENCE_SOURCES:
            return list(REFERENCE_SOURCES[org_lower].keys())
        return []
    
    def get_reference_status(
        self,
        organism: str,
        assembly: str,
    ) -> Dict[str, Any]:
        """
        Get comprehensive status of references for display.
        
        Returns dict with status for each resource type.
        """
        ref_info = self.check_references(organism, assembly)
        
        status = {
            "organism": organism,
            "assembly": assembly,
            "resources": {},
        }
        
        # Genome
        status["resources"]["genome"] = {
            "available": ref_info.genome_fasta is not None and ref_info.genome_fasta.exists(),
            "path": str(ref_info.genome_fasta) if ref_info.genome_fasta else None,
            "download_url": ref_info.download_urls.get("genome"),
        }
        
        # GTF
        status["resources"]["annotation"] = {
            "available": ref_info.annotation_gtf is not None and ref_info.annotation_gtf.exists(),
            "path": str(ref_info.annotation_gtf) if ref_info.annotation_gtf else None,
            "download_url": ref_info.download_urls.get("gtf"),
        }
        
        # Indexes
        for aligner in ["star", "salmon", "hisat2", "bwa", "kallisto"]:
            index_path = getattr(ref_info, f"{aligner}_index", None)
            status["resources"][f"{aligner}_index"] = {
                "available": index_path is not None and (
                    index_path.exists() if index_path else False
                ),
                "path": str(index_path) if index_path else None,
                "can_build": ref_info.genome_fasta is not None,
            }
        
        return status


# Convenience functions
def get_reference_info(
    organism: str,
    assembly: str,
    base_dir: Optional[str] = None,
) -> ReferenceInfo:
    """
    Quick function to get reference info.
    
    Args:
        organism: Organism name
        assembly: Genome assembly
        base_dir: Optional base directory
    
    Returns:
        ReferenceInfo
    """
    manager = ReferenceManager(base_dir=Path(base_dir) if base_dir else None)
    return manager.check_references(organism, assembly)


def list_organisms() -> List[str]:
    """List available organisms."""
    return list(REFERENCE_SOURCES.keys())


def list_assemblies(organism: str) -> List[str]:
    """List available assemblies for an organism."""
    org_lower = organism.lower()
    if org_lower in REFERENCE_SOURCES:
        return list(REFERENCE_SOURCES[org_lower].keys())
    return []
