"""
Ensembl Adapter
===============

Search and download reference data from Ensembl/Ensembl Genomes.

Provides:
- Reference genomes (FASTA)
- Gene annotations (GTF/GFF)
- cDNA, ncRNA, protein sequences
- Variation data

Ensembl REST API: https://rest.ensembl.org/
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import requests

from .base import BaseAdapter
from ..models import (
    SearchQuery, DatasetInfo, DownloadURL, DataSource,
    FileType, DownloadMethod, LocalReference
)

logger = logging.getLogger(__name__)


# Assembly mappings
ASSEMBLY_MAP = {
    "human": {"assembly": "GRCh38", "ensembl_name": "homo_sapiens"},
    "mouse": {"assembly": "GRCm39", "ensembl_name": "mus_musculus"},
    "rat": {"assembly": "mRatBN7.2", "ensembl_name": "rattus_norvegicus"},
    "zebrafish": {"assembly": "GRCz11", "ensembl_name": "danio_rerio"},
    "fly": {"assembly": "BDGP6.46", "ensembl_name": "drosophila_melanogaster"},
    "worm": {"assembly": "WBcel235", "ensembl_name": "caenorhabditis_elegans"},
    "yeast": {"assembly": "R64-1-1", "ensembl_name": "saccharomyces_cerevisiae"},
}


class EnsemblAdapter(BaseAdapter):
    """
    Adapter for Ensembl reference data.
    
    Usage:
        adapter = EnsemblAdapter()
        
        # Get reference genome
        refs = adapter.search(SearchQuery(
            organism="human",
            keywords=["genome"]
        ))
        
        # Get download URLs
        urls = adapter.get_download_urls("homo_sapiens/GRCh38")
    """
    
    SOURCE = DataSource.ENSEMBL
    BASE_URL = "https://rest.ensembl.org"
    FTP_BASE = "https://ftp.ensembl.org/pub/current_fasta"
    GTF_BASE = "https://ftp.ensembl.org/pub/current_gtf"
    
    def __init__(self, cache_enabled: bool = True, timeout: int = 30):
        """Initialize the Ensembl adapter."""
        super().__init__(cache_enabled, timeout)
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "BioPipelines/1.0"
        })
    
    def search(self, query: SearchQuery) -> List[DatasetInfo]:
        """
        Search for Ensembl reference data.
        
        Unlike experiment databases, Ensembl provides reference data,
        so we search by organism and data type.
        """
        cache_key = self._build_cache_key(query)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        datasets = []
        
        # Get organism info
        organism = query.organism or "human"
        org_info = ASSEMBLY_MAP.get(organism.lower(), {})
        ensembl_name = org_info.get("ensembl_name", organism.replace(" ", "_").lower())
        assembly = org_info.get("assembly", query.assembly or "")
        
        logger.info(f"Searching Ensembl for {ensembl_name} ({assembly})")
        
        try:
            # Genome reference
            if not query.keywords or any(kw in ["genome", "fasta", "dna", "reference"] 
                                          for kw in query.keywords):
                genome_ds = self._create_genome_dataset(ensembl_name, assembly)
                if genome_ds:
                    datasets.append(genome_ds)
            
            # Gene annotation
            if not query.keywords or any(kw in ["annotation", "gtf", "gff", "genes"] 
                                          for kw in query.keywords):
                gtf_ds = self._create_gtf_dataset(ensembl_name, assembly)
                if gtf_ds:
                    datasets.append(gtf_ds)
            
            # cDNA
            if query.keywords and any(kw in ["cdna", "transcripts", "mrna"] 
                                       for kw in query.keywords):
                cdna_ds = self._create_cdna_dataset(ensembl_name, assembly)
                if cdna_ds:
                    datasets.append(cdna_ds)
            
            # Protein
            if query.keywords and any(kw in ["protein", "peptide", "pep"] 
                                       for kw in query.keywords):
                pep_ds = self._create_protein_dataset(ensembl_name, assembly)
                if pep_ds:
                    datasets.append(pep_ds)
            
            self._set_cached(cache_key, datasets)
            return datasets
            
        except Exception as e:
            logger.error(f"Ensembl search failed: {e}")
            return []
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get detailed info about a reference dataset.
        
        dataset_id format: "{organism}/{data_type}" 
        e.g., "homo_sapiens/genome" or "mus_musculus/gtf"
        """
        cache_key = f"dataset:{dataset_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            parts = dataset_id.split("/")
            organism = parts[0]
            data_type = parts[1] if len(parts) > 1 else "genome"
            
            # Get assembly info
            info = self._get_species_info(organism)
            assembly = info.get("assembly", "")
            
            if data_type == "genome":
                dataset = self._create_genome_dataset(organism, assembly)
            elif data_type == "gtf":
                dataset = self._create_gtf_dataset(organism, assembly)
            elif data_type == "cdna":
                dataset = self._create_cdna_dataset(organism, assembly)
            elif data_type == "protein":
                dataset = self._create_protein_dataset(organism, assembly)
            else:
                return None
            
            if dataset:
                dataset.download_urls = self.get_download_urls(dataset_id)
                self._set_cached(cache_key, dataset)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to get Ensembl dataset {dataset_id}: {e}")
            return None
    
    def get_download_urls(self, dataset_id: str) -> List[DownloadURL]:
        """Get download URLs for Ensembl data."""
        urls = []
        
        try:
            parts = dataset_id.split("/")
            organism = parts[0]
            data_type = parts[1] if len(parts) > 1 else "genome"
            
            # Get species info
            info = self._get_species_info(organism)
            assembly = info.get("assembly", "")
            display_name = info.get("display_name", organism)
            
            # Capitalize organism name for FTP paths
            org_cap = organism.capitalize()
            
            if data_type == "genome":
                # Primary assembly and toplevel
                urls.append(DownloadURL(
                    url=f"{self.FTP_BASE}/{organism}/dna/{org_cap}.{assembly}.dna.primary_assembly.fa.gz",
                    filename=f"{org_cap}.{assembly}.dna.primary_assembly.fa.gz",
                    file_type=FileType.FASTA,
                    download_method=DownloadMethod.HTTPS,
                ))
                # Also offer toplevel (includes all sequences)
                urls.append(DownloadURL(
                    url=f"{self.FTP_BASE}/{organism}/dna/{org_cap}.{assembly}.dna.toplevel.fa.gz",
                    filename=f"{org_cap}.{assembly}.dna.toplevel.fa.gz",
                    file_type=FileType.FASTA,
                    download_method=DownloadMethod.HTTPS,
                ))
            
            elif data_type == "gtf":
                urls.append(DownloadURL(
                    url=f"{self.GTF_BASE}/{organism}/{org_cap}.{assembly}.gtf.gz",
                    filename=f"{org_cap}.{assembly}.gtf.gz",
                    file_type=FileType.GTF,
                    download_method=DownloadMethod.HTTPS,
                ))
            
            elif data_type == "cdna":
                urls.append(DownloadURL(
                    url=f"{self.FTP_BASE}/{organism}/cdna/{org_cap}.{assembly}.cdna.all.fa.gz",
                    filename=f"{org_cap}.{assembly}.cdna.all.fa.gz",
                    file_type=FileType.FASTA,
                    download_method=DownloadMethod.HTTPS,
                ))
            
            elif data_type == "protein":
                urls.append(DownloadURL(
                    url=f"{self.FTP_BASE}/{organism}/pep/{org_cap}.{assembly}.pep.all.fa.gz",
                    filename=f"{org_cap}.{assembly}.pep.all.fa.gz",
                    file_type=FileType.FASTA,
                    download_method=DownloadMethod.HTTPS,
                ))
            
            return urls
            
        except Exception as e:
            logger.error(f"Failed to get Ensembl download URLs for {dataset_id}: {e}")
            return []
    
    def get_species_list(self) -> List[Dict[str, Any]]:
        """Get list of all species available in Ensembl."""
        cache_key = "species_list"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/info/species",
                params={"content-type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            species = data.get("species", [])
            self._set_cached(cache_key, species)
            return species
            
        except Exception as e:
            logger.error(f"Failed to get Ensembl species list: {e}")
            return []
    
    def _get_species_info(self, organism: str) -> Dict[str, Any]:
        """Get info about a specific species."""
        species_list = self.get_species_list()
        
        for sp in species_list:
            if sp.get("name") == organism or sp.get("display_name", "").lower() == organism.lower():
                return sp
        
        # Default return for common organisms
        if organism in ASSEMBLY_MAP:
            return {
                "name": ASSEMBLY_MAP[organism]["ensembl_name"],
                "assembly": ASSEMBLY_MAP[organism]["assembly"],
                "display_name": organism.capitalize(),
            }
        
        return {"name": organism, "assembly": "", "display_name": organism}
    
    def _create_genome_dataset(self, organism: str, assembly: str) -> DatasetInfo:
        """Create a DatasetInfo for genome reference."""
        org_display = organism.replace("_", " ").title()
        return DatasetInfo(
            id=f"{organism}/genome",
            source=DataSource.ENSEMBL,
            title=f"{org_display} Genome Reference ({assembly})",
            description=f"Reference genome sequence for {org_display}, assembly {assembly}",
            organism=org_display,
            assembly=assembly,
            metadata={"data_type": "genome", "format": "fasta"},
            web_url=f"https://www.ensembl.org/{organism}/Info/Index",
        )
    
    def _create_gtf_dataset(self, organism: str, assembly: str) -> DatasetInfo:
        """Create a DatasetInfo for GTF annotation."""
        org_display = organism.replace("_", " ").title()
        return DatasetInfo(
            id=f"{organism}/gtf",
            source=DataSource.ENSEMBL,
            title=f"{org_display} Gene Annotation ({assembly})",
            description=f"Gene annotation (GTF) for {org_display}, assembly {assembly}",
            organism=org_display,
            assembly=assembly,
            metadata={"data_type": "annotation", "format": "gtf"},
            web_url=f"https://www.ensembl.org/{organism}/Info/Index",
        )
    
    def _create_cdna_dataset(self, organism: str, assembly: str) -> DatasetInfo:
        """Create a DatasetInfo for cDNA sequences."""
        org_display = organism.replace("_", " ").title()
        return DatasetInfo(
            id=f"{organism}/cdna",
            source=DataSource.ENSEMBL,
            title=f"{org_display} cDNA Sequences ({assembly})",
            description=f"Transcript (cDNA) sequences for {org_display}, assembly {assembly}",
            organism=org_display,
            assembly=assembly,
            metadata={"data_type": "cdna", "format": "fasta"},
            web_url=f"https://www.ensembl.org/{organism}/Info/Index",
        )
    
    def _create_protein_dataset(self, organism: str, assembly: str) -> DatasetInfo:
        """Create a DatasetInfo for protein sequences."""
        org_display = organism.replace("_", " ").title()
        return DatasetInfo(
            id=f"{organism}/protein",
            source=DataSource.ENSEMBL,
            title=f"{org_display} Protein Sequences ({assembly})",
            description=f"Protein sequences for {org_display}, assembly {assembly}",
            organism=org_display,
            assembly=assembly,
            metadata={"data_type": "protein", "format": "fasta"},
            web_url=f"https://www.ensembl.org/{organism}/Info/Index",
        )
    
    def validate_local_reference(self, path: str, ref_type: str = "genome") -> LocalReference:
        """
        Validate a local Ensembl reference file/directory.
        
        Args:
            path: Path to the reference file or directory
            ref_type: Type of reference (genome, annotation, index)
            
        Returns:
            LocalReference with validation status
        """
        path_obj = Path(path)
        
        ref = LocalReference(
            path=str(path_obj),
            name=path_obj.name,
            ref_type=ref_type,
        )
        
        if not path_obj.exists():
            ref.is_valid = False
            ref.validation_message = f"Path does not exist: {path}"
            return ref
        
        if ref_type == "genome":
            # Check for FASTA file
            if path_obj.is_file():
                if path_obj.suffix in [".fa", ".fasta", ".fna"] or path_obj.name.endswith((".fa.gz", ".fasta.gz")):
                    ref.is_valid = True
                    ref.size_bytes = path_obj.stat().st_size
                else:
                    ref.is_valid = False
                    ref.validation_message = "Expected FASTA file (.fa, .fasta, .fna)"
            else:
                ref.is_valid = False
                ref.validation_message = "Expected a file, got directory"
        
        elif ref_type == "annotation":
            if path_obj.is_file():
                if path_obj.suffix in [".gtf", ".gff", ".gff3"] or path_obj.name.endswith((".gtf.gz", ".gff.gz")):
                    ref.is_valid = True
                    ref.size_bytes = path_obj.stat().st_size
                else:
                    ref.is_valid = False
                    ref.validation_message = "Expected GTF/GFF file"
            else:
                ref.is_valid = False
                ref.validation_message = "Expected a file, got directory"
        
        elif ref_type == "index":
            # Index validation depends on aligner
            if path_obj.is_dir():
                files = list(path_obj.iterdir())
                if any(f.name.endswith(".fa") for f in files):
                    ref.aligner = "star"  # STAR index has genome files
                elif any(f.name.endswith(".bt2") for f in files):
                    ref.aligner = "bowtie2"
                elif any(f.name.endswith(".bwt") for f in files):
                    ref.aligner = "bwa"
                
                ref.is_valid = len(files) > 0
                ref.size_bytes = sum(f.stat().st_size for f in files if f.is_file())
            else:
                ref.is_valid = False
                ref.validation_message = "Expected directory for index"
        
        return ref


# Convenience functions
def get_human_genome_url() -> str:
    """Get URL for human genome FASTA (GRCh38)."""
    return "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"


def get_human_gtf_url() -> str:
    """Get URL for human GTF annotation."""
    return "https://ftp.ensembl.org/pub/current_gtf/homo_sapiens/Homo_sapiens.GRCh38.gtf.gz"


def get_mouse_genome_url() -> str:
    """Get URL for mouse genome FASTA (GRCm39)."""
    return "https://ftp.ensembl.org/pub/current_fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz"


def get_mouse_gtf_url() -> str:
    """Get URL for mouse GTF annotation."""
    return "https://ftp.ensembl.org/pub/current_gtf/mus_musculus/Mus_musculus.GRCm39.gtf.gz"
