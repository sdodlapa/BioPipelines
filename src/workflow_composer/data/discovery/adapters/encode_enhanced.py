"""
Enhanced ENCODE Adapter
=======================

Collects comprehensive metadata during initial search to avoid follow-up calls.

Improvements over base adapter:
1. Fetches file details in batch (parallel)
2. Includes quality metrics and audit information
3. Collects replicate and biosample details
4. Captures experimental conditions and controls
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .encode import ENCODEAdapter, ENCODE_ASSAY_MAP
from ..models import DataSource, FileType, DownloadMethod, SearchQuery
from ..rich_models import (
    RichDatasetInfo, 
    FileInfo, 
    ReplicateInfo,
    QualityMetrics,
    DataQuality,
    ExperimentCondition,
)

logger = logging.getLogger(__name__)


class EnhancedENCODEAdapter(ENCODEAdapter):
    """
    Enhanced ENCODE adapter that collects rich metadata upfront.
    
    Features:
    - Parallel file metadata fetching
    - Quality metrics extraction
    - Replicate information
    - Experimental conditions
    - Audit status (warnings/errors)
    """
    
    def __init__(
        self, 
        cache_enabled: bool = True, 
        timeout: int = 30,
        fetch_files: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize enhanced adapter.
        
        Args:
            cache_enabled: Enable caching
            timeout: Request timeout
            fetch_files: Whether to fetch file details (slower but more complete)
            max_workers: Max parallel requests for file fetching
        """
        super().__init__(cache_enabled, timeout)
        self.fetch_files = fetch_files
        self.max_workers = max_workers
    
    def search_rich(self, query: SearchQuery) -> List[RichDatasetInfo]:
        """
        Search ENCODE and return rich metadata.
        
        Args:
            query: Search query
            
        Returns:
            List of RichDatasetInfo with comprehensive metadata
        """
        # Use parent's search to get experiment list
        basic_results = self.search(query)
        
        if not basic_results:
            return []
        
        # Enhance each result with full metadata
        rich_results = []
        
        for dataset in basic_results:
            try:
                rich = self.get_rich_dataset(dataset.id)
                if rich:
                    rich_results.append(rich)
            except Exception as e:
                logger.warning(f"Failed to enrich {dataset.id}: {e}")
                # Create basic RichDatasetInfo from DatasetInfo
                rich_results.append(self._convert_to_rich(dataset))
        
        return rich_results
    
    def get_rich_dataset(self, dataset_id: str) -> Optional[RichDatasetInfo]:
        """
        Get comprehensive metadata for a single experiment.
        
        Fetches:
        - Experiment metadata
        - Biosample details
        - File list with sizes and formats
        - Quality metrics
        - Audit information
        - Replicate structure
        
        Args:
            dataset_id: ENCODE accession (e.g., ENCSR000ABC)
            
        Returns:
            RichDatasetInfo with all available metadata
        """
        cache_key = f"rich:{dataset_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Fetch experiment with embedded objects
        url = f"{self.BASE_URL}/experiments/{dataset_id}/"
        params = {
            "format": "json",
            "frame": "embedded",  # Include embedded objects
        }
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            rich = self._parse_rich_experiment(data)
            
            if rich and self.fetch_files:
                # Fetch file details in parallel
                rich.files = self._fetch_files_parallel(data.get("files", []))
                rich.file_count = len(rich.files)
                rich.total_size_bytes = sum(f.size_bytes or 0 for f in rich.files)
                rich.file_types = self._compute_file_types(rich.files)
            
            if rich:
                self._set_cached(cache_key, rich)
            
            return rich
            
        except Exception as e:
            logger.error(f"Failed to get rich data for {dataset_id}: {e}")
            return None
    
    def _parse_rich_experiment(self, data: Dict[str, Any]) -> Optional[RichDatasetInfo]:
        """Parse ENCODE JSON to RichDatasetInfo."""
        try:
            accession = data.get("accession")
            if not accession:
                return None
            
            # Basic metadata
            rich = RichDatasetInfo(
                id=accession,
                source="ENCODE",
                accession=accession,
                title=data.get("description", accession),
                description=data.get("description", ""),
                web_url=f"{self.BASE_URL}/experiments/{accession}/",
                status=data.get("status", "unknown"),
            )
            
            # Assay information
            rich.assay_type = ENCODE_ASSAY_MAP.get(
                data.get("assay_title", ""),
                data.get("assay_title", "")
            )
            rich.assay_title = data.get("assay_title", "")
            
            # Target (for ChIP-seq, CUT&RUN, etc.)
            target = data.get("target", {})
            if isinstance(target, dict):
                rich.target = target.get("name", "")
                rich.target_label = target.get("label", "")
            
            # Organism and assembly
            assemblies = data.get("assembly", [])
            if assemblies:
                rich.assembly = assemblies[0] if isinstance(assemblies, list) else assemblies
            
            # Biosample information
            biosample = data.get("biosample_ontology", {})
            if isinstance(biosample, dict):
                rich.biosample_type = biosample.get("classification", "")
                rich.tissue = biosample.get("term_name", "")
                rich.organ = biosample.get("organ_slims", [""])[0] if biosample.get("organ_slims") else ""
                
                # Cell slims
                cell_slims = biosample.get("cell_slims", [])
                if cell_slims:
                    rich.cell_type = cell_slims[0]
            
            # Get detailed biosample from replicates
            replicates = data.get("replicates", [])
            if replicates:
                self._parse_replicates(rich, replicates)
            
            # Lab and project info
            lab = data.get("lab", {})
            if isinstance(lab, dict):
                rich.lab = lab.get("title", "")
                pi = lab.get("pi", {})
                if isinstance(pi, dict):
                    rich.lab_pi = pi.get("title", "")
            
            award = data.get("award", {})
            if isinstance(award, dict):
                rich.project = award.get("project", "")
                rich.award = award.get("name", "")
            
            # Dates
            if data.get("date_released"):
                try:
                    rich.date_released = datetime.fromisoformat(
                        data["date_released"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            
            if data.get("date_created"):
                try:
                    rich.date_created = datetime.fromisoformat(
                        data["date_created"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            
            # Audit information (quality warnings/errors)
            self._parse_audit(rich, data.get("audit", {}))
            
            # Store raw metadata for edge cases
            rich.raw_metadata = {
                "status": data.get("status"),
                "assay_slims": data.get("assay_slims", []),
                "category_slims": data.get("category_slims", []),
                "objective_slims": data.get("objective_slims", []),
                "award_rfa": award.get("rfa", "") if isinstance(award, dict) else "",
            }
            
            return rich
            
        except Exception as e:
            logger.error(f"Failed to parse ENCODE experiment: {e}")
            return None
    
    def _parse_replicates(
        self, 
        rich: RichDatasetInfo, 
        replicates: List[Dict[str, Any]]
    ):
        """Parse replicate information."""
        bio_reps = set()
        tech_reps = set()
        has_control = False
        
        for rep in replicates:
            if not isinstance(rep, dict):
                continue
            
            # Count replicates
            bio_rep = rep.get("biological_replicate_number")
            tech_rep = rep.get("technical_replicate_number")
            if bio_rep:
                bio_reps.add(bio_rep)
            if tech_rep:
                tech_reps.add((bio_rep, tech_rep))
            
            # Get biosample details
            library = rep.get("library", {})
            if isinstance(library, dict):
                biosample = library.get("biosample", {})
                if isinstance(biosample, dict):
                    # Organism
                    organism = biosample.get("organism", {})
                    if isinstance(organism, dict):
                        rich.organism = organism.get("name", "")
                        rich.organism_scientific = organism.get("scientific_name", "")
                    
                    # Donor info
                    donor = biosample.get("donor", {})
                    if isinstance(donor, dict):
                        rich.sex = donor.get("sex", "")
                        rich.age = donor.get("age", "")
                        rich.life_stage = donor.get("life_stage", "")
                        rich.ethnicity = donor.get("ethnicity", "")
                        
                        # Health status
                        health = donor.get("health_status", "")
                        if health and health.lower() not in ("healthy", "normal", ""):
                            rich.disease_state = health
                    
                    # Disease ontology
                    disease_terms = biosample.get("disease_term_name", [])
                    if disease_terms:
                        rich.disease_state = disease_terms[0] if isinstance(disease_terms, list) else disease_terms
                    
                    # Treatments
                    treatments = biosample.get("treatments", [])
                    if treatments:
                        for treatment in treatments:
                            if isinstance(treatment, dict):
                                cond = ExperimentCondition(
                                    name=treatment.get("treatment_term_name", ""),
                                    treatment=treatment.get("treatment_term_name", ""),
                                    treatment_amount=treatment.get("amount", ""),
                                    treatment_duration=treatment.get("duration", ""),
                                )
                                rich.conditions.append(cond)
            
            # Create replicate info
            rep_info = ReplicateInfo(
                replicate_id=f"rep{bio_rep}_{tech_rep}",
                replicate_type="biological" if tech_rep == 1 else "technical",
                biosample_id=rep.get("biosample", {}).get("accession") if isinstance(rep.get("biosample"), dict) else None,
            )
            rich.replicates.append(rep_info)
        
        rich.biological_replicate_count = len(bio_reps)
        rich.technical_replicate_count = len(tech_reps)
        
        # Check for controls
        controls = []
        for rep in replicates:
            lib = rep.get("library", {}) if isinstance(rep, dict) else {}
            if isinstance(lib, dict):
                ctrl = lib.get("nucleic_acid_term_name", "")
                if "input" in ctrl.lower():
                    has_control = True
                    controls.append("input DNA")
        
        rich.has_controls = has_control or bool(controls)
        if controls:
            rich.control_type = controls[0]
    
    def _parse_audit(self, rich: RichDatasetInfo, audit: Dict[str, Any]):
        """Parse ENCODE audit information into quality metrics."""
        if not audit:
            return
        
        warnings = []
        errors = []
        
        # ENCODE audit levels: ERROR, NOT_COMPLIANT, WARNING, INTERNAL_ACTION
        for level, items in audit.items():
            if not isinstance(items, list):
                continue
            
            for item in items:
                if isinstance(item, dict):
                    msg = item.get("detail", item.get("category", "Unknown issue"))
                    
                    if level == "ERROR":
                        errors.append(msg[:100])
                    elif level in ("NOT_COMPLIANT", "WARNING"):
                        warnings.append(msg[:100])
        
        rich.quality.audit_warnings = warnings[:5]  # Limit
        rich.quality.audit_errors = errors[:3]
        
        # Determine quality level
        if errors:
            rich.quality.quality_level = DataQuality.POOR
        elif len(warnings) > 3:
            rich.quality.quality_level = DataQuality.ACCEPTABLE
        elif warnings:
            rich.quality.quality_level = DataQuality.GOOD
        else:
            rich.quality.quality_level = DataQuality.EXCELLENT
    
    def _fetch_files_parallel(self, file_refs: List[Any]) -> List[FileInfo]:
        """Fetch file details in parallel."""
        if not file_refs:
            return []
        
        files = []
        
        # Build file URLs
        file_urls = []
        for ref in file_refs:
            if isinstance(ref, str):
                file_url = f"{self.BASE_URL}{ref}" if ref.startswith("/") else ref
            elif isinstance(ref, dict):
                file_url = f"{self.BASE_URL}{ref.get('@id', '')}"
            else:
                continue
            file_urls.append(file_url)
        
        # Fetch in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_file_info, url): url
                for url in file_urls[:50]  # Limit to 50 files
            }
            
            for future in as_completed(futures, timeout=self.timeout * 2):
                try:
                    file_info = future.result()
                    if file_info:
                        files.append(file_info)
                except Exception as e:
                    logger.debug(f"Failed to fetch file: {e}")
        
        return files
    
    def _fetch_file_info(self, url: str) -> Optional[FileInfo]:
        """Fetch a single file's metadata."""
        try:
            response = self.session.get(
                url,
                params={"format": "json"},
                timeout=self.timeout
            )
            if not response.ok:
                return None
            
            data = response.json()
            
            # Only include released files
            if data.get("status") != "released":
                return None
            
            accession = data.get("accession", "")
            file_format = data.get("file_format", "other")
            
            # Get download URL
            href = data.get("href", "")
            download_url = f"{self.BASE_URL}{href}" if href.startswith("/") else href
            
            return FileInfo(
                accession=accession,
                filename=href.split("/")[-1] if "/" in href else accession,
                file_format=file_format,
                file_type=data.get("file_format_type", data.get("output_category", "")),
                output_type=data.get("output_type", ""),
                size_bytes=data.get("file_size"),
                md5sum=data.get("md5sum"),
                download_url=download_url,
                s3_uri=data.get("s3_uri"),
                biological_replicate=str(data.get("biological_replicates", [""])[0]) if data.get("biological_replicates") else None,
                technical_replicate=str(data.get("technical_replicates", [""])[0]) if data.get("technical_replicates") else None,
                assembly=data.get("assembly"),
                derived_from=[f.get("accession", "") for f in data.get("derived_from", []) if isinstance(f, dict)],
                audit_status="ok" if not data.get("audit") else "warning",
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse file from {url}: {e}")
            return None
    
    def _compute_file_types(self, files: List[FileInfo]) -> Dict[str, Dict[str, Any]]:
        """Compute file type breakdown."""
        breakdown = {}
        
        for f in files:
            fmt = f.file_format.upper()
            if fmt not in breakdown:
                breakdown[fmt] = {"count": 0, "size_bytes": 0}
            
            breakdown[fmt]["count"] += 1
            breakdown[fmt]["size_bytes"] += f.size_bytes or 0
        
        # Add human-readable sizes
        for fmt, info in breakdown.items():
            size = info["size_bytes"]
            if size > 0:
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if size < 1024:
                        info["size_human"] = f"{size:.1f} {unit}"
                        break
                    size /= 1024
            else:
                info["size_human"] = "Unknown"
        
        return breakdown
    
    def _convert_to_rich(self, dataset) -> RichDatasetInfo:
        """Convert basic DatasetInfo to RichDatasetInfo."""
        return RichDatasetInfo(
            id=dataset.id,
            source="ENCODE",
            accession=dataset.id,
            title=dataset.title,
            description=dataset.description,
            web_url=dataset.web_url or f"{self.BASE_URL}/experiments/{dataset.id}/",
            organism=dataset.organism,
            assembly=dataset.assembly,
            assay_type=dataset.assay_type,
            target=dataset.target,
            tissue=dataset.tissue,
            cell_line=dataset.cell_line,
            status=dataset.metadata.get("status", "unknown"),
            lab=dataset.metadata.get("lab", ""),
        )


# Convenience function
def search_encode_rich(
    organism: str = None,
    assay_type: str = None,
    target: str = None,
    tissue: str = None,
    **kwargs
) -> List[RichDatasetInfo]:
    """
    Search ENCODE with rich metadata.
    
    Returns comprehensive dataset information including files,
    quality metrics, and replicate structure.
    """
    query = SearchQuery(
        organism=organism,
        assay_type=assay_type,
        target=target,
        tissue=tissue,
        **kwargs
    )
    adapter = EnhancedENCODEAdapter()
    return adapter.search_rich(query)
