"""
Local Sample Scanner
====================

Scans local directories to discover FASTQ files and build sample manifests.

Features:
- Automatic FASTQ file discovery
- Paired-end read matching (R1/R2)
- Read length detection
- Condition inference from filenames
- Sample sheet generation

Usage:
    from workflow_composer.data.scanner import LocalSampleScanner
    
    scanner = LocalSampleScanner()
    
    # Scan a directory
    samples = scanner.scan_directory(Path("/data/raw"))
    
    # Build manifest
    manifest = scanner.build_manifest(Path("/data/raw"))
"""

import os
import re
import gzip
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from .manifest import SampleInfo, DataManifest, ReferenceInfo, DataSourceType

logger = logging.getLogger(__name__)


# Common patterns for FASTQ files
FASTQ_EXTENSIONS = ['.fastq', '.fq', '.fastq.gz', '.fq.gz']

# Patterns for matching R1/R2 pairs
PAIR_PATTERNS = [
    # Standard Illumina: sample_R1.fastq.gz / sample_R2.fastq.gz
    (r'(.+?)_R1([_.].*)?\.f(?:ast)?q(?:\.gz)?$', r'\1_R2\2.f\3q\4'),
    (r'(.+?)_R1_001\.f(?:ast)?q(?:\.gz)?$', r'\1_R2_001.f\2q\3'),
    
    # Alternative: sample_1.fastq.gz / sample_2.fastq.gz
    (r'(.+?)_1\.f(?:ast)?q(?:\.gz)?$', r'\1_2.f\2q\3'),
    
    # SRA format: SRR123_1.fastq.gz / SRR123_2.fastq.gz
    (r'(SRR\d+)_1\.f(?:ast)?q(?:\.gz)?$', r'\1_2.f\2q\3'),
    
    # With lane info: sample_L001_R1.fastq.gz
    (r'(.+?)_L\d+_R1([_.].*)?\.f(?:ast)?q(?:\.gz)?$', r'\1_L\2_R2\3.f\4q\5'),
]

# Patterns for extracting sample info from filenames
SAMPLE_NAME_PATTERNS = [
    # Remove _R1/_R2, _1/_2 suffixes
    r'^(.+?)(?:_R[12]|_[12])(?:_\d+)?\.f(?:ast)?q(?:\.gz)?$',
    # Remove lane info
    r'^(.+?)(?:_L\d+)?(?:_R[12]|_[12]).*\.f(?:ast)?q(?:\.gz)?$',
]

# Patterns for condition inference
CONDITION_PATTERNS = [
    # Control patterns
    (r'(?:^|[_\-])(?:ctrl|control|untreated|wt|wildtype|mock)(?:[_\-]|$)', 'control'),
    # Treatment patterns
    (r'(?:^|[_\-])(?:treat|treated|drug|stim|stimulated)(?:[_\-]|$)', 'treated'),
    # Knockdown/knockout
    (r'(?:^|[_\-])(?:kd|knockdown|ko|knockout|sirna)(?:[_\-]|$)', 'knockdown'),
    # Overexpression
    (r'(?:^|[_\-])(?:oe|overexpr|ox)(?:[_\-]|$)', 'overexpression'),
    # Time points
    (r'(?:^|[_\-])(\d+)\s*(?:h|hr|hour|min|d|day)(?:[_\-]|$)', r'\1_timepoint'),
]


@dataclass
class ScanResult:
    """Result of scanning a directory."""
    samples: List[SampleInfo]
    unpaired_files: List[Path]
    errors: List[str]
    warnings: List[str]
    scan_path: Path
    
    @property
    def sample_count(self) -> int:
        return len(self.samples)
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class LocalSampleScanner:
    """
    Scans local directories for FASTQ files and builds sample manifests.
    """
    
    def __init__(
        self,
        detect_read_length: bool = True,
        infer_conditions: bool = True,
        max_read_sample: int = 1000,
    ):
        """
        Initialize the scanner.
        
        Args:
            detect_read_length: Whether to sample files to detect read length
            infer_conditions: Whether to infer conditions from filenames
            max_read_sample: Max reads to sample for length detection
        """
        self.detect_read_length = detect_read_length
        self.infer_conditions = infer_conditions
        self.max_read_sample = max_read_sample
    
    def scan_directory(
        self,
        path: Path,
        recursive: bool = True,
        pattern: Optional[str] = None,
    ) -> ScanResult:
        """
        Scan a directory for FASTQ files.
        
        Args:
            path: Directory to scan
            recursive: Whether to scan subdirectories
            pattern: Optional glob pattern to filter files
        
        Returns:
            ScanResult with discovered samples
        """
        path = Path(path)
        
        if not path.exists():
            return ScanResult(
                samples=[],
                unpaired_files=[],
                errors=[f"Directory not found: {path}"],
                warnings=[],
                scan_path=path,
            )
        
        if not path.is_dir():
            return ScanResult(
                samples=[],
                unpaired_files=[],
                errors=[f"Not a directory: {path}"],
                warnings=[],
                scan_path=path,
            )
        
        # Find all FASTQ files
        fastq_files = self._find_fastq_files(path, recursive, pattern)
        
        if not fastq_files:
            return ScanResult(
                samples=[],
                unpaired_files=[],
                errors=[],
                warnings=[f"No FASTQ files found in {path}"],
                scan_path=path,
            )
        
        # Match pairs
        samples, unpaired = self._match_pairs(fastq_files)
        
        # Detect read lengths if enabled
        if self.detect_read_length:
            for sample in samples:
                sample.read_length = self._detect_read_length(sample.fastq_1)
        
        # Infer conditions if enabled
        if self.infer_conditions:
            for sample in samples:
                sample.condition = self._infer_condition(sample.sample_id)
        
        warnings = []
        if unpaired:
            warnings.append(
                f"Found {len(unpaired)} unpaired files "
                "(treating as single-end or check naming)"
            )
        
        return ScanResult(
            samples=samples,
            unpaired_files=unpaired,
            errors=[],
            warnings=warnings,
            scan_path=path,
        )
    
    def _find_fastq_files(
        self,
        path: Path,
        recursive: bool,
        pattern: Optional[str],
    ) -> List[Path]:
        """Find all FASTQ files in directory."""
        files = []
        
        if pattern:
            glob_method = path.rglob if recursive else path.glob
            files = list(glob_method(pattern))
        else:
            for ext in FASTQ_EXTENSIONS:
                glob_method = path.rglob if recursive else path.glob
                files.extend(glob_method(f"*{ext}"))
        
        # Filter to only files
        files = [f for f in files if f.is_file()]
        
        # Sort for consistent ordering
        files.sort()
        
        logger.info(f"Found {len(files)} FASTQ files in {path}")
        return files
    
    def _match_pairs(
        self,
        files: List[Path],
    ) -> Tuple[List[SampleInfo], List[Path]]:
        """
        Match R1/R2 pairs from a list of files.
        
        Returns:
            Tuple of (matched samples, unmatched files)
        """
        samples = []
        matched: Set[Path] = set()
        
        # Group files by potential sample name
        file_by_name: Dict[str, List[Path]] = defaultdict(list)
        
        for f in files:
            name = self._extract_sample_name(f)
            file_by_name[name].append(f)
        
        # Process each group
        for sample_name, group_files in file_by_name.items():
            if len(group_files) == 2:
                # Likely a pair - determine R1 vs R2
                r1, r2 = self._order_pair(group_files[0], group_files[1])
                if r1 and r2:
                    samples.append(SampleInfo(
                        sample_id=sample_name,
                        fastq_1=r1,
                        fastq_2=r2,
                        is_paired=True,
                        source=DataSourceType.LOCAL,
                    ))
                    matched.add(r1)
                    matched.add(r2)
                    continue
            
            elif len(group_files) == 1:
                # Single file - treat as single-end
                samples.append(SampleInfo(
                    sample_id=sample_name,
                    fastq_1=group_files[0],
                    fastq_2=None,
                    is_paired=False,
                    source=DataSourceType.LOCAL,
                ))
                matched.add(group_files[0])
                continue
            
            # Multiple files with same base - try to pair them
            for f in group_files:
                r2 = self._find_pair(f, group_files)
                if r2 and f not in matched and r2 not in matched:
                    samples.append(SampleInfo(
                        sample_id=self._extract_sample_name(f),
                        fastq_1=f,
                        fastq_2=r2,
                        is_paired=True,
                        source=DataSourceType.LOCAL,
                    ))
                    matched.add(f)
                    matched.add(r2)
        
        # Find unmatched files
        unmatched = [f for f in files if f not in matched]
        
        # Create single-end samples for unmatched R1-looking files
        for f in unmatched[:]:  # Copy list to avoid modification during iteration
            if self._is_r1_file(f):
                samples.append(SampleInfo(
                    sample_id=self._extract_sample_name(f),
                    fastq_1=f,
                    fastq_2=None,
                    is_paired=False,
                    source=DataSourceType.LOCAL,
                ))
                unmatched.remove(f)
        
        logger.info(f"Matched {len(samples)} samples, {len(unmatched)} unmatched files")
        return samples, unmatched
    
    def _extract_sample_name(self, file_path: Path) -> str:
        """Extract sample name from filename."""
        name = file_path.name
        
        # Remove extensions
        for ext in ['.gz', '.fastq', '.fq']:
            if name.endswith(ext):
                name = name[:-len(ext)]
        
        # Remove R1/R2, _1/_2 suffixes
        name = re.sub(r'[._]R[12](?:[._]\d+)?$', '', name)
        name = re.sub(r'[._][12]$', '', name)
        
        # Remove lane info
        name = re.sub(r'[._]L\d+', '', name)
        
        # Remove trailing underscores
        name = name.rstrip('_')
        
        return name or file_path.stem
    
    def _order_pair(
        self,
        file1: Path,
        file2: Path,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Order two files as R1, R2.
        
        Returns:
            Tuple of (R1, R2) or (None, None) if not a valid pair
        """
        name1 = file1.name.lower()
        name2 = file2.name.lower()
        
        # Check for R1/R2 pattern
        if '_r1' in name1 or '_r1.' in name1 or '.r1.' in name1:
            if '_r2' in name2 or '_r2.' in name2 or '.r2.' in name2:
                return file1, file2
        if '_r2' in name1 or '_r2.' in name1 or '.r2.' in name1:
            if '_r1' in name2 or '_r1.' in name2 or '.r1.' in name2:
                return file2, file1
        
        # Check for _1/_2 pattern
        if '_1.' in name1 or name1.endswith('_1'):
            if '_2.' in name2 or name2.endswith('_2'):
                return file1, file2
        if '_2.' in name1 or name1.endswith('_2'):
            if '_1.' in name2 or name2.endswith('_1'):
                return file2, file1
        
        return None, None
    
    def _find_pair(self, file: Path, candidates: List[Path]) -> Optional[Path]:
        """Find the pair file for a given file."""
        name = file.name
        
        # Generate expected pair name
        pair_name = None
        
        if '_R1' in name:
            pair_name = name.replace('_R1', '_R2')
        elif '_R2' in name:
            pair_name = name.replace('_R2', '_R1')
        elif '_1.' in name:
            pair_name = name.replace('_1.', '_2.')
        elif '_2.' in name:
            pair_name = name.replace('_2.', '_1.')
        
        if pair_name:
            for candidate in candidates:
                if candidate.name == pair_name and candidate != file:
                    return candidate
        
        return None
    
    def _is_r1_file(self, file: Path) -> bool:
        """Check if file looks like an R1 file."""
        name = file.name.lower()
        return '_r1' in name or '_1.' in name or name.endswith('_1')
    
    def _detect_read_length(self, fastq_path: Path) -> Optional[int]:
        """
        Detect read length by sampling the first few reads.
        
        Args:
            fastq_path: Path to FASTQ file
        
        Returns:
            Most common read length, or None if unable to detect
        """
        try:
            lengths = []
            
            # Handle gzipped files
            open_func = gzip.open if str(fastq_path).endswith('.gz') else open
            mode = 'rt' if str(fastq_path).endswith('.gz') else 'r'
            
            with open_func(fastq_path, mode) as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    # Sequence is every 4th line starting from line 2
                    if line_count % 4 == 2:
                        lengths.append(len(line.strip()))
                    
                    if len(lengths) >= self.max_read_sample:
                        break
            
            if lengths:
                # Return most common length
                from collections import Counter
                return Counter(lengths).most_common(1)[0][0]
            
        except Exception as e:
            logger.warning(f"Failed to detect read length for {fastq_path}: {e}")
        
        return None
    
    def _infer_condition(self, sample_id: str) -> Optional[str]:
        """
        Infer experimental condition from sample name.
        
        Args:
            sample_id: Sample identifier
        
        Returns:
            Inferred condition or None
        """
        sample_lower = sample_id.lower()
        
        for pattern, condition in CONDITION_PATTERNS:
            match = re.search(pattern, sample_lower, re.IGNORECASE)
            if match:
                # Handle capture groups (for time points)
                if match.groups():
                    return match.expand(condition)
                return condition
        
        return None
    
    def build_manifest(
        self,
        sample_dir: Path,
        reference: Optional[ReferenceInfo] = None,
        output_dir: Optional[Path] = None,
    ) -> DataManifest:
        """
        Build a complete DataManifest from a directory scan.
        
        Args:
            sample_dir: Directory containing samples
            reference: Optional reference information
            output_dir: Optional output directory for results
        
        Returns:
            DataManifest ready for workflow generation
        """
        result = self.scan_directory(sample_dir)
        
        manifest = DataManifest(
            samples=result.samples,
            reference=reference,
            created_from="local_scan",
        )
        
        if output_dir:
            manifest.output_dir = output_dir
        
        # Add warnings from scan
        manifest.warnings.extend(result.warnings)
        
        # Validate
        manifest.validate()
        
        return manifest


def scan_for_samples(
    directory: str,
    recursive: bool = True,
) -> List[SampleInfo]:
    """
    Convenience function to scan a directory for samples.
    
    Args:
        directory: Path to directory
        recursive: Whether to scan subdirectories
    
    Returns:
        List of discovered samples
    """
    scanner = LocalSampleScanner()
    result = scanner.scan_directory(Path(directory), recursive=recursive)
    return result.samples


def build_manifest_from_directory(
    directory: str,
    organism: Optional[str] = None,
    assembly: Optional[str] = None,
) -> DataManifest:
    """
    Convenience function to build a manifest from a directory.
    
    Args:
        directory: Path to sample directory
        organism: Optional organism name
        assembly: Optional genome assembly
    
    Returns:
        DataManifest
    """
    reference = None
    if organism and assembly:
        reference = ReferenceInfo(organism=organism, assembly=assembly)
    
    scanner = LocalSampleScanner()
    return scanner.build_manifest(Path(directory), reference=reference)
