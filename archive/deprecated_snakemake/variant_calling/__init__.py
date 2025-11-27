"""
VCF Statistics and Analysis Utilities
======================================

Tools for analyzing and summarizing VCF files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pysam
from collections import defaultdict, Counter
import pandas as pd


def count_variants_by_type(vcf_path: Path) -> Dict[str, int]:
    """
    Count variants by type (SNP, INDEL, etc.).
    
    Parameters
    ----------
    vcf_path : Path
        Path to VCF file
        
    Returns
    -------
    dict
        Dictionary with variant type counts
    """
    counts = Counter()
    
    vcf = pysam.VariantFile(str(vcf_path))
    
    for record in vcf:
        # Determine variant type
        ref_len = len(record.ref)
        alt_lens = [len(str(alt)) for alt in record.alts]
        
        if all(l == 1 for l in [ref_len] + alt_lens):
            counts['SNP'] += 1
        elif ref_len != max(alt_lens):
            counts['INDEL'] += 1
        else:
            counts['OTHER'] += 1
    
    vcf.close()
    return dict(counts)


def count_variants_by_chromosome(vcf_path: Path) -> Dict[str, int]:
    """
    Count variants per chromosome.
    
    Parameters
    ----------
    vcf_path : Path
        Path to VCF file
        
    Returns
    -------
    dict
        Dictionary with chromosome counts
    """
    counts = Counter()
    
    vcf = pysam.VariantFile(str(vcf_path))
    
    for record in vcf:
        counts[record.chrom] += 1
    
    vcf.close()
    return dict(counts)


def calculate_ti_tv_ratio(vcf_path: Path) -> float:
    """
    Calculate transition/transversion ratio for SNPs.
    
    Parameters
    ----------
    vcf_path : Path
        Path to VCF file
        
    Returns
    -------
    float
        Ti/Tv ratio
    """
    transitions = 0
    transversions = 0
    
    transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
    
    vcf = pysam.VariantFile(str(vcf_path))
    
    for record in vcf:
        if len(record.ref) == 1 and all(len(str(alt)) == 1 for alt in record.alts):
            for alt in record.alts:
                pair = (record.ref, str(alt))
                if pair in transition_pairs:
                    transitions += 1
                else:
                    transversions += 1
    
    vcf.close()
    
    if transversions == 0:
        return float('inf')
    return transitions / transversions


def get_variant_quality_stats(vcf_path: Path) -> Dict[str, float]:
    """
    Calculate summary statistics for variant quality scores.
    
    Parameters
    ----------
    vcf_path : Path
        Path to VCF file
        
    Returns
    -------
    dict
        Dictionary with quality statistics (mean, median, min, max)
    """
    qualities = []
    
    vcf = pysam.VariantFile(str(vcf_path))
    
    for record in vcf:
        if record.qual is not None:
            qualities.append(record.qual)
    
    vcf.close()
    
    if not qualities:
        return {'mean': 0, 'median': 0, 'min': 0, 'max': 0}
    
    qualities_array = pd.Series(qualities)
    
    return {
        'mean': float(qualities_array.mean()),
        'median': float(qualities_array.median()),
        'min': float(qualities_array.min()),
        'max': float(qualities_array.max()),
        'std': float(qualities_array.std())
    }


def filter_vcf_by_quality(
    vcf_path: Path,
    output_path: Path,
    min_qual: float = 30.0,
    min_depth: Optional[int] = None
) -> int:
    """
    Filter VCF file by quality score and optionally depth.
    
    Parameters
    ----------
    vcf_path : Path
        Input VCF file
    output_path : Path
        Output filtered VCF file
    min_qual : float
        Minimum quality score
    min_depth : int, optional
        Minimum depth (if specified)
        
    Returns
    -------
    int
        Number of variants passing filters
    """
    vcf_in = pysam.VariantFile(str(vcf_path))
    vcf_out = pysam.VariantFile(str(output_path), 'w', header=vcf_in.header)
    
    passed = 0
    
    for record in vcf_in:
        # Quality filter
        if record.qual is None or record.qual < min_qual:
            continue
        
        # Depth filter (if specified)
        if min_depth is not None:
            if 'DP' in record.info:
                if record.info['DP'] < min_depth:
                    continue
        
        vcf_out.write(record)
        passed += 1
    
    vcf_in.close()
    vcf_out.close()
    
    return passed


def vcf_summary_report(vcf_path: Path) -> pd.DataFrame:
    """
    Generate comprehensive summary report for VCF file.
    
    Parameters
    ----------
    vcf_path : Path
        Path to VCF file
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    stats = {}
    
    # Variant counts
    type_counts = count_variants_by_type(vcf_path)
    stats.update(type_counts)
    
    # Ti/Tv ratio
    stats['TiTv_ratio'] = calculate_ti_tv_ratio(vcf_path)
    
    # Quality stats
    qual_stats = get_variant_quality_stats(vcf_path)
    stats.update({f'QUAL_{k}': v for k, v in qual_stats.items()})
    
    # Chromosome counts
    chrom_counts = count_variants_by_chromosome(vcf_path)
    stats['total_variants'] = sum(chrom_counts.values())
    stats['num_chromosomes'] = len(chrom_counts)
    
    return pd.DataFrame([stats])
