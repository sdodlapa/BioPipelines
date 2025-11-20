"""
Peak Calling and ChIP-seq Utilities
====================================

Tools for peak analysis and ChIP-seq operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pybedtools


def parse_narrowpeak(narrowpeak_path: Path) -> pd.DataFrame:
    """
    Parse MACS2 narrowPeak file.
    
    Parameters
    ----------
    narrowpeak_path : Path
        Path to narrowPeak file
        
    Returns
    -------
    pd.DataFrame
        Peak annotations
    """
    columns = [
        'chrom', 'start', 'end', 'name', 'score',
        'strand', 'signal_value', 'pvalue', 'qvalue', 'peak'
    ]
    
    df = pd.read_csv(
        narrowpeak_path,
        sep='\t',
        header=None,
        names=columns
    )
    
    return df


def count_peaks_by_chromosome(narrowpeak_path: Path) -> Dict[str, int]:
    """
    Count peaks per chromosome.
    
    Parameters
    ----------
    narrowpeak_path : Path
        Path to narrowPeak file
        
    Returns
    -------
    dict
        Chromosome peak counts
    """
    df = parse_narrowpeak(narrowpeak_path)
    return df['chrom'].value_counts().to_dict()


def filter_peaks_by_score(
    narrowpeak_path: Path,
    output_path: Path,
    min_score: float = 100.0,
    min_qvalue: Optional[float] = None
) -> int:
    """
    Filter peaks by score and/or q-value.
    
    Parameters
    ----------
    narrowpeak_path : Path
        Input narrowPeak file
    output_path : Path
        Output filtered narrowPeak file
    min_score : float
        Minimum score threshold
    min_qvalue : float, optional
        Minimum -log10(qvalue) threshold
        
    Returns
    -------
    int
        Number of peaks passing filters
    """
    df = parse_narrowpeak(narrowpeak_path)
    
    # Score filter
    filtered = df[df['score'] >= min_score]
    
    # Q-value filter
    if min_qvalue is not None:
        filtered = filtered[filtered['qvalue'] >= min_qvalue]
    
    # Save
    filtered.to_csv(output_path, sep='\t', header=False, index=False)
    
    return len(filtered)


def get_peak_summit_bed(narrowpeak_path: Path, output_path: Path) -> None:
    """
    Extract peak summits to BED file.
    
    Parameters
    ----------
    narrowpeak_path : Path
        Input narrowPeak file
    output_path : Path
        Output BED file with summit positions
    """
    df = parse_narrowpeak(narrowpeak_path)
    
    # Calculate summit position
    summit_df = pd.DataFrame({
        'chrom': df['chrom'],
        'start': df['start'] + df['peak'],
        'end': df['start'] + df['peak'] + 1,
        'name': df['name'],
        'score': df['score']
    })
    
    summit_df.to_csv(output_path, sep='\t', header=False, index=False)


def calculate_frip(
    bam_path: Path,
    peaks_path: Path,
    threads: int = 1
) -> float:
    """
    Calculate Fraction of Reads in Peaks (FRiP score).
    
    Parameters
    ----------
    bam_path : Path
        Path to BAM file
    peaks_path : Path
        Path to peaks BED file
    threads : int
        Number of threads for bedtools
        
    Returns
    -------
    float
        FRiP score (0-1)
    """
    import pysam
    
    # Total reads
    bamfile = pysam.AlignmentFile(str(bam_path), 'rb')
    total_reads = bamfile.mapped
    bamfile.close()
    
    # Reads in peaks
    bam_bed = pybedtools.BedTool(str(bam_path))
    peaks_bed = pybedtools.BedTool(str(peaks_path))
    
    reads_in_peaks = len(bam_bed.intersect(peaks_bed, u=True))
    
    if total_reads == 0:
        return 0.0
    
    return reads_in_peaks / total_reads
