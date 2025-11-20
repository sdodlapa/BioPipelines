"""
Alignment Utilities
===================

Tools for BAM file operations and alignment statistics.
"""

from pathlib import Path
from typing import Dict, Optional
import pysam
import pandas as pd


def get_alignment_stats(bam_path: Path) -> Dict[str, int]:
    """
    Get basic alignment statistics from BAM file.
    
    Parameters
    ----------
    bam_path : Path
        Path to BAM file
        
    Returns
    -------
    dict
        Alignment statistics
    """
    bamfile = pysam.AlignmentFile(str(bam_path), 'rb')
    
    # Use samtools stats if available, otherwise count reads
    stats = {
        'total_reads': 0,
        'mapped_reads': 0,
        'unmapped_reads': 0,
        'properly_paired': 0,
        'duplicates': 0
    }
    
    try:
        # Try using index stats (faster)
        stats['mapped_reads'] = bamfile.mapped
        stats['unmapped_reads'] = bamfile.unmapped
        stats['total_reads'] = stats['mapped_reads'] + stats['unmapped_reads']
    except:
        # Fall back to counting
        for read in bamfile:
            stats['total_reads'] += 1
            if not read.is_unmapped:
                stats['mapped_reads'] += 1
                if read.is_proper_pair:
                    stats['properly_paired'] += 1
                if read.is_duplicate:
                    stats['duplicates'] += 1
            else:
                stats['unmapped_reads'] += 1
    
    bamfile.close()
    
    # Calculate mapping rate
    if stats['total_reads'] > 0:
        stats['mapping_rate'] = stats['mapped_reads'] / stats['total_reads']
    else:
        stats['mapping_rate'] = 0.0
    
    return stats


def calculate_insert_sizes(
    bam_path: Path,
    max_reads: int = 1000000
) -> pd.Series:
    """
    Calculate insert size distribution from paired-end BAM.
    
    Parameters
    ----------
    bam_path : Path
        Path to BAM file
    max_reads : int
        Maximum reads to sample
        
    Returns
    -------
    pd.Series
        Insert sizes
    """
    bamfile = pysam.AlignmentFile(str(bam_path), 'rb')
    
    insert_sizes = []
    count = 0
    
    for read in bamfile:
        if count >= max_reads:
            break
        
        if (read.is_proper_pair and 
            not read.is_unmapped and 
            not read.mate_is_unmapped and
            read.is_read1):
            
            insert_size = abs(read.template_length)
            if insert_size > 0 and insert_size < 2000:  # Filter extreme values
                insert_sizes.append(insert_size)
                count += 1
    
    bamfile.close()
    
    return pd.Series(insert_sizes)
