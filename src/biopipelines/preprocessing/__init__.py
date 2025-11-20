"""
Preprocessing Utilities
=======================

Tools for QC parsing and preprocessing operations.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd


def parse_fastqc_data(fastqc_data_path: Path) -> Dict[str, any]:
    """
    Parse FastQC data file.
    
    Parameters
    ----------
    fastqc_data_path : Path
        Path to fastqc_data.txt file
        
    Returns
    -------
    dict
        Parsed FastQC metrics
    """
    metrics = {}
    
    with open(fastqc_data_path, 'r') as f:
        for line in f:
            if line.startswith('Total Sequences'):
                metrics['total_sequences'] = int(line.split('\t')[1])
            elif line.startswith('Sequences flagged as poor quality'):
                metrics['poor_quality'] = int(line.split('\t')[1])
            elif line.startswith('%GC'):
                metrics['gc_content'] = float(line.split('\t')[1])
            elif line.startswith('Sequence length'):
                metrics['sequence_length'] = line.split('\t')[1].strip()
    
    return metrics


def parse_fastp_json(fastp_json_path: Path) -> Dict[str, any]:
    """
    Parse fastp JSON output.
    
    Parameters
    ----------
    fastp_json_path : Path
        Path to fastp JSON file
        
    Returns
    -------
    dict
        Parsed fastp metrics
    """
    with open(fastp_json_path, 'r') as f:
        data = json.load(f)
    
    metrics = {
        'total_reads_before': data['summary']['before_filtering']['total_reads'],
        'total_reads_after': data['summary']['after_filtering']['total_reads'],
        'total_bases_before': data['summary']['before_filtering']['total_bases'],
        'total_bases_after': data['summary']['after_filtering']['total_bases'],
        'q20_rate_before': data['summary']['before_filtering']['q20_rate'],
        'q20_rate_after': data['summary']['after_filtering']['q20_rate'],
        'q30_rate_before': data['summary']['before_filtering']['q30_rate'],
        'q30_rate_after': data['summary']['after_filtering']['q30_rate'],
        'gc_content_before': data['summary']['before_filtering']['gc_content'],
        'gc_content_after': data['summary']['after_filtering']['gc_content'],
    }
    
    return metrics


def aggregate_qc_metrics(qc_dir: Path, tool: str = 'fastp') -> pd.DataFrame:
    """
    Aggregate QC metrics from multiple samples.
    
    Parameters
    ----------
    qc_dir : Path
        Directory containing QC files
    tool : str
        QC tool: 'fastp' or 'fastqc'
        
    Returns
    -------
    pd.DataFrame
        QC metrics for all samples
    """
    metrics_list = []
    
    if tool == 'fastp':
        json_files = list(qc_dir.glob('*.json'))
        for json_file in json_files:
            sample_name = json_file.stem
            metrics = parse_fastp_json(json_file)
            metrics['sample'] = sample_name
            metrics_list.append(metrics)
    
    elif tool == 'fastqc':
        data_files = list(qc_dir.glob('*/fastqc_data.txt'))
        for data_file in data_files:
            sample_name = data_file.parent.name.replace('_fastqc', '')
            metrics = parse_fastqc_data(data_file)
            metrics['sample'] = sample_name
            metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)
