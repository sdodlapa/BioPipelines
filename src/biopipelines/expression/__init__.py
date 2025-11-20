"""
Expression Analysis Utilities
==============================

Tools for RNA-seq count matrix operations and analysis.
"""

from pathlib import Path
from typing import List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats


def load_count_matrix(
    count_file: Path,
    skiprows: int = 1,
    index_col: int = 0
) -> pd.DataFrame:
    """
    Load count matrix from featureCounts or similar output.
    
    Parameters
    ----------
    count_file : Path
        Path to count file
    skiprows : int
        Number of rows to skip (default: 1 for featureCounts header)
    index_col : int
        Column to use as index (default: 0 for gene IDs)
        
    Returns
    -------
    pd.DataFrame
        Count matrix with genes as rows, samples as columns
    """
    df = pd.read_csv(count_file, sep='\t', skiprows=skiprows, index_col=index_col)
    
    # Remove annotation columns (Chr, Start, End, Strand, Length)
    annotation_cols = ['Chr', 'Start', 'End', 'Strand', 'Length']
    df = df.drop(columns=[col for col in annotation_cols if col in df.columns])
    
    # Clean sample names (remove path and .bam extension)
    df.columns = [col.split('/')[-1].replace('.bam', '') for col in df.columns]
    
    return df


def normalize_counts(
    counts: pd.DataFrame,
    method: str = 'tpm',
    gene_lengths: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Normalize count matrix.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Raw count matrix
    method : str
        Normalization method: 'tpm', 'rpkm', or 'cpm'
    gene_lengths : pd.Series, optional
        Gene lengths (required for TPM and RPKM)
        
    Returns
    -------
    pd.DataFrame
        Normalized counts
    """
    if method == 'cpm':
        # Counts per million
        return counts.div(counts.sum(axis=0), axis=1) * 1e6
    
    elif method in ['tpm', 'rpkm']:
        if gene_lengths is None:
            raise ValueError(f"{method.upper()} requires gene lengths")
        
        # Rate per kilobase
        rpk = counts.div(gene_lengths / 1000, axis=0)
        
        if method == 'rpkm':
            # RPKM: RPK per million mapped reads
            return rpk.div(counts.sum(axis=0), axis=1) * 1e6
        else:
            # TPM: Normalize RPK values to sum to 1 million
            return rpk.div(rpk.sum(axis=0), axis=1) * 1e6
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def filter_low_counts(
    counts: pd.DataFrame,
    min_count: int = 10,
    min_samples: int = 2
) -> pd.DataFrame:
    """
    Filter genes with low counts across samples.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Count matrix
    min_count : int
        Minimum count threshold
    min_samples : int
        Minimum number of samples that must exceed threshold
        
    Returns
    -------
    pd.DataFrame
        Filtered count matrix
    """
    # Count how many samples have counts >= min_count for each gene
    passing_samples = (counts >= min_count).sum(axis=1)
    
    # Keep genes passing in at least min_samples
    return counts[passing_samples >= min_samples]


def calculate_fold_change(
    counts: pd.DataFrame,
    condition1_samples: List[str],
    condition2_samples: List[str],
    pseudocount: float = 1.0
) -> pd.Series:
    """
    Calculate log2 fold change between two conditions.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Count or normalized matrix
    condition1_samples : list
        Sample names for condition 1
    condition2_samples : list
        Sample names for condition 2
    pseudocount : float
        Pseudocount to add before log transformation
        
    Returns
    -------
    pd.Series
        Log2 fold changes
    """
    mean1 = counts[condition1_samples].mean(axis=1)
    mean2 = counts[condition2_samples].mean(axis=1)
    
    log2fc = np.log2((mean1 + pseudocount) / (mean2 + pseudocount))
    
    return log2fc


def perform_ttest(
    counts: pd.DataFrame,
    condition1_samples: List[str],
    condition2_samples: List[str]
) -> pd.DataFrame:
    """
    Perform t-test for differential expression.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Normalized count matrix
    condition1_samples : list
        Sample names for condition 1
    condition2_samples : list
        Sample names for condition 2
        
    Returns
    -------
    pd.DataFrame
        DataFrame with test statistics and p-values
    """
    results = []
    
    for gene in counts.index:
        group1 = counts.loc[gene, condition1_samples]
        group2 = counts.loc[gene, condition2_samples]
        
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        results.append({
            'gene': gene,
            'mean_condition1': group1.mean(),
            'mean_condition2': group2.mean(),
            'log2_fold_change': np.log2((group1.mean() + 1) / (group2.mean() + 1)),
            't_statistic': t_stat,
            'p_value': p_value
        })
    
    df = pd.DataFrame(results)
    
    # Add adjusted p-values (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    _, df['padj'], _, _ = multipletests(df['p_value'], method='fdr_bh')
    
    return df.sort_values('p_value')


def get_top_variable_genes(
    counts: pd.DataFrame,
    n_genes: int = 500,
    method: str = 'variance'
) -> List[str]:
    """
    Get top variable genes for dimensionality reduction.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Normalized count matrix
    n_genes : int
        Number of genes to return
    method : str
        Method: 'variance' or 'cv' (coefficient of variation)
        
    Returns
    -------
    list
        Gene IDs of most variable genes
    """
    if method == 'variance':
        var = counts.var(axis=1)
        top_genes = var.nlargest(n_genes).index.tolist()
    elif method == 'cv':
        cv = counts.std(axis=1) / (counts.mean(axis=1) + 1)
        top_genes = cv.nlargest(n_genes).index.tolist()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return top_genes


def calculate_sample_correlation(
    counts: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate pairwise sample correlations.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Count matrix
    method : str
        Correlation method: 'pearson' or 'spearman'
        
    Returns
    -------
    pd.DataFrame
        Sample correlation matrix
    """
    return counts.corr(method=method)
