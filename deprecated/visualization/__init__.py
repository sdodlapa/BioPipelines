"""
Visualization Utilities
=======================

Plotting functions for genomics data visualization.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_volcano(
    results: pd.DataFrame,
    log2fc_col: str = 'log2FoldChange',
    padj_col: str = 'padj',
    log2fc_threshold: float = 1.0,
    padj_threshold: float = 0.05,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create volcano plot for differential expression results.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results DataFrame with log2FC and adjusted p-value columns
    log2fc_col : str
        Column name for log2 fold change
    padj_col : str
        Column name for adjusted p-value
    log2fc_threshold : float
        Log2 fold change threshold for significance
    padj_threshold : float
        Adjusted p-value threshold
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remove NA values
    plot_data = results.dropna(subset=[log2fc_col, padj_col])
    
    # Calculate -log10(padj)
    plot_data = plot_data.copy()
    plot_data['-log10_padj'] = -np.log10(plot_data[padj_col])
    
    # Classify points
    plot_data['significant'] = (
        (np.abs(plot_data[log2fc_col]) > log2fc_threshold) &
        (plot_data[padj_col] < padj_threshold)
    )
    
    # Plot
    colors = {True: 'red', False: 'gray'}
    for sig, group in plot_data.groupby('significant'):
        ax.scatter(
            group[log2fc_col],
            group['-log10_padj'],
            c=colors[sig],
            alpha=0.6,
            s=20,
            label='Significant' if sig else 'Not significant'
        )
    
    # Add threshold lines
    ax.axhline(-np.log10(padj_threshold), color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(log2fc_threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-log2fc_threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Log2 Fold Change', fontsize=12)
    ax.set_ylabel('-Log10 Adjusted P-value', fontsize=12)
    ax.set_title('Volcano Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_ma(
    results: pd.DataFrame,
    basemean_col: str = 'baseMean',
    log2fc_col: str = 'log2FoldChange',
    padj_col: str = 'padj',
    padj_threshold: float = 0.05,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create MA plot (log fold change vs mean expression).
    
    Parameters
    ----------
    results : pd.DataFrame
        Results DataFrame
    basemean_col : str
        Column for base mean expression
    log2fc_col : str
        Column for log2 fold change
    padj_col : str
        Column for adjusted p-value
    padj_threshold : float
        Significance threshold
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_data = results.dropna(subset=[basemean_col, log2fc_col, padj_col])
    
    # Log scale base mean
    plot_data = plot_data.copy()
    plot_data['log_basemean'] = np.log10(plot_data[basemean_col] + 1)
    
    # Classify significance
    plot_data['significant'] = plot_data[padj_col] < padj_threshold
    
    # Plot
    colors = {True: 'red', False: 'gray'}
    for sig, group in plot_data.groupby('significant'):
        ax.scatter(
            group['log_basemean'],
            group[log2fc_col],
            c=colors[sig],
            alpha=0.5,
            s=20,
            label='Significant' if sig else 'Not significant'
        )
    
    ax.axhline(0, color='blue', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Log10 Mean Expression', fontsize=12)
    ax.set_ylabel('Log2 Fold Change', fontsize=12)
    ax.set_title('MA Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pca(
    counts: pd.DataFrame,
    sample_groups: Optional[pd.Series] = None,
    n_components: int = 2,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create PCA plot from count matrix.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Count matrix (genes x samples)
    sample_groups : pd.Series, optional
        Sample group labels for coloring
    n_components : int
        Number of components (2 or 3)
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Transpose for PCA (samples x genes)
    data = counts.T
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data_scaled)
    
    # Plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        if sample_groups is not None:
            for group in sample_groups.unique():
                mask = sample_groups == group
                ax.scatter(
                    components[mask, 0],
                    components[mask, 1],
                    label=group,
                    s=100,
                    alpha=0.7
                )
        else:
            ax.scatter(components[:, 0], components[:, 1], s=100, alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        ax.set_title('PCA Plot', fontsize=14, fontweight='bold')
        
        if sample_groups is not None:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_heatmap(
    data: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'RdYlBu_r',
    **kwargs
) -> plt.Figure:
    """
    Create heatmap from count matrix.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data matrix to plot
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    **kwargs
        Additional arguments for seaborn.clustermap
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Z-score normalization
    data_norm = (data - data.mean(axis=1).values.reshape(-1, 1)) / data.std(axis=1).values.reshape(-1, 1)
    
    # Create clustermap
    g = sns.clustermap(
        data_norm,
        cmap=cmap,
        figsize=figsize,
        **kwargs
    )
    
    plt.suptitle('Expression Heatmap', fontsize=14, fontweight='bold', y=0.98)
    
    if output_path:
        g.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return g.fig


def plot_qc_metrics(
    qc_df: pd.DataFrame,
    metric_col: str,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create QC metric visualization.
    
    Parameters
    ----------
    qc_df : pd.DataFrame
        QC metrics DataFrame
    metric_col : str
        Column to plot
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    ax1.bar(range(len(qc_df)), qc_df[metric_col])
    ax1.set_xlabel('Sample', fontsize=11)
    ax1.set_ylabel(metric_col, fontsize=11)
    ax1.set_title(f'{metric_col} by Sample', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Distribution
    ax2.hist(qc_df[metric_col], bins=20, edgecolor='black', alpha=0.7)
    ax2.set_xlabel(metric_col, fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'{metric_col} Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(qc_df[metric_col].median(), color='red', linestyle='--', label='Median')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
