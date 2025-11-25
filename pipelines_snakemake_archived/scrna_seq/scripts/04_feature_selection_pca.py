#!/usr/bin/env python3
"""
Feature selection and PCA
"""

import sys
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Snakemake inputs/outputs
input_h5ad = snakemake.input.h5ad
output_h5ad = snakemake.output.h5ad
hvg_plot = snakemake.output.hvg_plot
pca_plot = snakemake.output.pca_plot

# Parameters
n_top_genes = snakemake.params.n_top_genes
n_pcs = snakemake.params.n_pcs

print(f"Loading data from {input_h5ad}...")
adata = sc.read_h5ad(input_h5ad)

print(f"\nDataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Identify highly variable genes
print(f"\nIdentifying top {n_top_genes} highly variable genes...")
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=n_top_genes,
    flavor='seurat',
    batch_key=None  # Set to batch column if doing batch correction
)

n_hvg = adata.var['highly_variable'].sum()
print(f"Found {n_hvg} highly variable genes")

# Plot highly variable genes
fig = sc.pl.highly_variable_genes(adata, show=False)
plt.savefig(hvg_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"HVG plot saved to {hvg_plot}")

# Store full dataset
adata.raw = adata

# Subset to highly variable genes for downstream analysis
print(f"Subsetting to {n_hvg} highly variable genes...")
adata = adata[:, adata.var['highly_variable']]

# Regress out unwanted sources of variation (optional)
print("\nRegressing out unwanted variation (total counts, mitochondrial %)...")
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

# Scale data
print("Scaling data to unit variance...")
sc.pp.scale(adata, max_value=10)

# PCA
print(f"\nRunning PCA (computing {n_pcs} components)...")
sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack', random_state=42)

# Plot PCA variance ratio
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Explained variance - use scanpy without ax parameter
plt.subplot(1, 2, 1)
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=n_pcs, show=False)
plt.title('PCA Variance Ratio (log scale)')

# Cumulative variance
plt.subplot(1, 2, 2)
cumsum_var = np.cumsum(adata.uns['pca']['variance_ratio'])
plt.plot(range(1, len(cumsum_var)+1), cumsum_var, 'o-', markersize=3)
plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% variance')
plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95% variance')
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance')
plt.title('Cumulative Variance Explained')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(pca_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"PCA plot saved to {pca_plot}")

# Find optimal number of PCs (90% variance)
pcs_90_idx = np.where(cumsum_var >= 0.9)[0]
n_pcs_90 = pcs_90_idx[0] + 1 if len(pcs_90_idx) > 0 else n_pcs

pcs_95_idx = np.where(cumsum_var >= 0.95)[0]
n_pcs_95 = pcs_95_idx[0] + 1 if len(pcs_95_idx) > 0 else n_pcs

print("\n" + "="*80)
print("PCA SUMMARY")
print("="*80)
print(f"\nTotal PCs computed: {n_pcs}")
print(f"PCs for 90% variance: {n_pcs_90} (max variance: {cumsum_var[n_pcs_90-1]:.4f})")
print(f"PCs for 95% variance: {n_pcs_95} (max variance: {cumsum_var[n_pcs_95-1]:.4f})")
print(f"\nVariance explained by first PC: {adata.uns['pca']['variance_ratio'][0]:.4f}")
print(f"Variance explained by top 10 PCs: {cumsum_var[min(9, len(cumsum_var)-1)]:.4f}")
print(f"Variance explained by top 20 PCs: {cumsum_var[min(19, len(cumsum_var)-1)]:.4f}")
print(f"Variance explained by top 30 PCs: {cumsum_var[min(29, len(cumsum_var)-1)]:.4f}")

# Plot top PCs - use scanpy multi-panel plotting
sc.pl.pca(adata, color=['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'pct_counts_ribo'], 
          show=False, ncols=2)

pca_colored_plot = Path(pca_plot).parent / f"{Path(pca_plot).stem}_colored.png"
plt.savefig(pca_colored_plot, dpi=300, bbox_inches='tight')
plt.close()

# Save gene loadings for top PCs
top_loadings = []
for pc in range(5):  # Top 5 PCs
    loadings = pd.DataFrame({
        'gene': adata.var_names,
        f'PC{pc+1}': adata.varm['PCs'][:, pc]
    }).sort_values(f'PC{pc+1}', key=abs, ascending=False)
    top_loadings.append(loadings.head(20))

loadings_file = Path(output_h5ad).parent / f"{Path(output_h5ad).stem}_top_pc_loadings.csv"
pd.concat(top_loadings, axis=1).to_csv(loadings_file)
print(f"\nTop PC loadings saved to {loadings_file}")

# Save processed data
adata.write(output_h5ad)
print(f"Processed data saved to {output_h5ad}")

print("\nFeature selection and PCA complete!")
