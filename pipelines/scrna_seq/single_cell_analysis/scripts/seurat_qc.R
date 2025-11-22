# Seurat QC and Filtering Script
# Load libraries
library(Seurat)
library(ggplot2)
library(dplyr)

# Get parameters from Snakemake
matrix_file <- snakemake@input[[1]]
output_seurat <- snakemake@output[[1]]
output_metrics <- snakemake@output[[2]]
output_violin <- snakemake@output[[3]]

min_genes <- snakemake@params$min_genes
max_genes <- snakemake@params$max_genes
max_mt <- snakemake@params$max_mt
sample_name <- snakemake@params$sample_name

# Load 10x data
cat("Loading 10x data from:", matrix_file, "\n")
data <- Read10X_h5(matrix_file)

# Create Seurat object
seurat <- CreateSeuratObject(
  counts = data,
  project = sample_name,
  min.cells = 3,
  min.features = 200
)

cat("Cells before QC:", ncol(seurat), "\n")
cat("Genes:", nrow(seurat), "\n")

# Calculate mitochondrial percentage
seurat[["percent.mt"]] <- PercentageFeatureSet(seurat, pattern = "^MT-")

# QC metrics before filtering
qc_before <- data.frame(
  n_cells = ncol(seurat),
  n_genes = nrow(seurat),
  median_genes_per_cell = median(seurat$nFeature_RNA),
  median_umis_per_cell = median(seurat$nCount_RNA),
  median_mt_percent = median(seurat$percent.mt)
)

# QC violin plot
p <- VlnPlot(seurat, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
ggsave(output_violin, plot = p, width = 12, height = 4)

# Filter cells
seurat <- subset(seurat, subset = 
  nFeature_RNA > min_genes & 
  nFeature_RNA < max_genes & 
  percent.mt < max_mt
)

cat("Cells after QC:", ncol(seurat), "\n")

# QC metrics after filtering
qc_after <- data.frame(
  n_cells = ncol(seurat),
  n_genes = nrow(seurat),
  median_genes_per_cell = median(seurat$nFeature_RNA),
  median_umis_per_cell = median(seurat$nCount_RNA),
  median_mt_percent = median(seurat$percent.mt)
)

# Write metrics
sink(output_metrics)
cat("=== QC Metrics Before Filtering ===\n")
print(qc_before)
cat("\n=== QC Metrics After Filtering ===\n")
print(qc_after)
cat("\n=== Cells Removed ===\n")
cat("Removed:", qc_before$n_cells - qc_after$n_cells, "cells\n")
cat("Retention:", round(100 * qc_after$n_cells / qc_before$n_cells, 2), "%\n")
sink()

# Save Seurat object
saveRDS(seurat, file = output_seurat)
cat("QC complete. Seurat object saved to:", output_seurat, "\n")
