# Seurat PCA, UMAP, and Clustering Script
library(Seurat)
library(ggplot2)

# Get parameters
input_seurat <- snakemake@input[[1]]
output_seurat <- snakemake@output[[1]]
output_clusters <- snakemake@output[[2]]
output_pca <- snakemake@output[[3]]
output_umap <- snakemake@output[[4]]

n_pcs <- snakemake@params$n_pcs
dims <- snakemake@params$dims
resolution <- snakemake@params$resolution
algorithm <- snakemake@params$algorithm

# Load Seurat object
seurat <- readRDS(input_seurat)

# Run PCA
cat("Running PCA\n")
seurat <- RunPCA(seurat, features = VariableFeatures(object = seurat), npcs = n_pcs)

# PCA plot
p_pca <- DimPlot(seurat, reduction = "pca")
ggsave(output_pca, plot = p_pca, width = 8, height = 6)

# Clustering
cat("Finding neighbors and clusters (resolution =", resolution, ")\n")
seurat <- FindNeighbors(seurat, dims = 1:dims)
seurat <- FindClusters(seurat, resolution = resolution, algorithm = algorithm)

# Run UMAP
cat("Running UMAP\n")
seurat <- RunUMAP(seurat, dims = 1:dims)

# UMAP plot
p_umap <- DimPlot(seurat, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()
ggsave(output_umap, plot = p_umap, width = 8, height = 6)

# Save cluster assignments
clusters <- data.frame(
  cell = colnames(seurat),
  cluster = Idents(seurat)
)
write.csv(clusters, file = output_clusters, row.names = FALSE)

# Save Seurat object
saveRDS(seurat, file = output_seurat)
cat("Clustering complete. Found", length(unique(Idents(seurat))), "clusters\n")
cat("Seurat object saved to:", output_seurat, "\n")
