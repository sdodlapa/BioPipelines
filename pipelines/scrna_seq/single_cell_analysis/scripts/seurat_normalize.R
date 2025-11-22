# Seurat Normalization and Variable Features Script
library(Seurat)

# Get parameters
input_seurat <- snakemake@input[[1]]
output_seurat <- snakemake@output[[1]]
method <- snakemake@params$method
scale_factor <- snakemake@params$scale_factor
n_features <- snakemake@params$n_features

# Load Seurat object
seurat <- readRDS(input_seurat)

# Normalization
cat("Normalizing data using:", method, "\n")
seurat <- NormalizeData(seurat, normalization.method = method, scale.factor = scale_factor)

# Find variable features
cat("Finding", n_features, "variable features\n")
seurat <- FindVariableFeatures(seurat, selection.method = "vst", nfeatures = n_features)

# Scale data
cat("Scaling data\n")
all.genes <- rownames(seurat)
seurat <- ScaleData(seurat, features = all.genes)

# Save
saveRDS(seurat, file = output_seurat)
cat("Normalization complete. Seurat object saved to:", output_seurat, "\n")
