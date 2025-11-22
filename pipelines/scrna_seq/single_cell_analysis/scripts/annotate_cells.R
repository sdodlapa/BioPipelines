# Cell Type Annotation Script
library(Seurat)
library(ggplot2)
library(dplyr)

# Get parameters
input_seurat <- snakemake@input[[1]]
input_markers <- snakemake@input[[2]]
output_annotated <- snakemake@output[[1]]
output_umap <- snakemake@output[[2]]
output_composition <- snakemake@output[[3]]

marker_genes <- snakemake@params$marker_genes

# Load Seurat object and markers
seurat <- readRDS(input_seurat)
all.markers <- read.csv(input_markers)

# Get top marker for each cluster
top_markers <- all.markers %>%
  group_by(cluster) %>%
  top_n(n = 1, wt = avg_log2FC)

cat("Top markers per cluster:\n")
print(top_markers[, c("cluster", "gene", "avg_log2FC")])

# Annotate clusters based on marker genes
# This is a simplified approach - can be enhanced with automated methods
new.cluster.ids <- as.character(0:(length(unique(Idents(seurat))) - 1))

# Try to match clusters to cell types based on marker expression
for (cell_type in names(marker_genes)) {
  markers <- marker_genes[[cell_type]]
  
  # Calculate average expression of markers in each cluster
  for (cluster in unique(Idents(seurat))) {
    cells_in_cluster <- WhichCells(seurat, idents = cluster)
    expr <- FetchData(seurat, vars = markers, cells = cells_in_cluster)
    avg_expr <- colMeans(expr)
    
    # If markers are highly expressed, annotate this cluster
    if (mean(avg_expr > 0.5) > 0.5) {  # Simple threshold
      new.cluster.ids[as.numeric(cluster) + 1] <- cell_type
    }
  }
}

# Apply annotations
names(new.cluster.ids) <- levels(seurat)
seurat <- RenameIdents(seurat, new.cluster.ids)

# Save annotation metadata
seurat$cell_type <- Idents(seurat)

# UMAP plot with annotations
p_umap <- DimPlot(seurat, reduction = "umap", label = TRUE, pt.size = 0.5, repel = TRUE)
ggsave(output_umap, plot = p_umap, width = 10, height = 8)

# Cell composition
composition <- table(Idents(seurat))
comp_df <- data.frame(
  cell_type = names(composition),
  count = as.numeric(composition),
  percentage = round(100 * as.numeric(composition) / sum(composition), 2)
)

# Write composition
sink(output_composition)
cat("=== Cell Type Composition ===\n\n")
print(comp_df)
cat("\nTotal cells:", sum(comp_df$count), "\n")
sink()

# Save annotated Seurat object
saveRDS(seurat, file = output_annotated)
cat("Cell type annotation complete\n")
cat("Annotated", length(unique(Idents(seurat))), "cell types\n")
