# Seurat Marker Gene Discovery Script
library(Seurat)
library(ggplot2)
library(dplyr)

# Get parameters
input_seurat <- snakemake@input[[1]]
output_markers <- snakemake@output[[1]]
output_heatmap <- snakemake@output[[2]]
output_dotplot <- snakemake@output[[3]]

test <- snakemake@params$test
logfc_threshold <- snakemake@params$logfc_threshold
min_pct <- snakemake@params$min_pct
only_pos <- snakemake@params$only_pos

# Load Seurat object
seurat <- readRDS(input_seurat)

# Find all markers
cat("Finding marker genes for all clusters\n")
cat("Test method:", test, "\n")
all.markers <- FindAllMarkers(
  seurat,
  test.use = test,
  only.pos = only_pos,
  min.pct = min_pct,
  logfc.threshold = logfc_threshold
)

# Save markers
write.csv(all.markers, file = output_markers, row.names = FALSE)
cat("Found", nrow(all.markers), "marker genes across", length(unique(all.markers$cluster)), "clusters\n")

# Get top 10 markers per cluster for visualization
top10 <- all.markers %>%
  group_by(cluster) %>%
  top_n(n = 10, wt = avg_log2FC)

# Heatmap of top markers
p_heatmap <- DoHeatmap(seurat, features = top10$gene) + NoLegend()
ggsave(output_heatmap, plot = p_heatmap, width = 12, height = 10)

# Dot plot of top 5 markers per cluster
top5 <- all.markers %>%
  group_by(cluster) %>%
  top_n(n = 5, wt = avg_log2FC)

p_dot <- DotPlot(seurat, features = unique(top5$gene)) + RotatedAxis()
ggsave(output_dotplot, plot = p_dot, width = 14, height = 6)

cat("Marker discovery complete\n")
