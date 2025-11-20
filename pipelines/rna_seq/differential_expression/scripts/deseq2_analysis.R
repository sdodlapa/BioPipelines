#!/usr/bin/env Rscript
#
# DESeq2 Differential Expression Analysis
#

suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
  library(pheatmap)
  library(RColorBrewer)
})

# Snakemake inputs
counts_file <- snakemake@input[[1]]

# Snakemake outputs
results_file <- snakemake@output[["results"]]
normalized_file <- snakemake@output[["normalized"]]
volcano_file <- snakemake@output[["volcano"]]
ma_file <- snakemake@output[["ma"]]
heatmap_file <- snakemake@output[["heatmap"]]
pca_file <- snakemake@output[["pca"]]

# Parameters
treatment_samples <- snakemake@params[["treatment_samples"]]
control_samples <- snakemake@params[["control_samples"]]
padj_cutoff <- as.numeric(snakemake@params[["padj_cutoff"]])
lfc_cutoff <- as.numeric(snakemake@params[["lfc_cutoff"]])
top_genes <- as.numeric(snakemake@params[["top_genes"]])

# Read count matrix
message("Reading count data...")
count_data <- read.table(counts_file, header=TRUE, row.names=1, skip=1)

# Remove first 5 columns (Chr, Start, End, Strand, Length)
count_data <- count_data[, -(1:5)]

# Clean sample names (remove path and .bam extension)
colnames(count_data) <- gsub(".*\\/", "", colnames(count_data))
colnames(count_data) <- gsub("\\.Aligned\\.sortedByCoord\\.out\\.bam", "", colnames(count_data))

# Create sample metadata
coldata <- data.frame(
  sample = c(treatment_samples, control_samples),
  condition = factor(c(rep("treatment", length(treatment_samples)),
                       rep("control", length(control_samples))),
                     levels = c("control", "treatment"))
)
rownames(coldata) <- coldata$sample

# Ensure samples match
count_data <- count_data[, rownames(coldata)]

# Create DESeq2 dataset
message("Creating DESeq2 dataset...")
dds <- DESeqDataSetFromMatrix(countData = count_data,
                               colData = coldata,
                               design = ~ condition)

# Filter low count genes
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep,]

# Run DESeq2 analysis
message("Running DESeq2...")
dds <- DESeq(dds)

# Get normalized counts
normalized_counts <- counts(dds, normalized=TRUE)
write.csv(normalized_counts, normalized_file)

# Get results
res <- results(dds, contrast=c("condition", "treatment", "control"))
res_df <- as.data.frame(res)
res_df$gene <- rownames(res_df)
res_df <- res_df[order(res_df$padj),]

# Save results
write.csv(res_df, results_file, row.names=FALSE)

message(sprintf("Found %d genes with padj < %.3f", 
                sum(res_df$padj < padj_cutoff, na.rm=TRUE), 
                padj_cutoff))

# Volcano plot
message("Creating volcano plot...")
res_df$significant <- ifelse(res_df$padj < padj_cutoff & abs(res_df$log2FoldChange) > lfc_cutoff, 
                              "Significant", "Not Significant")
res_df$significant[is.na(res_df$significant)] <- "Not Significant"

png(volcano_file, width=800, height=600)
ggplot(res_df, aes(x=log2FoldChange, y=-log10(padj), color=significant)) +
  geom_point(alpha=0.6) +
  scale_color_manual(values=c("Significant"="red", "Not Significant"="gray")) +
  geom_vline(xintercept=c(-lfc_cutoff, lfc_cutoff), linetype="dashed") +
  geom_hline(yintercept=-log10(padj_cutoff), linetype="dashed") +
  theme_bw() +
  labs(title="Volcano Plot", 
       x="Log2 Fold Change", 
       y="-Log10 Adjusted P-value",
       color="")
dev.off()

# MA plot
message("Creating MA plot...")
png(ma_file, width=800, height=600)
plotMA(res, ylim=c(-5, 5), main="MA Plot")
dev.off()

# PCA plot
message("Creating PCA plot...")
vsd <- vst(dds, blind=FALSE)
png(pca_file, width=800, height=600)
plotPCA(vsd, intgroup="condition") +
  theme_bw() +
  labs(title="PCA Plot")
dev.off()

#Heatmap of top genes
message("Creating heatmap...")
top_genes_idx <- head(order(res_df$padj), top_genes)
top_genes_names <- rownames(res)[top_genes_idx]

mat <- assay(vsd)[top_genes_names, ]
mat <- mat - rowMeans(mat)

annotation_col <- data.frame(
  Condition = coldata[colnames(mat), "condition"]
)
rownames(annotation_col) <- colnames(mat)

png(heatmap_file, width=1000, height=800)
pheatmap(mat, 
         annotation_col = annotation_col,
         show_rownames = FALSE,
         show_colnames = FALSE,
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         color = colorRampPalette(rev(brewer.pal(n = 7, name = "RdYlBu")))(100),
         main = sprintf("Top %d Differentially Expressed Genes", top_genes))
dev.off()

message("Analysis complete!")
