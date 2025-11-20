# RNA-seq Differential Expression Pipeline Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Background](#biological-background)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Understanding the Output](#understanding-the-output)
6. [Running the Pipeline](#running-the-pipeline)

## Introduction

This tutorial guides you through RNA-seq differential expression analysis, which identifies genes that are expressed at different levels between conditions (e.g., treated vs. control, diseased vs. healthy).

### What You'll Learn
- What RNA sequencing is and why it's powerful
- How to process raw reads into gene expression counts
- How to identify differentially expressed genes
- How to interpret biological significance

### Prerequisites
- Basic understanding of gene expression and RNA
- Familiarity with command line
- Access to BioPipelines environment

## Biological Background

### What is RNA Sequencing?

**RNA-seq** sequences RNA molecules (mRNA) from cells to measure gene expression levels. Unlike microarrays, RNA-seq:
- Doesn't require prior knowledge of sequences
- Can detect novel transcripts and isoforms
- Has a larger dynamic range
- Can identify RNA editing and fusion genes

### The Central Dogma Refresher
```
DNA → (transcription) → RNA → (translation) → Protein
```

RNA-seq captures the RNA step, showing which genes are "turned on" and how strongly.

### Why Differential Expression?

**Differential expression analysis** answers questions like:
- Which genes are upregulated in cancer cells?
- How does a drug treatment change gene expression?
- What genes differ between cell types?
- Which pathways are activated in disease?

### Applications
- Disease biomarker discovery
- Drug target identification
- Understanding developmental processes
- Studying response to treatments

## Pipeline Overview

### The Complete Workflow

```
Raw Reads (FASTQ)
    ↓
1. Quality Control (FastQC)
    ↓
2. Read Trimming (fastp)
    ↓
3. Alignment to Transcriptome (STAR)
    ↓
4. Gene Quantification (featureCounts)
    ↓
5. Differential Expression (DESeq2)
    ↓
6. Functional Enrichment (GO/KEGG)
    ↓
Final Gene Lists + Visualizations
```

### Key Differences from DNA-seq
- Aligns to **transcriptome** (expressed regions)
- Handles **splice junctions** (intron removal)
- Quantifies **gene expression levels** (not variants)
- Requires **biological replicates** (3+ per condition)

### Time Estimates
- 6 samples (3+3 replicates): 4-8 hours
- Alignment is the slowest step

## Step-by-Step Walkthrough

### Step 1: Quality Control with FastQC

**Purpose**: Same as DNA-seq - assess read quality.

**Additional considerations for RNA-seq**:
- RNA-seq reads may have bias at the 5' or 3' end (normal)
- GC content varies by transcript abundance
- rRNA contamination should be minimal

**Command**:
```bash
fastqc -t 4 -o qc_output/ sample_R1.fastq.gz sample_R2.fastq.gz
```

**What to check**:
- ✅ Quality scores >28-30
- ✅ Low adapter content
- ⚠️ Duplication levels can be high (highly expressed genes)

---

### Step 2: Read Trimming with fastp

**Purpose**: Remove adapters and low-quality bases.

**RNA-seq specific considerations**:
- PolyA tails may need trimming
- Keep minimum length >50bp for reliable alignment

**Command**:
```bash
fastp \
    -i input_R1.fastq.gz \
    -I input_R2.fastq.gz \
    -o trimmed_R1.fastq.gz \
    -O trimmed_R2.fastq.gz \
    --qualified_quality_phred 20 \
    --length_required 50 \
    --thread 4
```

**Output**: Clean reads ready for alignment

---

### Step 3: Alignment with STAR

**Purpose**: Map reads to the reference genome, handling splice junctions.

**Why STAR?**
- **Fast**: Uses suffix arrays for quick alignment
- **Splice-aware**: Recognizes intron-exon boundaries
- **Accurate**: Handles complex splicing patterns

**Command breakdown**:
```bash
STAR \
    --runThreadN 8 \
    --genomeDir /path/to/star_index \
    --readFilesIn R1.fastq.gz R2.fastq.gz \
    --readFilesCommand zcat \
    --outFileNamePrefix sample_ \
    --outSAMtype BAM SortedByCoordinate \
    --quantMode GeneCounts
```

**Parameters explained**:
- `--runThreadN 8`: Use 8 CPU cores
- `--genomeDir`: Pre-built STAR index (includes splice junctions)
- `--readFilesCommand zcat`: Decompress gzipped files on-the-fly
- `--outSAMtype BAM SortedByCoordinate`: Output sorted BAM
- `--quantMode GeneCounts`: Also output gene counts

**How STAR handles splicing**:
1. Finds "seeds" (exact matches in genome)
2. Extends alignments across introns
3. Uses known splice junctions from GTF
4. Discovers novel junctions

**Output files**:
- `Aligned.sortedByCoord.out.bam`: Aligned reads
- `ReadsPerGene.out.tab`: Gene counts
- `Log.final.out`: Alignment statistics
- `SJ.out.tab`: Splice junctions detected

**Good alignment rates**:
- Uniquely mapped: >70-80%
- Multimapping: <20%
- Unmapped: <10%

---

### Step 4: Gene Quantification with featureCounts

**Purpose**: Count how many reads map to each gene.

**Why count reads?**
- Expression level ≈ number of reads
- More reads = higher expression
- Normalized counts allow comparison

**Command breakdown**:
```bash
featureCounts \
    -p \
    -t exon \
    -g gene_id \
    -a annotation.gtf \
    -o counts.txt \
    input.bam
```

**Parameters explained**:
- `-p`: Paired-end mode (count fragments, not reads)
- `-t exon`: Count only exonic regions
- `-g gene_id`: Summarize at gene level (not transcript)
- `-a annotation.gtf`: Gene annotations (GENCODE/Ensembl)
- `-o`: Output counts table

**Counting rules**:
- **Union mode**: Default - count if overlaps any exon
- Only count uniquely mapped reads (MAPQ >0)
- Handle multi-overlap carefully

**Output**: Count matrix
```
GeneID          Chr   Start    End    Length  Sample1  Sample2
ENSG00000223972 chr1  11869    14409  1735    45       52
ENSG00000227232 chr1  14404    29570  1351    123      98
```

**Understanding counts**:
- Raw counts: Actual read numbers
- High counts = high expression (e.g., housekeeping genes)
- Zero counts = not expressed or low coverage
- Counts depend on gene length and library size

---

### Step 5: Differential Expression with DESeq2

**Purpose**: Identify genes with significantly different expression between conditions.

**Statistical approach**:
1. Normalize for library size
2. Model count data with negative binomial distribution
3. Test for differential expression
4. Adjust p-values for multiple testing (FDR)

**R script overview**:
```R
library(DESeq2)

# Load count matrix
counts <- read.table("counts.txt", header=TRUE, row.names=1)

# Create metadata
coldata <- data.frame(
  condition = c("treated", "treated", "treated", 
                "control", "control", "control")
)

# Create DESeq2 object
dds <- DESeqDataSetFromMatrix(
  countData = counts,
  colData = coldata,
  design = ~ condition
)

# Run differential expression
dds <- DESeq(dds)
results <- results(dds, contrast=c("condition", "treated", "control"))
```

**Key statistics**:
- **baseMean**: Average normalized counts across samples
- **log2FoldChange**: Log2(treated/control)
  - >0: Upregulated in treated
  - <0: Downregulated in treated
- **pvalue**: Unadjusted significance
- **padj**: FDR-adjusted p-value (use this!)

**Normalization methods**:
- **Size factor**: Corrects for library size differences
- **TPM/FPKM**: Not recommended for DESeq2 (use raw counts)

**Significance thresholds**:
- **padj < 0.05**: Statistically significant
- **|log2FC| > 1**: 2-fold change (biologically meaningful)

---

### Step 6: Functional Enrichment Analysis

**Purpose**: Find biological themes in differentially expressed genes.

**Why enrichment?**
- Thousands of DE genes is overwhelming
- Genes work in pathways
- Identify affected biological processes

**Common databases**:
- **GO (Gene Ontology)**: Biological processes, molecular functions
- **KEGG**: Metabolic and signaling pathways
- **Reactome**: Curated pathway database

**Example - Over-representation analysis**:
```R
library(clusterProfiler)

# Get upregulated genes
up_genes <- results %>%
  filter(padj < 0.05, log2FoldChange > 1) %>%
  rownames()

# GO enrichment
ego <- enrichGO(
  gene = up_genes,
  OrgDb = org.Hs.eg.db,
  ont = "BP",  # Biological Process
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05
)
```

**Output**: Enriched pathways with:
- GeneRatio: Proportion of genes in pathway
- p.adjust: Significance
- Genes: Which genes are in each pathway

**Interpreting enrichment**:
- Focus on FDR < 0.05
- Look for biologically coherent themes
- Consider pathway overlap

---

## Understanding the Output

### Key Output Files

#### 1. **DESeq2 Results Table**
```
gene_id         baseMean  log2FC  lfcSE   pvalue     padj
ENSG00000223972 45.2      2.3     0.4     1.2e-6     0.003
ENSG00000227232 98.5      -1.8    0.3     4.5e-5     0.021
```

**Columns explained**:
- **baseMean**: Average expression level
- **log2FoldChange**: Effect size
  - log2FC=1 → 2x upregulated
  - log2FC=-1 → 2x downregulated
  - log2FC=2 → 4x upregulated
- **lfcSE**: Standard error of log2FC
- **padj**: FDR-corrected p-value (< 0.05 = significant)

#### 2. **Volcano Plot**
Visual representation of DE genes:
- X-axis: log2 Fold Change
- Y-axis: -log10(padj)
- Red dots: Significantly DE genes

#### 3. **MA Plot**
- X-axis: Average expression
- Y-axis: log2 Fold Change
- Shows expression-dependent effects

#### 4. **Heatmap**
Clustered expression of top DE genes across samples.

#### 5. **PCA Plot**
Shows sample clustering:
- Replicates should cluster together
- Conditions should separate

### Quality Control Checks

**1. Check replicates**:
```R
plotPCA(vst(dds), intgroup="condition")
```
- Replicates should cluster together
- Poor clustering = high variability

**2. Check dispersion**:
```R
plotDispEsts(dds)
```
- Points should follow the fitted curve
- Outliers indicate quality issues

**3. Sample correlation**:
```R
cor(counts(dds, normalized=TRUE))
```
- Within-group correlation should be >0.90

### Biological Interpretation

**Prioritizing genes**:
1. **High significance**: padj < 0.01
2. **Large effect size**: |log2FC| > 2
3. **High expression**: baseMean > 100
4. **Biological relevance**: Known function in your context

**Example interpretation**:
```
BRCA1: log2FC = -3.2, padj = 1e-8, baseMean = 450
```
- BRCA1 is strongly downregulated (8-fold decrease)
- Highly significant
- Well-expressed gene
- Known tumor suppressor → biologically relevant

---

## Running the Pipeline

### Quick Start

1. **Prepare sample sheet** (`config.yaml`):
```yaml
samples:
  treatment:
    - treat_rep1
    - treat_rep2
    - treat_rep3
  control:
    - ctrl_rep1
    - ctrl_rep2
    - ctrl_rep3

reference:
  genome: "/path/to/hg38.fa"
  gtf: "/path/to/gencode.v45.gtf"
  star_index: "/path/to/star_index"
```

2. **Ensure STAR index exists**:
```bash
# Check if index exists
ls /path/to/star_index/

# If not, build it first
sbatch ~/BioPipelines/scripts/build_star_index.sh
```

3. **Submit pipeline**:
```bash
cd ~/BioPipelines/pipelines/rna_seq/differential_expression
sbatch ~/BioPipelines/scripts/submit_rna_seq.sh
```

4. **Monitor progress**:
```bash
squeue --me
tail -f slurm_*.err
```

### Experimental Design Considerations

**Biological replicates**:
- Minimum: 3 per condition
- Recommended: 5-6 for complex designs
- Technical replicates not needed

**Sequencing depth**:
- Human/mouse: 20-30 million reads per sample
- More depth = detect lowly expressed genes
- More samples > more depth

**Paired vs. single-end**:
- Paired-end: Better for novel isoforms
- Single-end: Sufficient for gene-level DE

### Troubleshooting

**Low alignment rate (<70%)**:
- Wrong reference genome version
- Poor read quality
- rRNA contamination

**Too few DE genes**:
- High biological variability
- Subtle treatment effect
- Insufficient sample size

**Too many DE genes (>5000)**:
- Very strong treatment effect
- Batch effects
- Sample swaps/mislabeling

**PCA shows no separation**:
- Treatment has weak effect
- Check sample labels
- Consider alternative analysis

---

## Advanced Topics

### Alternative Methods

**Salmon (pseudo-alignment)**:
- Faster than STAR
- Doesn't produce BAM files
- Good for standard DE analysis

**edgeR (alternative to DESeq2)**:
- Similar approach
- Slightly different statistics
- Good for simple designs

### Multi-factor Designs

Example: Treatment + timepoint
```R
design = ~ timepoint + treatment + timepoint:treatment
```

### Batch Effect Correction

Use ComBat-seq or include batch in design:
```R
design = ~ batch + condition
```

---

## Additional Resources

### Further Reading
- [RNA-seqlopedia](https://rnaseq.uoregon.edu/)
- [DESeq2 Vignette](http://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)
- [STAR Manual](https://github.com/alexdobin/STAR/blob/master/doc/STARmanual.pdf)

### Tools Used
- **FastQC/fastp**: QC and trimming
- **STAR**: Spliced alignment
- **featureCounts**: Gene quantification
- **DESeq2**: Differential expression
- **clusterProfiler**: Functional enrichment

---

## Glossary

- **TPM**: Transcripts Per Million (normalized expression)
- **FPKM**: Fragments Per Kilobase Million
- **FDR**: False Discovery Rate (adjusted p-value)
- **GTF**: Gene Transfer Format (gene annotations)
- **log2FC**: Log2 Fold Change
- **Splice junction**: Boundary between exons in mRNA
- **Size factor**: DESeq2's normalization factor

---

*Last updated: November 2025*
