# Single-Cell RNA-seq Analysis Pipeline Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Background](#biological-background)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Understanding the Output](#understanding-the-output)
6. [Running the Pipeline](#running-the-pipeline)

## Introduction

This tutorial guides you through single-cell RNA sequencing (scRNA-seq) analysis, which measures gene expression in individual cells, revealing cellular heterogeneity invisible to bulk RNA-seq.

### What You'll Learn
- What single-cell RNA-seq is and why it's revolutionary
- How to process droplet-based scRNA-seq data (10X Genomics)
- How to identify cell types and states
- How to discover new cell populations

### Prerequisites
- Basic understanding of RNA-seq and gene expression
- Familiarity with command line and R/Python
- Access to BioPipelines environment
- STAR v2.7.10b or compatible with your STAR index

### Critical Setup Notes
- **STAR index compatibility**: STAR index version must match STAR binary version
- **Index location**: Use absolute paths in config.yaml
  ```yaml
  star_index: "/scratch/sdodl001/BioPipelines/data/references/star_index_hg38"
  ```
- **Building STAR index**: Requires ~40-50GB RAM, use SLURM job:
  ```bash
  sbatch scripts/build_star_index_scrna.sh  # ~20-30 minutes
  ```
- **10x Chemistry**: Auto-detection with `whitelist: "None"` and `EmptyDrops_CR`

## Biological Background

### What is Single-Cell RNA-seq?

**scRNA-seq** sequences RNA from individual cells (not tissue homogenate):

**Bulk RNA-seq:**
```
Tissue → Homogenize → Extract RNA → Sequence
Result: Average expression across all cells
```

**Single-cell RNA-seq:**
```
Tissue → Dissociate → Isolate individual cells → Sequence each
Result: Expression profile per cell
```

### Why Single Cells?

**Problems with bulk RNA-seq:**
- Masks cellular heterogeneity
- Can't identify rare cell types
- Averages across cell states
- Misses cell-to-cell variation

**scRNA-seq reveals:**
- Cell type composition
- Rare cell populations (<1%)
- Cell state transitions
- Developmental trajectories
- Disease-specific cell types

### Key Concepts

**1. Cell Types vs Cell States:**
```
Cell Type:  Neuron, T cell, Fibroblast (identity)
Cell State: Activated, Resting, Dying (condition)
```

**2. Transcriptomic Identity:**
- Cells defined by expression signature
- 100-500 marker genes per type
- Dimensionality reduction reveals structure

**3. Cellular Heterogeneity:**
- Even "same" cell type varies
- Microenvironment effects
- Cell cycle differences
- Technical noise

### Applications
- Developmental biology (cell lineages)
- Immunology (immune cell subtypes)
- Cancer (tumor heterogeneity, microenvironment)
- Neuroscience (brain cell diversity)
- Stem cell research
- Drug response heterogeneity
- Disease mechanisms

## Pipeline Overview

### The Complete Workflow

```
Raw Reads (FASTQ)
    ↓
1. Demultiplexing (if needed)
    ↓
2. Cell Barcode Processing (STARsolo/CellRanger)
    ↓
3. Read Alignment to Transcriptome
    ↓
4. UMI Counting (per cell per gene)
    ↓
5. Cell Quality Filtering
    ↓
6. Normalization & Feature Selection
    ↓
7. Dimensionality Reduction (PCA, UMAP)
    ↓
8. Clustering (Louvain/Leiden)
    ↓
9. Cell Type Annotation
    ↓
10. Differential Expression & Markers
    ↓
Final: Cell clusters + expression matrix
```

### Technologies

**Droplet-based (10X Genomics Chromium):**
- Gel beads in emulsion (GEMs)
- ~10,000 cells per run
- ~2,000 genes per cell (3' capture)
- Cell barcodes + UMIs

**Plate-based (Smart-seq2, Smart-seq3):**
- Full-length transcripts
- 96-384 cells per plate
- ~5,000 genes per cell
- No UMIs (higher sensitivity)

**This pipeline:** Optimized for 10X Chromium data

### Current Implementation Status

**⏸️ Status:** Pipeline under development
- Alternative to CellRanger: STARsolo
- No authentication/download required
- Uses standard conda/pip packages
- Expected completion: After dependency resolution

## Step-by-Step Walkthrough

### Step 1: Understanding 10X Data

**FASTQ files:**
```
sample_S1_L001_R1_001.fastq.gz    # Read 1: Cell barcode + UMI
sample_S1_L001_R2_001.fastq.gz    # Read 2: cDNA sequence
```

**Read 1 structure (28 bp):**
```
Position 1-16:   Cell barcode (CB) - which cell?
Position 17-26:  UMI (Unique Molecular Identifier) - which molecule?
Position 27-28:  Adapter
```

**Read 2:**
- Full cDNA sequence (50-100 bp)
- Aligns to genes

### Step 2: Cell Barcode Processing

**Tool:** STARsolo (alternative to CellRanger)

Identifies valid cell barcodes and counts UMIs.

**How it works:**

1. **Whitelist matching:**
   - Compares observed barcodes to known whitelist
   - Allows 1 mismatch (sequencing error)
   - ~737K valid barcodes for v3 chemistry

2. **Cell calling:**
   - Empty droplets vs cells
   - Cells have high UMI counts
   - Empty: <500 UMIs
   - Cells: >1000 UMIs (varies)

3. **UMI counting:**
   - Each molecule has unique UMI
   - Removes PCR duplicates
   - Counts = molecules, not reads

**Output:**
```
Gene x Cell matrix (sparse)
Genes: 20,000-30,000 (human genome)
Cells: 1,000-10,000 (typical)
```

### Step 3: Alignment

**Tool:** STAR (in STARsolo mode)

Aligns cDNA reads to reference transcriptome.

**Differences from bulk RNA-seq:**
- Only counts exonic reads
- Collapses isoforms to genes
- Handles multi-mapping differently

**Parameters:**
```yaml
--soloType: "CB_UMI_Simple"
--soloCBwhitelist: "737K-august-2016.txt"  # 10X whitelist
--soloUMIlen: 10                            # UMI length
--soloFeatures: "Gene"                      # Gene-level counts
```

### Step 4: Quality Control

**Cell-level QC:**

**Metrics:**

1. **Number of genes detected:**
   ```
   Good: 500-5000 genes
   Low quality: <200 genes (dying cells, empty droplets)
   High: >6000 genes (doublets - 2 cells in 1 droplet)
   ```

2. **Total UMI counts:**
   ```
   Good: 1000-50,000 UMIs
   Low: <500 (low RNA, poor capture)
   High: >100,000 (doublets)
   ```

3. **Mitochondrial percentage:**
   ```
   Good: <5-10%
   Bad: >20% (dying cells leak cytoplasmic RNA)
   ```

**Filtering thresholds:**
```yaml
min_genes: 200          # Minimum genes per cell
min_cells: 3            # Minimum cells expressing gene
max_genes: 6000         # Maximum (doublet threshold)
max_mito_pct: 20        # Maximum mitochondrial %
```

**Gene-level QC:**
- Remove mitochondrial genes (unless studying)
- Remove ribosomal genes (optional)
- Keep protein-coding genes

### Step 5: Normalization

**Why normalize:**
- Cells have different RNA content
- Sequencing depth varies per cell
- Technical noise

**Methods:**

1. **Log-normalization (Seurat):**
   ```
   normalized = log1p(counts * 10000 / total_counts)
   ```
   Scale to 10,000 UMIs per cell, log transform

2. **SCTransform:**
   ```
   Regresses out sequencing depth, mitochondrial content
   Better for downstream analysis
   ```

### Step 6: Feature Selection

**Highly Variable Genes (HVGs):**

Not all 20,000 genes are informative:
- Housekeeping genes: Same in all cells
- Lowly expressed: Noisy, uninformative

**Select ~2000 HVGs:**
- High variance across cells
- Enriched for cell type markers
- Used for clustering

**Example HVGs:**
- Cell type markers (CD3D, CD79A, MS4A1)
- Cell cycle genes (MKI67, TOP2A)
- Activation markers (IL2, IFNG)

### Step 7: Dimensionality Reduction

**Problem:** 20,000 dimensions (genes) is too many

**Solution:** Reduce to manageable number

**1. PCA (Principal Component Analysis):**
```
20,000 genes → 50 principal components
Captures main variation axes
Use top 20-50 PCs for downstream
```

**2. UMAP (Uniform Manifold Approximation and Projection):**
```
50 PCs → 2D visualization
Non-linear projection
Preserves local + global structure
```

**What UMAP shows:**
```
Each point = one cell
Close cells = similar expression
Clusters = cell types/states
```

### Step 8: Clustering

**Algorithm:** Louvain or Leiden

Groups cells with similar expression.

**How it works:**
1. Build k-nearest neighbor graph (in PC space)
2. Find communities (clusters)
3. Optimize modularity

**Parameters:**
```yaml
resolution: 0.8        # Higher = more clusters
                       # 0.4-0.6: broad types
                       # 0.8-1.2: subtypes
                       # 1.5+: fine-grained states
```

**Typical results:**
- 5-20 clusters in PBMC (immune cells)
- 30-100 clusters in complex tissues (brain)

### Step 9: Cell Type Annotation

**Manual annotation:**

For each cluster, find marker genes:
```
Cluster 0: CD3D, CD3E → T cells
Cluster 1: CD79A, MS4A1 → B cells
Cluster 2: CD14, LYZ → Monocytes
Cluster 3: CD8A, CD8B → CD8 T cells
Cluster 4: CD4, IL7R → CD4 T cells
Cluster 5: NKG7, GNLY → NK cells
```

**Reference-based annotation:**
- Compare to annotated datasets
- Use cell type databases (CellMarker, PanglaoDB)
- Automated tools: SingleR, CellTypist

### Step 10: Differential Expression

**Find cluster markers:**

For each cluster, compare to all others:
```
Cluster 0 vs rest: Which genes are upregulated?
Top markers define cluster identity
```

**Statistical test:**
- Wilcoxon rank-sum
- t-test
- Negative binomial (DESeq2, edgeR)

**Criteria:**
```yaml
log2_fold_change: >1    # 2x difference
adj_p_value: <0.05      # Significant
pct_expressed: >25%     # In at least 25% of cells
```

## Understanding the Output

### Main Output Files

```
data/results/scrna_seq/
├── qc/
│   ├── metrics_summary.csv           # Overall statistics
│   └── cell_qc_plots.pdf             # QC visualizations
├── counts/
│   ├── matrix.mtx.gz                 # Gene x Cell matrix (sparse)
│   ├── features.tsv.gz               # Gene names
│   └── barcodes.tsv.gz               # Cell barcodes
├── processed/
│   ├── normalized.h5ad               # Normalized counts
│   ├── hvg_genes.csv                 # Variable genes
│   └── pca_embeddings.csv            # PC coordinates
├── clusters/
│   ├── umap_coordinates.csv          # UMAP 2D positions
│   ├── cluster_assignments.csv       # Cell → cluster
│   └── umap_plot.pdf                 # Visualization
├── markers/
│   ├── cluster_markers.csv           # Top genes per cluster
│   └── heatmap.pdf                   # Expression heatmap
└── annotated/
    └── cell_types.csv                # Final annotations
```

### Interpreting UMAP Plots

**Example PBMC visualization:**
```
       NK cells
          ○
         ╱
    ○──○  CD8 T
   ╱
CD4 T
   ╲
    ○──○  Monocytes
         ╲
          ○
       B cells
```

**Reading the plot:**
- Each dot = one cell
- Colors = clusters or cell types
- Close dots = similar cells
- Separate islands = distinct types
- Bridges = transitional states

### Marker Gene Interpretation

**Top markers for T cells:**
```
Gene      Log2FC  Pct.1  Pct.2  P.adj
CD3D      3.2     0.95   0.05   1e-200
CD3E      3.1     0.94   0.04   1e-195
CD3G      2.8     0.89   0.06   1e-180
```

**Interpretation:**
- CD3D: 3.2x higher in cluster, expressed in 95% of cells
- Highly significant (P.adj ~ 0)
- Clear T cell identity

### Quality Metrics

**Good scRNA-seq run:**
- >80% cells pass QC
- 1000-3000 median genes per cell
- <10% median mitochondrial content
- Clear clusters in UMAP
- Known markers enrich expected clusters

**Potential issues:**
- Low cell count: Poor dissociation, cell death
- High doublet rate: Overloading
- High mito%: Tissue stress, ischemia
- No clear clusters: Batch effects, technical issues
- Unexpected cell types: Contamination

## Running the Pipeline

### Quick Start

```bash
# Navigate to project directory
cd ~/BioPipelines

# Ensure 10X data is present
ls data/raw/scrna_seq/sample1_S1_L001_R1_001.fastq.gz
ls data/raw/scrna_seq/sample1_S1_L001_R2_001.fastq.gz

# Submit to cluster using unified script
./scripts/submit_pipeline.sh --pipeline scrna_seq --mem 48G --cores 8 --time 10:00:00
```

### Configuration

Edit `pipelines/scrna_seq/config.yaml`:

```yaml
# Samples
samples:
  - sample1

# 10X chemistry
chemistry:
  version: "v3"           # or "v2"
  whitelist: "None"       # Use "None" for auto-detection with EmptyDrops_CR

# Reference - CRITICAL: Use absolute paths and matching STAR version!
reference:
  fasta: "/scratch/sdodl001/BioPipelines/data/references/refdata-gex-GRCh38-2024-A/fasta/genome.fa"
  gtf: "/scratch/sdodl001/BioPipelines/data/references/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz"
  star_index: "/scratch/sdodl001/BioPipelines/data/references/star_index_hg38"  # Must match STAR version!

# QC thresholds
filtering:
  min_genes: 250
  max_genes: 6000
  min_counts: 500
  max_mito_pct: 20

# Clustering
clustering:
  neighbors:
    n_neighbors: 15
    n_pcs: 40
  leiden:
    resolution: [0.4, 0.6, 0.8, 1.0]
```

### CRITICAL: STAR Index Setup

**STAR version compatibility issue:**
- STAR index version MUST match STAR binary version
- CellRanger indices use STAR 2.7.1a (incompatible with STAR 2.7.10b)

**Error you'll see if mismatched:**
```
EXITING because of FATAL ERROR: Genome version: 2.7.1a is INCOMPATIBLE with running STAR version: 2.7.10b
SOLUTION: please re-generate genome from scratch
```

**Solution: Build compatible STAR index**

1. **Check your STAR version:**
```bash
STAR --version
# STAR_2.7.10b
```

2. **Build matching index (requires 50GB RAM):**
```bash
# Submit as SLURM job (DO NOT run on login node!)
sbatch scripts/build_star_index_scrna.sh

# Monitor progress (~20-30 minutes)
tail -f star_index_build_*.out
```

3. **Script creates index at:**
```
/scratch/sdodl001/BioPipelines/data/references/star_index_hg38/
```

4. **Update config.yaml with new path:**
```yaml
star_index: "/scratch/sdodl001/BioPipelines/data/references/star_index_hg38"
```

**Index build requirements:**
- Memory: 40-50GB RAM
- Time: 20-30 minutes
- Disk: ~28GB for human genome
- Must run on compute node (via SLURM)

### Expected Runtime

- **STAR index build (one-time):** 20-30 minutes (50GB RAM)
- **STARsolo alignment:** 30-90 minutes (per sample)
- **QC + filtering:** 10-20 minutes
- **Normalization + feature selection:** 15-30 minutes
- **PCA + UMAP:** 10-20 minutes
- **Clustering + markers:** 20-40 minutes

**Total:** 2-4 hours for 10,000 cells (after index built)

**Memory requirements:**
- STAR index build: 50 GB
- STARsolo alignment: 32-64 GB
- Downstream analysis: 16-32 GB

### Monitoring Progress

```bash
# Check job status
squeue -u $USER | grep scrna

# View progress
tail -f scrna_seq_<jobid>.err

# Check intermediate files
ls data/processed/scrna_seq/
```

### Common Issues

**Issue:** Low cell count detected
```
Causes:
- Poor tissue dissociation
- Cell death during processing
- Wrong barcode whitelist

Solutions:
- Optimize dissociation protocol
- Minimize time from dissociation to sequencing
- Verify 10X chemistry version
```

**Issue:** High doublet rate (>10%)
```
Causes:
- Overloaded chips (>10K cells loaded)
- Sticky cells (not single-cell suspension)

Solutions:
- Load correct cell number (~17K for 10K recovery)
- Filter aggregates before loading
- Use doublet detection (DoubletFinder, Scrublet)
```

**Issue:** Batch effects
```
Symptoms:
- Samples cluster separately, not by biology
- Same cell type split across batches

Solutions:
- Integration methods (Harmony, Seurat integration)
- Include batch in analysis
- Process samples together when possible
```

## Advanced Topics

### Trajectory Analysis

**Pseudotime:**
```
Differentiation: Stem cell → Progenitor → Mature
Pseudotime orders cells along developmental path
```

**Tools:**
- Monocle3
- Slingshot
- RNA Velocity

**Applications:**
- Development
- Differentiation
- Disease progression

### Cell-Cell Communication

**Ligand-receptor analysis:**

Identifies cell-cell interactions:
```
Cell A expresses ligand (IL2)
Cell B expresses receptor (IL2R)
→ Potential interaction
```

**Tools:**
- CellPhoneDB
- NicheNet
- CellChat

### Spatial Transcriptomics

**Technologies:**
- 10X Visium
- MERFISH
- seqFISH

**Combines:**
- scRNA-seq expression
- Spatial location in tissue
- Tissue architecture

### Multi-modal Analysis

**CITE-seq:** RNA + protein (antibodies)
**scATAC-seq:** RNA + chromatin accessibility
**Patch-seq:** RNA + electrophysiology

**Integration:**
- Multiple data types per cell
- Richer cellular characterization
- Better cell type resolution

## Interpreting Biological Results

### Cell Type Discovery

**Novel cell types:**
- Unique marker signature
- Distinct cluster
- Biological validation needed

**Example: Pulmonary ionocytes**
- Discovered by scRNA-seq in lung
- Express CFTR (cystic fibrosis gene)
- <1% of lung cells (undetectable in bulk)

### Disease Applications

**Cancer:**
- Tumor heterogeneity
- Immune microenvironment
- Treatment resistance subclones

**Immunology:**
- Immune cell subtypes
- Activation states
- Dysfunction signatures

**Neuroscience:**
- Brain cell diversity (>100 types)
- Disease-affected populations
- Circuit organization

### Rare Cell Populations

**scRNA-seq advantage:**
- Detect cells at <0.1% frequency
- Characterize without purification
- Discover unexpected populations

**Examples:**
- Antigen-specific T cells
- Stem cell subsets
- Circulating tumor cells

## Next Steps

After initial analysis:

1. **Validation:**
   - Flow cytometry sorting
   - Immunofluorescence staining
   - Functional assays

2. **Integration:**
   - Compare to bulk RNA-seq
   - Integrate multiple samples
   - Cross-species comparison

3. **Functional analysis:**
   - Pathway enrichment
   - Gene regulatory networks
   - Trajectory inference

4. **Experimental follow-up:**
   - Isolate populations (FACS)
   - Perturbation experiments
   - Mechanistic studies

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: STAR Index Version Incompatibility
**Error:** `BAMoutput.cpp:27:BAMoutput: exiting because of *OUTPUT FILE* error: could not create output file`

**Cause:** STAR binary version doesn't match STAR index version

**Solution:**
```bash
# Check versions
STAR --version  # e.g., 2.7.10b
head -1 /path/to/star_index/Log.out  # Shows index version

# If mismatch, rebuild index:
sbatch scripts/build_star_index_scrna.sh  # Requires 40-50GB RAM
```

#### Issue 2: Missing Python Packages
**Errors:**
- `ModuleNotFoundError: No module named 'scanpy'`
- `ModuleNotFoundError: No module named 'scrublet'`
- `ModuleNotFoundError: No module named 'igraph'`

**Solution:**
```bash
# Install all required packages
conda activate biopipelines
pip install scanpy scrublet pandas matplotlib seaborn
conda install -c conda-forge python-igraph leidenalg louvain
```

#### Issue 3: Scanpy Plotting API Changes
**Error:** `TypeError: pca_variance_ratio() got an unexpected keyword argument 'ax'`

**Cause:** Newer scanpy versions (>1.9) don't accept matplotlib axes directly

**Solution:** Use scanpy's multi-panel plotting (already fixed in current version):
```python
# Old (broken):
sc.pl.pca(adata, color='gene', ax=axes[0], show=False)

# New (working):
sc.pl.pca(adata, color=['gene1', 'gene2'], show=False, ncols=2)
```

#### Issue 4: File Format Issues
**Error:** `FileNotFoundError: Did not find file matrix.mtx.gz`

**Cause:** STARsolo outputs uncompressed files, scanpy expects gzipped

**Solution:**
```bash
cd /path/to/starsolo/Gene/filtered/
gzip -k matrix.mtx barcodes.tsv features.tsv
```

#### Issue 5: Scrublet API Changes
**Error:** `TypeError: scrub_doublets() got unexpected keyword 'min_gene_variability_pct'`

**Cause:** Parameter removed in newer scrublet versions

**Solution:** Remove the parameter (already fixed in current version):
```python
# Old:
scrub.scrub_doublets(min_gene_variability_pct=85)

# New:
scrub.scrub_doublets(min_counts=2, min_cells=3, n_prin_comps=30)
```

#### Issue 6: Matplotlib Axes Indexing
**Error:** `AttributeError: 'numpy.ndarray' object has no attribute 'hist'`

**Cause:** Subplot axes indexing changed between matplotlib versions

**Solution:**
```python
# For 2x2 subplots:
fig, axes = plt.subplots(2, 2)

# Old (1D indexing):
axes[0].hist(...)  # Wrong for 2x2

# New (2D indexing):
axes[0, 0].hist(...)  # Correct for 2x2
axes[0, 1].hist(...)
axes[1, 0].hist(...)
axes[1, 1].hist(...)
```

### Performance Tips

1. **Memory requirements:**
   - STAR alignment: ~40GB RAM
   - Processing 10K cells: ~16GB RAM
   - Processing 50K+ cells: ~64GB RAM

2. **Runtime estimates:**
   - STARsolo alignment: 10-30 min (depends on read depth)
   - QC and filtering: 1-2 min
   - Clustering and annotation: 2-5 min
   - Total: ~15-40 min for typical dataset

3. **Disk space:**
   - Raw FASTQ: variable (5-50GB)
   - STAR output: ~2-10GB
   - Processed h5ad files: 100MB-2GB

## Resources

### Databases
- **10X Genomics datasets:** https://www.10xgenomics.com/resources/datasets
- **Single Cell Portal:** https://singlecell.broadinstitute.org/
- **CellMarker:** http://biocc.hrbmu.edu.cn/CellMarker/
- **PanglaoDB:** https://panglaodb.se/

### Publications
- Tang et al. (2009) "mRNA-Seq whole-transcriptome analysis of a single cell" (first scRNA-seq)
- Zheng et al. (2017) "Massively parallel digital transcriptional profiling of single cells" (10X)
- Regev et al. (2017) "The Human Cell Atlas" (HCA)

### Tools Documentation
- **STARsolo:** https://github.com/alexdobin/STAR/blob/master/docs/STARsolo.md
- **Seurat:** https://satijalab.org/seurat/
- **Scanpy:** https://scanpy.readthedocs.io/
- **Cell Ranger:** https://support.10xgenomics.com/single-cell-gene-expression

## Support

For pipeline-specific issues:
- Check 10X chemistry version matches config
- Verify reference genome version
- Review QC metrics in outputs
- Consult BioPipelines documentation

For biological interpretation:
- Review cell type marker databases
- Consult tissue-specific literature
- Consider contacting single-cell core facilities
- Join scRNA-seq community forums
