# Metagenomics Taxonomic Profiling Pipeline Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Background](#biological-background)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Understanding the Output](#understanding-the-output)
6. [Running the Pipeline](#running-the-pipeline)

## Introduction

This tutorial guides you through metagenomic analysis, which identifies and quantifies microbial communities from environmental or clinical samples without cultivation.

### What You'll Learn
- What metagenomics is and its applications
- How to identify microbial species from mixed DNA samples
- How to quantify taxonomic abundance
- How to interpret community composition

### Prerequisites
- Basic understanding of microbiology and taxonomy
- Familiarity with command line
- Access to BioPipelines environment

## Biological Background

### What is Metagenomics?

**Metagenomics** sequences DNA from entire microbial communities (bacteria, archaea, fungi, viruses) without isolation or cultivation. Unlike traditional microbiology:
- No need to culture organisms (many can't be cultured)
- Captures the full community diversity
- Quantifies relative abundance
- Can discover novel organisms

### Taxonomic Hierarchy
```
Domain (Bacteria/Archaea)
  ↓
Phylum
  ↓
Class
  ↓
Order
  ↓
Family
  ↓
Genus
  ↓
Species
```

### Why Study Microbiomes?

**Microbiome analysis** answers questions like:
- What bacteria live in the human gut?
- How does antibiotic treatment affect microbial diversity?
- What pathogens are present in a clinical sample?
- How do environmental factors shape communities?

### Applications
- Human microbiome health research
- Environmental monitoring
- Pathogen detection and outbreak tracking
- Agricultural soil analysis
- Industrial bioreactor optimization

## Pipeline Overview

### The Complete Workflow

```
Raw Reads (FASTQ)
    ↓
1. Quality Control (FastQC)
    ↓
2. Read Trimming (fastp)
    ↓
3. Host Contamination Removal (optional)
    ↓
4. Taxonomic Profiling (MetaPhlAn4)
    ↓
5. MultiQC Report Generation
    ↓
Final Taxonomic Profiles + Visualizations
```

### Key Tools

1. **MetaPhlAn4** (Metaphlan Phylogenetic Analysis)
   - Uses clade-specific marker genes
   - Fast and memory efficient
   - Species-level resolution
   - Relative abundance estimation

2. **Kraken2** (optional, disabled due to conflicts)
   - k-mer based classification
   - Extremely fast
   - Can use custom databases

3. **HUMAnN** (optional, disabled due to conflicts)
   - Functional profiling
   - Pathway abundance

## Step-by-Step Walkthrough

### Step 1: Quality Control

**Tool:** FastQC

Assesses raw read quality before processing.

**What it checks:**
- Per-base sequence quality
- Adapter contamination
- GC content distribution
- Sequence duplication

**Typical metagenomics characteristics:**
- Variable GC content (mixed species)
- Higher duplication (some abundant taxa)
- Quality should still be high (Q30+)

### Step 2: Read Trimming

**Tool:** fastp

Removes low-quality bases and adapters.

**Default parameters:**
```yaml
quality: 20        # Minimum base quality
length: 50         # Minimum read length
```

**Why trimming matters:**
- Improves classification accuracy
- Reduces false positive species calls
- Removes adapter sequences

### Step 3: Host Removal (Optional)

**Tool:** Bowtie2 (if HOST_REMOVAL enabled)

Removes host contamination (e.g., human DNA from gut samples).

**When to enable:**
- Clinical samples (remove human DNA)
- Animal microbiome studies
- Plant-associated microbiomes

**Configuration:**
```yaml
host_removal:
  enabled: true
  host_genome: "path/to/human_genome"
```

### Step 4: MetaPhlAn Taxonomic Profiling

**Tool:** MetaPhlAn v4.2.4

The core taxonomic classification step.

**How it works:**
1. Aligns reads to marker gene database (~1M markers)
2. Identifies species-specific markers
3. Estimates relative abundance
4. Reports from domain to species level

**Key parameters:**
```yaml
metaphlan:
  db_dir: "metaphlan_db"     # Database location
  tax_lev: "a"                # All taxonomic levels
```

**Output files:**
- `sample_profile.txt` - Taxonomic abundance table
- `sample.bowtie2.bz2` - Alignment mapping file

### Step 5: Report Generation

**Tool:** MultiQC

Aggregates all QC metrics into one HTML report.

**Sections include:**
- FastQC summary
- Trimming statistics
- Host removal stats (if enabled)
- Overall data quality

## Understanding the Output

### Main Output Files

```
data/results/metagenomics/
├── multiqc_report.html              # Comprehensive QC report
├── taxonomy/
│   └── metaphlan/
│       ├── sample1_profile.txt      # Taxonomic abundance
│       └── sample1.bowtie2.bz2      # Alignment data
└── qc/
    └── logs/
        └── metaphlan/
            └── sample1.log          # Processing log
```

### Interpreting MetaPhlAn Profiles

**Profile format:**
```
#clade_name                              NCBI_tax_id  relative_abundance
UNCLASSIFIED                             -1           18.93
k__Bacteria                              2            80.89
k__Archaea                               2157         0.18
k__Bacteria|p__Pseudomonadota            2|1224       30.36
k__Bacteria|p__Firmicutes                2|1239       22.53
k__Bacteria|p__Bacteroidota              2|976        20.41
```

**Reading the output:**
- Hierarchical format (| separates levels)
- Relative abundance (percentages, sum to 100)
- NCBI taxonomy IDs for validation
- UNCLASSIFIED = reads not matching database

**Example interpretation:**
- 80.9% bacteria, 0.2% archaea (typical for gut)
- Dominated by Pseudomonadota (Proteobacteria)
- Balanced Firmicutes/Bacteroidota (healthy gut signature)
- 18.9% unclassified (novel organisms or insufficient depth)

### Key Metrics to Report

1. **Total reads processed**
   - Found in MetaPhlAn log: `#27610944 reads processed`

2. **Taxonomic diversity**
   - Number of species detected
   - Shannon/Simpson diversity index
   - Evenness of distribution

3. **Dominant taxa**
   - Top 5-10 species
   - Phylum-level composition

4. **Classification rate**
   - 100% - UNCLASSIFIED%
   - Higher is better (>70% good)

### Quality Indicators

**Good metagenomics run:**
- >70% reads classified
- Smooth rank-abundance curve
- Expected taxa for sample type
- Sufficient sequencing depth (1-10M reads)

**Potential issues:**
- >50% unclassified (low depth or novel community)
- Single species dominance (contamination?)
- Unexpected human DNA (remove host)
- Very low diversity (PCR bias, antibiotics)

## Running the Pipeline

### Quick Start

```bash
# Navigate to metagenomics pipeline
cd ~/BioPipelines

# Submit to cluster using unified script
./scripts/submit_pipeline.sh --pipeline metagenomics --mem 64G --cores 8 --time 08:00:00
```

### Configuration

Edit `pipelines/metagenomics/taxonomic_profiling/config.yaml`:

```yaml
# Samples
samples:
  - sample1
  - sample2

# Enable/disable host removal
host_removal:
  enabled: false  # Set true for clinical samples

# Enable/disable optional tools
options:
  kraken2_enabled: false    # Has libcurl conflicts
  functional:
    enabled: false          # HUMAnN (has conflicts)
```

### Expected Runtime

- **Small dataset** (1-5M reads): 5-15 minutes
- **Medium dataset** (10-50M reads): 30-60 minutes
- **Large dataset** (100M+ reads): 2-4 hours

**Memory requirements:**
- MetaPhlAn: 8-16 GB
- With host removal: 16-32 GB
- Large datasets: 64-96 GB

**Current configuration:** 96 GB, 12 hour time limit

### Monitoring Progress

```bash
# Check job status
squeue -u $USER

# View log in real-time
tail -f metagenomics_pipeline_<jobid>.err

# Check specific step completion
grep "Finished job" metagenomics_pipeline_<jobid>.err
```

### Common Issues

**Issue:** Out of memory
```
Solution: Increase memory allocation in submit script
#SBATCH --mem=128G  # or use higher memory partition
```

**Issue:** MetaPhlAn database not found
```
Solution: Download database first
metaphlan --install --bowtie2db data/references/metaphlan_db
```

**Issue:** Low classification rate (<50%)
```
Possible causes:
- Novel/unusual microbial community
- Insufficient sequencing depth
- Poor read quality
- Non-microbial DNA (eukaryotes, viruses)
```

## Advanced Topics

### Comparing Multiple Samples

Merge MetaPhlAn profiles:
```bash
merge_metaphlan_tables.py \
    results/taxonomy/metaphlan/*_profile.txt \
    > merged_abundance_table.txt
```

### Functional Profiling

Enable HUMAnN (if conflicts resolved):
```yaml
functional:
  enabled: true
```

Provides:
- Gene family abundance
- Pathway coverage
- Metabolic potential

### Custom Databases

For specific environments:
```yaml
metaphlan:
  custom_db: "path/to/specialized_markers"
```

Examples:
- Marine microbiome databases
- Soil-specific markers
- Animal gut databases

## Interpreting Biological Results

### Healthy vs Diseased Microbiomes

**Healthy human gut:**
- High diversity (>100 species)
- Balanced Firmicutes/Bacteroidota
- Presence of beneficial taxa (Bifidobacterium, Faecalibacterium)

**Dysbiotic gut:**
- Reduced diversity
- Firmicutes/Bacteroidota imbalance
- Pathogen enrichment
- Loss of protective species

### Environmental Samples

**Soil microbiome:**
- Extremely diverse (1000+ species)
- High Actinobacteria, Proteobacteria
- Fungal component (if sequenced)

**Water samples:**
- Lower diversity than soil
- Cyanobacteria in freshwater
- Marine-specific taxa in ocean

### Clinical Applications

**Pathogen detection:**
- Look for known pathogens
- Check abundance levels
- Compare to healthy controls

**Antibiotic resistance:**
- Requires functional profiling
- Identify resistance genes
- Track transmission

## Next Steps

After metagenomics analysis:

1. **Statistical analysis**
   - Compare groups (healthy vs diseased)
   - Identify significantly different taxa
   - Use tools like LEfSe, ANCOM

2. **Visualization**
   - Taxonomic bar plots
   - Heatmaps of abundance
   - PCA/PCoA of beta diversity
   - Krona charts

3. **Functional analysis**
   - Enable HUMAnN for pathways
   - Gene family analysis
   - Metabolic reconstruction

4. **Integration**
   - Correlate with metadata
   - Time series analysis
   - Network analysis

## Resources

### Databases
- **MetaPhlAn markers:** http://segatalab.cibio.unitn.it/tools/metaphlan/
- **NCBI Taxonomy:** https://www.ncbi.nlm.nih.gov/taxonomy
- **Human Microbiome Project:** https://hmpdacc.org/

### Publications
- Beghini et al. (2021) "Integrating taxonomic, functional, and strain-level profiling of diverse microbial communities with bioBakery 3"
- Segata et al. (2012) "Metagenomic microbial community profiling using unique clade-specific marker genes"

### Tools Documentation
- **MetaPhlAn4:** https://github.com/biobakery/MetaPhlAn/wiki
- **HUMAnN3:** https://huttenhower.sph.harvard.edu/humann/
- **Kraken2:** https://ccb.jhu.edu/software/kraken2/

## Support

For pipeline-specific issues:
- Check logs in `data/results/metagenomics/qc/logs/`
- Review MultiQC report for quality issues
- Consult BioPipelines documentation

For biological interpretation:
- Consult microbiome literature for your sample type
- Compare to published studies
- Consider pilot testing with known samples
