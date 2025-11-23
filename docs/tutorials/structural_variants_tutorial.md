# Structural Variants Detection Pipeline Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Background](#biological-background)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Understanding the Output](#understanding-the-output)
6. [Running the Pipeline](#running-the-pipeline)

## Introduction

This tutorial guides you through structural variant (SV) detection, which identifies large-scale genomic rearrangements including deletions, duplications, inversions, and translocations.

### What You'll Learn
- What structural variants are and why they matter
- How to detect different types of SVs from sequencing data
- How to interpret and validate structural variant calls
- How SVs differ from single nucleotide variants (SNVs)

### Prerequisites
- Basic understanding of DNA structure and genomics
- Familiarity with command line and genomic file formats (BAM, VCF)
- Access to BioPipelines environment

## Biological Background

### What are Structural Variants?

**Structural Variants (SVs)** are genomic alterations affecting >50 base pairs (some define as >1 kb):

**Major SV types:**

1. **Deletion (DEL)**
   ```
   Reference: ---[SEGMENT]---
   Sample:    --------------
   ```
   Missing DNA segment

2. **Duplication (DUP)**
   ```
   Reference: ---[SEGMENT]---
   Sample:    ---[SEGMENT][SEGMENT]---
   ```
   Extra copy of DNA segment

3. **Inversion (INV)**
   ```
   Reference: ---ABCDE---
   Sample:    ---EDCBA---
   ```
   Reversed DNA segment

4. **Translocation (TRA)**
   ```
   Chr1: ---[A]---     Chr2: ---[B]---
   Chr1: ---[B]---     Chr2: ---[A]---
   ```
   Segments moved between chromosomes

5. **Insertion (INS)**
   ```
   Reference: ------
   Sample:    --[NEW]--
   ```
   Novel sequence inserted

### Why Study Structural Variants?

**SVs vs SNVs:**
- SNVs: Single base changes (most common)
- SVs: Large alterations (fewer but bigger impact)
- SVs affect more DNA per event
- SVs more likely to disrupt genes

**Clinical significance:**
- **Cancer:** Oncogene amplifications, tumor suppressor deletions
- **Genetic diseases:** DiGeorge syndrome (22q11.2 deletion), Charcot-Marie-Tooth (PMP22 duplication)
- **Evolution:** SVs drive species differences
- **Pharmacogenomics:** Gene copy number affects drug metabolism

### Applications
- Cancer genome characterization
- Rare disease diagnosis
- Population genetics studies
- Agricultural breeding
- Evolutionary biology

## Pipeline Overview

### The Complete Workflow

```
Aligned Reads (BAM)
    ↓
1. BAM Quality Control
    ↓
2. BAM Indexing
    ↓
3. SV Calling (Manta)
    ↓
4. VCF Filtering
    ↓
5. SV Annotation
    ↓
Final SV Calls (VCF)
```

### Current Implementation Status

**✅ Implemented:**
- BAM indexing and validation
- Manta SV caller (isolated Python 2.7 environment)

**⏸️ Planned (conflicts resolved in progress):**
- Delly (read-depth + paired-end)
- LUMPY (split-read + paired-end)
- SURVIVOR (consensus merging)

**Why multiple callers?**
- Each algorithm has strengths/weaknesses
- Manta: Balanced sensitivity/specificity
- Delly: Good for deletions
- LUMPY: Excellent for complex events
- Consensus calls increase confidence

## Step-by-Step Walkthrough

### Step 1: BAM Preparation

**Input requirement:** Aligned BAM file from DNA-seq pipeline

**Quality checks:**
- Proper paired-end alignment
- Sufficient coverage (30X+ recommended)
- PCR duplicates marked
- Base quality recalibration applied

**From DNA-seq pipeline:**
```
data/processed/sample1.recal.bam
data/processed/sample1.recal.bam.bai
```

### Step 2: BAM Indexing

**Tool:** samtools index

Creates BAI index for random access.

**Why needed:**
- SV callers need fast region access
- Enables parallel processing
- Required for IGV visualization

**Command:**
```bash
samtools index sample1.recal.bam
```

### Step 3: Manta SV Calling

**Tool:** Manta v1.6.0 (Illumina)

The primary SV detection algorithm.

**How Manta works:**

1. **Evidence collection:**
   - Paired-end read orientations
   - Split-read alignments
   - Read depth anomalies

2. **Candidate generation:**
   - Identifies potential SV breakpoints
   - Clusters supporting evidence
   - Filters low-quality candidates

3. **Local assembly:**
   - Assembles reads at breakpoints
   - Resolves exact breakpoint sequences
   - Determines SV type

4. **Scoring and filtering:**
   - Calculates quality scores
   - Applies size filters
   - Reports final calls

**Key parameters:**
```yaml
manta:
  min_candidate_spanning_count: 3    # Supporting read pairs
  min_scored_variant_size: 50        # Minimum SV size (bp)
```

**Manta implementation:**
- Pre-built binary (v1.6.0)
- Isolated Python 2.7 conda environment
- Custom wrapper script for compatibility
- No conflicts with main Python 3.10 environment

**Output:**
```
results/structural_variants/manta/sample1.vcf.gz
```

### Step 4: VCF Quality Control

**SV VCF format:**
```
CHROM  POS     ID  REF  ALT              QUAL   FILTER  INFO
chr1   10000   .   N    <DEL>            999    PASS    SVTYPE=DEL;SVLEN=-5000;END=15000
chr2   50000   .   N    <DUP>            450    PASS    SVTYPE=DUP;SVLEN=2000;END=52000
```

**Key INFO fields:**
- `SVTYPE`: Type of variant (DEL, DUP, INV, etc.)
- `SVLEN`: Length in bp (negative for deletions)
- `END`: End coordinate
- `CIPOS/CIEND`: Confidence intervals for breakpoints

**Quality metrics:**
- `QUAL`: Phred-scaled quality score
- Supporting read counts
- Allele frequency (if trio/population)

### Step 5: Filtering (Future)

**Recommended filters:**

1. **Size filters:**
   ```
   50 bp < SVLEN < 1 Mb
   ```
   Remove very small/large outliers

2. **Quality filters:**
   ```
   QUAL > 20 (or higher for stringency)
   ```
   High-confidence calls only

3. **Coverage filters:**
   ```
   Depth > 10X at breakpoints
   ```
   Sufficient evidence

4. **Repetitive region filters:**
   Remove SVs in low-complexity DNA

### Step 6: Annotation (Planned)

**What to annotate:**
- Overlapping genes
- Functional consequences
- Known pathogenic SVs (ClinVar, DECIPHER)
- Population frequency (gnomAD-SV, 1000 Genomes)

**Tools (to be integrated):**
- AnnotSV
- VEP (Variant Effect Predictor)
- Custom gene overlap scripts

## Understanding the Output

### Main Output Files

```
data/results/structural_variants/
├── manta/
│   └── sample1.vcf.gz              # Manta SV calls
│       └── sample1.vcf.gz.tbi      # Index
└── qc/
    └── logs/
        └── manta/
            └── sample1.log         # Processing log
```

### Interpreting Manta VCF

**Example deletion:**
```
chr17  41196312  MantaDEL:12345  N  <DEL>  999  PASS  
END=41277381;SVTYPE=DEL;SVLEN=-81069;IMPRECISE;CIPOS=-10,10
```

**Interpretation:**
- Chromosome 17, position 41,196,312
- Deletion of ~81 kb
- High quality (QUAL=999)
- Breakpoint uncertainty ±10 bp
- Passed all filters

**This could be:**
- BRCA1 gene region (tumor suppressor)
- Pathogenic if germline deletion
- Somatic in cancer cells

### SV Size Distribution

**Typical human genome SVs:**
- 50-100 bp: Most abundant (thousands)
- 100 bp - 1 kb: Hundreds
- 1 kb - 10 kb: Tens to hundreds
- 10 kb - 100 kb: Tens
- >100 kb: Rare (few to tens)

### Quality Indicators

**Good SV call:**
- QUAL > 50
- Multiple supporting reads (>5)
- Precise breakpoints (CIPOS < 50 bp)
- Not in repetitive regions

**Questionable call:**
- Low QUAL (<20)
- Single supporting read
- Large confidence interval (CIPOS > 100 bp)
- In centromere/telomere

## Running the Pipeline

### Quick Start

```bash
# Navigate to project directory
cd ~/BioPipelines

# Ensure BAM file exists
ls data/processed/sample1.recal.bam

# Submit to cluster using unified script
./scripts/submit_pipeline.sh --pipeline structural_variants --mem 48G --cores 8 --time 08:00:00
```

### Configuration

Edit `pipelines/structural_variants/sv_calling/config.yaml`:

```yaml
# Input BAM directory
raw_dir: "/scratch/.../data/processed"

# Samples
samples:
  - sample1

# Manta parameters
callers:
  manta:
    enabled: true
    min_candidate_spanning_count: 3
    min_scored_variant_size: 50
```

### Expected Runtime

- **Small genome region:** 5-15 minutes
- **Whole exome:** 30-60 minutes
- **Whole genome (30X):** 2-6 hours
- **High depth (100X+):** 8-12 hours

**Memory requirements:**
- Exome: 8-16 GB
- Whole genome: 16-32 GB

**Current configuration:** 32 GB, 1 day time limit

### Monitoring Progress

```bash
# Check job status
squeue -u $USER | grep sv_calling

# View Manta progress
tail -f sv_calling_<jobid>.err

# Check Manta log
cat data/results/structural_variants/qc/logs/manta/sample1.log
```

### Common Issues

**Issue:** Python 2 not found
```
Error: /usr/bin/env: 'python2': No such file or directory

Solution: Manta wrapper uses isolated Python 2.7 environment
Check: ~/miniconda3/envs/manta_py27/bin/python2 exists
```

**Issue:** BAM file not indexed
```
Solution: Run DNA-seq pipeline first to generate .bam.bai
Or manually: samtools index sample1.recal.bam
```

**Issue:** Low SV count (<10 in genome)
```
Possible causes:
- Low coverage (<20X)
- High-quality reference sample (few true SVs)
- Overly stringent filters
- Poor alignment quality
```

**Issue:** Too many SV calls (>10,000 in genome)
```
Possible causes:
- Low-quality alignments
- Mismapped reads
- Wrong reference genome
- Need stricter filtering
```

## Advanced Topics

### Somatic vs Germline SVs

**Germline SVs:**
- Present in all cells
- Inherited or de novo
- ~1000-2000 per individual
- Comparison: tumor vs matched normal

**Somatic SVs:**
- Only in subset of cells (e.g., tumor)
- Acquired mutations
- Highly variable (few to thousands in cancer)
- Requires matched normal sample

### Cancer-Specific SVs

**Oncogene amplifications:**
- HER2 amplification (breast cancer)
- MYC amplification (lymphomas)
- EGFR amplification (glioblastomas)

**Tumor suppressor deletions:**
- TP53 deletions
- PTEN deletions
- RB1 deletions

**Gene fusions:**
- BCR-ABL (chronic myeloid leukemia)
- EML4-ALK (lung cancer)
- TMPRSS2-ERG (prostate cancer)

### Complex SVs

**Chromothripsis:**
- Catastrophic chromosome shattering
- Tens to hundreds of breakpoints
- Seen in aggressive cancers

**Chromoplexy:**
- Multiple interconnected rearrangements
- Chain-like structure
- Common in prostate cancer

### Validation Methods

1. **PCR across breakpoints**
   - Gold standard validation
   - Sanger sequence for precision

2. **Orthogonal sequencing:**
   - Long-read sequencing (PacBio, Nanopore)
   - Optical mapping
   - Hi-C

3. **Array-based:**
   - Array CGH
   - SNP arrays (for CNVs)

## Interpreting Biological Results

### Clinical Interpretation Workflow

1. **Filter for quality:**
   - QUAL > 50
   - PASS in FILTER column

2. **Annotate with genes:**
   - Which genes overlap?
   - Is the gene disrupted or deleted?

3. **Check databases:**
   - ClinVar: Known pathogenic SVs
   - DECIPHER: Developmental disorders
   - gnomAD-SV: Population frequencies

4. **Assess pathogenicity:**
   - Gene function (tumor suppressor, oncogene?)
   - Previous reports in literature
   - Inheritance pattern (if family data)

5. **Clinical report:**
   - Tier 1: Pathogenic, actionable
   - Tier 2: Likely pathogenic
   - Tier 3: Uncertain significance
   - Tier 4: Likely benign
   - Tier 5: Benign

### Population Genetics

**Human SV diversity:**
- ~25,000 SVs per individual
- Most are benign and common
- ~1-2% genome differs due to SVs
- Structural variation > sequence variation in bp affected

**Evolutionary significance:**
- SVs create new genes (duplications)
- Gene family expansions
- Adaptation to environment
- Species differences (e.g., human-chimp)

## Next Steps

After SV calling:

1. **Validation:**
   - PCR confirmation of key variants
   - Long-read sequencing for complex SVs
   - Visualization in IGV

2. **Functional analysis:**
   - Check gene disruptions
   - Pathway enrichment
   - Driver vs passenger mutations (cancer)

3. **Integration:**
   - Combine with SNV/indel calls
   - Integrate expression data (RNA-seq)
   - Multi-omics analysis

4. **Reporting:**
   - Clinical interpretation
   - Patient report generation
   - Research publication

## Resources

### Databases
- **ClinVar:** https://www.ncbi.nlm.nih.gov/clinvar/
- **DECIPHER:** https://www.deciphergenomics.org/
- **gnomAD-SV:** https://gnomad.broadinstitute.org/
- **DGV (Database of Genomic Variants):** http://dgv.tcag.ca/

### Publications
- Chen et al. (2016) "Manta: rapid detection of structural variants and indels for germline and cancer sequencing applications"
- Collins et al. (2020) "A structural variation reference for medical and population genetics" (gnomAD-SV)

### Tools Documentation
- **Manta:** https://github.com/Illumina/manta
- **IGV (visualization):** https://software.broadinstitute.org/software/igv/
- **AnnotSV:** https://lbgi.fr/AnnotSV/

## Support

For pipeline-specific issues:
- Check logs in `data/results/structural_variants/qc/logs/`
- Verify BAM file quality from DNA-seq pipeline
- Consult BioPipelines documentation

For biological interpretation:
- Review ACMG/AMP guidelines for SV interpretation
- Consult clinical geneticist for patient cases
- Check literature for specific genes/SVs
