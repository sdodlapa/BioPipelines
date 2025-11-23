# Long-Read Sequencing Analysis Pipeline Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Background](#biological-background)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Understanding the Output](#understanding-the-output)
6. [Running the Pipeline](#running-the-pipeline)

## Introduction

This tutorial guides you through long-read sequencing analysis using Oxford Nanopore or PacBio technologies, which generate reads >10 kb (up to 100+ kb) enabling superior structural variant detection and de novo assembly.

### What You'll Learn
- What long-read sequencing is and its advantages
- How to process high-error-rate long reads
- How to detect structural variants from long reads
- How to perform de novo genome assembly

### Prerequisites
- Basic understanding of DNA sequencing technologies
- Familiarity with command line and genomic file formats
- Access to BioPipelines environment
- SURVIVOR tool installed for SV merging

### Important Notes
- **minimap2 syntax**: Use `-x` flag before preset: `minimap2 -x map-ont`
- **SURVIVOR**: Must be compiled from source, not available via conda/pip
- **Expected runtime**: ~2-3 minutes for 4.8GB ONT dataset

## Biological Background

### What is Long-Read Sequencing?

**Long-read technologies** generate DNA sequences thousands of base pairs long:

**Platform comparison:**

| Platform | Read Length | Accuracy | Throughput | Cost/Gb |
|----------|-------------|----------|------------|---------|
| Illumina | 150-300 bp | 99.9% | Very High | Low |
| PacBio HiFi | 10-25 kb | 99.9% | High | Medium |
| PacBio CLR | 10-100 kb | 85-90% | High | Medium |
| ONT | 10-100+ kb | 92-99% | High | Low |

**Key characteristics:**
- Much longer than short reads (100-1000x)
- Higher error rate (though HiFi approaching Illumina)
- Can span repetitive regions
- Capture structural variants directly

### Why Long Reads?

**Advantages over short reads:**

1. **Span repeat regions:**
   ```
   Short reads:  ========   ========   ========
   Repeat:          [==========REPEAT==========]
   Long read:   ====================================
   ```

2. **Phase variants:**
   - Determine which variants are on same chromosome
   - Critical for compound heterozygotes
   - Haplotype resolution

3. **Detect complex SVs:**
   - Large insertions/deletions
   - Inversions
   - Complex rearrangements
   - Full-length transcripts

4. **De novo assembly:**
   - Assemble novel genomes
   - Close gaps in reference
   - Resolve centromeres/telomeres

### Applications
- De novo genome assembly
- Structural variant discovery
- Haplotype phasing
- Full-length transcript sequencing
- Methylation detection (ONT)
- Chromosome-scale assembly
- Complex repeat region resolution

## Pipeline Overview

### The Complete Workflow

```
Raw Long Reads (FASTQ)
    ↓
1. Quality Control (NanoPlot/NanoStat)
    ↓
2. Read Filtering (minimum length/quality)
    ↓
3. Alignment (minimap2)
    ↓
4. Structural Variant Calling (Sniffles2 + cuteSV)
    ↓
5. SV Merging (SURVIVOR)
    ↓
6. Variant Phasing (WhatsHap)
    ↓
Final Variants + Phased Haplotypes
```

### Alternative Workflows

**De novo assembly:**
```
Raw Reads → Error Correction → Assembly (Flye/Canu) → Polishing → Final Assembly
```

**Targeted sequencing:**
```
Raw Reads → Region Extraction → Local Assembly → Variant Calling
```

### Current Implementation Status

**⏸️ Status:** Pipeline under development
- Conda environment conflicts being resolved
- Tools identified: minimap2, Sniffles2, NanoPlot
- Expected completion: After resolving dependency issues

## Step-by-Step Walkthrough

### Step 1: Quality Control

**Tools:** NanoPlot, NanoStat (Oxford Nanopore)

Assess long-read quality before processing.

**Key metrics:**

1. **Read length distribution:**
   - Mean read length (target: 10-50 kb)
   - N50 (length where 50% bases in longer reads)
   - Maximum read length

2. **Base quality:**
   - Mean quality score (Q10-Q20 typical for ONT)
   - Quality score distribution

3. **Yield:**
   - Total bases sequenced
   - Total reads
   - Gigabases of data

**Example output:**
```
Total reads: 500,000
Total bases: 15 Gb
Mean read length: 30 kb
N50: 45 kb
Mean quality: Q12
```

### Step 2: Read Filtering

**Tool:** NanoFilt, Chopper

Remove low-quality/short reads.

**Typical filters:**
```yaml
min_length: 1000      # Minimum 1 kb
min_quality: 7        # Q7 threshold
max_length: 100000    # Remove ultra-long outliers
```

**Why filter:**
- Short reads don't span features
- Low-quality reads cause false variants
- Reduces computational burden

**Expected loss:**
- 10-30% of reads typically filtered
- Depends on sequencing quality

### Step 3: Alignment

**Tool:** minimap2

The gold standard for long-read alignment.

**How minimap2 differs:**
- Tolerates high error rates
- Fast (~100x faster than BWA)
- Specialized for long reads
- Multiple modes (ont, pb, hifi)

**Key parameters:**
```yaml
preset: "map-ont"           # ONT reads
        "map-pb"            # PacBio CLR
        "asm20"             # PacBio HiFi / highly accurate

secondary: "no"             # No secondary alignments
soft_clipping: "yes"        # Allow soft clips at ends
```

**Output:**
```
aligned_reads.bam
```

### Step 4: Structural Variant Calling

**Tool:** Sniffles2

Purpose-built SV caller for long reads.

**How Sniffles works:**

1. **Read alignment analysis:**
   - Identifies split reads
   - Detects large insertions/deletions
   - Finds inversions from read orientation

2. **Clustering:**
   - Groups supporting reads
   - Determines breakpoints
   - Calculates confidence

3. **Genotyping:**
   - Homozygous vs heterozygous
   - Allele frequency

**Advantages for long reads:**
- Can see entire SV in single read
- No need for paired-end inference
- Higher precision breakpoints
- Detect complex nested SVs

**Parameters:**
```yaml
min_support: 3              # Minimum supporting reads
min_sv_size: 50             # Minimum SV size (bp)
min_mapq: 20                # Minimum mapping quality
```

**Output:**
```
structural_variants.vcf
```

### Step 5: Variant Phasing

**Tool:** WhatsHap

Links variants on same chromosome (haplotype).

**Why phasing matters:**

Example: Two heterozygous mutations
```
Variant 1: Position 1000, A→G
Variant 2: Position 2000, C→T

Phased:
Haplotype 1: A---C (maternal)
Haplotype 2: G---T (paternal)

Biological consequence: Different if in cis vs trans
```

**Long-read advantage:**
- Single read spans multiple variants
- Direct phasing without family data
- Mega-base scale phasing

**Output:**
```
phased_variants.vcf
haplotypes.gtf
```

## Understanding the Output

### Main Output Files

```
data/results/long_read/
├── qc/
│   ├── NanoPlot-report.html         # QC metrics
│   └── read_length_distribution.png
├── alignments/
│   ├── sample1.bam                  # Aligned reads
│   └── sample1.bam.bai
├── variants/
│   ├── sample1_sv.vcf               # Structural variants
│   └── sample1_phased.vcf           # Phased variants
└── assembly/                        # (if de novo assembly)
    └── sample1_assembly.fasta
```

### Interpreting Long-Read SVs

**VCF format:**
```
CHROM  POS     REF  ALT              QUAL  INFO
chr1   10000   N    <DEL>            60    SVTYPE=DEL;SVLEN=-5000;SUPPORT=15
chr2   50000   N    AGCT[100bp seq]  80    SVTYPE=INS;SVLEN=100;SUPPORT=20
```

**Key differences from short-read SVs:**
- Higher SUPPORT counts (more reads)
- More precise breakpoints
- Can report inserted sequence directly
- Better for complex SVs

### Quality Metrics

**Good long-read run:**
- N50 > 20 kb
- Mean quality > Q10
- >10 Gb data (for human genome)
- >30X coverage

**Potential issues:**
- Very short N50 (<5 kb): DNA degradation
- Low quality (<Q7): Flow cell issues
- Low yield: Insufficient DNA input
- Coverage gaps: Library prep issues

## Running the Pipeline

### Quick Start

```bash
# Navigate to project directory
cd ~/BioPipelines

# Place FASTQ in raw directory
ls data/raw/long_read/sample1.fastq.gz

# Submit to cluster using unified script
./scripts/submit_pipeline.sh --pipeline long_read --mem 48G --cores 8 --time 10:00:00
```

### Configuration

Edit `pipelines/long_read/config.yaml`:

```yaml
# Samples
samples:
  - sample1

# Platform
platform: "ont"        # or "pacbio", "hifi"

# Filtering
filtering:
  min_length: 1000
  min_quality: 7

# Alignment
alignment:
  preset: "map-ont"    # or "map-pb", "asm20"
  
# IMPORTANT: minimap2 requires -x flag before preset
# Correct:   minimap2 -t 16 -x map-ont reference.fa reads.fq.gz
# Incorrect: minimap2 -t 16 map-ont reference.fa reads.fq.gz

# SV calling (uses dual-caller approach)
sv_calling:
  callers:
    - sniffles2    # Primary SV caller
    - cutesv       # Secondary SV caller
  merge_tool: "SURVIVOR"  # Merges VCFs from both callers
  min_support: 3
  min_sv_size: 50
  max_distance: 1000    # For SURVIVOR merging
```

### Tool Installation Notes

**SURVIVOR** must be compiled from source:
```bash
git clone https://github.com/fritzsedlazeck/SURVIVOR.git
cd SURVIVOR/Debug
make
cp SURVIVOR ~/envs/biopipelines/bin/
```

**Note**: The Python package `survivor` on pip is NOT the SV merging tool!

### Expected Runtime

- **QC:** 10-30 minutes (depends on read count)
- **Alignment:** 1-4 hours (30X coverage)
- **SV calling (Sniffles2 + cuteSV):** 30-90 minutes
- **SV merging (SURVIVOR):** <1 minute
- **Phasing:** 30-60 minutes

**Example**: 4.8GB ONT dataset completed in ~2-3 minutes (most steps cached)

**De novo assembly:**
- Small bacterial genome: 1-4 hours
- Human genome: 24-72 hours

**Memory requirements:**
- Alignment: 16-32 GB
- De novo assembly: 64-256 GB

### Monitoring Progress

```bash
# Check job status
squeue -u $USER

# View alignment progress
tail -f long_read_<jobid>.err

# Check intermediate files
ls data/processed/long_read/
```

### Common Issues

**Issue:** Low N50 (<5 kb)
```
Causes:
- DNA degradation during extraction
- Aggressive mechanical shearing
- Enzyme damage

Solutions:
- Use high molecular weight DNA extraction
- Minimize vortexing
- Fresh samples preferred
```

**Issue:** High error rate
```
Causes:
- Old flow cell
- Incorrect basecalling model
- Poor sequencing chemistry

Solutions:
- Use latest basecaller (Guppy/Dorado)
- Rebasecall with updated model
- Consider polishing with short reads
```

**Issue:** Alignment gaps
```
Causes:
- Repetitive regions
- Structural variants in sample
- Reference genome gaps

Solutions:
- Increase coverage (>50X)
- Use de novo assembly
- Try different alignment parameters
```

## Advanced Topics

### De Novo Genome Assembly

**Assemblers:**
- **Flye:** General purpose, ONT/PacBio
- **Canu:** High-accuracy, resource intensive
- **Shasta:** Ultra-fast, ONT-optimized
- **Hifiasm:** PacBio HiFi specialist

**Workflow:**
```bash
# Assemble
flye --nano-raw reads.fastq.gz --genome-size 3g --out-dir assembly/

# Polish with long reads
medaka_consensus -i reads.fastq.gz -d assembly/assembly.fasta -o polished/

# (Optional) Polish with short reads
pilon --genome polished.fasta --frags illumina_reads.bam --output final
```

### Full-Length Transcript Sequencing

**Iso-Seq (PacBio) / Direct RNA-seq (ONT):**

**Advantages:**
- Full-length transcripts without assembly
- Novel isoform discovery
- Alternative splicing analysis
- Fusion transcript detection

**Tools:**
- IsoSeq3 (PacBio)
- FLAIR (isoform detection)
- TALON (transcript annotation)

### Methylation Detection

**ONT native methylation:**

Long reads capture modified bases (5mC, 6mA) without bisulfite conversion.

**Tools:**
- Nanopolish
- Megalodon
- DeepSignal

**Applications:**
- CpG methylation profiling
- Imprinting studies
- Cancer epigenetics
- Bacterial methylation

### Hybrid Assembly

**Combining long + short reads:**

```
Long reads (ONT/PacBio): Scaffolding, structure
Short reads (Illumina):  Error correction, polishing
```

**Advantages:**
- Long-read structure + short-read accuracy
- Best of both technologies
- Cost-effective approach

**Tools:**
- MaSuRCA
- Unicycler (bacterial genomes)
- Pilon (polishing)

## Interpreting Biological Results

### Structural Variant Discovery

**Long-read advantages:**
- Direct observation of variants
- Complex nested SVs resolved
- Repetitive region variants
- Mobile element insertions

**Clinical applications:**
- Resolve complex disease loci
- Detect pathogenic SVs missed by short reads
- Haplotype-resolved diagnosis

### De Novo Assembly Applications

**New genome sequencing:**
- Non-model organisms
- Personalized reference genomes
- Pan-genome construction

**Genome improvement:**
- Close reference genome gaps
- Resolve centromeres
- Complete telomeres
- T2T (telomere-to-telomere) assemblies

### Haplotype-Resolved Genomics

**Diploid genome assembly:**

Instead of one consensus genome:
```
Maternal genome (haplotype 1)
Paternal genome (haplotype 2)
```

**Applications:**
- Recessive disease compound heterozygotes
- Imprinted gene analysis
- Allele-specific expression
- Population genetics

## Next Steps

After long-read analysis:

1. **Validation:**
   - PCR confirmation of key SVs
   - Orthogonal technology (Illumina, Sanger)
   - Visualization in IGV

2. **Integration:**
   - Combine with short-read variants
   - Multi-omics integration
   - Functional validation

3. **Specialized analyses:**
   - Repeat expansion detection
   - Telomere length estimation
   - Chromosome-scale haplotyping

4. **Population studies:**
   - Catalog SVs across samples
   - Haplotype frequency analysis
   - Evolutionary studies

## Resources

### Databases
- **NCBI SRA:** Long-read datasets
- **PacBio SMRT Portal:** Analysis platform
- **ONT Community:** Nanopore resources

### Publications
- Sedlazeck et al. (2018) "Accurate detection of complex structural variations using single-molecule sequencing"
- Logsdon et al. (2020) "Long-read human genome sequencing and its applications"

### Tools Documentation
- **minimap2:** https://github.com/lh3/minimap2
- **Sniffles2:** https://github.com/fritzsedlazeck/Sniffles
- **Flye:** https://github.com/fenderglass/Flye
- **WhatsHap:** https://whatshap.readthedocs.io/

## Support

For pipeline-specific issues:
- Check sequencing platform documentation
- Review basecalling quality
- Consult BioPipelines documentation

For biological interpretation:
- Long-read community forums
- Platform-specific support (ONT, PacBio)
- Review technology-specific literature
