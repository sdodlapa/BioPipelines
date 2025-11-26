# BioPipelines

> **AI-Powered Bioinformatics Workflow Generation**  
> Natural language → Production Nextflow pipelines → Results

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Nextflow](https://img.shields.io/badge/nextflow-DSL2-green.svg)](https://www.nextflow.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Quick Start (30 seconds)

```bash
# 1. Activate environment
conda activate biopipelines

# 2. Set API key (Lightning.ai is FREE - 30M tokens!)
export LIGHTNING_API_KEY="your-key"

# 3. Launch web interface
./scripts/start_gradio.sh

# 4. Open browser and type:
#    "RNA-seq differential expression for mouse, paired-end reads"
```

📖 **[Complete Architecture & Guide](docs/ARCHITECTURE_AND_GUIDE.md)** - Everything in one document

---

## Features

### 🤖 AI-Powered Workflow Composer

Generate production-ready Nextflow pipelines from natural language:

```python
from workflow_composer import Composer

composer = Composer()  # Uses Lightning.ai by default (FREE)
workflow = composer.generate(
    "RNA-seq differential expression for mouse, treatment vs control"
)
workflow.save("my_rnaseq_workflow/")
```

**LLM Providers:**
- ⚡ **Lightning.ai** - 30M FREE tokens! (default)
- 🟢 **OpenAI** - GPT-4o, GPT-4-turbo
- 🔵 **Anthropic** - Claude 3.5 Sonnet, Opus
- 🟠 **Ollama** - Local models (llama3, mistral)
- 🟣 **vLLM** - GPU-accelerated local inference

See [LLM Setup Guide](docs/LLM_SETUP.md) for configuration.

### 🧬 10 Production-Ready Pipelines

(8 fully validated, 2 core complete):

- ✅ **DNA-seq**: Variant calling, structural variant detection (VALIDATED)
- ✅ **RNA-seq**: Differential expression, isoform analysis (VALIDATED)
- ✅ **scRNA-seq**: Single-cell analysis, clustering, cell-type annotation (VALIDATED)
- ✅ **ChIP-seq**: Peak calling, motif analysis, differential binding (VALIDATED)
- ✅ **ATAC-seq**: Chromatin accessibility, footprinting (VALIDATED)
- ⚠️ **Methylation**: WGBS/RRBS bisulfite sequencing analysis (CODE VALIDATED - needs production data)
- ⚠️ **Hi-C**: 3D genome organization, contact matrices (CORE COMPLETE - advanced tools optional)
- ✅ **Long-read**: Nanopore/PacBio structural variant detection (VALIDATED)
- ✅ **Metagenomics**: Taxonomic profiling with Kraken2 (VALIDATED)
- ✅ **Structural Variants**: Multi-tool SV calling pipeline (VALIDATED)

**Achievement**: 80% fully validated (8/10), 100% core functional (10/10)  
See `PIPELINE_STATUS_FINAL.md` for detailed validation report.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/BioPipelines.git
cd BioPipelines

# Create conda environment
conda env create -f environment.yml
conda activate biopipelines

# Install Python package
pip install -e .
```

### Running Your First Pipeline

**Option 1: Using Unified Scripts (Recommended)**

```bash
# Download test data
conda activate biopipelines
./scripts/download_data.py chipseq --test --output data/raw/chip_seq/

# Submit pipeline to SLURM
./scripts/submit_pipeline.sh --pipeline chip_seq --mem 32G --cores 8

# Check job status
squeue -u $USER

# View results
ls data/results/chip_seq/
```

**Option 2: Manual Execution**

```bash
# Navigate to pipeline directory
cd pipelines/dna_seq/variant_calling

# Edit config.yaml with your sample information
vim config.yaml

# Run with Snakemake
snakemake --cores 4
```

**Available Pipelines:**
- `atac_seq`, `chip_seq`, `dna_seq`, `rna_seq`, `scrna_seq`
- `methylation`, `hic`, `long_read`, `metagenomics`, `sv`

See `scripts/README.md` for detailed usage of unified scripts.

## Project Structure

```
BioPipelines/
├── src/workflow_composer/  # AI Workflow Composer (main package)
│   ├── llm/               # LLM adapters (OpenAI, vLLM, HuggingFace)
│   ├── core/              # Intent parsing, tool selection, workflow generation
│   ├── cli.py             # biocomposer CLI
│   └── composer.py        # Main Composer class
├── pipelines/             # Analysis pipelines (Snakemake workflows)
│   ├── dna_seq/           # Variant calling with GATK
│   ├── rna_seq/           # Differential expression with DESeq2
│   ├── scrna_seq/         # Single-cell analysis with Scanpy
│   ├── chip_seq/          # Peak calling with MACS2
│   └── ...                # More pipelines
├── containers/            # Singularity container definitions
├── config/                # Configuration files
│   └── composer.yaml      # Workflow Composer config
├── scripts/               # Utility scripts
│   └── llm/               # vLLM server scripts
├── data/                  # Data directory (gitignored)
├── docs/                  # Documentation
│   ├── LLM_SETUP.md       # LLM integration guide
│   ├── TUTORIALS.md       # Workflow Composer tutorials
│   └── COMPOSITION_PATTERNS.md  # 27 workflow patterns
├── examples/              # Example workflows
│   └── generated/         # AI-generated workflow examples
├── logs/                  # Job logs
└── tests/                 # Test suite
```

## AI Workflow Composer

### CLI Usage

```bash
# Generate workflow from natural language
biocomposer generate "ChIP-seq peak calling for human H3K4me3" -o chipseq_workflow/

# Interactive chat mode
biocomposer chat --llm openai

# Search available tools
biocomposer tools --search "alignment"

# List modules
biocomposer modules --list

# Check LLM providers
biocomposer providers --check
```

### Python API

```python
from workflow_composer import Composer
from workflow_composer.llm import get_llm, check_providers

# Check available providers
print(check_providers())
# {'openai': True, 'vllm': True, 'ollama': False, ...}

# Create composer with specific LLM
llm = get_llm("openai", model="gpt-4o")
composer = Composer(llm=llm)

# Generate and save workflow
workflow = composer.generate(
    "WGS germline variant calling for human samples"
)
workflow.save("variants_workflow/")
```

See [Workflow Composer Guide](docs/WORKFLOW_COMPOSER_GUIDE.md) for detailed documentation.

## Pipelines (Snakemake)

### DNA-seq Variant Calling
- Quality control (FastQC, MultiQC)
- Read trimming (fastp)
- Alignment (BWA-MEM)
- Variant calling (GATK, FreeBayes)
- Annotation (SnpEff, VEP)

### RNA-seq Differential Expression
- QC and trimming
- Alignment (STAR) or pseudo-alignment (Salmon)
- Quantification (featureCounts, RSEM)
- Differential expression (DESeq2, edgeR)
- Functional enrichment (GSEA)

## Documentation

📖 **[Architecture & Complete Guide](docs/ARCHITECTURE_AND_GUIDE.md)** - Start here! One document with everything.

### Additional References
- [LLM Setup Guide](docs/LLM_SETUP.md) - Configure API keys
- [Container Architecture](docs/CONTAINER_ARCHITECTURE.md) - Singularity containers
- [Quick Start Containers](docs/QUICK_START_CONTAINERS.md) - Build instructions

## Requirements

- Python >= 3.10
- Conda/Mamba
- Nextflow >= 23.0
- Singularity >= 3.8
- SLURM (for HPC execution)

## Contributing

Contributions welcome! Please open an issue first to discuss changes.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

For questions, please open an issue on GitHub.

