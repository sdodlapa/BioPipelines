"""
Documentation Agent
===================

Generates comprehensive documentation for workflows.

Produces:
- README.md with usage instructions
- Parameter documentation
- Input/output specifications
- DAG diagrams (mermaid)
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .planner import WorkflowPlan, WorkflowStep

logger = logging.getLogger(__name__)


class DocAgent:
    """
    Generates documentation for Nextflow workflows.
    
    Creates:
    - README.md with full usage guide
    - PARAMETERS.md with parameter details
    - DAG visualization (Mermaid format)
    """
    
    SYSTEM_PROMPT = """You are a technical documentation writer for bioinformatics workflows.
Generate clear, comprehensive documentation that includes:
1. Overview and purpose
2. Requirements and installation
3. Usage examples with real commands
4. Parameter descriptions
5. Input/output specifications
6. Troubleshooting tips

Use Markdown formatting with proper headings and code blocks."""

    def __init__(self, router=None):
        """
        Initialize documentation agent.
        
        Args:
            router: LLM provider router for enhanced documentation
        """
        self.router = router
    
    async def generate_readme(self, plan: WorkflowPlan, code: str = None) -> str:
        """
        Generate README.md for workflow.
        
        Args:
            plan: WorkflowPlan with workflow details
            code: Optional Nextflow code for parameter extraction
            
        Returns:
            README.md content as string
        """
        if self.router:
            try:
                return await self._generate_with_llm(plan, code)
            except Exception as e:
                logger.warning(f"LLM documentation failed: {e}")
        
        return self._generate_from_template(plan, code)
    
    def generate_readme_sync(self, plan: WorkflowPlan, code: str = None) -> str:
        """Synchronous version using templates."""
        return self._generate_from_template(plan, code)
    
    def generate_dag(self, plan: WorkflowPlan) -> str:
        """
        Generate Mermaid DAG diagram.
        
        Args:
            plan: WorkflowPlan with steps
            
        Returns:
            Mermaid diagram code
        """
        lines = [
            "```mermaid",
            "graph TD",
            "    INPUT([Input FASTQ]) --> STEP1",
        ]
        
        for i, step in enumerate(plan.steps):
            step_id = f"STEP{i+1}"
            next_id = f"STEP{i+2}" if i < len(plan.steps) - 1 else "OUTPUT"
            
            # Node definition
            lines.append(f"    {step_id}[{step.name}<br/>{step.tool}]")
            
            # Edge to next
            if i < len(plan.steps) - 1:
                lines.append(f"    {step_id} --> {next_id}")
            else:
                lines.append(f"    {step_id} --> OUTPUT([Output Files])")
        
        # Add styling
        lines.extend([
            "",
            "    classDef qc fill:#e1f5fe,stroke:#01579b",
            "    classDef align fill:#fff3e0,stroke:#e65100",
            "    classDef quant fill:#e8f5e9,stroke:#2e7d32",
        ])
        
        # Apply styles based on step type
        for i, step in enumerate(plan.steps):
            step_id = f"STEP{i+1}"
            if "qc" in step.name.lower() or "quality" in step.description.lower():
                lines.append(f"    class {step_id} qc")
            elif "align" in step.name.lower() or "map" in step.name.lower():
                lines.append(f"    class {step_id} align")
            elif "count" in step.name.lower() or "quant" in step.name.lower():
                lines.append(f"    class {step_id} quant")
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def generate_parameters_doc(self, plan: WorkflowPlan, code: str = None) -> str:
        """Generate PARAMETERS.md with detailed parameter documentation."""
        lines = [
            "# Pipeline Parameters",
            "",
            f"Parameter documentation for **{plan.name}**.",
            "",
            "## Required Parameters",
            "",
            "| Parameter | Description | Type | Default |",
            "|-----------|-------------|------|---------|",
            "| `--input` | Input files (glob pattern for FASTQ) | Path | *required* |",
        ]
        
        if plan.organism:
            lines.append(f"| `--genome` | Reference genome | String | `{plan.genome_build or 'GRCh38'}` |")
        
        lines.extend([
            "",
            "## Output Parameters",
            "",
            "| Parameter | Description | Type | Default |",
            "|-----------|-------------|------|---------|",
            "| `--outdir` | Output directory | Path | `./results` |",
            "",
            "## Resource Parameters",
            "",
            "| Parameter | Description | Type | Default |",
            "|-----------|-------------|------|---------|",
            f"| `--max_cpus` | Maximum CPUs per process | Integer | `{plan.recommended_cpus}` |",
            f"| `--max_memory` | Maximum memory per process | String | `'{plan.recommended_memory_gb} GB'` |",
            "| `--max_time` | Maximum time per process | String | `'4.h'` |",
            "",
            "## Process-Specific Parameters",
            "",
        ])
        
        # Add process-specific parameters
        for step in plan.steps:
            lines.append(f"### {step.name}")
            lines.append("")
            lines.append(f"*Tool: {step.tool}*")
            lines.append("")
            
            if step.params:
                lines.append("| Parameter | Value |")
                lines.append("|-----------|-------|")
                for key, value in step.params.items():
                    lines.append(f"| `{key}` | `{value}` |")
            else:
                lines.append("*Using default tool parameters.*")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_from_template(self, plan: WorkflowPlan, code: str = None) -> str:
        """Generate README from template."""
        dag = self.generate_dag(plan)
        
        # Build step descriptions
        steps_section = []
        for i, step in enumerate(plan.steps):
            steps_section.append(f"### {i+1}. {step.name}")
            steps_section.append("")
            steps_section.append(f"**Tool:** {step.tool}")
            steps_section.append("")
            steps_section.append(step.description)
            steps_section.append("")
            steps_section.append(f"- **Inputs:** {', '.join(step.inputs) if step.inputs else 'Previous step output'}")
            steps_section.append(f"- **Outputs:** {', '.join(step.outputs)}")
            steps_section.append("")
        
        return f"""# {plan.name}

{plan.description}

## Overview

This pipeline performs {plan.analysis_type} analysis{f' for {plan.organism}' if plan.organism else ''}.

**Analysis Type:** {plan.analysis_type}  
**Organism:** {plan.organism or 'Not specified'}  
**Genome Build:** {plan.genome_build or 'Not specified'}  
**Read Type:** {plan.read_type or 'Not specified'}

## Pipeline Workflow

{dag}

## Quick Start

### Basic Usage

```bash
nextflow run main.nf \\
    --input 'data/*_R{{1,2}}.fastq.gz' \\
    --outdir results
```

### With SLURM

```bash
nextflow run main.nf \\
    --input 'data/*_R{{1,2}}.fastq.gz' \\
    --outdir results \\
    -profile slurm
```

### With Docker

```bash
nextflow run main.nf \\
    --input 'data/*_R{{1,2}}.fastq.gz' \\
    --outdir results \\
    -profile docker
```

## Requirements

- **Nextflow:** >= 23.04.0
- **Container runtime:** Docker or Singularity

### Containers Used

{self._list_containers(plan)}

## Pipeline Steps

{chr(10).join(steps_section)}

## Parameters

### Input/Output

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Input FASTQ files (glob pattern) | *required* |
| `--outdir` | Output directory | `./results` |
{f'| `--genome` | Reference genome build | `{plan.genome_build}` |' if plan.genome_build else ''}

### Resource Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_cpus` | Maximum CPUs per process | `{plan.recommended_cpus}` |
| `--max_memory` | Maximum memory per process | `{plan.recommended_memory_gb} GB` |

## Output Structure

```
results/
├── {plan.steps[0].name if plan.steps else 'step1'}/
│   └── [QC reports]
{''.join([f'├── {step.name}/' + chr(10) + f'│   └── [{", ".join(step.outputs[:2])}]' + chr(10) for step in plan.steps[1:-1]])}
└── {plan.steps[-1].name if plan.steps else 'final'}/
    └── [Final outputs]
```

## Troubleshooting

### Common Issues

1. **Out of memory errors**
   - Increase `--max_memory` parameter
   - Use a profile with more resources

2. **Container not found**
   - Ensure Docker/Singularity is running
   - Check internet connection for pulling images

3. **Input files not found**
   - Verify the glob pattern matches your files
   - Use absolute paths if needed

## Citation

If you use this pipeline, please cite:

> BioPipelines Workflow Composer
> Generated on {datetime.now().strftime('%Y-%m-%d')}

## License

This pipeline is released under the MIT License.
"""
    
    def _list_containers(self, plan: WorkflowPlan) -> str:
        """Generate list of containers used."""
        containers = set()
        
        container_map = {
            "fastqc": "quay.io/biocontainers/fastqc:0.12.1",
            "fastp": "quay.io/biocontainers/fastp:0.23.4",
            "star": "quay.io/biocontainers/star:2.7.11a",
            "hisat2": "quay.io/biocontainers/hisat2:2.2.1",
            "salmon": "quay.io/biocontainers/salmon:1.10.2",
            "featurecounts": "quay.io/biocontainers/subread:2.0.6",
            "bowtie2": "quay.io/biocontainers/bowtie2:2.5.2",
            "macs2": "quay.io/biocontainers/macs2:2.2.9.1",
            "multiqc": "quay.io/biocontainers/multiqc:1.19",
        }
        
        for step in plan.steps:
            tool = step.tool.lower()
            if tool in container_map:
                containers.add(container_map[tool])
        
        if not containers:
            return "- Base Ubuntu container"
        
        return "\n".join(f"- `{c}`" for c in sorted(containers))
    
    async def _generate_with_llm(self, plan: WorkflowPlan, code: str = None) -> str:
        """Generate enhanced documentation using LLM."""
        code_section = ""
        if code:
            code_section = f"Code:\n```nextflow\n{code[:2000]}...\n```"
        
        prompt = f"""{self.SYSTEM_PROMPT}

Generate a README.md for this workflow:

Workflow Plan:
{plan.to_json()}

{code_section}

Include:
1. Clear title and description
2. Prerequisites
3. Usage examples with actual commands
4. Parameter table
5. Output descriptions
6. Troubleshooting section"""
        
        response = await self.router.route_async(prompt)
        return response
    
    def generate_changelog(self, version: str, changes: List[str]) -> str:
        """Generate CHANGELOG entry."""
        newline = chr(10)
        added = newline.join(f'- {c}' for c in changes if c.startswith('Add'))
        changed = newline.join(f'- {c}' for c in changes if c.startswith('Update') or c.startswith('Change'))
        fixed = newline.join(f'- {c}' for c in changes if c.startswith('Fix'))
        
        return f"""## [{version}] - {datetime.now().strftime('%Y-%m-%d')}

### Added
{added}

### Changed
{changed}

### Fixed
{fixed}
"""
    
    def generate_input_schema(self, plan: WorkflowPlan) -> Dict[str, Any]:
        """Generate nextflow_schema.json for nf-core compatibility."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema",
            "$id": f"https://raw.githubusercontent.com/user/{plan.name}/master/nextflow_schema.json",
            "title": plan.name,
            "description": plan.description,
            "type": "object",
            "definitions": {
                "input_output_options": {
                    "title": "Input/output options",
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "format": "file-path",
                            "description": "Input FASTQ files",
                            "help_text": "Path to input files using glob pattern"
                        },
                        "outdir": {
                            "type": "string",
                            "format": "directory-path",
                            "description": "Output directory",
                            "default": "./results"
                        }
                    },
                    "required": ["input"]
                },
                "reference_genome_options": {
                    "title": "Reference genome options",
                    "type": "object",
                    "properties": {
                        "genome": {
                            "type": "string",
                            "description": "Reference genome",
                            "default": plan.genome_build or "GRCh38"
                        }
                    }
                }
            },
            "allOf": [
                {"$ref": "#/definitions/input_output_options"},
                {"$ref": "#/definitions/reference_genome_options"}
            ]
        }
