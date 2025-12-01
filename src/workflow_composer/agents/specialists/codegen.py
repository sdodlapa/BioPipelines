"""
Code Generation Agent
=====================

Generates Nextflow DSL2 code from workflow plans.

Produces:
- main.nf with processes and workflow
- nextflow.config with resource configuration
- Module files for each tool
"""

import logging
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from .planner import WorkflowPlan, WorkflowStep

logger = logging.getLogger(__name__)


class CodeGenAgent:
    """
    Generates production-ready Nextflow DSL2 code.
    
    Takes a WorkflowPlan and generates:
    - main.nf: Main workflow with processes
    - nextflow.config: Configuration file
    - modules/: Individual tool modules
    """
    
    SYSTEM_PROMPT = """You are a Nextflow DSL2 expert.
Given a workflow plan, generate production-ready Nextflow code.

Requirements:
- Use DSL2 syntax with proper process and workflow blocks
- Include proper input/output channels with path() and tuple()
- Add resource directives (cpus, memory, time) 
- Use containers for reproducibility
- Include error handling with errorStrategy 'retry'
- Add publishDir for outputs
- Use proper channel operators (map, collect, join)

Generate complete, runnable main.nf code."""

    def __init__(self, router=None):
        """
        Initialize code generation agent.
        
        Args:
            router: LLM provider router
        """
        self.router = router
    
    async def generate(self, plan: WorkflowPlan) -> str:
        """
        Generate Nextflow code from plan.
        
        Args:
            plan: WorkflowPlan with steps and configuration
            
        Returns:
            Complete main.nf code as string
        """
        if self.router is None:
            # Use template-based generation
            return self._generate_from_template(plan)
        
        try:
            prompt = f"{self.SYSTEM_PROMPT}\n\nWorkflow Plan:\n{plan.to_json()}"
            response = await self.router.route_async(prompt)
            return self._extract_code(response)
        except Exception as e:
            logger.warning(f"LLM code generation failed: {e}, using template")
            return self._generate_from_template(plan)
    
    def generate_sync(self, plan: WorkflowPlan) -> str:
        """Synchronous version of generate."""
        return self._generate_from_template(plan)
    
    async def fix_issues(self, code: str, issues: List[str]) -> str:
        """
        Fix issues in generated code.
        
        Args:
            code: Original code with issues
            issues: List of issues to fix
            
        Returns:
            Fixed code
        """
        if self.router is None:
            # Apply basic fixes
            return self._apply_basic_fixes(code, issues)
        
        try:
            prompt = f"""Fix the following issues in this Nextflow code:

Issues:
{chr(10).join(f"- {issue}" for issue in issues)}

Code:
```nextflow
{code}
```

Return only the fixed code."""
            
            response = await self.router.route_async(prompt)
            return self._extract_code(response)
        except Exception as e:
            logger.warning(f"LLM fix failed: {e}")
            return self._apply_basic_fixes(code, issues)
    
    def _generate_from_template(self, plan: WorkflowPlan) -> str:
        """Generate Nextflow code using templates."""
        lines = [
            "#!/usr/bin/env nextflow",
            "",
            "/*",
            f" * {plan.name}",
            f" * {plan.description}",
            f" * Generated: {datetime.now().isoformat()}",
            " */",
            "",
            "nextflow.enable.dsl = 2",
            "",
        ]
        
        # Add parameters
        lines.extend(self._generate_params(plan))
        
        # Add process for each step
        for step in plan.steps:
            lines.extend(self._generate_process(step, plan))
        
        # Add workflow
        lines.extend(self._generate_workflow(plan))
        
        return "\n".join(lines)
    
    def _generate_params(self, plan: WorkflowPlan) -> List[str]:
        """Generate parameter definitions."""
        lines = [
            "// Pipeline parameters",
            "params.input = null",
            "params.outdir = './results'",
        ]
        
        if plan.organism:
            lines.append(f"params.genome = '{plan.genome_build or 'GRCh38'}'")
        
        if plan.read_type == "paired":
            lines.append("params.single_end = false")
        
        lines.extend([
            "",
            "// Validate inputs",
            "if (!params.input) {",
            "    error \"Please provide input with --input\"",
            "}",
            "",
        ])
        
        return lines
    
    def _generate_process(self, step: WorkflowStep, plan: WorkflowPlan) -> List[str]:
        """Generate a Nextflow process."""
        # Determine container
        container = self._get_container(step.tool, plan.analysis_type)
        
        # Resource directives
        memory = step.resources.get("memory", "8 GB")
        cpus = step.resources.get("cpus", 4)
        time = step.resources.get("time", "4h")
        
        lines = [
            f"/*",
            f" * {step.name}: {step.description}",
            f" */",
            f"process {step.name.upper()} {{",
            f"    tag \"$meta.id\"",
            f"    label 'process_medium'",
            f"    container '{container}'",
            "",
            f"    cpus {cpus}",
            f"    memory '{memory}'",
            f"    time '{time}'",
            "",
            "    errorStrategy 'retry'",
            "    maxRetries 2",
            "",
            f"    publishDir \"${{params.outdir}}/{step.name}\", mode: 'copy'",
            "",
        ]
        
        # Input block
        lines.append("    input:")
        if step.inputs:
            for inp in step.inputs:
                if inp in ["reads", "fastq"]:
                    lines.append("    tuple val(meta), path(reads)")
                elif inp in ["bam", "aligned_bam", "dedup_bam"]:
                    lines.append("    tuple val(meta), path(bam)")
                elif inp == "genome_index":
                    lines.append("    path(index)")
                elif inp == "annotation":
                    lines.append("    path(gtf)")
                else:
                    lines.append(f"    path({inp})")
        else:
            lines.append("    tuple val(meta), path(reads)")
        lines.append("")
        
        # Output block
        lines.append("    output:")
        for out in step.outputs:
            if out in ["fastqc_reports", "qc_reports"]:
                lines.append(f"    tuple val(meta), path('*.html'), emit: {out}")
                lines.append(f"    tuple val(meta), path('*.zip'), emit: {out}_zip")
            elif out in ["aligned_bam", "dedup_bam"]:
                lines.append(f"    tuple val(meta), path('*.bam'), path('*.bai'), emit: bam")
            elif out == "trimmed_reads":
                lines.append("    tuple val(meta), path('*.trimmed.fq.gz'), emit: reads")
            elif out == "count_matrix":
                lines.append(f"    path('counts.txt'), emit: counts")
            elif out == "vcf" or out == "filtered_vcf":
                lines.append(f"    tuple val(meta), path('*.vcf.gz'), emit: vcf")
            elif out == "peaks":
                lines.append(f"    tuple val(meta), path('*.narrowPeak'), emit: peaks")
            elif out == "bigwig":
                lines.append(f"    tuple val(meta), path('*.bw'), emit: bigwig")
            else:
                lines.append(f"    path('*'), emit: {out}")
        lines.append("    path('versions.yml'), emit: versions")
        lines.append("")
        
        # Script block
        lines.append("    script:")
        lines.extend(self._generate_script(step))
        
        lines.extend([
            "}",
            "",
        ])
        
        return lines
    
    def _generate_script(self, step: WorkflowStep) -> List[str]:
        """Generate script block for a process."""
        tool = step.tool.lower()
        
        # Tool-specific scripts
        if tool == "fastqc":
            return [
                '    """',
                '    fastqc -t $task.cpus -o . $reads',
                '',
                '    cat <<-END_VERSIONS > versions.yml',
                '    "${task.process}":',
                '        fastqc: \\$(fastqc --version | sed "s/FastQC v//")',
                '    END_VERSIONS',
                '    """',
            ]
        elif tool == "fastp":
            return [
                '    def prefix = meta.id',
                '    """',
                '    fastp \\',
                '        -i ${reads[0]} \\',
                '        -I ${reads[1]} \\',
                '        -o ${prefix}_1.trimmed.fq.gz \\',
                '        -O ${prefix}_2.trimmed.fq.gz \\',
                '        --thread $task.cpus \\',
                '        --json ${prefix}.fastp.json \\',
                '        --html ${prefix}.fastp.html',
                '',
                '    cat <<-END_VERSIONS > versions.yml',
                '    "${task.process}":',
                '        fastp: \\$(fastp --version 2>&1 | sed "s/fastp //")',
                '    END_VERSIONS',
                '    """',
            ]
        elif tool == "star":
            return [
                '    def prefix = meta.id',
                '    """',
                '    STAR \\',
                '        --runThreadN $task.cpus \\',
                '        --genomeDir $index \\',
                '        --readFilesIn $reads \\',
                '        --readFilesCommand zcat \\',
                '        --outSAMtype BAM SortedByCoordinate \\',
                '        --outFileNamePrefix ${prefix}.',
                '',
                '    samtools index ${prefix}.Aligned.sortedByCoord.out.bam',
                '',
                '    cat <<-END_VERSIONS > versions.yml',
                '    "${task.process}":',
                '        star: \\$(STAR --version | sed "s/STAR_//")',
                '        samtools: \\$(samtools --version | head -1 | sed "s/samtools //")',
                '    END_VERSIONS',
                '    """',
            ]
        elif tool == "featurecounts":
            return [
                '    """',
                '    featureCounts \\',
                '        -T $task.cpus \\',
                '        -p \\',
                '        -a $gtf \\',
                '        -o counts.txt \\',
                '        $bam',
                '',
                '    cat <<-END_VERSIONS > versions.yml',
                '    "${task.process}":',
                '        featurecounts: \\$(featureCounts -v 2>&1 | head -1 | sed "s/featureCounts v//")',
                '    END_VERSIONS',
                '    """',
            ]
        elif tool == "bowtie2":
            return [
                '    def prefix = meta.id',
                '    """',
                '    bowtie2 \\',
                '        -p $task.cpus \\',
                '        -x $index \\',
                '        -U $reads \\',
                '        | samtools sort -@ $task.cpus -o ${prefix}.bam -',
                '',
                '    samtools index ${prefix}.bam',
                '',
                '    cat <<-END_VERSIONS > versions.yml',
                '    "${task.process}":',
                '        bowtie2: \\$(bowtie2 --version | head -1 | sed "s/.*version //")',
                '    END_VERSIONS',
                '    """',
            ]
        elif tool == "macs2":
            return [
                '    def prefix = meta.id',
                '    """',
                '    macs2 callpeak \\',
                '        -t $bam \\',
                '        -f BAM \\',
                '        -g hs \\',
                '        -n $prefix \\',
                '        -q 0.05',
                '',
                '    cat <<-END_VERSIONS > versions.yml',
                '    "${task.process}":',
                '        macs2: \\$(macs2 --version | sed "s/macs2 //")',
                '    END_VERSIONS',
                '    """',
            ]
        elif tool == "multiqc":
            return [
                '    """',
                '    multiqc . -o .',
                '',
                '    cat <<-END_VERSIONS > versions.yml',
                '    "${task.process}":',
                '        multiqc: \\$(multiqc --version | sed "s/multiqc, version //")',
                '    END_VERSIONS',
                '    """',
            ]
        else:
            # Generic script
            return [
                '    """',
                f'    echo "Running {step.tool}"',
                '    # Add tool-specific commands here',
                '',
                '    cat <<-END_VERSIONS > versions.yml',
                '    "${task.process}":',
                f'        {tool}: "1.0.0"',
                '    END_VERSIONS',
                '    """',
            ]
    
    def _generate_workflow(self, plan: WorkflowPlan) -> List[str]:
        """Generate workflow block."""
        lines = [
            "/*",
            " * Main workflow",
            " */",
            "workflow {",
            "",
            "    // Create input channel",
            "    Channel",
            "        .fromFilePairs(params.input, checkIfExists: true)",
            "        .map { id, files -> [ [id: id], files ] }",
            "        .set { reads_ch }",
            "",
        ]
        
        # Call processes in order
        prev_output = "reads_ch"
        for i, step in enumerate(plan.steps):
            process_name = step.name.upper()
            
            if i == 0:
                lines.append(f"    // Step {i+1}: {step.description}")
                lines.append(f"    {process_name}({prev_output})")
            else:
                lines.append(f"")
                lines.append(f"    // Step {i+1}: {step.description}")
                
                # Determine input based on dependencies
                if step.dependencies:
                    dep_process = step.dependencies[0].upper()
                    lines.append(f"    {process_name}({dep_process}.out.{self._get_output_channel(step)})")
                else:
                    lines.append(f"    {process_name}({prev_output})")
            
            prev_output = f"{process_name}.out"
        
        lines.extend([
            "",
            "    // Collect versions",
            "    versions_ch = Channel.empty()",
        ])
        
        for step in plan.steps:
            lines.append(f"    versions_ch = versions_ch.mix({step.name.upper()}.out.versions)")
        
        lines.extend([
            "",
            "}",
        ])
        
        return lines
    
    def _get_container(self, tool: str, analysis_type: str) -> str:
        """Get container image for a tool."""
        tool_lower = tool.lower()
        
        containers = {
            "fastqc": "quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0",
            "fastp": "quay.io/biocontainers/fastp:0.23.4--h5f740d0_0",
            "star": "quay.io/biocontainers/star:2.7.11a--h0033a41_0",
            "hisat2": "quay.io/biocontainers/hisat2:2.2.1--he1b5a44_0",
            "salmon": "quay.io/biocontainers/salmon:1.10.2--hecfa306_0",
            "featurecounts": "quay.io/biocontainers/subread:2.0.6--he4a0461_0",
            "bowtie2": "quay.io/biocontainers/bowtie2:2.5.2--py39h6fed5c7_0",
            "bwa": "quay.io/biocontainers/bwa:0.7.17--h5bf99c6_8",
            "bwa-mem2": "quay.io/biocontainers/bwa-mem2:2.2.1--hd03093a_3",
            "samtools": "quay.io/biocontainers/samtools:1.18--h50ea8bc_1",
            "picard markduplicates": "quay.io/biocontainers/picard:3.1.1--hdfd78af_0",
            "gatk haplotypecaller": "broadinstitute/gatk:4.4.0.0",
            "macs2": "quay.io/biocontainers/macs2:2.2.9.1--py39hf95cd2a_0",
            "deeptools": "quay.io/biocontainers/deeptools:3.5.4--pyhdfd78af_1",
            "deseq2": "quay.io/biocontainers/bioconductor-deseq2:1.40.0--r43hf17093f_0",
            "multiqc": "quay.io/biocontainers/multiqc:1.19--pyhdfd78af_0",
        }
        
        return containers.get(tool_lower, "ubuntu:22.04")
    
    def _get_output_channel(self, step: WorkflowStep) -> str:
        """Get the main output channel name for a step."""
        if "bam" in step.outputs or "aligned_bam" in step.outputs:
            return "bam"
        if "reads" in step.outputs or "trimmed_reads" in step.outputs:
            return "reads"
        if step.outputs:
            return step.outputs[0]
        return "out"
    
    def _extract_code(self, response: str) -> str:
        """Extract Nextflow code from LLM response."""
        import re
        
        # Look for code block
        code_match = re.search(r'```(?:nextflow|groovy)?\s*([\s\S]*?)```', response)
        if code_match:
            return code_match.group(1).strip()
        
        # Return response as-is if no code block
        return response.strip()
    
    def _apply_basic_fixes(self, code: str, issues: List[str]) -> str:
        """Apply basic fixes to code."""
        for issue in issues:
            issue_lower = issue.lower()
            
            if "no process" in issue_lower:
                # Can't easily fix this without context
                pass
            
            if "no workflow" in issue_lower:
                # Add empty workflow if missing
                if "workflow {" not in code:
                    code += "\n\nworkflow {\n    // Add workflow logic\n}\n"
            
            if "container" in issue_lower:
                # Add default container if missing
                if "container" not in code:
                    code = code.replace("process ", "process {\n    container 'ubuntu:22.04'\n")
        
        return code
    
    def generate_config(self, plan: WorkflowPlan) -> str:
        """Generate nextflow.config file."""
        return f"""/*
 * Nextflow configuration for {plan.name}
 */

// Process defaults
process {{
    cpus = {plan.recommended_cpus}
    memory = '{plan.recommended_memory_gb} GB'
    time = '4h'
    
    errorStrategy = 'retry'
    maxRetries = 2
}}

// Execution profiles
profiles {{
    standard {{
        process.executor = 'local'
    }}
    
    slurm {{
        process.executor = 'slurm'
        process.queue = 'normal'
    }}
    
    docker {{
        docker.enabled = true
    }}
    
    singularity {{
        singularity.enabled = true
        singularity.autoMounts = true
    }}
}}

// Manifest
manifest {{
    name = '{plan.name}'
    description = '{plan.description}'
    version = '1.0.0'
    mainScript = 'main.nf'
    nextflowVersion = '>=23.04.0'
}}

// Reporting
report {{
    enabled = true
    file = "${{params.outdir}}/pipeline_report.html"
}}

timeline {{
    enabled = true
    file = "${{params.outdir}}/timeline.html"
}}

trace {{
    enabled = true
    file = "${{params.outdir}}/trace.txt"
}}
"""
