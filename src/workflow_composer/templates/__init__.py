"""
Workflow Templates Registry
============================

Maps analysis types to tested Nextflow workflow templates.
Templates are parameterized and customized by the LLM based on user intent.
"""

from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from ..core.query_parser import AnalysisType

# Base path for templates
TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "nextflow-pipelines" / "templates"
WORKFLOWS_DIR = Path(__file__).parent.parent.parent.parent / "nextflow-pipelines" / "workflows"


@dataclass
class WorkflowTemplate:
    """A workflow template that can be customized."""
    name: str
    template_file: Path
    description: str
    required_params: list
    optional_params: list
    
    def load(self) -> str:
        """Load the template content."""
        if self.template_file.exists():
            return self.template_file.read_text()
        return None
    
    def customize(self, params: Dict[str, str]) -> str:
        """Apply parameters to template."""
        content = self.load()
        if not content:
            return None
        
        for key, value in params.items():
            placeholder = "{{" + key.upper() + "}}"
            content = content.replace(placeholder, str(value))
        
        return content


# Registry of available templates
TEMPLATE_REGISTRY: Dict[AnalysisType, WorkflowTemplate] = {
    AnalysisType.CHIP_SEQ: WorkflowTemplate(
        name="chipseq",
        template_file=TEMPLATES_DIR / "chipseq_template.nf",
        description="ChIP-seq peak calling with MACS2",
        required_params=["ORGANISM", "GENOME_BUILD"],
        optional_params=["SINGLE_END", "DATE", "ANALYSIS_TYPE"]
    ),
    
    AnalysisType.RNA_SEQ_DE: WorkflowTemplate(
        name="rnaseq",
        template_file=WORKFLOWS_DIR / "rnaseq_simple.nf",  # Use existing
        description="RNA-seq differential expression analysis",
        required_params=["ORGANISM", "GENOME_BUILD"],
        optional_params=["STRANDEDNESS", "DATE"]
    ),
    
    AnalysisType.ATAC_SEQ: WorkflowTemplate(
        name="atacseq",
        template_file=WORKFLOWS_DIR / "atacseq.nf",
        description="ATAC-seq chromatin accessibility analysis",
        required_params=["ORGANISM", "GENOME_BUILD"],
        optional_params=["DATE"]
    ),
    
    AnalysisType.WGS_VARIANT_CALLING: WorkflowTemplate(
        name="dnaseq",
        template_file=WORKFLOWS_DIR / "dnaseq.nf",
        description="Whole genome sequencing variant calling",
        required_params=["ORGANISM", "GENOME_BUILD"],
        optional_params=["DATE"]
    ),
    
    AnalysisType.METAGENOMICS_PROFILING: WorkflowTemplate(
        name="metagenomics",
        template_file=WORKFLOWS_DIR / "metagenomics.nf",
        description="Metagenomic profiling and assembly",
        required_params=[],
        optional_params=["DATABASE", "DATE"]
    ),
    
    AnalysisType.SCRNA_SEQ: WorkflowTemplate(
        name="scrnaseq",
        template_file=WORKFLOWS_DIR / "scrnaseq.nf",
        description="Single-cell RNA-seq analysis",
        required_params=["ORGANISM", "GENOME_BUILD"],
        optional_params=["CHEMISTRY", "DATE"]
    ),
    
    AnalysisType.HIC: WorkflowTemplate(
        name="hic",
        template_file=WORKFLOWS_DIR / "hic.nf",
        description="Hi-C chromatin interaction analysis",
        required_params=["ORGANISM", "GENOME_BUILD"],
        optional_params=["RESOLUTION", "DATE"]
    ),
    
    AnalysisType.BISULFITE_SEQ: WorkflowTemplate(
        name="methylation",
        template_file=WORKFLOWS_DIR / "methylation.nf",
        description="DNA methylation (bisulfite-seq) analysis",
        required_params=["ORGANISM", "GENOME_BUILD"],
        optional_params=["DATE"]
    ),
    
    AnalysisType.LONG_READ_ASSEMBLY: WorkflowTemplate(
        name="longread",
        template_file=WORKFLOWS_DIR / "longread.nf",
        description="Long-read sequencing analysis (ONT/PacBio)",
        required_params=["ORGANISM", "GENOME_BUILD"],
        optional_params=["PLATFORM", "DATE"]
    ),
}


def get_template(analysis_type: AnalysisType) -> Optional[WorkflowTemplate]:
    """Get template for an analysis type."""
    return TEMPLATE_REGISTRY.get(analysis_type)


def list_templates() -> Dict[str, str]:
    """List all available templates."""
    return {
        at.value: t.description 
        for at, t in TEMPLATE_REGISTRY.items()
    }


def has_template(analysis_type: AnalysisType) -> bool:
    """Check if a template exists for the analysis type."""
    template = TEMPLATE_REGISTRY.get(analysis_type)
    if not template:
        return False
    return template.template_file.exists()
