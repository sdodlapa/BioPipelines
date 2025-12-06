"""
BioPipelines MCP Server
========================

Model Context Protocol server exposing BioPipelines capabilities
for integration with Claude Code, Cursor, and other MCP clients.

The server provides:
- Data discovery tools (ENCODE, GEO, TCGA search)
- Workflow generation (RNA-seq, ChIP-seq, etc.)
- Database queries (UniProt, STRING, KEGG, Reactome, PubMed, ClinVar)
- Job management (submit, status, cancel, logs)
- LLM strategy management
- Educational tools (concept explanation)
- MCP Prompts for common workflows

Protocol Version: 2025-06-18 (with 2024-11-05 compatibility)

Usage:
    # Run via stdio (default for Claude Code integration)
    python -m workflow_composer.mcp.server
    
    # Run via HTTP (for development/testing)
    python -m workflow_composer.mcp.server --transport http --port 8080
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Protocol version
MCP_PROTOCOL_VERSION = "2025-06-18"


@dataclass
class ToolAnnotations:
    """Annotations for MCP tools per 2025-06-18 spec."""
    requires_confirmation: bool = False
    destructive: bool = False
    read_only: bool = False
    category: str = "general"
    idempotent: bool = True
    open_world_hint: bool = False


@dataclass
class ToolDefinition:
    """Definition of an MCP tool with optional annotations and output schema."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    annotations: Optional[ToolAnnotations] = None
    output_schema: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    

@dataclass
class ResourceDefinition:
    """Definition of an MCP resource with optional annotations."""
    uri: str
    name: str
    description: str
    handler: Callable
    mime_type: str = "text/markdown"
    annotations: Optional[Dict[str, Any]] = None


@dataclass
class PromptArgument:
    """Argument for an MCP prompt."""
    name: str
    description: str
    required: bool = True


@dataclass
class PromptDefinition:
    """Definition of an MCP prompt."""
    name: str
    description: str
    handler: Callable
    arguments: List[PromptArgument] = field(default_factory=list)
    title: Optional[str] = None


class BioPipelinesMCPServer:
    """
    MCP Server exposing BioPipelines capabilities.
    
    This server implements the Model Context Protocol to expose
    BioPipelines tools to MCP-compatible clients like Claude Code.
    
    Features:
    - 18 tools including SLURM job management and LLM strategy control
    - 6 resources with dynamic URI pattern support
    - 5 prompts for common workflow patterns
    - Tool annotations for confirmation requirements
    - Output schemas for structured responses
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.resources: Dict[str, ResourceDefinition] = {}
        self.prompts: Dict[str, PromptDefinition] = {}
        self._setup_tools()
        self._setup_resources()
        self._setup_prompts()
    
    def _setup_tools(self):
        """Register all BioPipelines tools."""
        
        # ========================================
        # Data Discovery Tools
        # ========================================
        self._register_tool(
            name="search_encode",
            description="Search ENCODE database for chromatin accessibility, histone modifications, transcription factor binding, and gene expression data.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms (cell type, target, assay, etc.)"
                    },
                    "assay_type": {
                        "type": "string",
                        "enum": ["ChIP-seq", "ATAC-seq", "RNA-seq", "WGBS", "Hi-C"],
                        "description": "Filter by assay type"
                    },
                    "organism": {
                        "type": "string",
                        "enum": ["human", "mouse"],
                        "default": "human",
                        "description": "Filter by organism"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results to return"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_encode,
            annotations=ToolAnnotations(read_only=True, category="data_discovery"),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "accession": {"type": "string"},
                                "assay": {"type": "string"},
                                "biosample": {"type": "string"},
                                "target": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["success"]
            }
        )
        
        self._register_tool(
            name="search_geo",
            description="Search NCBI GEO database for gene expression datasets and experiments.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Filter by organism (e.g., 'Homo sapiens', 'Mus musculus')"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_geo,
            annotations=ToolAnnotations(read_only=True, category="data_discovery"),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "accession": {"type": "string"},
                                "title": {"type": "string"},
                                "organism": {"type": "string"},
                                "platform": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["success"]
            }
        )
        
        # ========================================
        # Workflow Generation Tools
        # ========================================
        self._register_tool(
            name="create_workflow",
            description="Generate a bioinformatics analysis workflow (Nextflow or Snakemake) for RNA-seq, ChIP-seq, variant calling, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": ["rnaseq", "chipseq", "methylation", "variant", "atacseq"],
                        "description": "Type of analysis"
                    },
                    "organism": {
                        "type": "string",
                        "default": "human",
                        "description": "Reference organism"
                    },
                    "input_dir": {
                        "type": "string",
                        "description": "Input data directory"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory"
                    },
                    "workflow_engine": {
                        "type": "string",
                        "enum": ["nextflow", "snakemake"],
                        "default": "nextflow",
                        "description": "Workflow engine"
                    }
                },
                "required": ["analysis_type"]
            },
            handler=self._handle_create_workflow,
            annotations=ToolAnnotations(category="workflow_generation"),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "workflow_path": {"type": "string"},
                    "engine": {"type": "string"},
                    "analysis_type": {"type": "string"},
                    "files_created": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["success"]
            }
        )
        
        self._register_tool(
            name="use_workflow_template",
            description="Generate a workflow from a pre-built template with customizable parameters.",
            parameters={
                "type": "object",
                "properties": {
                    "template_name": {
                        "type": "string",
                        "description": "Name of the template (e.g., 'basic_de', 'full_analysis', 'peak_calling')"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Template parameters"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for generated workflow"
                    }
                },
                "required": ["template_name"]
            },
            handler=self._handle_use_template
        )
        
        # ========================================
        # Database Query Tools
        # ========================================
        self._register_tool(
            name="search_uniprot",
            description="Search UniProt database for protein sequences, annotations, and functions.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gene name, protein name, or keywords"
                    },
                    "organism": {
                        "type": "string",
                        "default": "human",
                        "description": "Filter by organism"
                    },
                    "reviewed": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only return reviewed (Swiss-Prot) entries"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 25,
                        "description": "Maximum results"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_uniprot,
            annotations=ToolAnnotations(read_only=True, category="database"),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "proteins": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "accession": {"type": "string"},
                                "gene": {"type": "string"},
                                "name": {"type": "string"},
                                "organism": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["success"]
            }
        )
        
        self._register_tool(
            name="get_protein_interactions",
            description="Get protein-protein interactions from STRING database.",
            parameters={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene names"
                    },
                    "organism": {
                        "type": "string",
                        "default": "human",
                        "description": "Organism"
                    },
                    "score_threshold": {
                        "type": "integer",
                        "default": 400,
                        "description": "Minimum interaction score (0-1000)"
                    }
                },
                "required": ["genes"]
            },
            handler=self._handle_get_interactions,
            annotations=ToolAnnotations(read_only=True, category="database"),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "interactions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "protein_a": {"type": "string"},
                                "protein_b": {"type": "string"},
                                "score": {"type": "number"}
                            }
                        }
                    }
                },
                "required": ["success"]
            }
        )
        
        self._register_tool(
            name="get_functional_enrichment",
            description="Get Gene Ontology and pathway enrichment analysis for a gene list.",
            parameters={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene names"
                    },
                    "organism": {
                        "type": "string",
                        "default": "human",
                        "description": "Organism"
                    }
                },
                "required": ["genes"]
            },
            handler=self._handle_get_enrichment,
            annotations=ToolAnnotations(read_only=True, category="database")
        )
        
        self._register_tool(
            name="search_kegg_pathways",
            description="Search KEGG database for metabolic and signaling pathways.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pathway name or related terms"
                    },
                    "organism": {
                        "type": "string",
                        "default": "hsa",
                        "description": "KEGG organism code (hsa=human, mmu=mouse)"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_kegg,
            annotations=ToolAnnotations(read_only=True, category="database")
        )
        
        self._register_tool(
            name="search_pubmed",
            description="Search PubMed for scientific literature and publications.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results"
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "date"],
                        "default": "relevance",
                        "description": "Sort order"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_pubmed
        )
        
        self._register_tool(
            name="search_variants",
            description="Search ClinVar for variant pathogenicity information.",
            parameters={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol"
                    },
                    "significance": {
                        "type": "string",
                        "enum": ["pathogenic", "likely_pathogenic", "uncertain_significance", "benign"],
                        "description": "Filter by clinical significance"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 25,
                        "description": "Maximum results"
                    }
                },
                "required": ["gene"]
            },
            handler=self._handle_search_variants
        )
        
        # ========================================
        # Educational Tools
        # ========================================
        self._register_tool(
            name="explain_concept",
            description="Explain a bioinformatics concept, tool, or method in detail.",
            parameters={
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "The concept to explain (e.g., 'DESeq2', 'peak calling', 'GWAS')"
                    },
                    "level": {
                        "type": "string",
                        "enum": ["beginner", "intermediate", "advanced"],
                        "default": "intermediate",
                        "description": "Explanation level"
                    }
                },
                "required": ["concept"]
            },
            handler=self._handle_explain_concept,
            annotations=ToolAnnotations(read_only=True, category="education")
        )
        
        # ========================================
        # Job Management Tools
        # ========================================
        self._register_tool(
            name="check_job_status",
            description="Check the status of a submitted workflow job.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "SLURM job identifier"
                    }
                },
                "required": ["job_id"]
            },
            handler=self._handle_check_job,
            annotations=ToolAnnotations(read_only=True, category="job_management"),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "job_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"]},
                    "progress": {"type": "string"},
                    "runtime": {"type": "string"},
                    "exit_code": {"type": "integer"}
                },
                "required": ["success", "status"]
            }
        )
        
        self._register_tool(
            name="submit_job",
            description="Submit a workflow job to the SLURM cluster. Use dry_run=True to preview the command without submitting.",
            parameters={
                "type": "object",
                "properties": {
                    "workflow_path": {
                        "type": "string",
                        "description": "Path to workflow file (Nextflow .nf or Snakemake Snakefile)"
                    },
                    "partition": {
                        "type": "string",
                        "enum": ["t4flex", "gpu", "standard", "highmem"],
                        "default": "t4flex",
                        "description": "SLURM partition to use"
                    },
                    "time_limit": {
                        "type": "string",
                        "default": "4:00:00",
                        "description": "Job time limit (HH:MM:SS)"
                    },
                    "cpus": {
                        "type": "integer",
                        "default": 4,
                        "description": "Number of CPUs to request"
                    },
                    "memory": {
                        "type": "string",
                        "default": "16G",
                        "description": "Memory to request (e.g., '16G', '32G')"
                    },
                    "gpu": {
                        "type": "boolean",
                        "default": False,
                        "description": "Request GPU resources"
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, only show the sbatch command without submitting"
                    }
                },
                "required": ["workflow_path"]
            },
            handler=self._handle_submit_job,
            annotations=ToolAnnotations(
                requires_confirmation=True, 
                destructive=False, 
                category="job_management",
                idempotent=False
            ),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "job_id": {"type": "string"},
                    "command": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                    "message": {"type": "string"}
                },
                "required": ["success"]
            }
        )
        
        self._register_tool(
            name="list_jobs",
            description="List current SLURM jobs for the user.",
            parameters={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["all", "running", "pending", "completed", "failed"],
                        "default": "all",
                        "description": "Filter by job state"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of jobs to return"
                    }
                }
            },
            handler=self._handle_list_jobs,
            annotations=ToolAnnotations(read_only=True, category="job_management"),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "jobs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "job_id": {"type": "string"},
                                "name": {"type": "string"},
                                "state": {"type": "string"},
                                "partition": {"type": "string"},
                                "runtime": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["success"]
            }
        )
        
        self._register_tool(
            name="cancel_job",
            description="Cancel a running or pending SLURM job. This action cannot be undone.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "SLURM job ID to cancel"
                    }
                },
                "required": ["job_id"]
            },
            handler=self._handle_cancel_job,
            annotations=ToolAnnotations(
                requires_confirmation=True, 
                destructive=True, 
                category="job_management",
                idempotent=True
            ),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "job_id": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["success"]
            }
        )
        
        self._register_tool(
            name="get_job_logs",
            description="Retrieve logs from a SLURM job.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "SLURM job ID"
                    },
                    "log_type": {
                        "type": "string",
                        "enum": ["stdout", "stderr", "nextflow", "all"],
                        "default": "all",
                        "description": "Type of logs to retrieve"
                    },
                    "tail_lines": {
                        "type": "integer",
                        "default": 100,
                        "description": "Number of lines from end of log"
                    }
                },
                "required": ["job_id"]
            },
            handler=self._handle_get_job_logs,
            annotations=ToolAnnotations(read_only=True, category="job_management")
        )
        
        # ========================================
        # LLM Strategy Management Tools
        # ========================================
        self._register_tool(
            name="get_llm_strategy",
            description="Get current LLM routing strategy and metrics.",
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self._handle_get_strategy,
            annotations=ToolAnnotations(read_only=True, category="llm_management"),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "profile": {"type": "string"},
                    "strategy": {"type": "string"},
                    "cloud_fallback": {"type": "boolean"},
                    "local_models": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "cloud_providers": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["success"]
            }
        )
        
        self._register_tool(
            name="set_llm_strategy",
            description="Switch LLM routing strategy profile.",
            parameters={
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "enum": ["t4_hybrid", "t4_local_only", "cloud_only", "development"],
                        "description": "Strategy profile to activate"
                    }
                },
                "required": ["profile"]
            },
            handler=self._handle_set_strategy,
            annotations=ToolAnnotations(
                requires_confirmation=False, 
                category="llm_management",
                idempotent=True
            ),
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "previous_profile": {"type": "string"},
                    "new_profile": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["success"]
            }
        )
    
    def _setup_resources(self):
        """Register MCP resources with dynamic URI pattern support."""
        
        self._register_resource(
            uri="biopipelines://skills",
            name="Available Skills",
            description="List of all available BioPipelines skills and capabilities",
            handler=self._handle_get_skills
        )
        
        self._register_resource(
            uri="biopipelines://templates",
            name="Workflow Templates",
            description="List of available pre-built workflow templates",
            handler=self._handle_get_templates
        )
        
        self._register_resource(
            uri="biopipelines://databases",
            name="Database Integrations",
            description="List of integrated biological databases",
            handler=self._handle_get_databases
        )
        
        # NEW: Dynamic pipeline resources
        self._register_resource(
            uri="biopipelines://pipelines",
            name="Available Pipelines",
            description="List of all BioPipelines workflows with descriptions and requirements",
            handler=self._handle_get_pipelines
        )
        
        self._register_resource(
            uri="biopipelines://strategy",
            name="LLM Strategy Info",
            description="Current LLM routing strategy configuration",
            handler=self._handle_get_strategy_resource
        )
        
        self._register_resource(
            uri="biopipelines://jobs",
            name="Active Jobs",
            description="List of currently active SLURM jobs",
            handler=self._handle_get_jobs_resource
        )
    
    def _setup_prompts(self):
        """Register MCP prompts for common workflows."""
        
        self._register_prompt(
            name="analyze_rnaseq",
            title="RNA-seq Analysis Wizard",
            description="Guide through RNA-seq differential expression analysis from raw data to results",
            arguments=[
                PromptArgument(name="data_path", description="Path to FASTQ files", required=True),
                PromptArgument(name="organism", description="Organism (human, mouse, etc.)", required=True),
                PromptArgument(name="comparison", description="Comparison specification (e.g., 'treated_vs_control')", required=False)
            ],
            handler=self._handle_prompt_rnaseq
        )
        
        self._register_prompt(
            name="debug_workflow",
            title="Debug Failed Workflow",
            description="Analyze workflow errors and suggest fixes based on logs and error patterns",
            arguments=[
                PromptArgument(name="job_id", description="Failed SLURM job ID", required=True),
                PromptArgument(name="error_type", description="Type of error if known (memory, timeout, etc.)", required=False)
            ],
            handler=self._handle_prompt_debug
        )
        
        self._register_prompt(
            name="find_datasets",
            title="Find Public Datasets",
            description="Search ENCODE and GEO for datasets matching research criteria",
            arguments=[
                PromptArgument(name="topic", description="Research topic or keywords", required=True),
                PromptArgument(name="organism", description="Target organism", required=False),
                PromptArgument(name="assay_type", description="Preferred assay type", required=False)
            ],
            handler=self._handle_prompt_find_datasets
        )
        
        self._register_prompt(
            name="design_pipeline",
            title="Design Analysis Pipeline",
            description="Help design a complete analysis pipeline based on research question",
            arguments=[
                PromptArgument(name="research_question", description="What biological question are you investigating?", required=True),
                PromptArgument(name="data_type", description="Type of data you have (RNA-seq, ChIP-seq, etc.)", required=False)
            ],
            handler=self._handle_prompt_design_pipeline
        )
        
        self._register_prompt(
            name="explain_results",
            title="Explain Analysis Results",
            description="Help interpret and explain bioinformatics analysis results",
            arguments=[
                PromptArgument(name="analysis_type", description="Type of analysis (DE, enrichment, etc.)", required=True),
                PromptArgument(name="results_path", description="Path to results files", required=False)
            ],
            handler=self._handle_prompt_explain_results
        )
    
    def _register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
        annotations: Optional[ToolAnnotations] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None
    ):
        """Register a tool with the server."""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            annotations=annotations,
            output_schema=output_schema,
            title=title
        )
    
    def _register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        handler: Callable,
        mime_type: str = "text/markdown",
        annotations: Optional[Dict[str, Any]] = None
    ):
        """Register a resource with the server."""
        self.resources[uri] = ResourceDefinition(
            uri=uri,
            name=name,
            description=description,
            handler=handler,
            mime_type=mime_type,
            annotations=annotations
        )
    
    def _register_prompt(
        self,
        name: str,
        description: str,
        handler: Callable,
        arguments: Optional[List[PromptArgument]] = None,
        title: Optional[str] = None
    ):
        """Register a prompt with the server."""
        self.prompts[name] = PromptDefinition(
            name=name,
            description=description,
            handler=handler,
            arguments=arguments or [],
            title=title
        )
    
    # Tool Handlers
    async def _handle_search_encode(self, **kwargs) -> Dict[str, Any]:
        """Handle ENCODE search."""
        try:
            # Import BioPipelines data discovery
            from workflow_composer.data.discovery import DataDiscovery
            
            discovery = DataDiscovery()
            query = kwargs.get("query", "")
            assay_type = kwargs.get("assay_type")
            organism = kwargs.get("organism", "human")
            
            # Build search query
            search_query = f"{query}"
            if organism:
                search_query = f"{organism} {search_query}"
            if assay_type:
                search_query = f"{search_query} {assay_type}"
            
            result = discovery.search_encode(
                query=search_query,
                limit=kwargs.get("limit", 10)
            )
            
            return {
                "success": True,
                "content": self._format_search_results(result)
            }
        except Exception as e:
            logger.error(f"ENCODE search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_geo(self, **kwargs) -> Dict[str, Any]:
        """Handle GEO search."""
        try:
            from workflow_composer.data.discovery import DataDiscovery
            
            discovery = DataDiscovery()
            query = kwargs.get("query", "")
            organism = kwargs.get("organism")
            
            # Build search query
            search_query = f"{query}"
            if organism:
                search_query = f"{organism} {search_query}"
            
            result = discovery.search_geo(
                query=search_query,
                limit=kwargs.get("limit", 10)
            )
            
            return {
                "success": True,
                "content": self._format_search_results(result)
            }
        except Exception as e:
            logger.error(f"GEO search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_create_workflow(self, **kwargs) -> Dict[str, Any]:
        """Handle workflow creation."""
        try:
            from workflow_composer import BioPipelines
            
            bp = BioPipelines()
            workflow = bp.compose(
                analysis_type=kwargs.get("analysis_type"),
                organism=kwargs.get("organism", "human"),
                input_dir=kwargs.get("input_dir"),
                output_dir=kwargs.get("output_dir"),
                workflow_engine=kwargs.get("workflow_engine", "nextflow")
            )
            
            return {
                "success": True,
                "content": f"Generated {kwargs.get('analysis_type')} workflow:\n\n```{kwargs.get('workflow_engine', 'nextflow')}\n{workflow}\n```"
            }
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_use_template(self, **kwargs) -> Dict[str, Any]:
        """Handle workflow template usage."""
        try:
            # Add config to path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))
            from workflow_templates import get_template_engine
            
            engine = get_template_engine()
            template = engine.get_template(kwargs.get("template_name"))
            
            if not template:
                return {
                    "success": False,
                    "error": f"Template not found: {kwargs.get('template_name')}"
                }
            
            result = engine.generate(
                kwargs.get("template_name"),
                output_dir=kwargs.get("output_dir"),
                **kwargs.get("parameters", {})
            )
            
            return {
                "success": result.get("success", False),
                "content": f"Generated workflow from template '{kwargs.get('template_name')}'\n\nFiles:\n" + 
                          "\n".join(f"- {k}: {v}" for k, v in result.get("files", {}).items())
            }
        except Exception as e:
            logger.error(f"Template usage failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_uniprot(self, **kwargs) -> Dict[str, Any]:
        """Handle UniProt search."""
        try:
            from workflow_composer.agents.tools.databases import get_uniprot_client
            
            client = get_uniprot_client()
            result = client.search(
                query=kwargs.get("query", ""),
                organism=kwargs.get("organism", "human"),
                reviewed=kwargs.get("reviewed", True),
                limit=kwargs.get("limit", 25)
            )
            
            return {
                "success": result.success,
                "content": self._format_protein_results(result)
            }
        except Exception as e:
            logger.error(f"UniProt search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_interactions(self, **kwargs) -> Dict[str, Any]:
        """Handle STRING interaction query."""
        try:
            from workflow_composer.agents.tools.databases import get_string_client
            
            client = get_string_client()
            species_map = {"human": 9606, "mouse": 10090, "rat": 10116}
            organism = kwargs.get("organism", "human")
            species = species_map.get(organism, 9606)
            
            result = client.search(
                identifiers=kwargs.get("genes", []),
                species=species,
                required_score=kwargs.get("score_threshold", 400)
            )
            
            return {
                "success": result.success,
                "content": self._format_interaction_results(result)
            }
        except Exception as e:
            logger.error(f"STRING search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_enrichment(self, **kwargs) -> Dict[str, Any]:
        """Handle functional enrichment analysis."""
        try:
            from workflow_composer.agents.tools.databases import get_string_client
            
            client = get_string_client()
            species_map = {"human": 9606, "mouse": 10090, "rat": 10116}
            organism = kwargs.get("organism", "human")
            species = species_map.get(organism, 9606)
            
            result = client.get_enrichment(
                identifiers=kwargs.get("genes", []),
                species=species
            )
            
            return {
                "success": result.success,
                "content": self._format_enrichment_results(result)
            }
        except Exception as e:
            logger.error(f"Enrichment analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_kegg(self, **kwargs) -> Dict[str, Any]:
        """Handle KEGG pathway search."""
        try:
            from workflow_composer.agents.tools.databases import get_kegg_client
            
            client = get_kegg_client()
            result = client.search(
                query=kwargs.get("query", ""),
                organism=kwargs.get("organism", "hsa")
            )
            
            return {
                "success": result.success,
                "content": self._format_pathway_results(result)
            }
        except Exception as e:
            logger.error(f"KEGG search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_pubmed(self, **kwargs) -> Dict[str, Any]:
        """Handle PubMed search."""
        try:
            from workflow_composer.agents.tools.databases import get_pubmed_client
            
            client = get_pubmed_client()
            result = client.search(
                query=kwargs.get("query", ""),
                max_results=kwargs.get("limit", 10),
                sort=kwargs.get("sort", "relevance")
            )
            
            return {
                "success": result.success,
                "content": self._format_pubmed_results(result)
            }
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_variants(self, **kwargs) -> Dict[str, Any]:
        """Handle ClinVar variant search."""
        try:
            from workflow_composer.agents.tools.databases import get_clinvar_client
            
            client = get_clinvar_client()
            result = client.search_by_gene(
                gene=kwargs.get("gene", ""),
                significance=kwargs.get("significance"),
                limit=kwargs.get("limit", 25)
            )
            
            return {
                "success": result.success,
                "content": self._format_variant_results(result)
            }
        except Exception as e:
            logger.error(f"ClinVar search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_explain_concept(self, **kwargs) -> Dict[str, Any]:
        """Handle concept explanation."""
        try:
            from workflow_composer.agents.tools.education import explain_concept_impl
            
            result = explain_concept_impl(
                concept=kwargs.get("concept", "")
            )
            
            if result.success:
                return {
                    "success": True,
                    "content": result.data.get("explanation", result.message) if result.data else result.message
                }
            else:
                return {"success": False, "error": result.message}
        except Exception as e:
            logger.error(f"Concept explanation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_check_job(self, **kwargs) -> Dict[str, Any]:
        """Handle job status check."""
        try:
            job_id = kwargs.get("job_id", "")
            
            # Use squeue to check job status
            result = subprocess.run(
                ["squeue", "-j", job_id, "-o", "%i|%j|%T|%M|%P", "--noheader"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("|")
                status = parts[2] if len(parts) > 2 else "UNKNOWN"
                runtime = parts[3] if len(parts) > 3 else "N/A"
                
                return {
                    "success": True,
                    "content": f"**Job {job_id}**\n\nStatus: {status}\nRuntime: {runtime}",
                    "structuredContent": {
                        "success": True,
                        "job_id": job_id,
                        "status": status,
                        "runtime": runtime
                    }
                }
            else:
                # Check sacct for completed jobs
                result = subprocess.run(
                    ["sacct", "-j", job_id, "-o", "JobID,State,ExitCode,Elapsed", "--noheader", "-P"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    if lines:
                        parts = lines[0].split("|")
                        status = parts[1] if len(parts) > 1 else "UNKNOWN"
                        exit_code = parts[2] if len(parts) > 2 else "N/A"
                        elapsed = parts[3] if len(parts) > 3 else "N/A"
                        return {
                            "success": True,
                            "content": f"**Job {job_id}** (completed)\n\nStatus: {status}\nExit Code: {exit_code}\nElapsed: {elapsed}",
                            "structuredContent": {
                                "success": True,
                                "job_id": job_id,
                                "status": status,
                                "exit_code": exit_code,
                                "runtime": elapsed
                            }
                        }
                return {
                    "success": True,
                    "content": f"Job {job_id}: Not found or not accessible",
                    "structuredContent": {"success": True, "job_id": job_id, "status": "UNKNOWN"}
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "SLURM command timed out"}
        except FileNotFoundError:
            # SLURM not available, use mock for development
            return {
                "success": True,
                "content": f"**Job {kwargs.get('job_id')}** (mock mode)\n\nStatus: RUNNING\nProgress: 50%",
                "structuredContent": {
                    "success": True,
                    "job_id": kwargs.get("job_id"),
                    "status": "RUNNING",
                    "progress": "50%"
                }
            }
        except Exception as e:
            logger.error(f"Job status check failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_submit_job(self, **kwargs) -> Dict[str, Any]:
        """Handle job submission to SLURM."""
        try:
            workflow_path = kwargs.get("workflow_path", "")
            partition = kwargs.get("partition", "t4flex")
            time_limit = kwargs.get("time_limit", "4:00:00")
            cpus = kwargs.get("cpus", 4)
            memory = kwargs.get("memory", "16G")
            gpu = kwargs.get("gpu", False)
            dry_run = kwargs.get("dry_run", True)
            
            # Validate workflow path
            if not workflow_path:
                return {"success": False, "error": "workflow_path is required"}
            
            workflow_file = Path(workflow_path)
            if not dry_run and not workflow_file.exists():
                return {"success": False, "error": f"Workflow file not found: {workflow_path}"}
            
            # Determine workflow type
            is_nextflow = workflow_path.endswith(".nf") or "nextflow" in workflow_path.lower()
            
            # Build sbatch command
            job_name = workflow_file.stem if workflow_file.exists() else "biopipelines_job"
            
            sbatch_args = [
                "sbatch",
                f"--partition={partition}",
                f"--time={time_limit}",
                f"--cpus-per-task={cpus}",
                f"--mem={memory}",
                f"--job-name={job_name}",
                "--output=slurm-%j.out",
                "--error=slurm-%j.err"
            ]
            
            if gpu:
                sbatch_args.append("--gres=gpu:1")
            
            # Add workflow execution command
            if is_nextflow:
                run_cmd = f"nextflow run {workflow_path} -resume"
            else:
                run_cmd = f"snakemake -s {workflow_path} --cores {cpus}"
            
            sbatch_args.extend(["--wrap", run_cmd])
            
            command_str = " ".join(sbatch_args)
            
            if dry_run:
                return {
                    "success": True,
                    "content": f"**Dry Run - Command Preview**\n\n```bash\n{command_str}\n```\n\nSet `dry_run=False` to submit this job.",
                    "structuredContent": {
                        "success": True,
                        "command": command_str,
                        "dry_run": True,
                        "message": "Dry run mode - job not submitted"
                    }
                }
            
            # Actually submit
            result = subprocess.run(sbatch_args, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse job ID from "Submitted batch job 12345"
                output = result.stdout.strip()
                job_id = output.split()[-1] if output else "unknown"
                
                return {
                    "success": True,
                    "content": f"**Job Submitted Successfully**\n\nJob ID: {job_id}\nPartition: {partition}\nCommand: `{run_cmd}`",
                    "structuredContent": {
                        "success": True,
                        "job_id": job_id,
                        "command": command_str,
                        "dry_run": False,
                        "message": f"Job {job_id} submitted successfully"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"sbatch failed: {result.stderr}",
                    "structuredContent": {
                        "success": False,
                        "command": command_str,
                        "error": result.stderr
                    }
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Job submission timed out"}
        except FileNotFoundError:
            # SLURM not available
            if kwargs.get("dry_run", True):
                return {
                    "success": True,
                    "content": f"**Dry Run (SLURM not available)**\n\nWorkflow: {kwargs.get('workflow_path')}\nPartition: {kwargs.get('partition', 't4flex')}",
                    "structuredContent": {"success": True, "dry_run": True, "message": "SLURM not available - showing preview only"}
                }
            return {"success": False, "error": "SLURM (sbatch) not available on this system"}
        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_list_jobs(self, **kwargs) -> Dict[str, Any]:
        """Handle listing SLURM jobs."""
        try:
            state = kwargs.get("state", "all")
            limit = kwargs.get("limit", 20)
            
            # Map state filter
            state_filter = ""
            if state != "all":
                state_map = {
                    "running": "RUNNING",
                    "pending": "PENDING",
                    "completed": "COMPLETED",
                    "failed": "FAILED"
                }
                state_filter = f"-t {state_map.get(state, state)}"
            
            cmd = f"squeue -u $USER {state_filter} -o '%i|%j|%T|%P|%M' --noheader"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            jobs = []
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n")[:limit]:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        jobs.append({
                            "job_id": parts[0].strip(),
                            "name": parts[1].strip(),
                            "state": parts[2].strip(),
                            "partition": parts[3].strip(),
                            "runtime": parts[4].strip()
                        })
            
            if jobs:
                text = "# Current SLURM Jobs\n\n"
                text += "| Job ID | Name | State | Partition | Runtime |\n"
                text += "|--------|------|-------|-----------|----------|\n"
                for job in jobs:
                    text += f"| {job['job_id']} | {job['name']} | {job['state']} | {job['partition']} | {job['runtime']} |\n"
            else:
                text = "No jobs found matching criteria."
            
            return {
                "success": True,
                "content": text,
                "structuredContent": {
                    "success": True,
                    "count": len(jobs),
                    "jobs": jobs
                }
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "SLURM command timed out"}
        except FileNotFoundError:
            return {
                "success": True,
                "content": "# Jobs (Mock Mode)\n\nSLURM not available. Showing example output.\n\n| Job ID | Name | State |\n|--------|------|-------|\n| 12345 | rnaseq_job | RUNNING |",
                "structuredContent": {"success": True, "count": 0, "jobs": [], "note": "SLURM not available"}
            }
        except Exception as e:
            logger.error(f"List jobs failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_cancel_job(self, **kwargs) -> Dict[str, Any]:
        """Handle job cancellation."""
        try:
            job_id = kwargs.get("job_id", "")
            
            if not job_id:
                return {"success": False, "error": "job_id is required"}
            
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "content": f"**Job {job_id} cancelled successfully.**",
                    "structuredContent": {
                        "success": True,
                        "job_id": job_id,
                        "message": "Job cancelled successfully"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to cancel job: {result.stderr}",
                    "structuredContent": {
                        "success": False,
                        "job_id": job_id,
                        "error": result.stderr
                    }
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Cancel command timed out"}
        except FileNotFoundError:
            return {"success": False, "error": "SLURM (scancel) not available on this system"}
        except Exception as e:
            logger.error(f"Job cancellation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_job_logs(self, **kwargs) -> Dict[str, Any]:
        """Handle retrieving job logs."""
        try:
            job_id = kwargs.get("job_id", "")
            log_type = kwargs.get("log_type", "all")
            tail_lines = kwargs.get("tail_lines", 100)
            
            if not job_id:
                return {"success": False, "error": "job_id is required"}
            
            logs = {}
            content_parts = []
            
            # Expanded log file patterns - include download jobs, workflow jobs, etc.
            log_patterns = [
                # Standard SLURM patterns
                (f"slurm-{job_id}.out", "stdout"),
                (f"slurm-{job_id}.err", "stderr"),
                # Download job patterns (dl_XXXXX jobs write to logs/)
                (f"download_{job_id}.log", "download"),
                (f"dl_*_{job_id}.log", "download"),  # glob pattern
                # Workflow/Nextflow patterns
                (".nextflow.log", "nextflow"),
                (f"nextflow_{job_id}.log", "nextflow"),
                # vLLM and service logs
                (f"vllm_*_{job_id}.log", "vllm"),  # glob pattern
            ]
            
            # Expanded search directories
            search_dirs = [
                Path.cwd(),
                Path.cwd() / "logs",
                Path.home(),
                Path.home() / "logs",
                Path("/scratch") / os.environ.get("USER", ""),
                Path("/scratch") / os.environ.get("USER", "") / "logs",
            ]
            
            for pattern, log_name in log_patterns:
                if log_type != "all" and log_type != log_name:
                    continue
                    
                # Search in all locations
                found = False
                for search_dir in search_dirs:
                    if not search_dir.exists():
                        continue
                        
                    # Handle glob patterns (contain *)
                    if "*" in pattern:
                        import glob
                        matches = list(search_dir.glob(pattern))
                        if matches:
                            # Use most recent file
                            log_file = max(matches, key=lambda p: p.stat().st_mtime)
                        else:
                            continue
                    else:
                        log_file = search_dir / pattern
                        if not log_file.exists():
                            continue
                    
                    try:
                        with open(log_file, "r") as f:
                            lines = f.readlines()
                            tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
                            log_content = "".join(tail)
                            logs[log_name] = log_content
                            content_parts.append(
                                f"## {log_name.upper()} ({log_file.name})\n\n```\n{log_content}\n```"
                            )
                        found = True
                        break
                    except Exception as e:
                        logs[log_name] = f"Error reading {log_file}: {e}"
                        break
            
            if logs:
                return {
                    "success": True,
                    "content": f"# Logs for Job {job_id}\n\n" + "\n\n".join(content_parts),
                    "structuredContent": {
                        "success": True,
                        "job_id": job_id,
                        "logs": logs
                    }
                }
            else:
                return {
                    "success": True,
                    "content": f"No log files found for job {job_id}. The job may still be queued or logs may be in a different location.",
                    "structuredContent": {
                        "success": True,
                        "job_id": job_id,
                        "logs": {},
                        "message": "No log files found"
                    }
                }
                
        except Exception as e:
            logger.error(f"Get job logs failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_strategy(self, **kwargs) -> Dict[str, Any]:
        """Handle getting current LLM strategy."""
        try:
            from workflow_composer.llm.orchestrator import ModelOrchestrator
            from workflow_composer.llm.strategies import Strategy
            
            orchestrator = ModelOrchestrator()
            config = orchestrator.current_strategy
            
            return {
                "success": True,
                "content": f"""# Current LLM Strategy

**Profile**: {config.profile or 'default'}
**Strategy**: {config.strategy.value}
**Cloud Fallback**: {'Enabled' if config.enable_cloud_fallback else 'Disabled'}

## Local Models
{chr(10).join(f'- {m}' for m in config.local_model_priorities) if config.local_model_priorities else '- None configured'}

## Cloud Providers
{chr(10).join(f'- {p}' for p in config.cloud_provider_cascade) if config.cloud_provider_cascade else '- None configured'}
""",
                "structuredContent": {
                    "success": True,
                    "profile": config.profile or "default",
                    "strategy": config.strategy.value,
                    "cloud_fallback": config.enable_cloud_fallback,
                    "local_models": config.local_model_priorities,
                    "cloud_providers": config.cloud_provider_cascade
                }
            }
        except ImportError:
            return {
                "success": True,
                "content": "# LLM Strategy\n\nLLM orchestrator not configured. Using default settings.",
                "structuredContent": {
                    "success": True,
                    "profile": "default",
                    "strategy": "cloud_only",
                    "message": "LLM orchestrator not loaded"
                }
            }
        except Exception as e:
            logger.error(f"Get strategy failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_set_strategy(self, **kwargs) -> Dict[str, Any]:
        """Handle setting LLM strategy."""
        try:
            profile = kwargs.get("profile", "")
            
            if not profile:
                return {"success": False, "error": "profile is required"}
            
            from workflow_composer.llm.orchestrator import ModelOrchestrator
            
            orchestrator = ModelOrchestrator()
            previous = orchestrator.current_strategy.profile or "default"
            
            orchestrator.switch_strategy(profile)
            
            return {
                "success": True,
                "content": f"**Strategy Changed**\n\nPrevious: {previous}\nNew: {profile}",
                "structuredContent": {
                    "success": True,
                    "previous_profile": previous,
                    "new_profile": profile,
                    "message": f"Switched from {previous} to {profile}"
                }
            }
        except ValueError as e:
            return {"success": False, "error": f"Invalid profile: {e}"}
        except ImportError:
            return {"success": False, "error": "LLM orchestrator not available"}
        except Exception as e:
            logger.error(f"Set strategy failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Resource Handlers
    async def _handle_get_skills(self) -> str:
        """Get available skills."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))
            from skills import get_skill_registry
            
            registry = get_skill_registry()
            
            text = "# Available BioPipelines Skills\n\n"
            
            categories = ["data_discovery", "workflow_generation", "job_management", "education"]
            for category in categories:
                skills = registry.get_skills_by_category(category)
                if skills:
                    text += f"## {category.replace('_', ' ').title()}\n\n"
                    for skill in skills:
                        text += f"- **{skill.name}**: {skill.description[:100]}...\n"
                    text += "\n"
            
            return text
        except Exception as e:
            return f"Failed to load skills: {e}"
    
    async def _handle_get_templates(self) -> str:
        """Get available workflow templates."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))
            from workflow_templates import get_template_engine
            
            engine = get_template_engine()
            templates = engine.list_templates()
            
            text = "# Available Workflow Templates\n\n"
            
            for template in templates:
                text += f"## {template.display_name}\n"
                text += f"{template.description}\n\n"
                text += f"**Category**: {template.category}\n"
                text += f"**Tags**: {', '.join(template.tags)}\n"
                text += f"**Engine**: {template.engine}\n\n"
            
            return text
        except Exception as e:
            return f"Failed to load templates: {e}"
    
    async def _handle_get_databases(self) -> str:
        """Get available database integrations."""
        databases = """# Integrated Biological Databases

## UniProt
- Protein sequences, annotations, and functions
- Swiss-Prot (reviewed) and TrEMBL entries
- Gene Ontology annotations

## STRING
- Protein-protein interactions
- Functional enrichment analysis
- Network visualization

## KEGG
- Metabolic pathways
- Signaling pathways
- Disease pathways

## Reactome
- Biological pathways
- Reaction networks
- Gene set analysis

## PubMed
- Scientific literature search
- Citation information
- Abstract retrieval

## ClinVar
- Variant pathogenicity
- Clinical significance
- Disease associations
"""
        return databases
    
    async def _handle_get_pipelines(self) -> str:
        """Get available pipelines from analysis definitions."""
        try:
            import yaml
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "analysis_definitions.yaml"
            
            if config_path.exists():
                with open(config_path) as f:
                    definitions = yaml.safe_load(f)
                
                text = "# Available BioPipelines Workflows\n\n"
                
                # Group by category
                categories = {
                    "RNA Analysis": ["rna_seq_basic", "rna_seq_differential_expression", "single_cell_rna_seq", "long_read_rna_seq"],
                    "DNA/Variant Calling": ["wgs_variant_calling", "somatic_variant_calling", "structural_variant_detection"],
                    "Epigenetics": ["chip_seq_peak_calling", "atac_seq", "bisulfite_seq_methylation", "hic_chromatin_interaction"],
                    "Metagenomics": ["metagenomics_profiling", "metagenomics_assembly"],
                    "Spatial Transcriptomics": ["spatial_transcriptomics", "spatial_visium", "spatial_xenium"],
                    "Multi-omics": ["multi_omics_integration", "rna_atac_integration", "proteogenomics"]
                }
                
                for cat_name, pipeline_ids in categories.items():
                    matching = [pid for pid in pipeline_ids if pid in definitions]
                    if matching:
                        text += f"## {cat_name}\n\n"
                        for pid in matching:
                            defn = definitions[pid]
                            required = defn.get("required", [])
                            required_str = ", ".join(required[:4]) if required else "None"
                            if len(required) > 4:
                                required_str += "..."
                            text += f"### {pid.replace('_', ' ').title()}\n"
                            text += f"- **Required tools**: {required_str}\n"
                            if defn.get("recommended"):
                                text += f"- **Recommended**: {', '.join(defn['recommended'][:3])}\n"
                            text += "\n"
                
                return text
            else:
                return "# Available Pipelines\n\nPipeline definitions not found."
        except Exception as e:
            return f"# Available Pipelines\n\nFailed to load pipeline definitions: {e}"
    
    async def _handle_get_strategy_resource(self) -> str:
        """Get LLM strategy info as a resource."""
        try:
            result = await self._handle_get_strategy()
            return result.get("content", "Strategy information not available")
        except Exception as e:
            return f"# LLM Strategy\n\nFailed to load strategy: {e}"
    
    async def _handle_get_jobs_resource(self) -> str:
        """Get active jobs as a resource."""
        try:
            result = await self._handle_list_jobs(state="running", limit=10)
            return result.get("content", "No job information available")
        except Exception as e:
            return f"# Active Jobs\n\nFailed to load jobs: {e}"
    
    # Prompt Handlers
    async def _handle_prompt_rnaseq(self, data_path: str, organism: str, comparison: str = None) -> Dict:
        """Generate RNA-seq analysis prompt."""
        comparison_text = f"\n**Comparison**: {comparison}" if comparison else ""
        
        return {
            "description": "RNA-seq differential expression analysis setup",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""I want to analyze RNA-seq data for differential expression.

**Data Location**: {data_path}
**Organism**: {organism}{comparison_text}

Please help me:
1. Validate my input data
2. Select appropriate workflow template
3. Configure parameters for my organism
4. Set up the analysis and submit the job

Start by checking if the data path exists and contains FASTQ files."""
                    }
                }
            ]
        }
    
    async def _handle_prompt_debug(self, job_id: str, error_type: str = None) -> Dict:
        """Generate debug workflow prompt."""
        error_hint = f"\n**Known Error Type**: {error_type}" if error_type else ""
        
        return {
            "description": "Debug failed workflow job",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""I have a failed workflow job that needs debugging.

**Job ID**: {job_id}{error_hint}

Please help me:
1. Check the job status and retrieve logs
2. Analyze the error messages
3. Identify the root cause
4. Suggest fixes or recovery steps

Start by checking the job status and retrieving the logs."""
                    }
                }
            ]
        }
    
    async def _handle_prompt_find_datasets(self, topic: str, organism: str = None, assay_type: str = None) -> Dict:
        """Generate dataset search prompt."""
        filters = []
        if organism:
            filters.append(f"**Organism**: {organism}")
        if assay_type:
            filters.append(f"**Assay Type**: {assay_type}")
        filter_text = "\n".join(filters) if filters else ""
        
        return {
            "description": "Search for public datasets",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""I'm looking for public datasets related to my research.

**Research Topic**: {topic}
{filter_text}

Please help me:
1. Search ENCODE and GEO databases for relevant datasets
2. Filter results by quality and relevance
3. Summarize the best matching datasets
4. Provide download instructions if needed

Start by searching both databases with the topic."""
                    }
                }
            ]
        }
    
    async def _handle_prompt_design_pipeline(self, research_question: str, data_type: str = None) -> Dict:
        """Generate pipeline design prompt."""
        data_text = f"\n**Data Type**: {data_type}" if data_type else ""
        
        return {
            "description": "Design analysis pipeline",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""I need help designing an analysis pipeline for my research.

**Research Question**: {research_question}{data_text}

Please help me:
1. Understand what type of analysis would answer my question
2. Recommend the appropriate pipeline(s)
3. Explain the key steps and tools involved
4. Suggest parameters and considerations for my specific case

Start by listing available pipelines that might be relevant."""
                    }
                }
            ]
        }
    
    async def _handle_prompt_explain_results(self, analysis_type: str, results_path: str = None) -> Dict:
        """Generate results explanation prompt."""
        path_text = f"\n**Results Location**: {results_path}" if results_path else ""
        
        return {
            "description": "Explain analysis results",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""I need help understanding my analysis results.

**Analysis Type**: {analysis_type}{path_text}

Please help me:
1. Explain what the key output files contain
2. Interpret the main findings
3. Identify significant results
4. Suggest next steps or follow-up analyses

Start by explaining what to expect from a {analysis_type} analysis."""
                    }
                }
            ]
        }
    
    # Formatting helpers
    def _format_search_results(self, result) -> str:
        """Format search results."""
        if not hasattr(result, 'success') or not result.success:
            return f"Search failed: {getattr(result, 'message', 'Unknown error')}"
        
        text = f"Found {result.count} results:\n\n"
        for item in result.data[:10]:
            if isinstance(item, dict):
                text += f"- **{item.get('id', 'N/A')}**: {item.get('title', item.get('name', 'N/A'))}\n"
            else:
                text += f"- {item}\n"
        
        return text
    
    def _format_protein_results(self, result) -> str:
        """Format UniProt results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} proteins:\n\n"
        for protein in result.data[:10]:
            accession = protein.get("primaryAccession", "N/A")
            name = "N/A"
            if "proteinDescription" in protein:
                rec_name = protein["proteinDescription"].get("recommendedName", {})
                if "fullName" in rec_name:
                    name = rec_name["fullName"].get("value", "N/A")
            
            gene = "N/A"
            if protein.get("genes"):
                gene_data = protein["genes"][0].get("geneName", {})
                gene = gene_data.get("value", "N/A")
            
            text += f"- **{accession}** ({gene}): {name}\n"
        
        return text
    
    def _format_interaction_results(self, result) -> str:
        """Format STRING interaction results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} interactions:\n\n"
        for interaction in result.data[:20]:
            p1 = interaction.get("preferredName_A", interaction.get("stringId_A", "?"))
            p2 = interaction.get("preferredName_B", interaction.get("stringId_B", "?"))
            score = interaction.get("score", 0)
            text += f"- {p1} ↔ {p2} (score: {score})\n"
        
        return text
    
    def _format_enrichment_results(self, result) -> str:
        """Format enrichment results."""
        if not result.success:
            return f"Enrichment failed: {result.message}"
        
        text = f"Found {result.count} enriched terms:\n\n"
        
        # Group by category
        categories = {}
        for term in result.data[:50]:
            cat = term.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(term)
        
        for cat, terms in categories.items():
            text += f"## {cat}\n"
            for term in terms[:5]:
                desc = term.get("description", term.get("term", "N/A"))
                pvalue = term.get("p_value", term.get("fdr", 1.0))
                try:
                    text += f"- {desc} (p={float(pvalue):.2e})\n"
                except:
                    text += f"- {desc}\n"
            text += "\n"
        
        return text
    
    def _format_pathway_results(self, result) -> str:
        """Format KEGG pathway results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} pathways:\n\n"
        for pathway in result.data[:15]:
            pid = pathway.get("id", "N/A")
            name = pathway.get("name", "N/A")
            text += f"- **{pid}**: {name}\n"
        
        return text
    
    def _format_pubmed_results(self, result) -> str:
        """Format PubMed results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} articles:\n\n"
        for article in result.data[:10]:
            pmid = article.get("pmid", article.get("id", "N/A"))
            title = article.get("title", "N/A")
            authors = article.get("authors", [])
            author_str = authors[0] if authors else "Unknown"
            if len(authors) > 1:
                author_str += " et al."
            year = article.get("year", "")
            
            text += f"- **PMID:{pmid}** ({year}) {author_str}: {title[:100]}...\n"
        
        return text
    
    def _format_variant_results(self, result) -> str:
        """Format ClinVar variant results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} variants:\n\n"
        for variant in result.data[:15]:
            vid = variant.get("id", variant.get("variation_id", "N/A"))
            name = variant.get("name", variant.get("title", "N/A"))
            sig = variant.get("clinical_significance", variant.get("significance", "N/A"))
            text += f"- **{vid}**: {name} [{sig}]\n"
        
        return text
    
    # Protocol methods
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Get list of tools in MCP format with annotations and output schemas."""
        tools_list = []
        for tool in self.tools.values():
            tool_dict = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters
            }
            # Add annotations if present (MCP 2025-06-18 spec)
            if tool.annotations:
                tool_dict["annotations"] = {
                    "readOnly": tool.annotations.read_only,
                    "requiresConfirmation": tool.annotations.requires_confirmation,
                    "destructive": tool.annotations.destructive,
                    "idempotent": tool.annotations.idempotent,
                    "category": tool.annotations.category
                }
            # Add output schema if present
            if tool.output_schema:
                tool_dict["outputSchema"] = tool.output_schema
            tools_list.append(tool_dict)
        return tools_list
    
    def get_resources_list(self) -> List[Dict[str, Any]]:
        """Get list of resources in MCP format."""
        return [
            {
                "uri": res.uri,
                "name": res.name,
                "description": res.description
            }
            for res in self.resources.values()
        ]
    
    def get_prompts_list(self) -> List[Dict[str, Any]]:
        """Get list of prompts in MCP format."""
        prompts_list = []
        for prompt in self.prompts.values():
            prompt_dict = {
                "name": prompt.name,
                "description": prompt.description,
            }
            if prompt.title:
                prompt_dict["title"] = prompt.title
            if prompt.arguments:
                prompt_dict["arguments"] = [
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "required": arg.required
                    }
                    for arg in prompt.arguments
                ]
            prompts_list.append(prompt_dict)
        return prompts_list
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get a prompt by name with arguments."""
        if name not in self.prompts:
            return {"error": f"Unknown prompt: {name}"}
        
        prompt = self.prompts[name]
        try:
            result = await prompt.handler(**(arguments or {}))
            return {
                "description": prompt.description,
                "messages": result
            }
        except Exception as e:
            logger.error(f"Error executing prompt {name}: {e}")
            return {"error": str(e)}
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name."""
        if name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {name}"}
        
        tool = self.tools[name]
        return await tool.handler(**arguments)
    
    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI."""
        if uri not in self.resources:
            return f"Unknown resource: {uri}"
        
        resource = self.resources[uri]
        return await resource.handler()
    
    # Server run methods
    async def run_stdio(self):
        """Run server using stdio transport (for Claude Code integration)."""
        logger.info("Starting BioPipelines MCP Server (stdio)")
        
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self._handle_request(request)
                
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Error handling request: {e}")
    
    async def run_http(self, host: str = "0.0.0.0", port: int = 8080):
        """Run server using HTTP transport (for development/testing)."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp required for HTTP transport. Install with: pip install aiohttp")
            return
        
        async def handle_request(request):
            data = await request.json()
            response = await self._handle_request(data)
            return web.json_response(response)
        
        async def handle_tools(request):
            return web.json_response({"tools": self.get_tools_list()})
        
        async def handle_resources(request):
            return web.json_response({"resources": self.get_resources_list()})
        
        async def handle_prompts(request):
            return web.json_response({"prompts": self.get_prompts_list()})
        
        async def handle_health(request):
            """Health check endpoint."""
            return web.json_response({
                "status": "healthy",
                "server": "biopipelines",
                "version": "2.0.0",
                "tools_count": len(self.tools),
                "resources_count": len(self.resources),
                "prompts_count": len(self.prompts)
            })
        
        app = web.Application()
        app.router.add_post("/", handle_request)
        app.router.add_get("/tools", handle_tools)
        app.router.add_get("/resources", handle_resources)
        app.router.add_get("/prompts", handle_prompts)
        app.router.add_get("/health", handle_health)
        
        logger.info(f"Starting BioPipelines MCP Server (HTTP) on {host}:{port}")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        # Keep running
        while True:
            await asyncio.sleep(3600)
    
    async def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "biopipelines",
                        "version": "2.0.0"
                    },
                    "capabilities": {
                        "tools": {
                            "listChanged": True  # We support tool list notifications
                        },
                        "resources": {
                            "subscribe": False,  # Not yet implemented
                            "listChanged": False
                        },
                        "prompts": {
                            "listChanged": False
                        }
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": self.get_tools_list()}
            }
        
        elif method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = await self.call_tool(name, arguments)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result.get("content", result.get("error", "No result"))
                        }
                    ],
                    "isError": not result.get("success", False)
                }
            }
        
        elif method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"resources": self.get_resources_list()}
            }
        
        elif method == "resources/read":
            uri = params.get("uri", "")
            content = await self.read_resource(uri)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": content
                        }
                    ]
                }
            }
        
        elif method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"prompts": self.get_prompts_list()}
            }
        
        elif method == "prompts/get":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = await self.get_prompt(name, arguments)
            
            if "error" in result:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": result["error"]
                    }
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }


def create_server() -> BioPipelinesMCPServer:
    """Create a new MCP server instance."""
    return BioPipelinesMCPServer()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BioPipelines MCP Server")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "http"], 
        default="stdio",
        help="Transport method"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port (for http transport)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP host (for http transport)"
    )
    
    args = parser.parse_args()
    
    server = create_server()
    
    if args.transport == "stdio":
        await server.run_stdio()
    else:
        await server.run_http(host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
