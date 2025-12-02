"""
BioPipelines Workflow Template Engine
=====================================

Pre-built workflow templates that users can invoke with minimal customization.
Templates generate ready-to-run Nextflow/Snakemake pipelines.

Usage:
    from config.workflow_templates import get_template_engine
    
    engine = get_template_engine()
    workflow = engine.generate("rnaseq_basic", input_dir="/data", organism="human")
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

# Try to use PyYAML
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not available - template loading will be limited")


@dataclass
class TemplateParameter:
    """Definition of a template parameter."""
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class TemplateStep:
    """A step in the workflow template."""
    name: str
    description: str
    tool: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None


@dataclass
class WorkflowTemplate:
    """Complete workflow template definition."""
    name: str
    display_name: str
    version: str
    category: str
    description: str
    tags: List[str]
    inputs: Dict[str, List[TemplateParameter]]
    steps: List[TemplateStep]
    outputs: Dict[str, Any] = field(default_factory=dict)
    engine: str = "nextflow"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTemplate":
        """Create WorkflowTemplate from dictionary (parsed YAML)."""
        # Parse inputs
        inputs_data = data.get("inputs", {})
        inputs = {}
        for key in ["required", "optional"]:
            if key in inputs_data:
                inputs[key] = [
                    TemplateParameter(
                        name=p.get("name", ""),
                        type=p.get("type", "string"),
                        description=p.get("description", ""),
                        required=key == "required",
                        default=p.get("default"),
                        enum=p.get("enum")
                    )
                    for p in inputs_data[key]
                ]
        
        # Parse steps
        steps_data = data.get("steps", [])
        steps = [
            TemplateStep(
                name=s.get("name", ""),
                description=s.get("description", ""),
                tool=s.get("tool", ""),
                inputs=s.get("inputs", {}),
                outputs=s.get("outputs", {}),
                params=s.get("params", {}),
                condition=s.get("condition")
            )
            for s in steps_data
        ]
        
        return cls(
            name=data.get("name", "unknown"),
            display_name=data.get("display_name", data.get("name", "Unknown")),
            version=data.get("version", "1.0.0"),
            category=data.get("category", "general"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            inputs=inputs,
            steps=steps,
            outputs=data.get("outputs", {}),
            engine=data.get("engine", "nextflow")
        )
    
    def validate_inputs(self, provided: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that required inputs are provided.
        
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        for param in self.inputs.get("required", []):
            if param.name not in provided:
                errors.append(f"Missing required parameter: {param.name}")
            elif param.enum and provided[param.name] not in param.enum:
                errors.append(f"Invalid value for {param.name}: must be one of {param.enum}")
        
        return len(errors) == 0, errors
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for optional parameters."""
        defaults = {}
        for param in self.inputs.get("optional", []):
            if param.default is not None:
                defaults[param.name] = param.default
        return defaults


class TemplateEngine:
    """
    Engine for loading and generating workflows from templates.
    """
    
    def __init__(self, templates_dir: Path = None):
        """
        Initialize template engine.
        
        Args:
            templates_dir: Directory containing template YAML files
        """
        self.templates_dir = templates_dir or Path(__file__).parent
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._loaded = False
    
    def _load_templates(self):
        """Load all template YAML files."""
        if not HAS_YAML:
            logger.warning("Cannot load templates - PyYAML not available")
            self._loaded = True
            return
        
        if self._loaded:
            return
        
        for yaml_file in self.templates_dir.rglob("*.yaml"):
            # Skip schema files
            if yaml_file.name in ["schema.yaml", "__init__.py"]:
                continue
            
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                if not data or not isinstance(data, dict):
                    continue
                
                # Check if it's a workflow template (has steps)
                if "steps" not in data:
                    continue
                
                template = WorkflowTemplate.from_dict(data)
                self._templates[template.name] = template
                
                logger.debug(f"Loaded template: {template.name}")
                
            except Exception as e:
                logger.warning(f"Failed to load template {yaml_file}: {e}")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._templates)} workflow templates")
    
    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a template by name."""
        self._load_templates()
        return self._templates.get(name)
    
    def list_templates(self) -> List[WorkflowTemplate]:
        """List all available templates."""
        self._load_templates()
        return list(self._templates.values())
    
    def list_templates_by_category(self, category: str) -> List[WorkflowTemplate]:
        """List templates in a category."""
        self._load_templates()
        return [t for t in self._templates.values() if t.category == category]
    
    def get_categories(self) -> List[str]:
        """Get all template categories."""
        self._load_templates()
        return list(set(t.category for t in self._templates.values()))
    
    def generate(
        self,
        template_name: str,
        output_dir: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a workflow from a template.
        
        Args:
            template_name: Name of the template to use
            output_dir: Directory to write workflow files
            **kwargs: Template parameters
            
        Returns:
            Dictionary with workflow info and file paths
        """
        template = self.get_template(template_name)
        if not template:
            return {
                "success": False,
                "error": f"Template not found: {template_name}",
                "available_templates": [t.name for t in self.list_templates()]
            }
        
        # Merge defaults with provided values
        params = template.get_defaults()
        params.update(kwargs)
        
        # Validate inputs
        is_valid, errors = template.validate_inputs(params)
        if not is_valid:
            return {
                "success": False,
                "error": "Validation failed",
                "errors": errors
            }
        
        # Generate output directory
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./workflows/{template_name}_{timestamp}"
        
        output_path = Path(output_dir)
        
        # Generate workflow based on engine type
        if template.engine == "nextflow":
            result = self._generate_nextflow(template, params, output_path)
        elif template.engine == "snakemake":
            result = self._generate_snakemake(template, params, output_path)
        else:
            result = {
                "success": False,
                "error": f"Unsupported engine: {template.engine}"
            }
        
        if result.get("success"):
            result["template_name"] = template_name
            result["parameters"] = params
        
        return result
    
    def _generate_nextflow(
        self,
        template: WorkflowTemplate,
        params: Dict[str, Any],
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate a Nextflow workflow."""
        try:
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate main.nf
            main_nf = self._render_nextflow_main(template, params)
            main_path = output_path / "main.nf"
            main_path.write_text(main_nf)
            
            # Generate nextflow.config
            config = self._render_nextflow_config(template, params)
            config_path = output_path / "nextflow.config"
            config_path.write_text(config)
            
            # Generate params.yaml
            params_yaml = self._render_params_yaml(params)
            params_path = output_path / "params.yaml"
            params_path.write_text(params_yaml)
            
            return {
                "success": True,
                "workflow_dir": str(output_path),
                "files": {
                    "main": str(main_path),
                    "config": str(config_path),
                    "params": str(params_path)
                },
                "run_command": f"nextflow run {output_path}/main.nf -params-file {output_path}/params.yaml"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _render_nextflow_main(self, template: WorkflowTemplate, params: Dict[str, Any]) -> str:
        """Render the main.nf file."""
        lines = [
            "#!/usr/bin/env nextflow",
            "",
            f"// {template.display_name}",
            f"// Generated by BioPipelines Template Engine",
            f"// Version: {template.version}",
            "",
            "nextflow.enable.dsl=2",
            "",
            "// Parameters",
        ]
        
        # Add parameter definitions
        for key in ["required", "optional"]:
            for param in template.inputs.get(key, []):
                value = params.get(param.name, param.default)
                if isinstance(value, str):
                    lines.append(f'params.{param.name} = "{value}"')
                elif value is None:
                    lines.append(f"params.{param.name} = null")
                else:
                    lines.append(f"params.{param.name} = {value}")
        
        lines.extend(["", "// Log parameters", "log.info \"\"\""])
        lines.append(f"    {template.display_name}")
        lines.append("    ========================")
        for param in template.inputs.get("required", []):
            lines.append(f"    {param.name}: ${{params.{param.name}}}")
        lines.append('"""')
        
        # Add process definitions for each step
        lines.extend(["", "// Processes", ""])
        
        for step in template.steps:
            lines.extend(self._render_nextflow_process(step, params))
        
        # Add workflow definition
        lines.extend([
            "",
            "// Workflow",
            "workflow {",
        ])
        
        # Simple linear workflow
        prev_output = None
        for i, step in enumerate(template.steps):
            process_name = step.name.upper().replace(" ", "_").replace("-", "_")
            if prev_output:
                lines.append(f"    {process_name}({prev_output})")
            else:
                # First process uses input channel
                lines.append(f"    {process_name}(Channel.fromPath(params.input_dir + '/*'))")
            prev_output = f"{process_name}.out"
        
        lines.append("}")
        lines.append("")
        
        return "\n".join(lines)
    
    def _render_nextflow_process(self, step: TemplateStep, params: Dict[str, Any]) -> List[str]:
        """Render a Nextflow process."""
        process_name = step.name.upper().replace(" ", "_").replace("-", "_")
        
        lines = [
            f"process {process_name} {{",
            f'    tag "$sample_id"',
            "",
            "    input:",
            "    path(reads)",
            "",
            "    output:",
            "    path('*'), emit: results",
            "",
            "    script:",
            '    """',
            f"    # {step.description}",
            f"    echo 'Running {step.tool}'",
            f"    # {step.tool} command here",
            '    """',
            "}",
            "",
        ]
        
        return lines
    
    def _render_nextflow_config(self, template: WorkflowTemplate, params: Dict[str, Any]) -> str:
        """Render nextflow.config."""
        return f"""// Nextflow configuration for {template.display_name}
// Generated by BioPipelines Template Engine

// Profiles
profiles {{
    standard {{
        process.executor = 'local'
    }}
    
    slurm {{
        process.executor = 'slurm'
        process.queue = 'normal'
        process.time = '24h'
        process.memory = '32 GB'
        process.cpus = 8
    }}
    
    docker {{
        docker.enabled = true
    }}
    
    singularity {{
        singularity.enabled = true
        singularity.autoMounts = true
    }}
}}

// Default resources
process {{
    cpus = 4
    memory = '16 GB'
    time = '12h'
}}

// Manifest
manifest {{
    name = '{template.name}'
    description = '{template.description}'
    version = '{template.version}'
    mainScript = 'main.nf'
}}
"""
    
    def _render_params_yaml(self, params: Dict[str, Any]) -> str:
        """Render params.yaml."""
        if HAS_YAML:
            return yaml.dump(params, default_flow_style=False)
        else:
            # Simple YAML-like output
            lines = []
            for key, value in params.items():
                if isinstance(value, str):
                    lines.append(f'{key}: "{value}"')
                elif isinstance(value, list):
                    lines.append(f"{key}:")
                    for item in value:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)
    
    def _generate_snakemake(
        self,
        template: WorkflowTemplate,
        params: Dict[str, Any],
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate a Snakemake workflow."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate Snakefile
            snakefile = self._render_snakefile(template, params)
            snake_path = output_path / "Snakefile"
            snake_path.write_text(snakefile)
            
            # Generate config.yaml
            config_yaml = self._render_params_yaml(params)
            config_path = output_path / "config.yaml"
            config_path.write_text(config_yaml)
            
            return {
                "success": True,
                "workflow_dir": str(output_path),
                "files": {
                    "snakefile": str(snake_path),
                    "config": str(config_path)
                },
                "run_command": f"snakemake -s {snake_path} --configfile {config_path} --cores 8"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _render_snakefile(self, template: WorkflowTemplate, params: Dict[str, Any]) -> str:
        """Render a Snakefile."""
        lines = [
            f"# {template.display_name}",
            f"# Generated by BioPipelines Template Engine",
            f"# Version: {template.version}",
            "",
            "configfile: 'config.yaml'",
            "",
        ]
        
        # Add rules for each step
        for i, step in enumerate(template.steps):
            rule_name = step.name.lower().replace(" ", "_").replace("-", "_")
            lines.extend([
                f"rule {rule_name}:",
                f'    """',
                f'    {step.description}',
                f'    """',
                "    input:",
                '        "input/{sample}.fastq.gz"',
                "    output:",
                f'        "output/{rule_name}/{{sample}}.out"',
                "    shell:",
                f'        "echo Running {step.tool}"',
                "",
            ])
        
        return "\n".join(lines)


# Singleton instance
_template_engine: Optional[TemplateEngine] = None


@lru_cache(maxsize=1)
def get_template_engine(templates_dir: str = None) -> TemplateEngine:
    """
    Get the singleton template engine.
    
    Args:
        templates_dir: Optional path to templates directory
        
    Returns:
        TemplateEngine instance
    """
    global _template_engine
    if _template_engine is None:
        path = Path(templates_dir) if templates_dir else None
        _template_engine = TemplateEngine(templates_dir=path)
    return _template_engine


def reset_engine():
    """Reset the singleton engine (for testing)."""
    global _template_engine
    _template_engine = None
    get_template_engine.cache_clear()


__all__ = [
    "WorkflowTemplate",
    "TemplateParameter",
    "TemplateStep",
    "TemplateEngine",
    "get_template_engine",
    "reset_engine",
]
