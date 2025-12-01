"""
Main Composer Class
===================

The main entry point for the AI Workflow Composer.

Orchestrates all components to:
1. Parse natural language intent
2. Select appropriate tools
3. Map to modules
4. Generate complete workflows

Example:
    from workflow_composer import Composer
    from workflow_composer.llm import get_llm
    
    composer = Composer(llm=get_llm("ollama"))
    workflow = composer.generate(
        "RNA-seq differential expression, mouse, paired-end"
    )
    workflow.save("my_workflow/")
    
Data-First Workflow Example:
    from workflow_composer import Composer
    from workflow_composer.data import DataManifest, LocalSampleScanner
    
    # Scan for data first
    scanner = LocalSampleScanner()
    samples = scanner.scan_directory("/path/to/fastq")
    manifest = DataManifest(samples=samples)
    
    # Generate workflow with real paths
    composer = Composer()
    workflow = composer.generate(
        "RNA-seq differential expression",
        data_manifest=manifest
    )

Multi-Agent Generation Example:
    from workflow_composer import Composer
    
    composer = Composer()
    
    # Use multi-agent system for advanced generation
    result = composer.generate_with_agents(
        "RNA-seq differential expression for human",
        output_dir="workflows/rnaseq"
    )
    print(result.code)  # Nextflow DSL2
    print(result.documentation)  # README.md
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING, AsyncIterator

from .config import Config
from .llm import LLMAdapter, get_llm
from .core import (
    IntentParser, ParsedIntent,
    ToolSelector, Tool,
    ModuleMapper, Module,
    WorkflowGenerator, Workflow
)

# Import pre-flight validator (optional - may not be available)
try:
    from .core import PreflightValidator, ValidationResult, PREFLIGHT_AVAILABLE
except ImportError:
    PREFLIGHT_AVAILABLE = False
    PreflightValidator = None
    ValidationResult = None

# Import DataManifest (optional - for data-first workflow)
try:
    from .data.manifest import DataManifest
    DATA_MANIFEST_AVAILABLE = True
except ImportError:
    DATA_MANIFEST_AVAILABLE = False
    DataManifest = None

# Import multi-agent specialists (Phase 2.4)
try:
    from .agents.specialists import (
        SupervisorAgent,
        PlannerAgent,
        WorkflowResult as AgentWorkflowResult,
        WorkflowPlan,
    )
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False
    SupervisorAgent = None
    AgentWorkflowResult = None

logger = logging.getLogger(__name__)


class Composer:
    """
    AI Workflow Composer - main orchestrator class.
    
    Takes natural language descriptions and generates complete
    Nextflow bioinformatics pipelines.
    """
    
    def __init__(
        self,
        llm: Optional[LLMAdapter] = None,
        config: Optional[Config] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the Workflow Composer.
        
        Args:
            llm: LLM adapter to use (optional, will use config default)
            config: Configuration object (optional)
            config_path: Path to config file (optional)
        """
        # Load configuration
        self.config = config or Config.load(config_path)
        
        # Set up LLM
        if llm:
            self.llm = llm
        else:
            provider = self.config.llm.default_provider
            model = self.config.get_llm_config(provider).model
            self.llm = get_llm(provider, model)
        
        # Initialize components
        self._init_components()
        
        logger.info(f"Composer initialized with LLM: {self.llm}")
    
    def _init_components(self) -> None:
        """Initialize all composer components."""
        base_path = self.config.base_path
        
        # Intent parser
        self.intent_parser = IntentParser(self.llm)
        
        # Tool selector
        catalog_path = self.config.resolve_path(
            self.config.knowledge_base.tool_catalog
        )
        self.tool_selector = ToolSelector(str(catalog_path))
        
        # Module mapper
        module_path = self.config.resolve_path(
            self.config.knowledge_base.module_library
        )
        self.module_mapper = ModuleMapper(str(module_path))
        
        # Workflow generator
        patterns_path = self.config.resolve_path(
            self.config.knowledge_base.workflow_patterns
        )
        self.workflow_generator = WorkflowGenerator(
            str(patterns_path) if patterns_path.exists() else None
        )
        
        # Pre-flight validator (optional)
        if PREFLIGHT_AVAILABLE and PreflightValidator:
            self.preflight_validator = PreflightValidator(
                containers_dir=str(self.config.base_path / "containers"),
                module_library_path=str(module_path)
            )
            logger.info("Pre-flight validator initialized")
        else:
            self.preflight_validator = None
            logger.warning("Pre-flight validator not available")
    
    def generate(
        self,
        description: str,
        output_dir: Optional[str] = None,
        auto_create_modules: bool = True,
        interactive: bool = False,
        data_manifest: Optional["DataManifest"] = None
    ) -> Workflow:
        """
        Generate a workflow from a natural language description.
        
        Args:
            description: Natural language description of the analysis
            output_dir: Directory to save workflow (optional)
            auto_create_modules: Auto-create missing modules using LLM
            interactive: Enable interactive clarification
            data_manifest: Optional DataManifest with sample/reference paths.
                          When provided, workflow parameters will be populated
                          with actual paths from the manifest.
            
        Returns:
            Generated Workflow object
        """
        logger.info(f"Generating workflow from: {description[:100]}...")
        
        # Log data manifest if provided
        if data_manifest:
            logger.info(f"  Data manifest provided: {len(data_manifest.samples)} samples")
            if data_manifest.reference:
                logger.info(f"  Reference: {data_manifest.reference.organism} ({data_manifest.reference.build})")
        
        # Step 1: Parse intent
        logger.info("Step 1: Parsing intent...")
        intent = self.intent_parser.parse(description)
        logger.info(f"  Analysis type: {intent.analysis_type.value}")
        logger.info(f"  Organism: {intent.organism}")
        logger.info(f"  Confidence: {intent.confidence:.2f}")
        
        # Enhance intent with manifest data if available
        if data_manifest:
            intent = self._enhance_intent_from_manifest(intent, data_manifest)
        
        # Interactive clarification if needed
        if interactive and intent.confidence < 0.7:
            logger.info("  Low confidence - would prompt for clarification")
            # TODO: Implement interactive mode
        
        # Step 2: Select tools
        logger.info("Step 2: Selecting tools...")
        tool_map = self.tool_selector.find_tools_for_analysis(
            intent.analysis_type.value
        )
        
        # Flatten tool list
        all_tools = []
        for category, tools in tool_map.items():
            logger.info(f"  {category}: {[t.name for t in tools]}")
            all_tools.extend(tools)
        
        # Step 3: Map to modules
        logger.info("Step 3: Mapping to modules...")
        tool_names = [t.name for t in all_tools]
        module_map = self.module_mapper.find_modules_for_tools(tool_names)
        
        modules = []
        missing = []
        for tool_name, module in module_map.items():
            if module:
                modules.append(module)
                logger.info(f"  ✓ {tool_name} -> {module.name}")
            else:
                missing.append(tool_name)
                logger.warning(f"  ✗ {tool_name} -> module not found")
        
        # Auto-create missing modules
        if missing and auto_create_modules:
            logger.info(f"Creating {len(missing)} missing modules...")
            for tool_name in missing:
                tool = self.tool_selector.find_tool(tool_name)
                if tool:
                    container = tool.container
                else:
                    container = "base"
                
                try:
                    module = self.module_mapper.create_module(
                        tool_name, container, self.llm
                    )
                    modules.append(module)
                    logger.info(f"  Created: {module.name}")
                except Exception as e:
                    logger.error(f"  Failed to create {tool_name}: {e}")
        
        # Step 4: Generate workflow
        logger.info("Step 4: Generating workflow...")
        workflow = self.workflow_generator.generate(
            intent, modules, self.llm, data_manifest=data_manifest
        )
        
        # If manifest provided, also generate samplesheet from it
        if data_manifest and data_manifest.samples:
            logger.info("Step 5: Generating samplesheet from manifest...")
            samplesheet_content = data_manifest.to_samplesheet()
            workflow.samplesheet_template = samplesheet_content
            logger.info(f"  Generated samplesheet with {len(data_manifest.samples)} samples")
        
        # Save if output_dir specified
        if output_dir:
            workflow.save(output_dir)
        
        logger.info(f"Workflow generation complete: {workflow.name}")
        return workflow
    
    def _enhance_intent_from_manifest(
        self,
        intent: ParsedIntent,
        manifest: "DataManifest"
    ) -> ParsedIntent:
        """
        Enhance parsed intent with information from the data manifest.
        
        This enriches the intent with concrete data about:
        - Organism (from samples or reference)
        - Paired-end vs single-end (from samples)
        - Genome build (from reference)
        - Sample count (for parameter suggestions)
        
        Args:
            intent: Original parsed intent
            manifest: Data manifest with sample/reference info
            
        Returns:
            Enhanced ParsedIntent
        """
        # Infer organism from manifest
        if manifest.reference and manifest.reference.organism:
            # Reference organism takes precedence
            inferred_organism = manifest.reference.organism
        elif manifest.samples:
            # Try to get organism from samples metadata
            organisms = set()
            for sample in manifest.samples:
                if sample.metadata.get("organism"):
                    organisms.add(sample.metadata["organism"])
            if len(organisms) == 1:
                inferred_organism = organisms.pop()
            else:
                inferred_organism = None
        else:
            inferred_organism = None
        
        # Update intent if we found organism and it wasn't specified
        if inferred_organism and not intent.organism:
            logger.info(f"  Inferred organism from manifest: {inferred_organism}")
            intent.organism = inferred_organism
        
        # Infer paired-end from samples
        if manifest.samples:
            paired_samples = sum(1 for s in manifest.samples if s.is_paired)
            total_samples = len(manifest.samples)
            intent.paired_end = paired_samples > total_samples / 2
            logger.info(f"  Paired-end: {intent.paired_end} ({paired_samples}/{total_samples} samples)")
        
        # Infer genome build from reference
        if manifest.reference and manifest.reference.assembly:
            if not intent.genome_build:
                intent.genome_build = manifest.reference.assembly
                logger.info(f"  Genome build from manifest: {intent.genome_build}")
        
        # Store manifest metadata in intent for template use
        intent.metadata = intent.metadata or {}
        intent.metadata["sample_count"] = len(manifest.samples)
        intent.metadata["has_manifest"] = True
        
        if manifest.reference:
            intent.metadata["genome_path"] = str(manifest.reference.genome_fasta or "")
            intent.metadata["annotation_path"] = str(manifest.reference.annotation_gtf or "")
        
        return intent
    
    def parse_intent(self, description: str) -> ParsedIntent:
        """
        Parse a description to extract intent (without generating workflow).
        
        Args:
            description: Natural language description
            
        Returns:
            ParsedIntent object
        """
        return self.intent_parser.parse(description)
    
    def find_tools(self, analysis_type: str) -> Dict[str, List[Tool]]:
        """
        Find tools for an analysis type.
        
        Args:
            analysis_type: Analysis type string
            
        Returns:
            Dict mapping categories to tool lists
        """
        return self.tool_selector.find_tools_for_analysis(analysis_type)
    
    def find_modules(self, tool_names: List[str]) -> Dict[str, Optional[Module]]:
        """
        Find modules for a list of tools.
        
        Args:
            tool_names: List of tool names
            
        Returns:
            Dict mapping tool names to modules (or None)
        """
        return self.module_mapper.find_modules_for_tools(tool_names)
    
    def check_readiness(self, description: str) -> Dict[str, Any]:
        """
        Check if all components are ready to generate a workflow.
        
        This now uses the PreflightValidator for comprehensive validation
        including tool availability, container images, and resource estimation.
        
        Args:
            description: Natural language description
            
        Returns:
            Dict with readiness status, issues, and resource estimates
        """
        result = {
            "ready": True,
            "issues": [],
            "warnings": [],
            "auto_fixable": False,
            "resources": {}
        }
        
        # Parse intent first
        try:
            intent = self.intent_parser.parse(description)
            result["intent"] = intent.to_dict()
        except Exception as e:
            result["ready"] = False
            result["issues"].append(f"Intent parsing failed: {e}")
            return result
        
        if intent.confidence < 0.5:
            result["warnings"].append(
                f"Low confidence ({intent.confidence:.2f}) in intent parsing"
            )
        
        # Use pre-flight validator if available
        if self.preflight_validator:
            validation_result = self.preflight_validator.validate_query(description)
            
            result["ready"] = validation_result.can_proceed
            result["auto_fixable"] = validation_result.auto_fixable
            result["resources"] = validation_result.resources
            
            # Categorize validation items
            for item in validation_result.items:
                if item.status == "missing":
                    if item.severity == "critical":
                        result["issues"].append(
                            f"[{item.category}] {item.name}: {item.message}"
                        )
                    else:
                        result["warnings"].append(
                            f"[{item.category}] {item.name}: {item.message}"
                        )
                elif item.status == "warning":
                    result["warnings"].append(
                        f"[{item.category}] {item.name}: {item.message}"
                    )
            
            result["validation"] = {
                "total_items": validation_result.total_items,
                "valid": validation_result.valid,
                "missing": validation_result.missing,
                "warnings": validation_result.warnings_count,
                "tools_found": [i.name for i in validation_result.items 
                               if i.category == "tool" and i.status == "valid"],
                "tools_missing": [i.name for i in validation_result.items 
                                 if i.category == "tool" and i.status == "missing"],
                "containers_available": [i.name for i in validation_result.items 
                                        if i.category == "container" and i.status == "valid"],
                "containers_missing": [i.name for i in validation_result.items 
                                      if i.category == "container" and i.status == "missing"],
                "modules_available": [i.name for i in validation_result.items 
                                     if i.category == "module" and i.status == "valid"],
                "modules_missing": [i.name for i in validation_result.items 
                                   if i.category == "module" and i.status == "missing"]
            }
            
            return result
        
        # Fallback to basic validation if pre-flight not available
        # Find tools
        tool_map = self.tool_selector.find_tools_for_analysis(
            intent.analysis_type.value
        )
        
        all_tools = []
        for tools in tool_map.values():
            all_tools.extend(tools)
        
        if not all_tools:
            result["ready"] = False
            result["issues"].append(
                f"No tools found for analysis type: {intent.analysis_type.value}"
            )
        
        result["tools_found"] = len(all_tools)
        
        # Find modules
        tool_names = [t.name for t in all_tools]
        module_map = self.module_mapper.find_modules_for_tools(tool_names)
        
        missing = [name for name, mod in module_map.items() if mod is None]
        
        if missing:
            result["warnings"].append(
                f"Missing modules for: {', '.join(missing)}"
            )
        
        result["modules_found"] = sum(1 for m in module_map.values() if m)
        result["modules_missing"] = missing
        
        return result
    
    def validate_and_prepare(
        self, 
        description: str, 
        auto_fix: bool = False
    ) -> Dict[str, Any]:
        """
        Validate readiness and optionally auto-fix issues.
        
        This is a higher-level method that:
        1. Checks readiness using pre-flight validation
        2. If auto_fix=True, attempts to resolve fixable issues
        3. Returns updated status and preparation report
        
        Args:
            description: Natural language description
            auto_fix: Automatically fix resolvable issues
            
        Returns:
            Dict with validation status, fixes applied, and readiness
        """
        # Initial validation
        readiness = self.check_readiness(description)
        
        if readiness["ready"]:
            return {
                "status": "ready",
                "readiness": readiness,
                "fixes_applied": [],
                "message": "All components ready for workflow generation"
            }
        
        if not auto_fix or not readiness.get("auto_fixable", False):
            return {
                "status": "not_ready",
                "readiness": readiness,
                "fixes_applied": [],
                "message": "Issues found but auto-fix not enabled or not possible"
            }
        
        # Attempt auto-fix
        fixes_applied = []
        
        # Parse intent to get tools needed
        intent = self.intent_parser.parse(description)
        tool_map = self.tool_selector.find_tools_for_analysis(
            intent.analysis_type.value
        )
        
        all_tools = []
        for tools in tool_map.values():
            all_tools.extend(tools)
        
        # Try to create missing modules
        validation = readiness.get("validation", {})
        missing_modules = validation.get("modules_missing", [])
        
        for tool_name in missing_modules:
            tool = self.tool_selector.find_tool(tool_name)
            if tool:
                try:
                    module = self.module_mapper.create_module(
                        tool_name, tool.container, self.llm
                    )
                    fixes_applied.append({
                        "type": "module_created",
                        "name": tool_name,
                        "status": "success",
                        "details": f"Created module: {module.name}"
                    })
                except Exception as e:
                    fixes_applied.append({
                        "type": "module_created",
                        "name": tool_name,
                        "status": "failed",
                        "details": str(e)
                    })
        
        # Re-check readiness after fixes
        updated_readiness = self.check_readiness(description)
        
        return {
            "status": "ready" if updated_readiness["ready"] else "partial",
            "readiness": updated_readiness,
            "fixes_applied": fixes_applied,
            "message": f"Applied {len([f for f in fixes_applied if f['status'] == 'success'])} fixes"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about available resources."""
        return {
            "llm": str(self.llm),
            "tool_catalog": self.tool_selector.get_stats(),
            "modules": {
                "total": len(self.module_mapper.modules),
                "by_category": self.module_mapper.list_by_category()
            }
        }
    
    def switch_llm(self, provider: str, model: Optional[str] = None) -> None:
        """
        Switch to a different LLM provider.
        
        Args:
            provider: Provider name (ollama, openai, anthropic, huggingface)
            model: Optional model name
        """
        self.llm = get_llm(provider, model)
        self.intent_parser = IntentParser(self.llm)
        logger.info(f"Switched to LLM: {self.llm}")
    
    # =========================================================================
    # Multi-Agent Generation (Phase 2.4)
    # =========================================================================
    
    def _get_supervisor(self) -> "SupervisorAgent":
        """Get or create the multi-agent supervisor."""
        if not MULTI_AGENT_AVAILABLE:
            raise ImportError(
                "Multi-agent specialists not available. "
                "Install required dependencies."
            )
        
        if not hasattr(self, '_supervisor') or self._supervisor is None:
            # Try to get provider router for LLM operations
            try:
                from .providers import get_router
                router = get_router()
            except Exception:
                router = None
            
            self._supervisor = SupervisorAgent(router=router)
        
        return self._supervisor
    
    def generate_with_agents(
        self,
        description: str,
        output_dir: Optional[str] = None,
    ) -> "AgentWorkflowResult":
        """
        Generate a workflow using multi-agent coordination (synchronous).
        
        This method uses the multi-agent system with:
        - PlannerAgent: Designs workflow architecture
        - CodeGenAgent: Generates Nextflow DSL2 code
        - ValidatorAgent: Reviews code with fix loops
        - DocAgent: Creates documentation
        - QCAgent: Configures quality checks
        
        Args:
            description: Natural language description of the workflow
            output_dir: Optional directory to save generated files
            
        Returns:
            AgentWorkflowResult with code, config, documentation
            
        Example:
            composer = Composer()
            result = composer.generate_with_agents(
                "RNA-seq differential expression for human",
                output_dir="workflows/rnaseq"
            )
            if result.success:
                print(f"Generated: {result.plan.name}")
                print(f"Code lines: {len(result.code.split(chr(10)))}")
        """
        if not MULTI_AGENT_AVAILABLE:
            raise ImportError(
                "Multi-agent generation not available. "
                "Ensure agents.specialists module is properly installed."
            )
        
        logger.info(f"Multi-agent generation: {description[:100]}...")
        supervisor = self._get_supervisor()
        
        return supervisor.execute_sync(description, output_dir)
    
    async def generate_with_agents_async(
        self,
        description: str,
        output_dir: Optional[str] = None,
    ) -> "AgentWorkflowResult":
        """
        Generate a workflow using multi-agent coordination (async).
        
        Same as generate_with_agents but fully async with LLM calls.
        
        Args:
            description: Natural language description
            output_dir: Optional directory for output files
            
        Returns:
            AgentWorkflowResult with generated artifacts
        """
        if not MULTI_AGENT_AVAILABLE:
            raise ImportError("Multi-agent generation not available.")
        
        logger.info(f"Async multi-agent generation: {description[:100]}...")
        supervisor = self._get_supervisor()
        
        return await supervisor.execute(description, output_dir)
    
    async def generate_with_agents_streaming(
        self,
        description: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate a workflow with streaming progress updates.
        
        Yields progress updates as the multi-agent system works:
        - Planning phase
        - Code generation phase
        - Validation phase (with fix attempts)
        - Documentation phase
        
        Args:
            description: Natural language description
            
        Yields:
            Dict with phase, status, and progress info
            
        Example:
            async for update in composer.generate_with_agents_streaming("RNA-seq"):
                print(f"{update['phase']}: {update['status']}")
                if update['phase'] == 'complete':
                    result = update['result']
        """
        if not MULTI_AGENT_AVAILABLE:
            raise ImportError("Multi-agent generation not available.")
        
        supervisor = self._get_supervisor()
        
        async for update in supervisor.execute_streaming(description):
            yield update
    
    def get_agent_plan(self, description: str) -> "WorkflowPlan":
        """
        Get just the workflow plan without full generation.
        
        Useful for previewing what the multi-agent system would create.
        
        Args:
            description: Natural language description
            
        Returns:
            WorkflowPlan with steps, tools, and resources
        """
        if not MULTI_AGENT_AVAILABLE:
            raise ImportError("Multi-agent planning not available.")
        
        supervisor = self._get_supervisor()
        return supervisor.planner.create_plan_sync(description)
