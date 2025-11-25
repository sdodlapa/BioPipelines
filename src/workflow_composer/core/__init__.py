"""
Core Components
===============

Core functionality for the Workflow Composer:

Query Parsing:
- QueryParser: Rule-based intent parsing from natural language
- EnsembleQueryParser: Multi-model ensemble (BiomedBERT + SciBERT + BioMistral)
- AdaptiveQueryParser: Auto-selects best available parsing strategy

Model Management:
- ModelServiceManager: GPU/CPU model lifecycle and orchestration

Workflow Generation:
- ToolSelector: Query tool catalog and select appropriate tools
- ModuleMapper: Map tools to Nextflow modules
- WorkflowGenerator: Generate Nextflow DSL2 workflows

File naming convention:
- query_parser.py: Rule-based parsing
- query_parser_ensemble.py: Multi-model ensemble parsing
- model_service_manager.py: Model lifecycle management
- tool_selector.py: Tool catalog and selection
- module_mapper.py: Tool to module mapping
- workflow_generator.py: Nextflow workflow generation
"""

# Core query parser (rule-based + LLM hybrid)
from .query_parser import IntentParser, ParsedIntent, AnalysisType

# Backward compatibility aliases
QueryParser = IntentParser

from .tool_selector import ToolSelector, Tool, ToolMatch
from .module_mapper import ModuleMapper, Module
from .workflow_generator import WorkflowGenerator, Workflow

# Ensemble query parser (optional, requires transformers)
try:
    from .query_parser_ensemble import (
        EnsembleIntentParser,
        EnsembleResult,
        BioEntity,
        BioEntityType,
        create_hybrid_parser,
    )
    # Aliases for new naming convention
    EnsembleQueryParser = EnsembleIntentParser
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    EnsembleIntentParser = None
    EnsembleQueryParser = None
    EnsembleResult = None

# Model service manager (manages GPU/CPU model loading)
try:
    from .model_service_manager import (
        ModelOrchestrator,
        AdaptiveIntentParser,
        get_orchestrator,
        ServiceStatus,
        ModelStatus,
        OrchestratorConfig,
    )
    # Aliases for new naming convention
    ModelServiceManager = ModelOrchestrator
    AdaptiveQueryParser = AdaptiveIntentParser
    get_model_manager = get_orchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    ModelOrchestrator = None
    ModelServiceManager = None
    AdaptiveIntentParser = None
    AdaptiveQueryParser = None
    get_orchestrator = None
    get_model_manager = None

# Pre-flight validator (validates prerequisites before execution)
try:
    from .preflight_validator import (
        PreflightValidator,
        ValidationReport,
        ValidationItem,
        ValidationStatus,
        ResourceEstimate,
        validate_workflow_prerequisites,
    )
    PREFLIGHT_AVAILABLE = True
except ImportError:
    PREFLIGHT_AVAILABLE = False
    PreflightValidator = None
    ValidationReport = None
    validate_workflow_prerequisites = None

__all__ = [
    # Query Parsing
    "IntentParser",        # Original name
    "QueryParser",         # New alias
    "ParsedIntent",
    "AnalysisType",
    
    # Tool Selection & Mapping
    "ToolSelector",
    "Tool",
    "ToolMatch",
    "ModuleMapper",
    "Module",
    
    # Workflow Generation
    "WorkflowGenerator",
    "Workflow",
    
    # Ensemble Parsing (conditional)
    "EnsembleIntentParser",  # Original name
    "EnsembleQueryParser",   # New alias
    "EnsembleResult",
    "BioEntity",
    "BioEntityType",
    "create_hybrid_parser",
    "ENSEMBLE_AVAILABLE",
    
    # Model Service Management (conditional)
    "ModelOrchestrator",     # Original name  
    "ModelServiceManager",   # New alias
    "AdaptiveIntentParser",  # Original name
    "AdaptiveQueryParser",   # New alias
    "get_orchestrator",      # Original name
    "get_model_manager",     # New alias
    "ServiceStatus",
    "ModelStatus",
    "OrchestratorConfig",
    "ORCHESTRATOR_AVAILABLE",
    
    # Pre-flight Validation (conditional)
    "PreflightValidator",
    "ValidationReport",
    "ValidationItem",
    "ValidationStatus",
    "ResourceEstimate",
    "validate_workflow_prerequisites",
    "PREFLIGHT_AVAILABLE",
]
