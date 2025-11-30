"""
Intent Parsing & Context Management
====================================

Enhanced natural language understanding for the BioPipelines agent.

Components:
- IntentParser: Pattern-based intent detection with entity extraction
- HybridQueryParser: Production-grade hybrid parser (pattern + semantic + NER)
- UnifiedEnsembleParser: Multi-method ensemble with weighted voting
- SemanticIntentClassifier: FAISS-based semantic similarity classification
- BioinformaticsNER: Domain-specific named entity recognition
- ConversationContext: Semantic memory and coreference resolution
- DialogueManager: Conversation flow and multi-turn task tracking
- ChatIntegration: Integration layer for BioPipelines facade

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                 UnifiedEnsembleParser                       │
    │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
    │  │ Rule     │ Semantic │ NER      │ LLM      │ RAG      │  │
    │  │ Patterns │ FAISS    │ BioBERT  │ Fallback │ History  │  │
    │  │ (0.25)   │ (0.30)   │ (0.20)   │ (0.15)   │ (0.10)   │  │
    │  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
    │                          ↓                                  │
    │              Confidence-Weighted Fusion                     │
    │              + Agreement Boosting                           │
    │                          ↓                                  │
    │               Final Intent + Confidence                     │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Production usage (recommended)
    from workflow_composer.agents.intent import UnifiedEnsembleParser
    
    parser = UnifiedEnsembleParser()
    result = parser.parse("search for human brain RNA-seq data")
    print(result.intent)      # "DATA_SEARCH"
    print(result.confidence)  # 0.92
    print(result.agreement_level)  # 0.80 (4/5 methods agreed)
    
    # With chat integration
    from workflow_composer.agents.intent import ChatIntegration
    
    intent_system = ChatIntegration()
    result = intent_system.process_message(message, session_id)
"""

from .parser import IntentParser, IntentType, IntentResult, Entity, EntityType
from .context import ConversationContext, ContextMemory, EntityTracker, MemoryItem, EntityReference
from .dialogue import (
    DialogueManager, 
    DialogueResult,
    ConversationState, 
    ConversationPhase,
    TaskState,
    TaskStatus,
    SlotFiller,
)
from .integration import ChatIntegration, route_to_tool, format_clarification_response
from .semantic import (
    HybridQueryParser,
    SemanticIntentClassifier,
    BioinformaticsNER,
    BioEntity,
    QueryParseResult,
    INTENT_EXAMPLES,
    FAISS_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)
from .learning import (
    LearningHybridParser,
    QueryLogger,
    FeedbackManager,
    LLMIntentClassifier,
    FineTuningExporter,
)
from .unified_ensemble import (
    UnifiedEnsembleParser,
    EnsembleParseResult,
    MethodVote,
    ParsingMethod,
    create_ensemble_parser,
)

__all__ = [
    # High-level (recommended)
    "UnifiedEnsembleParser",  # NEW: Multi-method ensemble parser
    "HybridQueryParser",      # Production-grade hybrid parser
    "LearningHybridParser",   # With active learning & feedback
    "ChatIntegration",        # Chat handler integration
    
    # Ensemble components
    "EnsembleParseResult",
    "MethodVote",
    "ParsingMethod",
    "create_ensemble_parser",
    
    # Learning components
    "QueryLogger",
    "FeedbackManager",
    "LLMIntentClassifier",
    "FineTuningExporter",
    
    # Semantic components
    "SemanticIntentClassifier",
    "BioinformaticsNER",
    "BioEntity",
    "QueryParseResult",
    "INTENT_EXAMPLES",
    "FAISS_AVAILABLE",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    
    # Pattern-based parser
    "IntentParser",
    "IntentType", 
    "IntentResult",
    "Entity",
    "EntityType",
    
    # Context management
    "ConversationContext",
    "ContextMemory",
    "EntityTracker",
    "MemoryItem",
    "EntityReference",
    
    # Dialogue management
    "DialogueManager",
    "DialogueResult",
    "ConversationState",
    "ConversationPhase",
    "TaskState",
    "TaskStatus",
    "SlotFiller",
    
    # Integration helpers
    "route_to_tool",
    "format_clarification_response",
]
