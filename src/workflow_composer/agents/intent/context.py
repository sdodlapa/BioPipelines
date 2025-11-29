"""
Conversation Context Management
================================

Semantic memory and coreference resolution for coherent multi-turn conversations.

Features:
- Entity tracking across turns
- Coreference resolution ("it", "that", "the data")
- Conversation summarization
- Working memory for active tasks
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque
from enum import Enum

from .parser import Entity, EntityType, IntentType

logger = logging.getLogger(__name__)


# =============================================================================
# MEMORY STRUCTURES
# =============================================================================

@dataclass
class MemoryItem:
    """A single item in conversation memory."""
    content: str
    timestamp: datetime
    role: str  # "user", "assistant"
    entities: List[Entity] = field(default_factory=list)
    intent: Optional[IntentType] = None
    salience: float = 1.0  # How important/recent this item is
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class EntityReference:
    """A reference to an entity with context."""
    entity: Entity
    first_mentioned: datetime
    last_mentioned: datetime
    mention_count: int = 1
    aliases: Set[str] = field(default_factory=set)
    
    def update(self, new_entity: Entity):
        """Update reference with new mention."""
        self.last_mentioned = datetime.now()
        self.mention_count += 1
        if new_entity.value != self.entity.value:
            self.aliases.add(new_entity.value)


class ReferentType(Enum):
    """Types of referential expressions."""
    PRONOUN_IT = "it"           # "it", "that", "this"
    PRONOUN_THEY = "they"       # "they", "them", "those"
    DEFINITE_NP = "definite"    # "the data", "the workflow"
    DEMONSTRATIVE = "demo"      # "this data", "that file"
    ELLIPSIS = "ellipsis"       # Omitted reference


# =============================================================================
# COREFERENCE RESOLVER
# =============================================================================

class CoreferenceResolver:
    """
    Resolve pronouns and definite references to their antecedents.
    
    Handles:
    - "it" → most recent singular entity
    - "them" → most recent plural/collection
    - "the data" → most recent data-related entity
    - "this workflow" → current workflow in context
    """
    
    # Patterns for referential expressions
    REFERENT_PATTERNS = {
        # Pronouns
        ReferentType.PRONOUN_IT: [
            r'\bit\b', r'\bthis\b(?!\s+\w)', r'\bthat\b(?!\s+\w)',
        ],
        ReferentType.PRONOUN_THEY: [
            r'\bthey\b', r'\bthem\b', r'\bthose\b', r'\bthese\b(?!\s+\w)',
        ],
        # Definite NPs (the + noun)
        ReferentType.DEFINITE_NP: [
            r'\bthe\s+(data|dataset|file|sample|workflow|pipeline|job|results?|output)\b',
        ],
        # Demonstrative NPs (this/that + noun)
        ReferentType.DEMONSTRATIVE: [
            r'\b(this|that)\s+(data|dataset|file|sample|workflow|pipeline|job)\b',
        ],
    }
    
    # Entity type affinity for resolution
    NOUN_ENTITY_MAP = {
        "data": [EntityType.DATASET_ID, EntityType.FILE_PATH, EntityType.DIRECTORY_PATH],
        "dataset": [EntityType.DATASET_ID, EntityType.PROJECT_ID],
        "file": [EntityType.FILE_PATH],
        "sample": [EntityType.SAMPLE_ID],
        "workflow": [EntityType.WORKFLOW_TYPE],
        "pipeline": [EntityType.WORKFLOW_TYPE],
        "job": [EntityType.JOB_ID],
        "results": [EntityType.FILE_PATH, EntityType.DIRECTORY_PATH],
        "output": [EntityType.FILE_PATH, EntityType.DIRECTORY_PATH],
    }
    
    def __init__(self, context: "ConversationContext"):
        self.context = context
        import re
        self._compiled_patterns = {
            rtype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for rtype, patterns in self.REFERENT_PATTERNS.items()
        }
    
    def resolve(self, message: str) -> Dict[str, Entity]:
        """
        Resolve referential expressions in message.
        
        Returns:
            Dict mapping surface forms to resolved entities
        """
        resolutions = {}
        
        for rtype, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(message):
                    surface = match.group(0)
                    
                    if rtype == ReferentType.PRONOUN_IT:
                        resolved = self._resolve_pronoun_it()
                    elif rtype == ReferentType.PRONOUN_THEY:
                        resolved = self._resolve_pronoun_they()
                    elif rtype == ReferentType.DEFINITE_NP:
                        noun = match.group(1).lower()
                        resolved = self._resolve_definite_np(noun)
                    elif rtype == ReferentType.DEMONSTRATIVE:
                        noun = match.group(2).lower()
                        resolved = self._resolve_definite_np(noun)
                    else:
                        resolved = None
                    
                    if resolved:
                        resolutions[surface] = resolved
        
        return resolutions
    
    def _resolve_pronoun_it(self) -> Optional[Entity]:
        """Resolve 'it' to most salient singular entity."""
        # Priority order for 'it' resolution
        priority_types = [
            EntityType.DATASET_ID,
            EntityType.WORKFLOW_TYPE,
            EntityType.JOB_ID,
            EntityType.FILE_PATH,
            EntityType.DIRECTORY_PATH,
        ]
        
        for etype in priority_types:
            entity = self.context.get_recent_entity(etype)
            if entity:
                return entity
        
        # Fall back to any recent entity
        return self.context.get_most_recent_entity()
    
    def _resolve_pronoun_they(self) -> Optional[Entity]:
        """Resolve 'they/them' to most recent collection."""
        # Look for plural entities (samples, files)
        for etype in [EntityType.SAMPLE_ID, EntityType.FILE_PATH]:
            entities = self.context.get_recent_entities(etype, limit=5)
            if len(entities) > 1:
                # Return first as representative
                return entities[0]
        return None
    
    def _resolve_definite_np(self, noun: str) -> Optional[Entity]:
        """Resolve 'the X' to most recent entity of type X."""
        target_types = self.NOUN_ENTITY_MAP.get(noun, [])
        
        for etype in target_types:
            entity = self.context.get_recent_entity(etype)
            if entity:
                return entity
        
        return None


# =============================================================================
# ENTITY TRACKER
# =============================================================================

class EntityTracker:
    """
    Track entities across conversation turns.
    
    Maintains:
    - All mentioned entities with recency
    - Active entities (likely to be referenced)
    - Entity coreference chains
    """
    
    def __init__(self, max_entities: int = 100):
        self.max_entities = max_entities
        self._entities: Dict[EntityType, List[EntityReference]] = {}
        self._all_entities: List[EntityReference] = []
    
    def add(self, entity: Entity):
        """Add or update an entity reference."""
        # Check if entity already exists
        existing = self._find_existing(entity)
        
        if existing:
            existing.update(entity)
        else:
            ref = EntityReference(
                entity=entity,
                first_mentioned=datetime.now(),
                last_mentioned=datetime.now(),
            )
            
            # Add to type-specific list
            if entity.type not in self._entities:
                self._entities[entity.type] = []
            self._entities[entity.type].append(ref)
            
            # Add to global list
            self._all_entities.append(ref)
            
            # Prune if too many
            if len(self._all_entities) > self.max_entities:
                self._prune_oldest()
    
    def _find_existing(self, entity: Entity) -> Optional[EntityReference]:
        """Find existing reference for an entity."""
        if entity.type not in self._entities:
            return None
        
        for ref in self._entities[entity.type]:
            if ref.entity.canonical == entity.canonical:
                return ref
            if entity.value in ref.aliases:
                return ref
        
        return None
    
    def _prune_oldest(self):
        """Remove oldest entities when over limit."""
        # Sort by last mentioned
        self._all_entities.sort(key=lambda r: r.last_mentioned)
        
        # Remove oldest 20%
        prune_count = len(self._all_entities) // 5
        to_remove = self._all_entities[:prune_count]
        self._all_entities = self._all_entities[prune_count:]
        
        # Remove from type-specific lists
        for ref in to_remove:
            if ref.entity.type in self._entities:
                try:
                    self._entities[ref.entity.type].remove(ref)
                except ValueError:
                    pass
    
    def get_recent(self, entity_type: EntityType, limit: int = 5) -> List[Entity]:
        """Get recent entities of a specific type."""
        if entity_type not in self._entities:
            return []
        
        refs = sorted(
            self._entities[entity_type],
            key=lambda r: r.last_mentioned,
            reverse=True
        )
        
        return [r.entity for r in refs[:limit]]
    
    def get_most_recent(self, entity_type: EntityType = None) -> Optional[Entity]:
        """Get the most recently mentioned entity."""
        if entity_type:
            entities = self.get_recent(entity_type, limit=1)
            return entities[0] if entities else None
        
        if not self._all_entities:
            return None
        
        most_recent = max(self._all_entities, key=lambda r: r.last_mentioned)
        return most_recent.entity
    
    def get_salient_entities(self, limit: int = 5) -> List[Entity]:
        """Get most salient (recent + frequently mentioned) entities."""
        if not self._all_entities:
            return []
        
        # Score by recency and frequency
        now = datetime.now()
        scored = []
        for ref in self._all_entities:
            age = (now - ref.last_mentioned).total_seconds()
            recency_score = 1.0 / (1.0 + age / 60.0)  # Decay over minutes
            frequency_score = min(ref.mention_count / 5.0, 1.0)
            score = 0.7 * recency_score + 0.3 * frequency_score
            scored.append((score, ref))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ref.entity for score, ref in scored[:limit]]


# =============================================================================
# CONTEXT MEMORY
# =============================================================================

class ContextMemory:
    """
    Working memory for the conversation.
    
    Tracks:
    - Recent messages (sliding window)
    - Active task state
    - Important facts/constraints
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._messages: deque = deque(maxlen=window_size)
        self._facts: Dict[str, Any] = {}
        self._active_task: Optional[Dict] = None
    
    def add_message(
        self, 
        role: str, 
        content: str, 
        entities: List[Entity] = None,
        intent: IntentType = None
    ):
        """Add a message to memory."""
        item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            role=role,
            entities=entities or [],
            intent=intent,
        )
        self._messages.append(item)
    
    def get_recent_messages(self, limit: int = 10) -> List[MemoryItem]:
        """Get recent messages."""
        messages = list(self._messages)[-limit:]
        return messages
    
    def get_context_for_llm(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get messages formatted for LLM."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.get_recent_messages(limit)
        ]
    
    def set_fact(self, key: str, value: Any):
        """Store a fact in working memory."""
        self._facts[key] = value
    
    def get_fact(self, key: str) -> Optional[Any]:
        """Retrieve a fact from working memory."""
        return self._facts.get(key)
    
    def set_active_task(self, task: Dict):
        """Set the current active task."""
        self._active_task = task
    
    def get_active_task(self) -> Optional[Dict]:
        """Get the current active task."""
        return self._active_task
    
    def clear_active_task(self):
        """Clear the active task."""
        self._active_task = None
    
    def summarize(self) -> str:
        """Generate a summary of current context."""
        parts = []
        
        # Recent topic
        if self._messages:
            recent = self._messages[-1]
            if recent.intent:
                parts.append(f"Recent: {recent.intent.name}")
        
        # Active task
        if self._active_task:
            task_type = self._active_task.get("type", "unknown")
            parts.append(f"Task: {task_type}")
        
        # Key facts
        for key in ["data_path", "workflow", "organism"]:
            if key in self._facts:
                parts.append(f"{key}: {self._facts[key]}")
        
        return " | ".join(parts) if parts else "No context"


# =============================================================================
# CONVERSATION CONTEXT
# =============================================================================

class ConversationContext:
    """
    Complete conversation context manager.
    
    Integrates:
    - Entity tracking
    - Coreference resolution
    - Working memory
    - State management
    """
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.created_at = datetime.now()
        
        # Core components
        self.memory = ContextMemory()
        self.entity_tracker = EntityTracker()
        self.coreference_resolver = CoreferenceResolver(self)
        
        # Application state
        self.state: Dict[str, Any] = {
            "data_path": None,
            "samples": [],
            "current_workflow": None,
            "workflow_path": None,
            "jobs": {},
            "last_search_results": None,
            "pending_confirmation": None,
        }
    
    def add_turn(
        self, 
        role: str, 
        content: str,
        entities: List[Entity] = None,
        intent: IntentType = None
    ):
        """Add a conversation turn."""
        # Add to memory
        self.memory.add_message(role, content, entities, intent)
        
        # Track entities
        if entities:
            for entity in entities:
                self.entity_tracker.add(entity)
    
    def resolve_references(self, message: str) -> Tuple[str, Dict[str, Entity]]:
        """
        Resolve coreferences in a message.
        
        Returns:
            Tuple of (resolved_message, resolution_map)
        """
        resolutions = self.coreference_resolver.resolve(message)
        
        resolved = message
        for surface, entity in resolutions.items():
            # Replace pronoun with explicit reference
            # Be careful not to over-replace
            if surface.lower() in ["it", "this", "that"]:
                # Only replace in specific contexts
                pass
        
        return resolved, resolutions
    
    def get_recent_entity(self, entity_type: EntityType) -> Optional[Entity]:
        """Get most recent entity of a type."""
        return self.entity_tracker.get_most_recent(entity_type)
    
    def get_recent_entities(self, entity_type: EntityType, limit: int = 5) -> List[Entity]:
        """Get recent entities of a type."""
        return self.entity_tracker.get_recent(entity_type, limit)
    
    def get_most_recent_entity(self) -> Optional[Entity]:
        """Get the most recently mentioned entity of any type."""
        return self.entity_tracker.get_most_recent()
    
    def get_salient_entities(self, limit: int = 5) -> List[Entity]:
        """Get most salient entities for context."""
        return self.entity_tracker.get_salient_entities(limit)
    
    def update_state(self, key: str, value: Any):
        """Update application state."""
        self.state[key] = value
        self.memory.set_fact(key, value)
    
    def get_state(self, key: str) -> Optional[Any]:
        """Get application state value."""
        return self.state.get(key)
    
    def get_context_summary(self) -> str:
        """Get a summary of current context for LLM."""
        parts = []
        
        # Data info
        if self.state.get("data_path"):
            samples = self.state.get("samples", [])
            parts.append(f"📁 Data: {self.state['data_path']} ({len(samples)} samples)")
        
        # Workflow info
        if self.state.get("current_workflow"):
            parts.append(f"📋 Workflow: {self.state['current_workflow']}")
        
        # Active jobs
        jobs = self.state.get("jobs", {})
        running = len([j for j in jobs.values() if j.get("status") == "running"])
        if running:
            parts.append(f"🔄 Jobs: {running} running")
        
        # Salient entities
        entities = self.get_salient_entities(3)
        if entities:
            entity_strs = [f"{e.type.name}:{e.canonical}" for e in entities]
            parts.append(f"🏷️ Context: {', '.join(entity_strs)}")
        
        # Memory summary
        memory_summary = self.memory.summarize()
        if memory_summary and memory_summary != "No context":
            parts.append(f"💭 {memory_summary}")
        
        return " | ".join(parts) if parts else "Ready - no data loaded"
    
    def get_messages_for_llm(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent messages formatted for LLM."""
        return self.memory.get_context_for_llm(limit)
    
    def set_pending_confirmation(self, action: str, data: Dict):
        """Set a pending action requiring confirmation."""
        self.state["pending_confirmation"] = {
            "action": action,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_pending_confirmation(self) -> Optional[Dict]:
        """Get pending confirmation if any."""
        return self.state.get("pending_confirmation")
    
    def clear_pending_confirmation(self):
        """Clear pending confirmation."""
        self.state["pending_confirmation"] = None
    
    def clear(self):
        """Clear all context."""
        self.memory = ContextMemory()
        self.entity_tracker = EntityTracker()
        self.state = {
            "data_path": None,
            "samples": [],
            "current_workflow": None,
            "workflow_path": None,
            "jobs": {},
            "last_search_results": None,
            "pending_confirmation": None,
        }
