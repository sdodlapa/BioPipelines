"""
Agent Memory System
====================

Vector-based RAG memory for the agentic system:
- Conversation history with semantic search
- Error history for learning from past mistakes
- User preferences tracking
- Workflow context persistence

Uses BGE-small embeddings (runs on CPU) for efficient similarity search.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Entry Types
# =============================================================================

@dataclass
class MemoryEntry:
    """A single memory entry with embedding."""
    id: str
    content: str
    memory_type: str  # "conversation", "error", "preference", "workflow"
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp,
            "metadata": json.dumps(self.metadata),
        }
    
    @classmethod
    def from_row(cls, row: tuple, embedding: Optional[np.ndarray] = None) -> "MemoryEntry":
        """Create from database row."""
        return cls(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            timestamp=row[3],
            metadata=json.loads(row[4]) if row[4] else {},
            embedding=embedding.tolist() if embedding is not None else None
        )


@dataclass
class SearchResult:
    """Result from memory search."""
    entry: MemoryEntry
    score: float  # Similarity score (0-1)


# =============================================================================
# Embedding Model
# =============================================================================

class EmbeddingModel:
    """
    Lightweight embedding model for semantic search.
    Uses sentence-transformers with BGE-small (runs on CPU).
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # BGE-small dimension
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedding dimension: {self._dimension}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using fallback")
                self._model = None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self._model = None
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    def encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding vector."""
        if self.model is None:
            return self._fallback_encode(text)
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return self._fallback_encode(text)
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts."""
        if self.model is None:
            return [self._fallback_encode(t) for t in texts]
        
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return [self._fallback_encode(t) for t in texts]
    
    def _fallback_encode(self, text: str) -> np.ndarray:
        """Simple hash-based fallback when model unavailable."""
        # Use hash of text to create pseudo-random but deterministic embedding
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Expand to dimension
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        embedding = np.random.randn(self._dimension).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


# =============================================================================
# Agent Memory
# =============================================================================

class AgentMemory:
    """
    Long-term memory for the agentic system.
    
    Features:
    - Semantic search using embeddings
    - SQLite storage for persistence
    - Memory types: conversation, error, preference, workflow
    - Automatic pruning of old memories
    
    Usage:
        memory = AgentMemory()
        memory.add("User asked about RNA-seq", "conversation")
        results = memory.search("RNA sequencing analysis")
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        max_memories: int = 10000,
    ):
        """
        Initialize agent memory.
        
        Args:
            db_path: Path to SQLite database (default: ~/.biopipelines/memory.db)
            embedding_model: Sentence transformer model for embeddings
            max_memories: Maximum number of memories to keep
        """
        self.max_memories = max_memories
        
        # Set up database path
        if db_path is None:
            db_dir = Path.home() / ".biopipelines"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "agent_memory.db")
        self.db_path = db_path
        
        # Initialize embedding model
        self.embedder = EmbeddingModel(embedding_model)
        
        # Initialize database
        self._init_db()
        
        # In-memory embedding cache for fast search
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._load_embeddings()
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB
            )
        """)
        
        # Create index on memory_type
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
        """)
        
        # Create index on timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Memory database initialized: {self.db_path}")
    
    def _load_embeddings(self):
        """Load embeddings into memory for fast search."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        
        for row in rows:
            if row[1]:
                self._embeddings_cache[row[0]] = np.frombuffer(row[1], dtype=np.float32)
        
        conn.close()
        logger.info(f"Loaded {len(self._embeddings_cache)} embeddings into cache")
    
    def add(
        self,
        content: str,
        memory_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a memory entry.
        
        Args:
            content: Text content to remember
            memory_type: Type of memory (conversation, error, preference, workflow)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        import uuid
        
        memory_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # Generate embedding
        embedding = self.embedder.encode(content)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO memories (id, content, memory_type, timestamp, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            content,
            memory_type,
            timestamp,
            json.dumps(metadata) if metadata else None,
            embedding.tobytes() if embedding is not None else None
        ))
        
        conn.commit()
        conn.close()
        
        # Update cache
        if embedding is not None:
            self._embeddings_cache[memory_id] = embedding
        
        # Prune if needed
        self._prune_if_needed()
        
        logger.debug(f"Added memory {memory_id}: {content[:50]}...")
        return memory_id
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[SearchResult]:
        """
        Search memories by semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            memory_type: Filter by memory type
            min_score: Minimum similarity score
            
        Returns:
            List of SearchResult sorted by relevance
        """
        # Get query embedding
        query_embedding = self.embedder.encode(query)
        if query_embedding is None:
            return []
        
        # Get candidate IDs
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute(
                "SELECT id, content, memory_type, timestamp, metadata FROM memories WHERE memory_type = ?",
                (memory_type,)
            )
        else:
            cursor.execute(
                "SELECT id, content, memory_type, timestamp, metadata FROM memories"
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Compute similarities
        results = []
        for row in rows:
            memory_id = row[0]
            embedding = self._embeddings_cache.get(memory_id)
            
            if embedding is not None:
                # Cosine similarity (embeddings are normalized)
                score = float(np.dot(query_embedding, embedding))
                
                if score >= min_score:
                    entry = MemoryEntry.from_row(row, embedding)
                    results.append(SearchResult(entry=entry, score=score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def get_context_for_query(self, query: str, max_tokens: int = 500) -> str:
        """
        Get relevant context for a query as a formatted string.
        
        Args:
            query: The user's query
            max_tokens: Approximate max tokens for context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Get relevant conversations
        conv_results = self.search(query, top_k=2, memory_type="conversation")
        if conv_results:
            context_parts.append("**Recent Relevant Conversations:**")
            for r in conv_results:
                context_parts.append(f"  - {r.entry.content[:200]}...")
        
        # Get relevant errors
        error_results = self.search(query, top_k=1, memory_type="error")
        if error_results:
            context_parts.append("\n**Past Errors to Avoid:**")
            for r in error_results:
                context_parts.append(f"  - {r.entry.content[:200]}...")
        
        # Get preferences
        pref_results = self.search(query, top_k=1, memory_type="preference")
        if pref_results:
            context_parts.append("\n**User Preferences:**")
            for r in pref_results:
                context_parts.append(f"  - {r.entry.content[:150]}")
        
        # Get workflow context
        workflow_results = self.search(query, top_k=1, memory_type="workflow")
        if workflow_results:
            context_parts.append("\n**Recent Workflows:**")
            for r in workflow_results:
                context_parts.append(f"  - {r.entry.content[:150]}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def add_conversation(self, user_msg: str, assistant_msg: str) -> str:
        """Add a conversation exchange to memory."""
        content = f"User: {user_msg}\nAssistant: {assistant_msg}"
        return self.add(content, "conversation", {"user": user_msg[:100]})
    
    def add_error(
        self,
        error_msg: str,
        diagnosis: Optional[str] = None,
        fix: Optional[str] = None,
    ) -> str:
        """Add an error and its resolution to memory."""
        content = f"Error: {error_msg}"
        if diagnosis:
            content += f"\nDiagnosis: {diagnosis}"
        if fix:
            content += f"\nFix: {fix}"
        
        return self.add(content, "error", {
            "has_diagnosis": diagnosis is not None,
            "has_fix": fix is not None,
        })
    
    def add_preference(self, preference: str, category: str = "general") -> str:
        """Add a user preference."""
        return self.add(preference, "preference", {"category": category})
    
    def add_workflow(self, workflow_desc: str, workflow_type: str) -> str:
        """Add workflow information."""
        return self.add(workflow_desc, "workflow", {"type": workflow_type})
    
    def get_recent(
        self,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Get most recent memories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute(
                """SELECT id, content, memory_type, timestamp, metadata 
                   FROM memories WHERE memory_type = ? 
                   ORDER BY timestamp DESC LIMIT ?""",
                (memory_type, limit)
            )
        else:
            cursor.execute(
                """SELECT id, content, memory_type, timestamp, metadata 
                   FROM memories ORDER BY timestamp DESC LIMIT ?""",
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [MemoryEntry.from_row(row) for row in rows]
    
    def clear(self, memory_type: Optional[str] = None):
        """Clear memories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute("DELETE FROM memories WHERE memory_type = ?", (memory_type,))
            # Clear from cache
            for mid in list(self._embeddings_cache.keys()):
                if mid in self._embeddings_cache:
                    del self._embeddings_cache[mid]
        else:
            cursor.execute("DELETE FROM memories")
            self._embeddings_cache.clear()
        
        conn.commit()
        conn.close()
        logger.info(f"Cleared memories: type={memory_type or 'all'}")
    
    def _prune_if_needed(self):
        """Prune old memories if over limit."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]
        
        if count > self.max_memories:
            # Delete oldest memories, keeping max_memories
            to_delete = count - self.max_memories
            cursor.execute("""
                DELETE FROM memories WHERE id IN (
                    SELECT id FROM memories ORDER BY timestamp ASC LIMIT ?
                )
            """, (to_delete,))
            conn.commit()
            
            # Rebuild cache
            self._embeddings_cache.clear()
            self._load_embeddings()
            
            logger.info(f"Pruned {to_delete} old memories")
        
        conn.close()
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
        type_counts = dict(cursor.fetchall())
        
        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total": total,
            "by_type": type_counts,
            "embeddings_cached": len(self._embeddings_cache),
            "db_path": self.db_path,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_global_memory: Optional[AgentMemory] = None


def get_memory() -> AgentMemory:
    """Get global memory instance."""
    global _global_memory
    if _global_memory is None:
        _global_memory = AgentMemory()
    return _global_memory


def remember(content: str, memory_type: str = "conversation") -> str:
    """Quick function to add a memory."""
    return get_memory().add(content, memory_type)


def recall(query: str, top_k: int = 5) -> List[SearchResult]:
    """Quick function to search memories."""
    return get_memory().search(query, top_k)
