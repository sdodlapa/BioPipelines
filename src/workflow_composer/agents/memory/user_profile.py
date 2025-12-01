"""
User Profile System
===================

Persistent user profiles that track:
- Preferred organisms, read types, aligners
- Analysis type usage patterns
- Successful workflow templates
- Session history

Used to personalize the experience and provide smart defaults.
"""

import json
import sqlite3
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """
    Persistent user preferences and history.
    
    Tracks:
    - Inferred preferences (organism, read type, tools)
    - Usage patterns (which analyses are most common)
    - Successful workflows for reuse
    """
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    # Inferred preferences (updated from queries)
    preferred_organism: Optional[str] = None  # "human", "mouse", etc.
    preferred_read_type: Optional[str] = None  # "paired", "single"
    preferred_aligner: Optional[str] = None  # "bwa", "bowtie2", "star"
    preferred_caller: Optional[str] = None  # "gatk", "freebayes", "deepvariant"
    
    # Usage patterns
    query_count: int = 0
    successful_workflows: int = 0
    analysis_types: Dict[str, int] = field(default_factory=dict)  # {"rna-seq": 5, "chip-seq": 2}
    
    # Tool preferences (learned from usage)
    tool_preferences: Dict[str, str] = field(default_factory=dict)  # {"alignment": "bwa", "qc": "fastqc"}
    
    # Recent queries (for context)
    recent_queries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "preferred_organism": self.preferred_organism,
            "preferred_read_type": self.preferred_read_type,
            "preferred_aligner": self.preferred_aligner,
            "preferred_caller": self.preferred_caller,
            "query_count": self.query_count,
            "successful_workflows": self.successful_workflows,
            "analysis_types": self.analysis_types,
            "tool_preferences": self.tool_preferences,
            "recent_queries": self.recent_queries[-10:],  # Keep last 10
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            preferred_organism=data.get("preferred_organism"),
            preferred_read_type=data.get("preferred_read_type"),
            preferred_aligner=data.get("preferred_aligner"),
            preferred_caller=data.get("preferred_caller"),
            query_count=data.get("query_count", 0),
            successful_workflows=data.get("successful_workflows", 0),
            analysis_types=data.get("analysis_types", {}),
            tool_preferences=data.get("tool_preferences", {}),
            recent_queries=data.get("recent_queries", []),
        )
    
    def get_most_common_analysis(self) -> Optional[str]:
        """Get the user's most frequently used analysis type."""
        if not self.analysis_types:
            return None
        return max(self.analysis_types, key=self.analysis_types.get)
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get user's default preferences for query parsing."""
        return {
            "organism": self.preferred_organism,
            "read_type": self.preferred_read_type,
            "aligner": self.preferred_aligner,
            "variant_caller": self.preferred_caller,
            "most_common_analysis": self.get_most_common_analysis(),
        }


class PersistentProfileStore:
    """
    SQLite-backed persistent storage for user profiles and history.
    
    Stores:
    - User profiles with preferences
    - Conversation history
    - Successful workflows for template reuse
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the profile store.
        
        Args:
            db_path: Path to SQLite database. Defaults to ~/.biopipelines/profiles.db
        """
        if db_path is None:
            db_dir = Path.home() / ".biopipelines"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "profiles.db")
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # User profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Conversation history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Successful workflows table (for template reuse)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS successful_workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    organism TEXT,
                    workflow_json TEXT NOT NULL,
                    rating INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_user 
                ON conversation_history(user_id, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_type 
                ON successful_workflows(analysis_type)
            """)
            
            conn.commit()
        
        logger.info(f"Profile store initialized: {self.db_path}")
    
    # =========================================================================
    # Profile Management
    # =========================================================================
    
    def save_profile(self, profile: UserProfile) -> None:
        """Save or update a user profile."""
        profile.last_active = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_profiles (user_id, profile_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (profile.user_id, json.dumps(profile.to_dict())))
            conn.commit()
        
        logger.debug(f"Saved profile for user: {profile.user_id}")
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load a user profile by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT profile_json FROM user_profiles WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            
            if row:
                return UserProfile.from_dict(json.loads(row[0]))
        return None
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create a new one."""
        profile = self.load_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
            self.save_profile(profile)
        return profile
    
    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile and all associated data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM conversation_history WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM successful_workflows WHERE user_id = ?", (user_id,))
            conn.commit()
        return True
    
    # =========================================================================
    # Conversation History
    # =========================================================================
    
    def save_conversation(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Save a conversation message."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO conversation_history (user_id, session_id, role, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                session_id,
                role,
                content,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_conversation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        with sqlite3.connect(self.db_path) as conn:
            if session_id:
                rows = conn.execute("""
                    SELECT role, content, timestamp, metadata
                    FROM conversation_history
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, session_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT role, content, timestamp, metadata
                    FROM conversation_history
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, limit)).fetchall()
        
        return [
            {
                "role": row[0],
                "content": row[1],
                "timestamp": row[2],
                "metadata": json.loads(row[3]) if row[3] else None,
            }
            for row in reversed(rows)  # Chronological order
        ]
    
    # =========================================================================
    # Successful Workflows
    # =========================================================================
    
    def save_successful_workflow(
        self,
        query: str,
        analysis_type: str,
        workflow: dict,
        user_id: Optional[str] = None,
        organism: Optional[str] = None,
    ) -> int:
        """Save a successfully generated workflow as a template."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO successful_workflows 
                (user_id, query, analysis_type, organism, workflow_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                query,
                analysis_type,
                organism,
                json.dumps(workflow)
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_similar_workflows(
        self,
        analysis_type: str,
        organism: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get similar successful workflows for template reuse."""
        with sqlite3.connect(self.db_path) as conn:
            if organism:
                rows = conn.execute("""
                    SELECT query, workflow_json, organism, rating, created_at
                    FROM successful_workflows
                    WHERE analysis_type = ? AND organism = ?
                    ORDER BY rating DESC, created_at DESC
                    LIMIT ?
                """, (analysis_type, organism, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT query, workflow_json, organism, rating, created_at
                    FROM successful_workflows
                    WHERE analysis_type = ?
                    ORDER BY rating DESC, created_at DESC
                    LIMIT ?
                """, (analysis_type, limit)).fetchall()
        
        return [
            {
                "query": row[0],
                "workflow": json.loads(row[1]),
                "organism": row[2],
                "rating": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]
    
    def rate_workflow(self, workflow_id: int, rating: int) -> None:
        """Rate a workflow (for ranking templates)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE successful_workflows SET rating = ? WHERE id = ?",
                (rating, workflow_id)
            )
            conn.commit()
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with sqlite3.connect(self.db_path) as conn:
            profile_count = conn.execute(
                "SELECT COUNT(*) FROM user_profiles"
            ).fetchone()[0]
            
            conv_count = conn.execute(
                "SELECT COUNT(*) FROM conversation_history"
            ).fetchone()[0]
            
            workflow_count = conn.execute(
                "SELECT COUNT(*) FROM successful_workflows"
            ).fetchone()[0]
            
            # Analysis type distribution
            type_dist = dict(conn.execute("""
                SELECT analysis_type, COUNT(*) 
                FROM successful_workflows 
                GROUP BY analysis_type
            """).fetchall())
        
        return {
            "profiles": profile_count,
            "conversations": conv_count,
            "workflows": workflow_count,
            "workflow_by_type": type_dist,
            "db_path": self.db_path,
        }


# =============================================================================
# Preference Learner
# =============================================================================

class PreferenceLearner:
    """
    Learns user preferences from interactions.
    
    Updates user profile based on:
    - Parsed query intents (organism, read type, etc.)
    - Tool selections
    - Successful workflow completions
    """
    
    def __init__(self, store: Optional[PersistentProfileStore] = None):
        """
        Initialize preference learner.
        
        Args:
            store: Profile store instance. Creates new one if not provided.
        """
        self.store = store or PersistentProfileStore()
    
    def update_from_query(
        self,
        user_id: str,
        parsed_intent: Dict[str, Any],
    ) -> UserProfile:
        """
        Update user profile based on parsed query intent.
        
        Args:
            user_id: User identifier
            parsed_intent: Parsed intent from query parser
            
        Returns:
            Updated user profile
        """
        profile = self.store.get_or_create_profile(user_id)
        
        # Update preferred organism if specified
        if organism := parsed_intent.get("organism"):
            profile.preferred_organism = organism
        
        # Update preferred read type
        if read_type := parsed_intent.get("read_type"):
            profile.preferred_read_type = read_type
        
        # Track analysis type usage
        if analysis_type := parsed_intent.get("analysis_type"):
            profile.analysis_types[analysis_type] = \
                profile.analysis_types.get(analysis_type, 0) + 1
        
        # Update tool preferences
        if tools := parsed_intent.get("tools"):
            for tool_category, tool_name in tools.items():
                profile.tool_preferences[tool_category] = tool_name
        
        # Track query
        if query := parsed_intent.get("query"):
            profile.recent_queries.append(query)
            profile.recent_queries = profile.recent_queries[-10:]  # Keep last 10
        
        profile.query_count += 1
        profile.last_active = datetime.now()
        
        self.store.save_profile(profile)
        return profile
    
    def update_from_workflow_success(
        self,
        user_id: str,
        workflow: Dict[str, Any],
    ) -> UserProfile:
        """
        Update profile when a workflow completes successfully.
        
        Args:
            user_id: User identifier
            workflow: Successful workflow data
            
        Returns:
            Updated user profile
        """
        profile = self.store.get_or_create_profile(user_id)
        profile.successful_workflows += 1
        
        # Save workflow as template
        self.store.save_successful_workflow(
            query=workflow.get("query", ""),
            analysis_type=workflow.get("analysis_type", "unknown"),
            workflow=workflow,
            user_id=user_id,
            organism=workflow.get("organism"),
        )
        
        self.store.save_profile(profile)
        return profile
    
    def get_context_for_query(self, user_id: str) -> Dict[str, Any]:
        """
        Get user context to enhance query parsing.
        
        Provides defaults based on user's history.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with default values for query parsing
        """
        profile = self.store.load_profile(user_id)
        if not profile:
            return {}
        
        context = profile.get_defaults()
        
        # Add recent query context
        if profile.recent_queries:
            context["recent_queries"] = profile.recent_queries[-3:]
        
        return context
    
    def get_workflow_suggestions(
        self,
        user_id: str,
        analysis_type: str,
        organism: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get workflow suggestions based on past successes.
        
        Args:
            user_id: User identifier
            analysis_type: Type of analysis
            organism: Optional organism filter
            
        Returns:
            List of similar successful workflows
        """
        return self.store.get_similar_workflows(
            analysis_type=analysis_type,
            organism=organism,
            limit=3,
        )


# =============================================================================
# Global Instances
# =============================================================================

_profile_store: Optional[PersistentProfileStore] = None
_preference_learner: Optional[PreferenceLearner] = None


def get_profile_store() -> PersistentProfileStore:
    """Get global profile store instance."""
    global _profile_store
    if _profile_store is None:
        _profile_store = PersistentProfileStore()
    return _profile_store


def get_preference_learner() -> PreferenceLearner:
    """Get global preference learner instance."""
    global _preference_learner
    if _preference_learner is None:
        _preference_learner = PreferenceLearner(get_profile_store())
    return _preference_learner


def get_user_profile(user_id: str) -> UserProfile:
    """Quick function to get user profile."""
    return get_profile_store().get_or_create_profile(user_id)


def update_preferences(user_id: str, parsed_intent: Dict[str, Any]) -> UserProfile:
    """Quick function to update preferences from a parsed intent."""
    return get_preference_learner().update_from_query(user_id, parsed_intent)
