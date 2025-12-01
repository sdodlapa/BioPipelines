"""
Session Manager
===============

Manages conversation sessions with:
- Session state persistence
- Conversation history within sessions
- Session-level context injection
- Multi-turn conversation support

Integrates with UserProfile for preference learning.
"""

import json
import uuid
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .user_profile import (
    PersistentProfileStore,
    PreferenceLearner,
    get_profile_store,
    get_preference_learner,
)

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    def to_chat_format(self) -> dict:
        """Convert to format expected by LLM providers."""
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    """
    A conversation session with context and history.
    
    Sessions track:
    - Conversation messages
    - Current context (parsed intents, entities)
    - Session-level state (current workflow, etc.)
    """
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Conversation history
    messages: List[Message] = field(default_factory=list)
    
    # Session context
    current_organism: Optional[str] = None
    current_analysis_type: Optional[str] = None
    current_workflow: Optional[Dict[str, Any]] = None
    
    # Session state
    state: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired (1 hour of inactivity)."""
        return datetime.now() - self.last_activity > timedelta(hours=1)
    
    @property
    def message_count(self) -> int:
        """Get number of messages in session."""
        return len(self.messages)
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to the session."""
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self.last_activity = datetime.now()
        return message
    
    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add a user message."""
        return self.add_message("user", content, metadata)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add an assistant message."""
        return self.add_message("assistant", content, metadata)
    
    def add_system_message(self, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add a system message."""
        return self.add_message("system", content, metadata)
    
    def get_recent_messages(
        self,
        limit: int = 10,
        include_system: bool = False,
    ) -> List[Message]:
        """Get recent messages for context."""
        messages = self.messages[-limit:] if limit else self.messages
        if not include_system:
            messages = [m for m in messages if m.role != "system"]
        return messages
    
    def get_chat_history(
        self,
        limit: int = 10,
        include_system: bool = True,
    ) -> List[dict]:
        """Get chat history in LLM format."""
        messages = self.get_recent_messages(limit, include_system)
        return [m.to_chat_format() for m in messages]
    
    def update_context(self, parsed_intent: Dict[str, Any]) -> None:
        """Update session context from parsed intent."""
        if organism := parsed_intent.get("organism"):
            self.current_organism = organism
        if analysis_type := parsed_intent.get("analysis_type"):
            self.current_analysis_type = analysis_type
        self.last_activity = datetime.now()
    
    def set_current_workflow(self, workflow: Dict[str, Any]) -> None:
        """Set the current workflow being built/discussed."""
        self.current_workflow = workflow
        self.last_activity = datetime.now()
    
    def get_context_summary(self) -> str:
        """Get a text summary of session context for prompts."""
        parts = []
        
        if self.current_organism:
            parts.append(f"Current organism: {self.current_organism}")
        if self.current_analysis_type:
            parts.append(f"Current analysis: {self.current_analysis_type}")
        if self.current_workflow:
            parts.append(f"Active workflow: {self.current_workflow.get('name', 'unnamed')}")
        
        return "; ".join(parts) if parts else "No active context"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
            "current_organism": self.current_organism,
            "current_analysis_type": self.current_analysis_type,
            "current_workflow": self.current_workflow,
            "state": self.state,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create from dictionary."""
        session = cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            current_organism=data.get("current_organism"),
            current_analysis_type=data.get("current_analysis_type"),
            current_workflow=data.get("current_workflow"),
            state=data.get("state", {}),
        )
        
        for msg_data in data.get("messages", []):
            session.messages.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                metadata=msg_data.get("metadata"),
            ))
        
        return session


class SessionManager:
    """
    Manages conversation sessions with persistence.
    
    Features:
    - Create/resume sessions
    - Persist sessions to database
    - Inject user preferences into sessions
    - Multi-turn conversation tracking
    """
    
    def __init__(
        self,
        store: Optional[PersistentProfileStore] = None,
        learner: Optional[PreferenceLearner] = None,
    ):
        """
        Initialize session manager.
        
        Args:
            store: Profile store for persistence
            learner: Preference learner for context
        """
        self.store = store or get_profile_store()
        self.learner = learner or get_preference_learner()
        
        # In-memory session cache (for active sessions)
        self._sessions: Dict[str, Session] = {}
    
    def create_session(self, user_id: str) -> Session:
        """
        Create a new session for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            New session instance
        """
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, user_id=user_id)
        
        # Inject user context as system message
        profile = self.store.get_or_create_profile(user_id)
        defaults = profile.get_defaults()
        
        if any(defaults.values()):
            context_parts = []
            if defaults.get("organism"):
                context_parts.append(f"preferred organism: {defaults['organism']}")
            if defaults.get("read_type"):
                context_parts.append(f"preferred read type: {defaults['read_type']}")
            if defaults.get("most_common_analysis"):
                context_parts.append(f"most common analysis: {defaults['most_common_analysis']}")
            
            session.add_system_message(
                f"User context: {', '.join(context_parts)}. "
                f"Use these as defaults unless the user specifies otherwise."
            )
        
        self._sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get an existing session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if found and not expired, None otherwise
        """
        session = self._sessions.get(session_id)
        
        if session and session.is_expired:
            logger.info(f"Session {session_id} expired, removing")
            self.end_session(session_id)
            return None
        
        return session
    
    def get_or_create_session(self, session_id: Optional[str], user_id: str) -> Session:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Optional session ID to resume
            user_id: User identifier
            
        Returns:
            Session instance
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(user_id)
    
    def end_session(self, session_id: str) -> None:
        """
        End a session and persist history.
        
        Args:
            session_id: Session to end
        """
        session = self._sessions.pop(session_id, None)
        
        if session:
            # Persist all messages to database
            for msg in session.messages:
                self.store.save_conversation(
                    user_id=session.user_id,
                    role=msg.role,
                    content=msg.content,
                    session_id=session_id,
                    metadata=msg.metadata,
                )
            
            logger.info(f"Ended session {session_id} with {session.message_count} messages")
    
    def add_user_message(
        self,
        session_id: str,
        content: str,
        parsed_intent: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """
        Add a user message and update context.
        
        Args:
            session_id: Session identifier
            content: User message content
            parsed_intent: Optional parsed intent for context update
            
        Returns:
            Message if session found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        message = session.add_user_message(content, metadata=parsed_intent)
        
        # Update session context
        if parsed_intent:
            session.update_context(parsed_intent)
            # Update user preferences
            self.learner.update_from_query(session.user_id, parsed_intent)
        
        return message
    
    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        workflow: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """
        Add an assistant message.
        
        Args:
            session_id: Session identifier
            content: Assistant message content
            workflow: Optional workflow that was generated
            
        Returns:
            Message if session found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        metadata = {"workflow": workflow} if workflow else None
        message = session.add_assistant_message(content, metadata)
        
        if workflow:
            session.set_current_workflow(workflow)
        
        return message
    
    def get_chat_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[dict]:
        """
        Get chat history for LLM context.
        
        Args:
            session_id: Session identifier
            limit: Max messages to return
            
        Returns:
            List of messages in chat format
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        return session.get_chat_history(limit)
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get full session context for query enhancement.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context dict with session state and user preferences
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Combine session context with user preferences
        user_context = self.learner.get_context_for_query(session.user_id)
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "message_count": session.message_count,
            "current_organism": session.current_organism or user_context.get("organism"),
            "current_analysis_type": session.current_analysis_type,
            "current_workflow": session.current_workflow,
            "user_defaults": user_context,
            "context_summary": session.get_context_summary(),
        }
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired
        ]
        
        for sid in expired:
            self.end_session(sid)
        
        return len(expired)
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        self.cleanup_expired()
        return len(self._sessions)


# =============================================================================
# Global Instance
# =============================================================================

_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
