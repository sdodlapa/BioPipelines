#!/usr/bin/env python3
"""
Test Phase 2.2: Session Memory

Tests:
- User profile creation and persistence
- Session management
- Preference learning
- Conversation history
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_user_profile():
    """Test user profile creation and persistence."""
    from workflow_composer.agents.memory import (
        UserProfile,
        PersistentProfileStore,
    )
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        store = PersistentProfileStore(db_path)
        
        # Create profile
        profile = store.get_or_create_profile("test_user_1")
        assert profile.user_id == "test_user_1"
        assert profile.query_count == 0
        
        # Update profile
        profile.preferred_organism = "human"
        profile.preferred_aligner = "bwa"
        profile.analysis_types["rna-seq"] = 5
        profile.query_count = 10
        store.save_profile(profile)
        
        # Reload profile
        reloaded = store.load_profile("test_user_1")
        assert reloaded.preferred_organism == "human"
        assert reloaded.preferred_aligner == "bwa"
        assert reloaded.analysis_types["rna-seq"] == 5
        assert reloaded.query_count == 10
        
        print("✅ User profile creation and persistence: PASS")
        return True
        
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_session_manager():
    """Test session management."""
    from workflow_composer.agents.memory import (
        SessionManager,
        PersistentProfileStore,
    )
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        store = PersistentProfileStore(db_path)
        manager = SessionManager(store)
        
        # Create session
        session = manager.create_session("test_user")
        assert session.user_id == "test_user"
        assert session.session_id is not None
        
        # Add messages
        manager.add_user_message(session.session_id, "Hello, I need help with RNA-seq")
        manager.add_assistant_message(session.session_id, "I can help with RNA-seq analysis!")
        
        # Get history
        history = manager.get_chat_history(session.session_id)
        # Filter out system messages
        user_assistant = [m for m in history if m["role"] in ("user", "assistant")]
        assert len(user_assistant) == 2
        assert user_assistant[0]["role"] == "user"
        assert "RNA-seq" in user_assistant[0]["content"]
        
        print("✅ Session management: PASS")
        return True
        
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_preference_learning():
    """Test preference learning from queries."""
    from workflow_composer.agents.memory import (
        PreferenceLearner,
        PersistentProfileStore,
    )
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        store = PersistentProfileStore(db_path)
        learner = PreferenceLearner(store)
        
        # Simulate parsed intent from query
        parsed_intent = {
            "query": "Run RNA-seq on human samples",
            "organism": "human",
            "analysis_type": "rna-seq",
            "read_type": "paired",
        }
        
        profile = learner.update_from_query("test_user", parsed_intent)
        
        assert profile.preferred_organism == "human"
        assert profile.preferred_read_type == "paired"
        assert profile.analysis_types.get("rna-seq") == 1
        assert profile.query_count == 1
        
        # Another query should increment counts
        parsed_intent2 = {
            "query": "Run another RNA-seq",
            "analysis_type": "rna-seq",
        }
        profile = learner.update_from_query("test_user", parsed_intent2)
        assert profile.analysis_types["rna-seq"] == 2
        assert profile.query_count == 2
        
        # Get context for new query
        context = learner.get_context_for_query("test_user")
        assert context["organism"] == "human"
        assert context["read_type"] == "paired"
        
        print("✅ Preference learning: PASS")
        return True
        
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_workflow_caching():
    """Test successful workflow caching."""
    from workflow_composer.agents.memory import (
        PersistentProfileStore,
    )
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        store = PersistentProfileStore(db_path)
        
        # Save a successful workflow
        workflow = {
            "name": "RNA-seq Analysis",
            "steps": ["fastqc", "star", "featureCounts"],
            "parameters": {"genome": "GRCh38"},
        }
        
        workflow_id = store.save_successful_workflow(
            query="Run RNA-seq on human samples",
            analysis_type="rna-seq",
            workflow=workflow,
            user_id="test_user",
            organism="human",
        )
        
        assert workflow_id > 0
        
        # Retrieve similar workflows
        similar = store.get_similar_workflows("rna-seq", organism="human")
        assert len(similar) == 1
        assert similar[0]["workflow"]["name"] == "RNA-seq Analysis"
        
        print("✅ Workflow caching: PASS")
        return True
        
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_conversation_history():
    """Test conversation history persistence."""
    from workflow_composer.agents.memory import (
        PersistentProfileStore,
    )
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        store = PersistentProfileStore(db_path)
        
        # Save conversation messages
        store.save_conversation(
            user_id="test_user",
            role="user",
            content="Hello!",
            session_id="session_1",
        )
        store.save_conversation(
            user_id="test_user",
            role="assistant",
            content="Hi there!",
            session_id="session_1",
        )
        
        # Retrieve history
        history = store.get_conversation_history("test_user", session_id="session_1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        
        print("✅ Conversation history: PASS")
        return True
        
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_global_functions():
    """Test global convenience functions."""
    from workflow_composer.agents.memory import (
        get_profile_store,
        get_session_manager,
        get_preference_learner,
    )
    
    # These should return singleton instances
    store1 = get_profile_store()
    store2 = get_profile_store()
    
    manager1 = get_session_manager()
    manager2 = get_session_manager()
    
    learner1 = get_preference_learner()
    learner2 = get_preference_learner()
    
    # Verify singletons (same id)
    assert id(store1) == id(store2), "ProfileStore should be singleton"
    assert id(manager1) == id(manager2), "SessionManager should be singleton"
    assert id(learner1) == id(learner2), "PreferenceLearner should be singleton"
    
    print("✅ Global singleton functions: PASS")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Phase 2.2: Session Memory Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("User Profile", test_user_profile),
        ("Session Manager", test_session_manager),
        ("Preference Learning", test_preference_learning),
        ("Workflow Caching", test_workflow_caching),
        ("Conversation History", test_conversation_history),
        ("Global Functions", test_global_functions),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {name}: FAIL - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
