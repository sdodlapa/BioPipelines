"""
Evaluation Module for BioPipelines
==================================

Systematic testing and evaluation of chat agent capabilities.

Modules:
- conversation_test_suite: Comprehensive conversation evaluation
"""

from .conversation_test_suite import (
    ConversationEvaluator,
    TestConversation,
    EvaluationReport,
    get_test_conversations,
    print_report,
    save_report,
)

__all__ = [
    "ConversationEvaluator",
    "TestConversation", 
    "EvaluationReport",
    "get_test_conversations",
    "print_report",
    "save_report",
]
