"""
Learning Module for Chat Agent.

This module provides infrastructure for continuous learning and improvement:
1. Production Query Collection - captures real user queries for analysis
2. Active Learning - prioritizes difficult queries for manual review
3. Feedback Collection - captures user corrections and preferences

Author: BioPipelines Team
Date: November 2025
"""
from .production_queries import ProductionQueryCollector, ProductionQuery
from .active_learner import ActiveLearner, LearningSignal, QueryDifficulty

__all__ = [
    "ProductionQueryCollector",
    "ProductionQuery",
    "ActiveLearner",
    "LearningSignal",
    "QueryDifficulty",
]
