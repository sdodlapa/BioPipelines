"""
Training Data Collection Module
===============================

This module provides infrastructure for collecting, generating, and
preparing training data for fine-tuning the BioPipelines LLM.

Components:
- TrainingDataGenerator: Generate synthetic training examples
- ConversationGenerator: Generate multi-turn dialogues
- ConversationRunner: Run conversations through actual system
- ResultAnalyzer: Analyze results and identify gaps
- InteractionLogger: Log real user interactions
- TrainingDataPipeline: Process and validate training data
- DataExporter: Export to various training formats

Example:
    >>> from workflow_composer.training import TrainingDataCollector
    >>> collector = TrainingDataCollector(num_conversations=100)
    >>> summary = await collector.run_full_pipeline()
"""

from .config import (
    TrainingConfig,
    GeneratorConfig,
    LoggerConfig,
    PipelineConfig,
    ExportConfig,
    VariationType,
    get_default_config,
)

from .data_generator import (
    TrainingDataGenerator,
)

from .conversation_generator import (
    ConversationGenerator,
    GeneratedConversation,
    ConversationScenario,
    ConversationPattern,
    ConversationTurn,
    CONVERSATION_SCENARIOS,
)

from .conversation_runner import (
    ConversationRunner,
    ConversationResult,
    TurnResult,
    RunnerConfig,
)

from .result_analyzer import (
    ResultAnalyzer,
    AnalysisReport,
    GapReport,
)

from .interaction_logger import (
    InteractionLogger,
)

from .data_pipeline import (
    TrainingDataPipeline,
    DataValidator,
    QualityScorer,
    QualityMetrics,
    process_training_data,
)

from .export import (
    TrainingDataExporter,
    ExportResult,
    OpenAIChatExporter,
    AlpacaExporter,
    ShareGPTExporter,
    AxolotlExporter,
    export_training_data,
    export_all_formats,
)

from .collect_training_data import (
    TrainingDataCollector,
)

__all__ = [
    # Config
    "TrainingConfig",
    "GeneratorConfig",
    "LoggerConfig",
    "PipelineConfig",
    "ExportConfig",
    "VariationType",
    "get_default_config",
    # Conversation Generation
    "ConversationGenerator",
    "GeneratedConversation",
    "ConversationScenario",
    "ConversationPattern",
    "ConversationTurn",
    "CONVERSATION_SCENARIOS",
    # Conversation Running
    "ConversationRunner",
    "ConversationResult",
    "TurnResult",
    "RunnerConfig",
    # Result Analysis
    "ResultAnalyzer",
    "AnalysisReport",
    "GapReport",
    # Single-turn Generator
    "TrainingDataGenerator",
    # Logger
    "InteractionLogger",
    # Pipeline
    "TrainingDataPipeline",
    "DataValidator",
    "QualityScorer",
    "QualityMetrics",
    "process_training_data",
    # Export
    "TrainingDataExporter",
    "ExportResult",
    "OpenAIChatExporter",
    "AlpacaExporter",
    "ShareGPTExporter",
    "AxolotlExporter",
    "export_training_data",
    "export_all_formats",
    # Main Collector
    "TrainingDataCollector",
]


def get_data_generator(config: GeneratorConfig = None) -> TrainingDataGenerator:
    """Factory function to create a data generator."""
    return TrainingDataGenerator(config)


def get_interaction_logger(config: LoggerConfig = None) -> InteractionLogger:
    """Factory function to create an interaction logger."""
    return InteractionLogger(config)


def get_pipeline(config: PipelineConfig = None) -> TrainingDataPipeline:
    """Factory function to create a data pipeline."""
    return TrainingDataPipeline(config)


def get_exporter(config: ExportConfig = None) -> TrainingDataExporter:
    """Factory function to create a data exporter."""
    return TrainingDataExporter(config)


def get_conversation_generator(config: GeneratorConfig = None) -> ConversationGenerator:
    """Factory function to create a conversation generator."""
    return ConversationGenerator(config)


def get_conversation_runner(config: RunnerConfig = None) -> ConversationRunner:
    """Factory function to create a conversation runner."""
    return ConversationRunner(config)
