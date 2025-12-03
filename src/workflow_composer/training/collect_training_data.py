#!/usr/bin/env python
"""
Training Data Collection Script
===============================

Main script to:
1. Generate multi-turn conversations
2. Run them through the actual system
3. Collect results and training data
4. Analyze gaps and generate reports

Usage:
    python -m workflow_composer.training.collect_training_data --conversations 100
    python -m workflow_composer.training.collect_training_data --analyze-only /path/to/results
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from workflow_composer.training.conversation_generator import (
    ConversationGenerator,
    GeneratedConversation,
    CONVERSATION_SCENARIOS,
)
from workflow_composer.training.conversation_runner import (
    ConversationRunner,
    RunnerConfig,
)
from workflow_composer.training.result_analyzer import (
    ResultAnalyzer,
    AnalysisReport,
)
from workflow_composer.training.config import GeneratorConfig

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """Orchestrates the full training data collection pipeline."""
    
    def __init__(
        self,
        output_dir: Path = None,
        num_conversations: int = 100,
        min_turns: int = 5,
        max_turns: int = 15,
    ):
        self.output_dir = output_dir or Path("data/training")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_conversations = num_conversations
        self.min_turns = min_turns
        self.max_turns = max_turns
        
        # Create subdirectories
        self.conversations_dir = self.output_dir / "conversations"
        self.results_dir = self.output_dir / "run_results"
        self.analysis_dir = self.output_dir / "analysis"
        self.training_data_dir = self.output_dir / "training_data"
        
        for d in [self.conversations_dir, self.results_dir, self.analysis_dir, self.training_data_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.generator = ConversationGenerator(
            GeneratorConfig(output_dir=self.conversations_dir)
        )
        self.runner = ConversationRunner(
            RunnerConfig(output_dir=self.results_dir)
        )
        self.analyzer = ResultAnalyzer(output_dir=self.analysis_dir)
    
    async def generate_conversations(self) -> list[GeneratedConversation]:
        """Generate multi-turn conversations."""
        
        logger.info(f"Generating {self.num_conversations} conversations...")
        
        conversations = await self.generator.generate_dataset(
            num_conversations=self.num_conversations,
            output_path=self.conversations_dir,
            min_turns=self.min_turns,
            max_turns=self.max_turns,
        )
        
        logger.info(f"Generated {len(conversations)} conversations")
        return conversations
    
    async def run_conversations(
        self,
        conversations: list[GeneratedConversation],
    ) -> list:
        """Run conversations through the actual system."""
        
        logger.info(f"Running {len(conversations)} conversations through system...")
        
        results = await self.runner.run_all(conversations)
        
        successful = sum(1 for r in results if r.completed_successfully)
        logger.info(f"Completed: {successful}/{len(results)} successful")
        
        return results
    
    def analyze_results(self, results: list) -> AnalysisReport:
        """Analyze results and identify gaps."""
        
        logger.info("Analyzing results...")
        
        # Convert to dicts if needed
        result_dicts = [
            r.to_dict() if hasattr(r, 'to_dict') else r 
            for r in results
        ]
        
        report = self.analyzer.analyze(result_dicts)
        self.analyzer.save_report(report)
        
        return report
    
    def extract_training_data(
        self,
        conversations: list[GeneratedConversation],
        results: list,
    ) -> Path:
        """Extract high-quality training data from successful runs."""
        
        logger.info("Extracting training data...")
        
        training_examples = []
        
        for conv, result in zip(conversations, results):
            # Skip failed conversations
            if hasattr(result, 'completed_successfully'):
                if not result.completed_successfully:
                    continue
            elif isinstance(result, dict):
                if not result.get('completed_successfully'):
                    continue
            
            # Convert to training format
            example = {
                "id": conv.id if hasattr(conv, 'id') else conv.get('id'),
                "messages": [],
                "metadata": {
                    "category": conv.category if hasattr(conv, 'category') else conv.get('category'),
                    "analysis_type": conv.analysis_type if hasattr(conv, 'analysis_type') else conv.get('analysis_type'),
                    "total_turns": len(conv.turns if hasattr(conv, 'turns') else conv.get('turns', [])),
                },
            }
            
            # Add system prompt
            example["messages"].append({
                "role": "system",
                "content": self._get_system_prompt(),
            })
            
            # Add conversation turns
            turns = conv.turns if hasattr(conv, 'turns') else conv.get('turns', [])
            for turn in turns:
                if hasattr(turn, 'role'):
                    role = turn.role
                    content = turn.content
                else:
                    role = turn.get('role', '')
                    content = turn.get('content', '')
                
                example["messages"].append({
                    "role": role,
                    "content": content,
                })
            
            training_examples.append(example)
        
        # Save training data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.training_data_dir / f"training_conversations_{timestamp}.jsonl"
        
        with open(output_file, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(training_examples)} training examples to {output_file}")
        
        # Also save in OpenAI format
        openai_file = self.training_data_dir / f"openai_format_{timestamp}.jsonl"
        with open(openai_file, 'w') as f:
            for example in training_examples:
                f.write(json.dumps({"messages": example["messages"]}) + '\n')
        
        logger.info(f"Saved OpenAI format to {openai_file}")
        
        return output_file
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for training."""
        
        return """You are BioPipelines, an expert bioinformatics workflow assistant. You help researchers design and execute computational biology pipelines using Nextflow.

Your capabilities include:
- Understanding various analysis types: RNA-seq, ChIP-seq, ATAC-seq, variant calling, metagenomics, single-cell, methylation, and more
- Recommending appropriate tools for each analysis step
- Generating Nextflow DSL2 workflows
- Explaining tool choices and pipeline configurations
- Helping troubleshoot errors and optimize parameters

When helping users:
1. First understand their analysis goal and data type
2. Ask clarifying questions if needed (organism, replicates, sequencing type, etc.)
3. Recommend appropriate tools with brief explanations
4. Generate complete Nextflow workflows when requested
5. Provide helpful guidance throughout the process"""
    
    async def run_full_pipeline(self) -> dict:
        """Run the complete training data collection pipeline."""
        
        start_time = datetime.now()
        
        # Step 1: Generate conversations
        conversations = await self.generate_conversations()
        
        # Step 2: Run through system
        results = await self.run_conversations(conversations)
        
        # Step 3: Analyze results
        report = self.analyze_results(results)
        
        # Step 4: Extract training data
        training_file = self.extract_training_data(conversations, results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print summary
        summary = {
            "duration_seconds": duration,
            "conversations_generated": len(conversations),
            "results_collected": len(results),
            "success_rate": report.success_rate,
            "workflow_generation_rate": report.workflow_generation_rate,
            "gaps_identified": len(report.gaps),
            "priority_fixes": report.priority_fixes[:5],
            "training_file": str(training_file),
        }
        
        print("\n" + "="*60)
        print("TRAINING DATA COLLECTION COMPLETE")
        print("="*60)
        print(f"\nDuration: {duration:.1f} seconds")
        print(f"Conversations: {len(conversations)}")
        print(f"Success Rate: {report.success_rate:.1%}")
        print(f"Workflow Rate: {report.workflow_generation_rate:.1%}")
        print(f"\nGaps Identified: {len(report.gaps)}")
        for gap in report.gaps[:3]:
            print(f"  - [{gap.severity.upper()}] {gap.category}: {gap.description}")
        print(f"\nTraining data saved to: {training_file}")
        print(f"Analysis report saved to: {self.analysis_dir}")
        
        return summary


async def main():
    parser = argparse.ArgumentParser(
        description="Collect training data by running conversations through BioPipelines"
    )
    
    parser.add_argument(
        "--conversations", "-n",
        type=int,
        default=50,
        help="Number of conversations to generate (default: 50)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/training",
        help="Output directory for training data"
    )
    
    parser.add_argument(
        "--min-turns",
        type=int,
        default=5,
        help="Minimum turns per conversation (default: 5)"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum turns per conversation (default: 15)"
    )
    
    parser.add_argument(
        "--analyze-only",
        type=str,
        default=None,
        help="Only analyze existing results from this path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.analyze_only:
        # Just analyze existing results
        analyzer = ResultAnalyzer(Path(args.output_dir) / "analysis")
        results = analyzer.load_results(Path(args.analyze_only))
        report = analyzer.analyze(results)
        analyzer.save_report(report)
        print(report.to_markdown())
    else:
        # Run full pipeline
        collector = TrainingDataCollector(
            output_dir=Path(args.output_dir),
            num_conversations=args.conversations,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
        )
        
        summary = await collector.run_full_pipeline()
        
        # Save summary
        summary_file = Path(args.output_dir) / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
