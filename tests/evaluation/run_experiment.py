#!/usr/bin/env python
"""Simple script to run evaluation experiments."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_runner import ExperimentRunner, ExperimentConfig
from database import EvaluationDatabase

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='baseline', help='Experiment name')
    parser.add_argument('--description', default='', help='Description')
    parser.add_argument('--db', default='tests/evaluation/evaluation.db', help='Database path')
    args = parser.parse_args()
    
    config = ExperimentConfig(
        name=args.name,
        description=args.description,
        max_conversations=0,  # 0 = all conversations
    )
    
    db = EvaluationDatabase(args.db)
    runner = ExperimentRunner(db)
    experiment_id = runner.run_experiment(config)
    
    # Get results from database
    exp = runner.db.get_experiment(experiment_id)
    
    print('=' * 60)
    print('RESULTS:')
    print(f'  Experiment ID: {experiment_id}')
    print(f'  Pass Rate: {exp.passed_conversations / exp.total_conversations * 100:.1f}%')
    print(f'  Intent Accuracy: {exp.overall_intent_accuracy*100:.1f}%')
    print(f'  Entity F1: {exp.overall_entity_f1*100:.1f}%')
    print(f'  Tool Accuracy: {exp.overall_tool_accuracy*100:.1f}%')
    print(f'  Avg Latency: {exp.avg_latency_ms:.1f}ms')
    print('=' * 60)

if __name__ == '__main__':
    main()
