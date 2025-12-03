"""
Result Analyzer
===============

Analyzes conversation run results to identify:
1. System gaps and weaknesses
2. Common error patterns
3. Performance bottlenecks
4. Missing capabilities
5. Training data quality issues
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


@dataclass
class GapReport:
    """Report on identified gaps."""
    
    category: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "frequency": self.frequency,
            "examples": self.examples[:3],  # Limit examples
            "recommendation": self.recommendation,
        }


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Summary stats
    total_conversations: int = 0
    total_turns: int = 0
    success_rate: float = 0.0
    workflow_generation_rate: float = 0.0
    
    # Gaps
    gaps: List[GapReport] = field(default_factory=list)
    
    # Performance
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    slow_operations: List[Dict] = field(default_factory=list)
    
    # Coverage
    analysis_type_coverage: Dict[str, float] = field(default_factory=dict)
    tool_coverage: Dict[str, int] = field(default_factory=dict)
    missing_capabilities: List[str] = field(default_factory=list)
    
    # Error patterns
    error_patterns: Dict[str, int] = field(default_factory=dict)
    common_warnings: Dict[str, int] = field(default_factory=dict)
    
    # Recommendations
    priority_fixes: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "summary": {
                "total_conversations": self.total_conversations,
                "total_turns": self.total_turns,
                "success_rate": self.success_rate,
                "workflow_generation_rate": self.workflow_generation_rate,
            },
            "gaps": [g.to_dict() for g in self.gaps],
            "performance": {
                "avg_latency_ms": self.avg_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "slow_operations": self.slow_operations[:5],
            },
            "coverage": {
                "analysis_types": self.analysis_type_coverage,
                "tools": self.tool_coverage,
                "missing": self.missing_capabilities,
            },
            "errors": {
                "patterns": self.error_patterns,
                "warnings": self.common_warnings,
            },
            "recommendations": {
                "priority_fixes": self.priority_fixes,
                "improvements": self.improvement_areas,
            },
        }
        return data
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        
        lines = [
            "# BioPipelines Conversation Analysis Report",
            f"\n*Generated: {self.timestamp}*\n",
            
            "## Summary",
            f"- **Total Conversations**: {self.total_conversations}",
            f"- **Total Turns**: {self.total_turns}",
            f"- **Success Rate**: {self.success_rate:.1%}",
            f"- **Workflow Generation Rate**: {self.workflow_generation_rate:.1%}",
            
            "\n## Performance",
            f"- **Average Latency**: {self.avg_latency_ms:.0f}ms",
            f"- **95th Percentile**: {self.p95_latency_ms:.0f}ms",
        ]
        
        if self.gaps:
            lines.append("\n## Identified Gaps\n")
            for gap in sorted(self.gaps, key=lambda g: 
                              {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(g.severity, 4)):
                severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(gap.severity, "⚪")
                lines.append(f"### {severity_emoji} {gap.category}: {gap.description}")
                lines.append(f"- **Severity**: {gap.severity}")
                lines.append(f"- **Frequency**: {gap.frequency} occurrences")
                if gap.recommendation:
                    lines.append(f"- **Recommendation**: {gap.recommendation}")
                if gap.examples:
                    lines.append("- **Examples**:")
                    for ex in gap.examples[:2]:
                        lines.append(f"  - {ex[:100]}...")
                lines.append("")
        
        if self.error_patterns:
            lines.append("\n## Error Patterns\n")
            lines.append("| Error Type | Count |")
            lines.append("|------------|-------|")
            for error, count in sorted(self.error_patterns.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"| {error} | {count} |")
        
        if self.priority_fixes:
            lines.append("\n## Priority Fixes\n")
            for i, fix in enumerate(self.priority_fixes, 1):
                lines.append(f"{i}. {fix}")
        
        if self.improvement_areas:
            lines.append("\n## Improvement Areas\n")
            for area in self.improvement_areas:
                lines.append(f"- {area}")
        
        return "\n".join(lines)


class ResultAnalyzer:
    """Analyzes conversation results to identify gaps."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("data/training/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Known analysis types and tools for coverage calculation
        self._load_known_capabilities()
    
    def _load_known_capabilities(self):
        """Load known analysis types and tools."""
        
        self.known_analysis_types = set()
        self.known_tools = set()
        
        try:
            from ..core.query_parser import AnalysisType
            self.known_analysis_types = {at.value for at in AnalysisType}
        except ImportError:
            self.known_analysis_types = {
                "rna_seq", "chip_seq", "atac_seq", "dna_seq", "variant_calling",
                "methylation", "metagenomics", "scrna_seq", "long_read",
            }
        
        try:
            from ..agents.rag.tool_catalog_indexer import TOOL_DESCRIPTIONS
            self.known_tools = set(TOOL_DESCRIPTIONS.keys())
        except ImportError:
            self.known_tools = {
                "fastp", "star", "bwa", "gatk", "deseq2", "macs2",
                "bowtie2", "samtools", "bcftools", "deeptools",
            }
    
    def load_results(self, results_path: Path) -> List[Dict]:
        """Load results from file."""
        
        results = []
        
        if results_path.suffix == '.jsonl':
            with open(results_path) as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        elif results_path.suffix == '.json':
            with open(results_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results = data
                else:
                    results = [data]
        else:
            # Try to load all .json and .jsonl in directory
            for file in results_path.glob("*.jsonl"):
                results.extend(self.load_results(file))
            for file in results_path.glob("result_*.json"):
                results.extend(self.load_results(file))
        
        return results
    
    def analyze(self, results: List[Dict]) -> AnalysisReport:
        """Analyze results and generate report."""
        
        report = AnalysisReport()
        report.total_conversations = len(results)
        
        # Collect data for analysis
        all_turns = []
        all_errors = []
        all_warnings = []
        all_gaps = []
        all_latencies = []
        tools_used = Counter()
        analysis_types_seen = Counter()
        
        successful = 0
        workflow_generated = 0
        
        for result in results:
            # Count successes
            if result.get('completed_successfully'):
                successful += 1
            if result.get('workflow_generated'):
                workflow_generated += 1
            
            # Track analysis type
            analysis_types_seen[result.get('analysis_type', 'unknown')] += 1
            
            # Collect turn data
            for turn in result.get('turn_results', []):
                all_turns.append(turn)
                
                # Latency
                latency = turn.get('latency_ms', 0)
                if latency > 0:
                    all_latencies.append(latency)
                
                # Tools
                for tool in turn.get('tools_selected', []):
                    tools_used[tool.lower()] += 1
                
                # Errors
                if not turn.get('success', True):
                    all_errors.append({
                        'type': turn.get('error_type', 'Unknown'),
                        'message': turn.get('error_message', ''),
                        'conversation': result.get('conversation_id'),
                    })
                
                # Warnings
                for warning in turn.get('warnings', []):
                    all_warnings.append(warning)
            
            # Collect gaps
            for gap in result.get('gaps_identified', []):
                all_gaps.append(gap)
            
            # Collect errors from result level
            for error in result.get('errors', []):
                all_errors.append(error)
        
        # Calculate summary stats
        report.total_turns = len(all_turns)
        report.success_rate = successful / len(results) if results else 0
        report.workflow_generation_rate = workflow_generated / len(results) if results else 0
        
        # Performance stats
        if all_latencies:
            all_latencies.sort()
            report.avg_latency_ms = sum(all_latencies) / len(all_latencies)
            p95_idx = int(len(all_latencies) * 0.95)
            report.p95_latency_ms = all_latencies[p95_idx] if p95_idx < len(all_latencies) else all_latencies[-1]
            
            # Find slow operations
            slow_threshold = 3000  # 3 seconds
            for turn in all_turns:
                if turn.get('latency_ms', 0) > slow_threshold:
                    report.slow_operations.append({
                        'turn': turn.get('turn_index'),
                        'latency_ms': turn.get('latency_ms'),
                        'message': turn.get('user_message', '')[:50],
                    })
        
        # Error patterns
        error_types = Counter()
        for error in all_errors:
            etype = error.get('type') or error.get('error_type', 'Unknown')
            error_types[etype] += 1
        report.error_patterns = dict(error_types)
        
        # Warning patterns
        warning_patterns = Counter()
        for warning in all_warnings:
            # Normalize warnings
            normalized = re.sub(r'\d+', 'N', warning)  # Replace numbers
            normalized = normalized[:50]  # Truncate
            warning_patterns[normalized] += 1
        report.common_warnings = dict(warning_patterns.most_common(10))
        
        # Coverage analysis
        for analysis_type in self.known_analysis_types:
            seen = analysis_types_seen.get(analysis_type, 0)
            report.analysis_type_coverage[analysis_type] = seen / len(results) if results else 0
        
        report.tool_coverage = dict(tools_used.most_common(20))
        
        # Find missing capabilities
        seen_tools = set(tools_used.keys())
        missing = self.known_tools - seen_tools
        report.missing_capabilities = list(missing)[:10]
        
        # Identify gaps
        report.gaps = self._identify_gaps(results, all_errors, all_warnings, all_gaps)
        
        # Generate recommendations
        report.priority_fixes, report.improvement_areas = self._generate_recommendations(report)
        
        return report
    
    def _identify_gaps(
        self,
        results: List[Dict],
        errors: List[Dict],
        warnings: List[str],
        gaps: List[str],
    ) -> List[GapReport]:
        """Identify and categorize gaps."""
        
        gap_reports = []
        
        # Gap 1: Intent parsing issues
        intent_issues = [w for w in warnings if 'intent' in w.lower()]
        if intent_issues:
            gap_reports.append(GapReport(
                category="Intent Parsing",
                severity="high" if len(intent_issues) > 5 else "medium",
                description="Intent parsing failures or low confidence",
                frequency=len(intent_issues),
                examples=intent_issues[:3],
                recommendation="Review intent parser coverage and add more patterns",
            ))
        
        # Gap 2: Tool selection issues
        tool_issues = [w for w in warnings if 'tool' in w.lower()]
        if tool_issues:
            gap_reports.append(GapReport(
                category="Tool Selection",
                severity="medium",
                description="Tool selection warnings or failures",
                frequency=len(tool_issues),
                examples=tool_issues[:3],
                recommendation="Expand tool mappings and improve selection logic",
            ))
        
        # Gap 3: Workflow generation failures
        workflow_issues = [w for w in warnings if 'workflow' in w.lower()]
        workflow_failures = [r for r in results if not r.get('workflow_generated')]
        if workflow_failures:
            gap_reports.append(GapReport(
                category="Workflow Generation",
                severity="critical" if len(workflow_failures) > len(results) * 0.3 else "high",
                description=f"Failed to generate workflow for {len(workflow_failures)} conversations",
                frequency=len(workflow_failures),
                examples=[r.get('analysis_type', 'unknown') for r in workflow_failures[:3]],
                recommendation="Review workflow generator templates and error handling",
            ))
        
        # Gap 4: Timeout issues
        timeout_issues = [w for w in warnings if 'timeout' in w.lower()]
        if timeout_issues:
            gap_reports.append(GapReport(
                category="Performance",
                severity="high",
                description="Operations timing out",
                frequency=len(timeout_issues),
                examples=timeout_issues[:3],
                recommendation="Optimize slow operations or increase timeouts",
            ))
        
        # Gap 5: Missing analysis types
        analysis_types = Counter(r.get('analysis_type', 'unknown') for r in results)
        common_types = {"rna_seq", "chip_seq", "variant_calling", "atac_seq"}
        missing_types = common_types - set(analysis_types.keys())
        if missing_types:
            gap_reports.append(GapReport(
                category="Coverage",
                severity="medium",
                description=f"No test coverage for analysis types: {', '.join(missing_types)}",
                frequency=len(missing_types),
                examples=list(missing_types),
                recommendation="Add test scenarios for missing analysis types",
            ))
        
        # Gap 6: Error recovery
        error_conversations = [r for r in results if r.get('errors')]
        if error_conversations:
            gap_reports.append(GapReport(
                category="Error Handling",
                severity="high" if len(error_conversations) > 5 else "medium",
                description="Conversations with unhandled errors",
                frequency=len(error_conversations),
                examples=[e.get('message', '')[:50] for r in error_conversations[:3] for e in r.get('errors', [])[:1]],
                recommendation="Improve error handling and recovery logic",
            ))
        
        # Gap 7: Low confidence responses
        low_confidence = sum(1 for g in gaps if 'confidence' in g.lower())
        if low_confidence > 0:
            gap_reports.append(GapReport(
                category="Response Quality",
                severity="medium",
                description="Low confidence in intent or response",
                frequency=low_confidence,
                examples=[g for g in gaps if 'confidence' in g.lower()][:3],
                recommendation="Train on more examples or improve parsing rules",
            ))
        
        return gap_reports
    
    def _generate_recommendations(
        self,
        report: AnalysisReport,
    ) -> Tuple[List[str], List[str]]:
        """Generate prioritized recommendations."""
        
        priority_fixes = []
        improvements = []
        
        # Priority based on severity
        critical_gaps = [g for g in report.gaps if g.severity == "critical"]
        high_gaps = [g for g in report.gaps if g.severity == "high"]
        
        for gap in critical_gaps:
            priority_fixes.append(f"[CRITICAL] {gap.category}: {gap.recommendation}")
        
        for gap in high_gaps:
            priority_fixes.append(f"[HIGH] {gap.category}: {gap.recommendation}")
        
        # Success rate based recommendations
        if report.success_rate < 0.7:
            priority_fixes.append(
                "[HIGH] Overall success rate below 70% - review core pipeline components"
            )
        
        if report.workflow_generation_rate < 0.5:
            priority_fixes.append(
                "[HIGH] Workflow generation rate below 50% - check generator templates"
            )
        
        # Performance recommendations
        if report.avg_latency_ms > 2000:
            improvements.append(
                "Average latency is high - consider caching or optimization"
            )
        
        if report.p95_latency_ms > 10000:
            improvements.append(
                "P95 latency exceeds 10s - investigate slow operations"
            )
        
        # Coverage recommendations
        if report.missing_capabilities:
            improvements.append(
                f"Add support for tools: {', '.join(report.missing_capabilities[:5])}"
            )
        
        low_coverage = [
            at for at, cov in report.analysis_type_coverage.items() 
            if cov < 0.1 and at in {"rna_seq", "chip_seq", "variant_calling"}
        ]
        if low_coverage:
            improvements.append(
                f"Increase test coverage for: {', '.join(low_coverage)}"
            )
        
        # Error pattern recommendations
        if report.error_patterns:
            top_error = max(report.error_patterns.items(), key=lambda x: x[1])
            improvements.append(
                f"Most common error '{top_error[0]}' ({top_error[1]} occurrences) - investigate root cause"
            )
        
        return priority_fixes, improvements
    
    def save_report(self, report: AnalysisReport) -> Tuple[Path, Path]:
        """Save analysis report."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_path = self.output_dir / f"analysis_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Save Markdown
        md_path = self.output_dir / f"analysis_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(report.to_markdown())
        
        logger.info(f"Saved analysis to {json_path} and {md_path}")
        
        return json_path, md_path


def analyze_results(
    results_path: Path,
    output_dir: Path = None,
) -> AnalysisReport:
    """Convenience function to analyze results."""
    
    analyzer = ResultAnalyzer(output_dir)
    results = analyzer.load_results(results_path)
    report = analyzer.analyze(results)
    analyzer.save_report(report)
    
    return report


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
        report = analyze_results(results_path)
        print(report.to_markdown())
    else:
        print("Usage: python -m workflow_composer.training.result_analyzer <results_path>")
