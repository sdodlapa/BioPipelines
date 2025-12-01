"""
Quality Control Agent
=====================

Validates workflow outputs and quality metrics.

Checks:
- Expected outputs exist
- QC metrics are within acceptable ranges
- Sample tracking consistency
- MultiQC report analysis
"""

import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QCMetric:
    """A QC metric measurement."""
    name: str
    value: float
    unit: str = ""
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    status: str = "unknown"  # pass, warn, fail, unknown
    
    def evaluate(self) -> str:
        """Evaluate metric against thresholds."""
        if self.threshold_min is not None and self.value < self.threshold_min:
            self.status = "fail" if self.value < self.threshold_min * 0.8 else "warn"
        elif self.threshold_max is not None and self.value > self.threshold_max:
            self.status = "fail" if self.value > self.threshold_max * 1.2 else "warn"
        else:
            self.status = "pass"
        return self.status


@dataclass
class QCReport:
    """Quality control report for a workflow run."""
    sample_id: str
    metrics: List[QCMetric] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    passed: bool = True
    
    def add_metric(self, metric: QCMetric):
        """Add a metric and evaluate."""
        metric.evaluate()
        self.metrics.append(metric)
        
        if metric.status == "fail":
            self.errors.append(f"{metric.name}: {metric.value} {metric.unit} (threshold: {metric.threshold_min}-{metric.threshold_max})")
            self.passed = False
        elif metric.status == "warn":
            self.warnings.append(f"{metric.name}: {metric.value} {metric.unit} approaching limits")


class QCAgent:
    """
    Validates workflow outputs and quality metrics.
    
    Responsibilities:
    - Check expected output files exist
    - Parse and evaluate QC metrics
    - Analyze MultiQC reports
    - Flag samples with issues
    """
    
    # Default QC thresholds by analysis type
    DEFAULT_THRESHOLDS = {
        "rna-seq": {
            "total_reads": {"min": 10_000_000},
            "mapping_rate": {"min": 70.0, "max": 100.0},
            "percent_duplicates": {"max": 60.0},
            "percent_gc": {"min": 30.0, "max": 70.0},
            "percent_exonic": {"min": 50.0},
            "ribosomal_fraction": {"max": 10.0},
        },
        "chip-seq": {
            "total_reads": {"min": 10_000_000},
            "mapping_rate": {"min": 70.0},
            "percent_duplicates": {"max": 40.0},
            "frip": {"min": 0.01},  # Fraction of reads in peaks
            "nrf": {"min": 0.8},    # Non-redundant fraction
        },
        "atac-seq": {
            "total_reads": {"min": 25_000_000},
            "mapping_rate": {"min": 80.0},
            "percent_mitochondrial": {"max": 20.0},
            "tss_enrichment": {"min": 5.0},
        },
        "wgs": {
            "total_reads": {"min": 300_000_000},
            "mapping_rate": {"min": 95.0},
            "mean_coverage": {"min": 30.0},
            "percent_duplicates": {"max": 20.0},
        },
        "wes": {
            "total_reads": {"min": 50_000_000},
            "mapping_rate": {"min": 95.0},
            "mean_coverage": {"min": 50.0},
            "on_target_rate": {"min": 60.0},
        },
    }

    def __init__(self, analysis_type: str = "rna-seq", custom_thresholds: Dict = None):
        """
        Initialize QC agent.
        
        Args:
            analysis_type: Type of analysis for threshold selection
            custom_thresholds: Override default thresholds
        """
        self.analysis_type = analysis_type
        self.thresholds = self.DEFAULT_THRESHOLDS.get(analysis_type, {})
        
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
    
    def validate_outputs(self, output_dir: str, expected_files: List[str] = None) -> Dict[str, bool]:
        """
        Check that expected output files exist.
        
        Args:
            output_dir: Path to output directory
            expected_files: List of expected file patterns
            
        Returns:
            Dict mapping file patterns to existence status
        """
        output_path = Path(output_dir)
        results = {}
        
        # Default expected files by analysis type
        if expected_files is None:
            expected_files = self._get_expected_files()
        
        for pattern in expected_files:
            matches = list(output_path.glob(pattern))
            results[pattern] = len(matches) > 0
        
        return results
    
    def _get_expected_files(self) -> List[str]:
        """Get expected output files for analysis type."""
        common = [
            "**/multiqc_report.html",
            "**/pipeline_report.html",
        ]
        
        type_specific = {
            "rna-seq": [
                "**/*.bam",
                "**/counts*.txt",
                "**/*fastqc*.html",
            ],
            "chip-seq": [
                "**/*.bam",
                "**/*.narrowPeak",
                "**/*.bw",
            ],
            "atac-seq": [
                "**/*.bam",
                "**/*.narrowPeak",
                "**/*.bw",
            ],
            "wgs": [
                "**/*.bam",
                "**/*.vcf.gz",
            ],
            "wes": [
                "**/*.bam",
                "**/*.vcf.gz",
            ],
        }
        
        return common + type_specific.get(self.analysis_type, [])
    
    def parse_fastqc(self, fastqc_data: str, sample_id: str = "sample") -> QCReport:
        """
        Parse FastQC data file.
        
        Args:
            fastqc_data: Content of fastqc_data.txt
            sample_id: Sample identifier
            
        Returns:
            QCReport with extracted metrics
        """
        report = QCReport(sample_id=sample_id)
        
        # Extract total sequences
        match = re.search(r'Total Sequences\s+(\d+)', fastqc_data)
        if match:
            report.add_metric(QCMetric(
                name="total_reads",
                value=int(match.group(1)),
                threshold_min=self.thresholds.get("total_reads", {}).get("min")
            ))
        
        # Extract %GC
        match = re.search(r'%GC\s+(\d+)', fastqc_data)
        if match:
            thresholds = self.thresholds.get("percent_gc", {})
            report.add_metric(QCMetric(
                name="percent_gc",
                value=float(match.group(1)),
                unit="%",
                threshold_min=thresholds.get("min"),
                threshold_max=thresholds.get("max")
            ))
        
        # Check module status
        for line in fastqc_data.split('\n'):
            if line.startswith('>>') and line.endswith('fail'):
                module_name = line.split('\t')[0].replace('>>', '')
                report.warnings.append(f"FastQC {module_name} failed")
        
        return report
    
    def parse_star_log(self, log_content: str, sample_id: str = "sample") -> QCReport:
        """
        Parse STAR aligner final log.
        
        Args:
            log_content: Content of Log.final.out
            sample_id: Sample identifier
            
        Returns:
            QCReport with alignment metrics
        """
        report = QCReport(sample_id=sample_id)
        
        # Input reads
        match = re.search(r'Number of input reads\s+\|\s+(\d+)', log_content)
        if match:
            report.add_metric(QCMetric(
                name="input_reads",
                value=int(match.group(1))
            ))
        
        # Uniquely mapped
        match = re.search(r'Uniquely mapped reads %\s+\|\s+([\d.]+)', log_content)
        if match:
            thresholds = self.thresholds.get("mapping_rate", {})
            report.add_metric(QCMetric(
                name="mapping_rate",
                value=float(match.group(1)),
                unit="%",
                threshold_min=thresholds.get("min"),
                threshold_max=thresholds.get("max")
            ))
        
        # Multi-mapped
        match = re.search(r'% of reads mapped to multiple loci\s+\|\s+([\d.]+)', log_content)
        if match:
            report.add_metric(QCMetric(
                name="multi_mapped_rate",
                value=float(match.group(1)),
                unit="%",
                threshold_max=30.0
            ))
        
        return report
    
    def parse_picard_metrics(self, metrics_content: str, sample_id: str = "sample") -> QCReport:
        """
        Parse Picard MarkDuplicates metrics.
        
        Args:
            metrics_content: Content of metrics file
            sample_id: Sample identifier
            
        Returns:
            QCReport with duplication metrics
        """
        report = QCReport(sample_id=sample_id)
        
        # Find the metrics line (after METRICS CLASS header)
        lines = metrics_content.strip().split('\n')
        header_idx = None
        
        for i, line in enumerate(lines):
            if line.startswith('LIBRARY'):
                header_idx = i
                break
        
        if header_idx and len(lines) > header_idx + 1:
            headers = lines[header_idx].split('\t')
            values = lines[header_idx + 1].split('\t')
            
            # Find PERCENT_DUPLICATION
            if 'PERCENT_DUPLICATION' in headers:
                idx = headers.index('PERCENT_DUPLICATION')
                if idx < len(values):
                    thresholds = self.thresholds.get("percent_duplicates", {})
                    report.add_metric(QCMetric(
                        name="percent_duplicates",
                        value=float(values[idx]) * 100,  # Convert to percentage
                        unit="%",
                        threshold_max=thresholds.get("max")
                    ))
        
        return report
    
    def parse_multiqc_data(self, multiqc_data: Dict[str, Any]) -> List[QCReport]:
        """
        Parse MultiQC general stats data.
        
        Args:
            multiqc_data: Parsed multiqc_data.json
            
        Returns:
            List of QCReports, one per sample
        """
        reports = []
        
        general_stats = multiqc_data.get('report_general_stats_data', [])
        
        for sample_data in general_stats:
            for sample_id, metrics in sample_data.items():
                report = QCReport(sample_id=sample_id)
                
                # Map MultiQC metrics to our format
                metric_mapping = {
                    'total_reads': 'total_reads',
                    'percent_aligned': 'mapping_rate',
                    'percent_gc': 'percent_gc',
                    'percent_duplicates': 'percent_duplicates',
                }
                
                for mqc_key, our_key in metric_mapping.items():
                    if mqc_key in metrics:
                        threshold = self.thresholds.get(our_key, {})
                        report.add_metric(QCMetric(
                            name=our_key,
                            value=float(metrics[mqc_key]),
                            unit="%" if "percent" in our_key else "",
                            threshold_min=threshold.get("min"),
                            threshold_max=threshold.get("max")
                        ))
                
                reports.append(report)
        
        return reports
    
    def generate_qc_summary(self, reports: List[QCReport]) -> str:
        """
        Generate QC summary report.
        
        Args:
            reports: List of QCReports
            
        Returns:
            Markdown summary
        """
        lines = [
            "# Quality Control Summary",
            "",
            f"**Analysis Type:** {self.analysis_type}",
            f"**Samples Analyzed:** {len(reports)}",
            f"**Samples Passed:** {sum(1 for r in reports if r.passed)}",
            f"**Samples with Warnings:** {sum(1 for r in reports if r.warnings)}",
            f"**Samples Failed:** {sum(1 for r in reports if not r.passed)}",
            "",
            "## Sample Status",
            "",
            "| Sample | Status | Warnings | Errors |",
            "|--------|--------|----------|--------|",
        ]
        
        for report in reports:
            status = "✅ PASS" if report.passed else "❌ FAIL"
            warnings = len(report.warnings)
            errors = len(report.errors)
            lines.append(f"| {report.sample_id} | {status} | {warnings} | {errors} |")
        
        # Add metrics summary
        if reports:
            lines.extend([
                "",
                "## Metrics Summary",
                "",
                "| Metric | Min | Mean | Max | Threshold |",
                "|--------|-----|------|-----|-----------|",
            ])
            
            # Aggregate metrics across samples
            metric_values = {}
            for report in reports:
                for metric in report.metrics:
                    if metric.name not in metric_values:
                        metric_values[metric.name] = []
                    metric_values[metric.name].append(metric.value)
            
            for name, values in metric_values.items():
                if values:
                    threshold = self.thresholds.get(name, {})
                    threshold_str = ""
                    if "min" in threshold:
                        threshold_str = f"≥{threshold['min']}"
                    if "max" in threshold:
                        threshold_str += f" ≤{threshold['max']}" if threshold_str else f"≤{threshold['max']}"
                    
                    lines.append(
                        f"| {name} | {min(values):.2f} | {sum(values)/len(values):.2f} | {max(values):.2f} | {threshold_str} |"
                    )
        
        # Add warnings and errors
        all_warnings = []
        all_errors = []
        for report in reports:
            all_warnings.extend([f"{report.sample_id}: {w}" for w in report.warnings])
            all_errors.extend([f"{report.sample_id}: {e}" for e in report.errors])
        
        if all_warnings:
            lines.extend([
                "",
                "## Warnings",
                "",
            ])
            for w in all_warnings[:20]:  # Limit to 20
                lines.append(f"- ⚠️ {w}")
            if len(all_warnings) > 20:
                lines.append(f"- ... and {len(all_warnings) - 20} more")
        
        if all_errors:
            lines.extend([
                "",
                "## Errors",
                "",
            ])
            for e in all_errors[:20]:
                lines.append(f"- ❌ {e}")
            if len(all_errors) > 20:
                lines.append(f"- ... and {len(all_errors) - 20} more")
        
        return "\n".join(lines)
    
    def check_sample_consistency(self, sample_ids: List[str], output_dir: str) -> Dict[str, List[str]]:
        """
        Check that all samples have consistent outputs.
        
        Args:
            sample_ids: List of expected sample IDs
            output_dir: Path to output directory
            
        Returns:
            Dict mapping samples to missing outputs
        """
        output_path = Path(output_dir)
        missing = {}
        
        for sample_id in sample_ids:
            sample_missing = []
            
            # Check for expected per-sample files
            expected_patterns = [
                f"**/{sample_id}*.bam",
                f"**/{sample_id}*fastqc*",
            ]
            
            for pattern in expected_patterns:
                if not list(output_path.glob(pattern)):
                    sample_missing.append(pattern.replace("**", "..."))
            
            if sample_missing:
                missing[sample_id] = sample_missing
        
        return missing
