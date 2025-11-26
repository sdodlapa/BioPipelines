"""
Main Error Diagnosis Agent.

Provides AI-powered error diagnosis for bioinformatics workflow failures
using a tiered approach:
1. Pattern matching (fast, offline)
2. Historical learning (boost confidence from past diagnoses)
3. LLM analysis (comprehensive, contextual)
"""

import re
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .categories import (
    ErrorCategory, 
    ErrorDiagnosis, 
    FixSuggestion,
    FixRiskLevel,
)
from .patterns import ERROR_PATTERNS, get_all_patterns
from .log_collector import LogCollector, CollectedLogs
from .prompts import (
    DIAGNOSIS_PROMPT,
    DIAGNOSIS_PROMPT_SIMPLE,
    SYSTEM_PROMPT_DIAGNOSIS,
    build_diagnosis_prompt,
)
from .history import get_diagnosis_history, record_diagnosis, DiagnosisHistory

logger = logging.getLogger(__name__)


# Provider configuration with priority
DIAGNOSIS_PROVIDERS = {
    "lightning": {
        "priority": 1,
        "cost": "free_tier",
        "use_for": ["general_diagnosis"],
        "env_var": "LIGHTNING_API_KEY",
    },
    "gemini": {
        "priority": 2,
        "cost": "free_tier", 
        "use_for": ["quick_analysis"],
        "env_var": "GOOGLE_API_KEY",
    },
    "ollama": {
        "priority": 3,
        "cost": "free",
        "use_for": ["offline_fallback"],
        "env_var": None,
    },
    "vllm": {
        "priority": 4,
        "cost": "free",
        "use_for": ["local_inference"],
        "env_var": "VLLM_API_BASE",
    },
    "openai": {
        "priority": 5,
        "cost": "paid",
        "use_for": ["complex_analysis"],
        "env_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "priority": 6,
        "cost": "paid",
        "use_for": ["complex_analysis"],
        "env_var": "ANTHROPIC_API_KEY",
    },
}


class ErrorDiagnosisAgent:
    """
    AI-powered error diagnosis agent for bioinformatics workflows.
    
    Uses a tiered approach:
    1. Pattern matching (fast, offline) - tries first
    2. LLM analysis (comprehensive) - for complex/unknown errors
    
    Example:
        agent = ErrorDiagnosisAgent()
        diagnosis = await agent.diagnose(failed_job)
        
        print(f"Error: {diagnosis.category}")
        for fix in diagnosis.suggested_fixes:
            print(f"  Fix: {fix.description}")
    """
    
    def __init__(
        self,
        llm=None,
        provider_priority: List[str] = None,
        pattern_confidence_threshold: float = 0.75,
        enable_history: bool = True,
        history_boost_factor: float = 0.1,
    ):
        """
        Initialize the diagnosis agent.
        
        Args:
            llm: Pre-configured LLM adapter (optional)
            provider_priority: Ordered list of LLM providers to try
            pattern_confidence_threshold: Min confidence to use pattern match
            enable_history: Whether to use historical learning
            history_boost_factor: How much to boost confidence from history (0.0-0.2)
        """
        self.llm = llm
        self.provider_priority = provider_priority or [
            "lightning",  # Free tier - priority
            "gemini",     # Free tier
            "ollama",     # Local/free
            "vllm",       # Local/free
            "openai",     # Paid backup
        ]
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.log_collector = LogCollector()
        self._llm_cache = {}
        
        # History-based learning
        self.enable_history = enable_history
        self.history_boost_factor = min(0.2, max(0.0, history_boost_factor))
        self._history: Optional[DiagnosisHistory] = None
        if enable_history:
            try:
                self._history = get_diagnosis_history()
            except Exception as e:
                logger.warning(f"Failed to load diagnosis history: {e}")
    
    def _get_historical_confidence_boost(self, category: ErrorCategory) -> float:
        """
        Get confidence boost based on historical success rate.
        
        Args:
            category: Error category to check
            
        Returns:
            Confidence boost value (0.0 to history_boost_factor)
        """
        if not self._history:
            return 0.0
        
        try:
            success_rates = self._history.get_fix_success_rate()
            cat_success = success_rates.get(category.value, 0.0)
            
            # Also check how often this category appears (frequency)
            category_counts = self._history.get_common_errors(20)
            total = sum(c["count"] for c in category_counts)
            cat_count = next(
                (c["count"] for c in category_counts if c["category"] == category.value),
                0
            )
            frequency_factor = cat_count / total if total > 0 else 0.0
            
            # Boost = success_rate * frequency_factor * boost_factor
            # High success rate + high frequency = more confident
            boost = cat_success * frequency_factor * self.history_boost_factor
            
            if boost > 0:
                logger.debug(
                    f"Historical boost for {category.value}: {boost:.3f} "
                    f"(success={cat_success:.2f}, freq={frequency_factor:.2f})"
                )
            
            return boost
        except Exception as e:
            logger.warning(f"Error calculating historical boost: {e}")
            return 0.0
    
    def _get_historical_fixes(self, category: ErrorCategory) -> List[str]:
        """
        Get fixes that have worked for this category in the past.
        
        Args:
            category: Error category
            
        Returns:
            List of successful fix descriptions
        """
        if not self._history:
            return []
        
        try:
            records = self._history.get_by_category(category.value)
            successful = [
                r.fix_applied for r in records
                if r.fix_success and r.fix_applied
            ]
            # Return unique fixes, most recent first
            seen = set()
            unique = []
            for fix in reversed(successful):
                if fix not in seen:
                    seen.add(fix)
                    unique.append(fix)
            return unique[:5]  # Top 5 successful fixes
        except Exception:
            return []
    
    def _record_diagnosis(self, diagnosis: ErrorDiagnosis, job) -> None:
        """Record diagnosis to history for future learning."""
        if not self._history:
            return
        
        try:
            record_diagnosis(
                diagnosis=diagnosis,
                job_id=getattr(job, 'job_id', 'unknown'),
                workflow_name=getattr(job, 'workflow_name', 'Unknown'),
            )
        except Exception as e:
            logger.warning(f"Failed to record diagnosis: {e}")
    
    def _enhance_with_history(self, diagnosis: ErrorDiagnosis) -> ErrorDiagnosis:
        """
        Enhance diagnosis with historical learning.
        
        Args:
            diagnosis: Original diagnosis
            
        Returns:
            Enhanced diagnosis with boosted confidence and historical fixes
        """
        if not self._history:
            return diagnosis
        
        # Boost confidence based on historical success
        history_boost = self._get_historical_confidence_boost(diagnosis.category)
        boosted_confidence = min(0.99, diagnosis.confidence + history_boost)
        
        # Get historically successful fixes
        historical_fixes = self._get_historical_fixes(diagnosis.category)
        
        # Add note about historical learning
        enhanced_explanation = diagnosis.user_explanation
        if history_boost > 0:
            enhanced_explanation += (
                f"\n\n📊 *Historical confidence boost: +{history_boost:.1%} "
                f"based on past successful diagnoses.*"
            )
        
        if historical_fixes:
            enhanced_explanation += (
                f"\n\n✅ *Previously successful fixes for this error type:*\n"
                + "\n".join(f"  • {fix}" for fix in historical_fixes[:3])
            )
        
        # Create enhanced diagnosis
        return ErrorDiagnosis(
            category=diagnosis.category,
            confidence=boosted_confidence,
            root_cause=diagnosis.root_cause,
            user_explanation=enhanced_explanation,
            log_excerpt=diagnosis.log_excerpt,
            suggested_fixes=diagnosis.suggested_fixes,
            pattern_matched=diagnosis.pattern_matched,
            failed_process=diagnosis.failed_process,
            work_directory=diagnosis.work_directory,
            llm_provider_used=diagnosis.llm_provider_used,
            historical_boost=history_boost,
        )
    
    async def diagnose(self, job) -> ErrorDiagnosis:
        """
        Perform full error diagnosis on a failed job.
        
        Args:
            job: PipelineJob object with failure information
            
        Returns:
            ErrorDiagnosis with root cause and fix suggestions
        """
        # Step 1: Collect all logs
        logger.info(f"Collecting logs for job diagnosis...")
        logs = self.log_collector.collect(job)
        
        if not logs.has_errors():
            return self._create_no_logs_diagnosis()
        
        # Step 2: Try pattern matching first (fast)
        pattern_result = self._match_patterns(logs)
        
        # Step 3: Apply historical learning boost
        if pattern_result and self.enable_history:
            pattern_result = self._enhance_with_history(pattern_result)
        
        if pattern_result and pattern_result.confidence >= self.pattern_confidence_threshold:
            logger.info(
                f"Pattern match found: {pattern_result.category.value} "
                f"(confidence: {pattern_result.confidence:.0%})"
            )
            # Record for future learning
            self._record_diagnosis(pattern_result, job)
            return pattern_result
        
        # Step 4: Use LLM for complex errors
        llm = self._get_available_llm()
        if llm:
            try:
                logger.info(f"Using LLM for deep analysis...")
                llm_result = await self._llm_diagnosis(logs, job, llm)
                if llm_result and llm_result.confidence > 0.5:
                    # Apply historical enhancement to LLM result too
                    if self.enable_history:
                        llm_result = self._enhance_with_history(llm_result)
                    # Record for future learning
                    self._record_diagnosis(llm_result, job)
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM diagnosis failed: {e}")
        
        # Step 5: Return pattern result or unknown
        if pattern_result:
            self._record_diagnosis(pattern_result, job)
            return pattern_result
        
        return self._create_unknown_diagnosis(logs)
    
    def diagnose_sync(self, job) -> ErrorDiagnosis:
        """
        Synchronous wrapper for diagnose().
        
        Args:
            job: PipelineJob object
            
        Returns:
            ErrorDiagnosis
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.diagnose(job))
    
    def diagnose_from_logs(self, log_text: str) -> ErrorDiagnosis:
        """
        Diagnose from raw log text (pattern matching only).
        
        Args:
            log_text: Raw log content
            
        Returns:
            ErrorDiagnosis based on pattern matching
        """
        logs = CollectedLogs(nextflow_log=log_text)
        result = self._match_patterns(logs)
        
        if result:
            return result
        
        return self._create_unknown_diagnosis(logs)
    
    async def diagnose_from_logs_with_llm(self, log_text: str) -> ErrorDiagnosis:
        """
        Full diagnosis from raw log text including LLM analysis.
        
        Args:
            log_text: Raw log content
            
        Returns:
            ErrorDiagnosis with LLM-enhanced analysis if available
        """
        logs = CollectedLogs(nextflow_log=log_text)
        
        # Step 1: Try pattern matching first
        pattern_result = self._match_patterns(logs)
        
        if pattern_result and pattern_result.confidence >= self.pattern_confidence_threshold:
            logger.info(
                f"Pattern match found: {pattern_result.category.value} "
                f"(confidence: {pattern_result.confidence:.0%})"
            )
            # Still enhance with LLM if available for better explanation
            llm = self._get_available_llm()
            if llm:
                try:
                    # Create mock job for LLM context
                    class MockJob:
                        job_id = "log_analysis"
                        workflow_name = "unknown"
                        workflow_dir = "."
                        slurm_job_id = None
                        error_message = log_text[:500]
                    
                    enhanced = await self._llm_diagnosis(logs, MockJob(), llm)
                    if enhanced and enhanced.confidence > pattern_result.confidence:
                        return enhanced
                except Exception as e:
                    logger.warning(f"LLM enhancement failed: {e}")
            
            return pattern_result
        
        # Step 2: Use LLM for complex/unmatched errors
        llm = self._get_available_llm()
        if llm:
            try:
                logger.info(f"Using LLM for deep analysis...")
                class MockJob:
                    job_id = "log_analysis"
                    workflow_name = "unknown"
                    workflow_dir = "."
                    slurm_job_id = None
                    error_message = log_text[:500]
                
                llm_result = await self._llm_diagnosis(logs, MockJob(), llm)
                if llm_result and llm_result.confidence > 0.5:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM diagnosis failed: {e}")
        
        # Step 3: Return pattern result or unknown
        if pattern_result:
            return pattern_result
        
        return self._create_unknown_diagnosis(logs)
    
    def diagnose_from_logs_sync(self, log_text: str, use_llm: bool = False) -> ErrorDiagnosis:
        """
        Synchronous diagnosis from raw log text.
        
        Args:
            log_text: Raw log content
            use_llm: Whether to use LLM for enhanced analysis
            
        Returns:
            ErrorDiagnosis
        """
        if not use_llm:
            return self.diagnose_from_logs(log_text)
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.diagnose_from_logs_with_llm(log_text))
    
    def _match_patterns(self, logs: CollectedLogs) -> Optional[ErrorDiagnosis]:
        """
        Match error patterns against log content.
        
        Args:
            logs: Collected log content
            
        Returns:
            ErrorDiagnosis if pattern matched, None otherwise
        """
        combined_logs = logs.get_full_log_text()
        
        if not combined_logs:
            return None
        
        best_match = None
        best_confidence = 0.0
        matched_text = ""
        match_count = 0
        
        for category, pattern_def in ERROR_PATTERNS.items():
            category_matches = 0
            category_matched_text = ""
            
            for pattern in pattern_def.patterns:
                try:
                    matches = re.findall(pattern, combined_logs, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        category_matches += len(matches)
                        # Store first meaningful match
                        if not category_matched_text:
                            match_val = matches[0]
                            if isinstance(match_val, tuple):
                                match_val = match_val[0] if match_val else ""
                            category_matched_text = str(match_val)[:200]
                except re.error as e:
                    logger.warning(f"Invalid regex pattern: {pattern} - {e}")
                    continue
            
            if category_matches > 0:
                # Calculate confidence based on:
                # - Number of matches (more = more confident)
                # - Number of patterns matched (diversity)
                confidence = min(0.5 + (category_matches * 0.1), 0.95)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = pattern_def
                    matched_text = category_matched_text
                    match_count = category_matches
        
        if best_match:
            # Build diagnosis from pattern
            return ErrorDiagnosis(
                category=best_match.category,
                confidence=best_confidence,
                root_cause=f"{best_match.description}: {matched_text}".strip(": "),
                user_explanation=self._humanize_explanation(best_match, matched_text),
                log_excerpt=matched_text[:500] if matched_text else "",
                suggested_fixes=list(best_match.suggested_fixes),  # Copy list
                pattern_matched=True,
                failed_process=logs.failed_process,
                work_directory=logs.work_directory,
            )
        
        return None
    
    def _humanize_explanation(self, pattern, matched_text: str) -> str:
        """Convert technical error to user-friendly explanation."""
        explanations = {
            ErrorCategory.FILE_NOT_FOUND: (
                f"A required file could not be found. This often happens when "
                f"reference data hasn't been downloaded or the file path is incorrect."
            ),
            ErrorCategory.OUT_OF_MEMORY: (
                f"The analysis ran out of memory. Your input files may be too large "
                f"for the current memory allocation, or too many processes are running."
            ),
            ErrorCategory.CONTAINER_ERROR: (
                f"There's a problem with the software container. The container image "
                f"may not be built or Singularity may not be loaded."
            ),
            ErrorCategory.PERMISSION_DENIED: (
                f"Access to a file or directory was denied. Check file permissions "
                f"and ensure you have write access to the output directory."
            ),
            ErrorCategory.DEPENDENCY_MISSING: (
                f"A required software tool or package is not available. Make sure "
                f"all dependencies are installed in the container or environment."
            ),
            ErrorCategory.SLURM_ERROR: (
                f"The job scheduler (SLURM) reported an error. The job may have "
                f"exceeded time or memory limits, or been cancelled."
            ),
            ErrorCategory.NETWORK_ERROR: (
                f"A network connection failed. This could be due to server issues "
                f"or the compute node not having internet access."
            ),
            ErrorCategory.TOOL_ERROR: (
                f"A bioinformatics tool failed during execution. Check the input "
                f"data format and tool parameters."
            ),
            ErrorCategory.DATA_FORMAT_ERROR: (
                f"The input data format is incorrect or the file is corrupted. "
                f"Verify your input files are in the expected format."
            ),
        }
        
        return explanations.get(pattern.category, pattern.description)
    
    def _get_available_llm(self):
        """Get the first available LLM from priority list."""
        if self.llm:
            return self.llm
        
        # Try to use new unified providers
        try:
            from ..providers import get_provider, check_providers, get_available_providers
            
            available = get_available_providers()
            
            for provider in self.provider_priority:
                if provider in available:
                    # Check cache first
                    if provider in self._llm_cache:
                        return self._llm_cache[provider]
                    
                    try:
                        llm = get_provider(provider)
                        self._llm_cache[provider] = llm
                        logger.info(f"Using LLM provider: {provider}")
                        return llm
                    except Exception as e:
                        logger.debug(f"Failed to initialize {provider}: {e}")
                        continue
        except ImportError:
            # Fallback to old llm module
            try:
                from ..llm import check_providers as llm_check, get_llm
                
                available = llm_check()
                
                for provider in self.provider_priority:
                    if available.get(provider):
                        if provider in self._llm_cache:
                            return self._llm_cache[provider]
                        
                        try:
                            llm = get_llm(provider)
                            self._llm_cache[provider] = llm
                            logger.info(f"Using LLM provider: {provider}")
                            return llm
                        except Exception as e:
                            logger.debug(f"Failed to initialize {provider}: {e}")
                            continue
            except ImportError:
                logger.debug("Neither providers nor llm module available")
        
        return None
    
    async def _llm_diagnosis(
        self, 
        logs: CollectedLogs, 
        job, 
        llm
    ) -> Optional[ErrorDiagnosis]:
        """
        Use LLM for deep error analysis.
        
        Args:
            logs: Collected logs
            job: Job object
            llm: LLM adapter
            
        Returns:
            ErrorDiagnosis or None
        """
        # Try new providers Message, fallback to llm Message
        try:
            from ..providers import Message
        except ImportError:
            try:
                from ..llm import Message
            except ImportError:
                logger.warning("Message class not available")
                return None
        
        # Build context
        workflow_name = getattr(job, 'name', 'Unknown')
        analysis_type = getattr(job, 'analysis_type', 'Unknown')
        
        # Use simple prompt for free tier / faster response
        provider_name = llm.__class__.__name__.lower()
        use_simple = any(p in provider_name for p in ['lightning', 'gemini', 'ollama'])
        
        prompt = build_diagnosis_prompt(
            logs=logs,
            workflow_name=workflow_name,
            analysis_type=analysis_type,
            simple=use_simple,
            error_categories=", ".join([c.value for c in ErrorCategory]),
        )
        
        messages = [
            Message.system(SYSTEM_PROMPT_DIAGNOSIS),
            Message.user(prompt),
        ]
        
        # Get response
        response = llm.chat(messages)
        
        # Parse structured response
        return self._parse_llm_response(response.content, provider_name, logs)
    
    def _parse_llm_response(
        self, 
        content: str, 
        provider: str,
        logs: CollectedLogs
    ) -> Optional[ErrorDiagnosis]:
        """
        Parse LLM response into ErrorDiagnosis.
        
        Args:
            content: LLM response text
            provider: Name of LLM provider
            logs: Original logs for context
            
        Returns:
            ErrorDiagnosis or None
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            logger.warning("No JSON found in LLM response")
            return None
        
        try:
            data = json.loads(json_match.group())
            
            # Map category string to enum
            category_str = data.get('error_category', 'unknown')
            category = ErrorCategory.from_string(category_str)
            
            # Parse fixes
            fixes = []
            for fix_data in data.get('suggested_fixes', []):
                risk_str = fix_data.get('risk_level', 'medium')
                risk = FixRiskLevel.from_string(risk_str)
                
                fixes.append(FixSuggestion(
                    description=fix_data.get('description', ''),
                    command=fix_data.get('command'),
                    risk_level=risk,
                    auto_executable=fix_data.get('auto_executable', False),
                ))
            
            # If no fixes from LLM, get from pattern database
            if not fixes and category in ERROR_PATTERNS:
                fixes = list(ERROR_PATTERNS[category].suggested_fixes)
            
            return ErrorDiagnosis(
                category=category,
                confidence=float(data.get('confidence', 0.7)),
                root_cause=data.get('root_cause', ''),
                user_explanation=data.get('user_explanation', ''),
                log_excerpt=data.get('log_excerpt', '')[:500],
                suggested_fixes=fixes,
                llm_provider_used=provider,
                pattern_matched=False,
                failed_process=logs.failed_process,
                work_directory=logs.work_directory,
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None
    
    def _create_no_logs_diagnosis(self) -> ErrorDiagnosis:
        """Create diagnosis when no logs are available."""
        return ErrorDiagnosis(
            category=ErrorCategory.UNKNOWN,
            confidence=0.0,
            root_cause="No log files found",
            user_explanation=(
                "Unable to find log files for this job. The job may still be "
                "running, or the logs may have been deleted."
            ),
            suggested_fixes=[
                FixSuggestion(
                    description="Check if job is still running",
                    command="squeue -u $USER",
                    risk_level=FixRiskLevel.SAFE,
                    auto_executable=True,
                ),
                FixSuggestion(
                    description="Check job history",
                    command="sacct -j {job_id} --format=JobID,State,ExitCode",
                    risk_level=FixRiskLevel.SAFE,
                    auto_executable=True,
                ),
            ],
        )
    
    def _create_unknown_diagnosis(self, logs: CollectedLogs) -> ErrorDiagnosis:
        """Create diagnosis for unknown errors."""
        return ErrorDiagnosis(
            category=ErrorCategory.UNKNOWN,
            confidence=0.3,
            root_cause="Unable to automatically determine root cause",
            user_explanation=(
                "The error could not be automatically classified. Please review "
                "the log files manually for more details."
            ),
            log_excerpt=logs.get_combined_error_context(30),
            suggested_fixes=[
                FixSuggestion(
                    description="View full Nextflow log",
                    command="cat .nextflow.log | tail -200",
                    risk_level=FixRiskLevel.SAFE,
                    auto_executable=True,
                ),
                FixSuggestion(
                    description="Check work directory for process logs",
                    command=f"ls -la {logs.work_directory}" if logs.work_directory else "ls -la work/",
                    risk_level=FixRiskLevel.SAFE,
                    auto_executable=True,
                ),
            ],
            failed_process=logs.failed_process,
            work_directory=logs.work_directory,
        )


# Convenience function
def diagnose_job(job) -> ErrorDiagnosis:
    """
    Convenience function to diagnose a job.
    
    Args:
        job: PipelineJob or similar object
        
    Returns:
        ErrorDiagnosis
    """
    agent = ErrorDiagnosisAgent()
    return agent.diagnose_sync(job)


def diagnose_log(log_text: str, use_llm: bool = False) -> ErrorDiagnosis:
    """
    Convenience function to diagnose from log text.
    
    Args:
        log_text: Raw log content
        use_llm: Whether to use LLM for enhanced analysis (default: False)
        
    Returns:
        ErrorDiagnosis
    """
    agent = ErrorDiagnosisAgent()
    return agent.diagnose_from_logs_sync(log_text, use_llm=use_llm)
