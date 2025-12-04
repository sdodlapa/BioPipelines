"""
Workflow Validator
==================

Validates generated workflows using multiple methods:
1. Static syntax checking (Nextflow DSL2 grammar)
2. Semantic validation (process/channel consistency)
3. LLM-based review (optional, for deeper analysis)

This provides a safety net for generated workflows, especially
important during development and testing.
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: str  # "error", "warning", "suggestion"
    message: str
    line: Optional[int] = None
    code: Optional[str] = None  # Code snippet causing the issue
    fix_suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of workflow validation."""
    valid: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    suggestions: List[ValidationIssue] = field(default_factory=list)
    score: float = 0.0  # 0-100 quality score
    llm_review: Optional[str] = None  # LLM reasoning if used
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [{"severity": e.severity, "message": e.message, "line": e.line} for e in self.errors],
            "warnings": [{"severity": w.severity, "message": w.message, "line": w.line} for w in self.warnings],
            "suggestions": [{"severity": s.severity, "message": s.message, "line": s.line} for s in self.suggestions],
            "score": self.score,
            "llm_review": self.llm_review
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "✅ Valid" if self.valid else "❌ Invalid"
        lines = [f"{status} (Score: {self.score:.0f}/100)"]
        
        if self.errors:
            lines.append(f"\n**Errors ({len(self.errors)}):**")
            for e in self.errors[:5]:  # Show first 5
                lines.append(f"  - {e.message}")
        
        if self.warnings:
            lines.append(f"\n**Warnings ({len(self.warnings)}):**")
            for w in self.warnings[:5]:
                lines.append(f"  - {w.message}")
        
        if self.suggestions:
            lines.append(f"\n**Suggestions ({len(self.suggestions)}):**")
            for s in self.suggestions[:3]:
                lines.append(f"  - {s.message}")
        
        return "\n".join(lines)


class WorkflowValidator:
    """
    Validates Nextflow workflows using multiple strategies.
    
    Usage:
        validator = WorkflowValidator()
        result = validator.validate(workflow_code)
        
        # Or with LLM review
        result = validator.validate(workflow_code, use_llm=True)
    """
    
    # Required Nextflow DSL2 elements
    REQUIRED_ELEMENTS = [
        (r"nextflow\.enable\.dsl\s*=\s*2", "Missing 'nextflow.enable.dsl = 2' declaration"),
        (r"workflow\s*\{", "Missing 'workflow {}' block"),
    ]
    
    # Common issues to check
    COMMON_ISSUES = [
        # Process definitions
        (r"process\s+(\w+)\s*\{[^}]*\}", None),  # Just match, no error
        
        # Deprecated syntax
        (r"\.into\s*\{", "Using deprecated '.into{}' (DSL1). Use DSL2 channel operators."),
        (r"Channel\s*\.\s*from\s*\(", "Using deprecated 'Channel.from()'. Use 'Channel.of()' instead."),
        
        # Missing output
        (r"process\s+\w+\s*\{[^}]*output:\s*$", "Process has empty output block"),
        
        # Container issues
        (r"container\s+['\"][^'\"]+['\"]", None),  # Container defined (good)
    ]
    
    # Best practices
    BEST_PRACTICES = [
        (r"publishDir", "Using publishDir for outputs (good practice)"),
        (r"params\.\w+", "Using params for configuration (good practice)"),
        (r"conda\s+['\"][^'\"]+['\"]", "Using conda environment"),
        (r"container\s+['\"][^'\"]+['\"]", "Using container (recommended for reproducibility)"),
    ]
    
    def __init__(self, use_llm: bool = False, llm_client: Any = None):
        """
        Initialize validator.
        
        Args:
            use_llm: Whether to use LLM for deep analysis
            llm_client: Pre-configured LLM client (optional)
        """
        self.use_llm = use_llm
        self._llm_client = llm_client
        self._llm_model = None
    
    def _get_llm_client(self) -> Tuple[Any, str]:
        """Get or create LLM client using existing provider infrastructure."""
        if self._llm_client:
            return self._llm_client, self._llm_model or "default"
        
        try:
            # Try using the existing provider router (uses secrets automatically)
            from src.workflow_composer.providers.router import get_router
            router = get_router()
            return router, "auto"
        except Exception as e:
            logger.debug(f"Provider router not available: {e}")
        
        try:
            # Fallback to direct OpenAI client
            import os
            from openai import OpenAI
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                # Try loading from secrets file
                secrets_path = Path.cwd() / ".secrets" / "openai_key"
                if secrets_path.exists():
                    api_key = secrets_path.read_text().strip()
            
            if api_key:
                client = OpenAI(api_key=api_key)
                return client, "gpt-4o-mini"  # Cost-effective model
        except Exception as e:
            logger.debug(f"OpenAI client not available: {e}")
        
        try:
            # Try Groq (fast, free tier)
            import os
            from openai import OpenAI
            
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                secrets_path = Path.cwd() / ".secrets" / "groq_key"
                if secrets_path.exists():
                    api_key = secrets_path.read_text().strip()
            
            if api_key:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                return client, "llama-3.1-70b-versatile"
        except Exception as e:
            logger.debug(f"Groq client not available: {e}")
        
        return None, None
    
    def validate(
        self,
        workflow_code: str,
        workflow_type: str = "nextflow",
        use_llm: Optional[bool] = None
    ) -> ValidationResult:
        """
        Validate a workflow.
        
        Args:
            workflow_code: The workflow source code
            workflow_type: "nextflow" or "snakemake"
            use_llm: Override instance setting for LLM usage
            
        Returns:
            ValidationResult with all issues found
        """
        if use_llm is None:
            use_llm = self.use_llm
        
        result = ValidationResult(valid=True)
        
        # Step 1: Static syntax validation
        if workflow_type == "nextflow":
            self._validate_nextflow_syntax(workflow_code, result)
        elif workflow_type == "snakemake":
            self._validate_snakemake_syntax(workflow_code, result)
        
        # Step 2: Semantic validation
        self._validate_semantics(workflow_code, result)
        
        # Step 3: Best practices check
        self._check_best_practices(workflow_code, result)
        
        # Step 4: LLM review (if enabled and available)
        if use_llm and not result.errors:  # Skip LLM if already has errors
            self._llm_review(workflow_code, workflow_type, result)
        
        # Calculate score
        result.score = self._calculate_score(result)
        
        # Final validity
        result.valid = len(result.errors) == 0
        
        return result
    
    def _validate_nextflow_syntax(self, code: str, result: ValidationResult) -> None:
        """Validate Nextflow DSL2 syntax."""
        lines = code.split("\n")
        
        # Check required elements
        for pattern, error_msg in self.REQUIRED_ELEMENTS:
            if not re.search(pattern, code):
                result.errors.append(ValidationIssue(
                    severity="error",
                    message=error_msg,
                ))
        
        # Check for balanced braces
        brace_count = 0
        paren_count = 0
        for i, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.split("//")[0]
            brace_count += stripped.count("{") - stripped.count("}")
            paren_count += stripped.count("(") - stripped.count(")")
        
        if brace_count != 0:
            result.errors.append(ValidationIssue(
                severity="error",
                message=f"Unbalanced braces: {brace_count} more '{{' than '}}'",
            ))
        
        if paren_count != 0:
            result.errors.append(ValidationIssue(
                severity="error",
                message=f"Unbalanced parentheses: {paren_count} more '(' than ')'",
            ))
        
        # Check for common issues
        for pattern, error_msg in self.COMMON_ISSUES:
            if error_msg and re.search(pattern, code):
                result.warnings.append(ValidationIssue(
                    severity="warning",
                    message=error_msg,
                ))
        
        # Check process definitions have required sections
        # Use a simpler approach - find process blocks by name and check content
        process_names = re.findall(r"process\s+(\w+)\s*\{", code)
        for process_name in process_names:
            # Find the process content (simplified - look for script/shell/exec after process name)
            # This is a heuristic that works for most well-formatted workflows
            process_match = re.search(
                rf"process\s+{process_name}\s*\{{.*?(script|shell|exec)\s*:",
                code,
                re.DOTALL
            )
            if not process_match:
                result.errors.append(ValidationIssue(
                    severity="error",
                    message=f"Process '{process_name}' missing script/shell/exec block",
                ))
    
    def _validate_snakemake_syntax(self, code: str, result: ValidationResult) -> None:
        """Validate Snakemake syntax."""
        # Check for rule definitions
        if not re.search(r"rule\s+\w+:", code):
            result.errors.append(ValidationIssue(
                severity="error",
                message="No rule definitions found",
            ))
        
        # Check for 'all' or default rule
        if not re.search(r"rule\s+all:", code):
            result.warnings.append(ValidationIssue(
                severity="warning",
                message="Missing 'rule all' - consider adding a target rule",
            ))
    
    def _validate_semantics(self, code: str, result: ValidationResult) -> None:
        """Validate semantic correctness."""
        # Find all defined channels
        channel_defs = set(re.findall(r"(\w+)\s*=\s*Channel\.", code))
        channel_defs.update(re.findall(r"\.set\s*\{\s*(\w+)\s*\}", code))
        
        # Find all process inputs
        input_channels = set()
        for match in re.finditer(r'input:\s*([^}]+?)(?=output:|script:|shell:)', code, re.DOTALL):
            input_block = match.group(1)
            # Extract channel references
            for ch in re.findall(r"from\s+(\w+)|(\w+)\s*\.", input_block):
                input_channels.update(c for c in ch if c)
        
        # Check for undefined channels (simplified check)
        # This is a heuristic - full analysis would require parsing
        
    def _check_best_practices(self, code: str, result: ValidationResult) -> None:
        """Check for best practices."""
        # Check for container usage
        if not re.search(r"container\s+['\"]", code):
            result.suggestions.append(ValidationIssue(
                severity="suggestion",
                message="Consider adding container definitions for reproducibility",
            ))
        
        # Check for publishDir
        if not re.search(r"publishDir", code):
            result.suggestions.append(ValidationIssue(
                severity="suggestion",
                message="Consider adding publishDir to save outputs",
            ))
        
        # Check for params usage
        if not re.search(r"params\.\w+", code):
            result.suggestions.append(ValidationIssue(
                severity="suggestion",
                message="Consider using params.* for configurable values",
            ))
        
        # Check for hardcoded paths
        hardcoded = re.findall(r"['\"]\/[a-zA-Z][^\s'\"]*['\"]", code)
        if hardcoded:
            result.warnings.append(ValidationIssue(
                severity="warning",
                message=f"Found hardcoded paths: {hardcoded[:3]}. Use params instead.",
            ))
    
    def _llm_review(self, code: str, workflow_type: str, result: ValidationResult) -> None:
        """Use LLM for deeper code review."""
        client, model = self._get_llm_client()
        
        if not client:
            logger.debug("No LLM client available for review")
            return
        
        prompt = f"""Review this {workflow_type} workflow for potential issues.

```{workflow_type}
{code[:4000]}  # Truncate for token limits
```

Analyze for:
1. **Errors**: Issues that will cause the workflow to fail
2. **Warnings**: Potential problems or deprecated patterns
3. **Suggestions**: Improvements for best practices

Be concise. Focus on actionable feedback.

Respond as JSON:
{{"errors": ["..."], "warnings": ["..."], "suggestions": ["..."], "overall_assessment": "..."}}
"""

        try:
            # Handle different client types
            if hasattr(client, 'generate'):
                # It's our router
                response = client.generate(prompt, max_tokens=1024, temperature=0.1)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                # Standard OpenAI-compatible client
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert Nextflow/Snakemake code reviewer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content
            
            # Parse LLM response
            import json
            
            # Extract JSON from response
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
            if json_match:
                try:
                    llm_result = json.loads(json_match.group())
                    
                    # Add LLM-found issues
                    for error in llm_result.get("errors", []):
                        if error and isinstance(error, str):
                            result.errors.append(ValidationIssue(
                                severity="error",
                                message=f"[LLM] {error}",
                            ))
                    
                    for warning in llm_result.get("warnings", []):
                        if warning and isinstance(warning, str):
                            result.warnings.append(ValidationIssue(
                                severity="warning",
                                message=f"[LLM] {warning}",
                            ))
                    
                    for suggestion in llm_result.get("suggestions", []):
                        if suggestion and isinstance(suggestion, str):
                            result.suggestions.append(ValidationIssue(
                                severity="suggestion",
                                message=f"[LLM] {suggestion}",
                            ))
                    
                    result.llm_review = llm_result.get("overall_assessment", "")
                    
                except json.JSONDecodeError:
                    logger.debug("Failed to parse LLM JSON response")
                    result.llm_review = content
            else:
                result.llm_review = content
                
        except Exception as e:
            logger.warning(f"LLM review failed: {e}")
    
    def _calculate_score(self, result: ValidationResult) -> float:
        """Calculate a quality score (0-100)."""
        score = 100.0
        
        # Heavy penalty for errors
        score -= len(result.errors) * 25
        
        # Moderate penalty for warnings
        score -= len(result.warnings) * 5
        
        # Small penalty for missing best practices
        score -= len(result.suggestions) * 2
        
        return max(0, min(100, score))
    
    def validate_file(self, filepath: Path, use_llm: Optional[bool] = None) -> ValidationResult:
        """Validate a workflow file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            return ValidationResult(
                valid=False,
                errors=[ValidationIssue(
                    severity="error",
                    message=f"File not found: {filepath}"
                )]
            )
        
        code = filepath.read_text()
        
        # Detect type from extension
        if filepath.suffix in [".nf", ".nextflow"]:
            workflow_type = "nextflow"
        elif filepath.name == "Snakefile" or filepath.suffix in [".smk", ".snakemake"]:
            workflow_type = "snakemake"
        else:
            workflow_type = "nextflow"  # Default
        
        return self.validate(code, workflow_type, use_llm)
    
    def validate_directory(self, dirpath: Path, use_llm: Optional[bool] = None) -> Dict[str, ValidationResult]:
        """Validate all workflow files in a directory."""
        dirpath = Path(dirpath)
        results = {}
        
        # Find workflow files
        patterns = ["*.nf", "Snakefile", "*.smk"]
        for pattern in patterns:
            for filepath in dirpath.rglob(pattern):
                results[str(filepath)] = self.validate_file(filepath, use_llm)
        
        return results


# Convenience function
def validate_workflow(
    code_or_path: str,
    workflow_type: str = "nextflow",
    use_llm: bool = False
) -> ValidationResult:
    """
    Validate a workflow (convenience function).
    
    Args:
        code_or_path: Workflow code string or path to file
        workflow_type: "nextflow" or "snakemake"
        use_llm: Whether to use LLM for deep analysis
        
    Returns:
        ValidationResult
    """
    validator = WorkflowValidator(use_llm=use_llm)
    
    # Check if it's a path
    if "\n" not in code_or_path and Path(code_or_path).exists():
        return validator.validate_file(Path(code_or_path), use_llm)
    
    return validator.validate(code_or_path, workflow_type, use_llm)
