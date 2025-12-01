"""
Validator Agent
===============

Validates generated Nextflow code using both static analysis and LLM review.

Checks:
- Syntax correctness
- DSL2 compliance
- Resource requirements
- Container specifications
- Channel flow integrity
"""

import logging
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _ValidationIssue:
    """A single validation issue (internal use)."""
    severity: str  # 'error', 'warning', 'info'
    message: str
    line: Optional[int] = None
    rule: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class ValidatorAgent:
    """
    Validates Nextflow code for correctness and best practices.
    
    Uses:
    - Static analysis for syntax and structure
    - LLM review for semantic correctness
    - nf-core lint rules for best practices
    """
    
    SYSTEM_PROMPT = """You are a Nextflow code reviewer.
Analyze the code for:
1. Correctness: Will it run without errors?
2. Best practices: Does it follow nf-core standards?
3. Resource usage: Are resources properly specified?
4. Reproducibility: Are containers and versions specified?

Report issues as JSON:
{
  "issues": [
    {"severity": "error|warning|info", "message": "...", "line": N, "rule": "..."}
  ],
  "suggestions": ["..."]
}"""

    # Validation rules
    REQUIRED_PATTERNS = [
        (r'nextflow\.enable\.dsl\s*=\s*2', 'DSL2 declaration required'),
        (r'process\s+\w+\s*{', 'At least one process required'),
        (r'workflow\s*{', 'Workflow block required'),
    ]
    
    RECOMMENDED_PATTERNS = [
        (r'container\s+[\'"]', 'Processes should specify containers'),
        (r'errorStrategy', 'Consider adding errorStrategy directive'),
        (r'publishDir', 'Consider adding publishDir for outputs'),
        (r'cpus\s+\d+|label\s+[\'"]process_', 'Resource specifications recommended'),
    ]
    
    PROBLEMATIC_PATTERNS = [
        (r'file\s*\(\s*[\'"]', 'Use path() instead of file() for inputs', 'warning'),
        (r'Channel\.from\s*\(', 'Channel.from is deprecated, use Channel.of', 'warning'),
        (r'set\s*\{\s*\w+\s*\}', 'Prefer collect() over set{}', 'info'),
    ]

    def __init__(self, router=None):
        """
        Initialize validator agent.
        
        Args:
            router: LLM provider router for semantic validation
        """
        self.router = router
    
    async def validate(self, code: str) -> ValidationResult:
        """
        Validate Nextflow code.
        
        Args:
            code: Nextflow code to validate
            
        Returns:
            ValidationResult with issues and success status
        """
        issues = []
        
        # Static analysis
        static_issues = self._static_analysis(code)
        issues.extend(static_issues)
        
        # LLM semantic review
        if self.router:
            try:
                llm_issues = await self._llm_review(code)
                issues.extend(llm_issues)
            except Exception as e:
                logger.warning(f"LLM review failed: {e}")
        
        # Determine success
        errors = [i for i in issues if i.severity == 'error']
        warnings = [i for i in issues if i.severity == 'warning']
        
        return ValidationResult(
            valid=len(errors) == 0,
            issues=[i.message for i in issues],
            warnings=[i.message for i in warnings],
            suggestions=self._generate_suggestions(code, issues)
        )
    
    def validate_sync(self, code: str) -> ValidationResult:
        """Synchronous validation using static analysis only."""
        issues = self._static_analysis(code)
        errors = [i for i in issues if i.severity == 'error']
        warnings = [i for i in issues if i.severity == 'warning']
        
        return ValidationResult(
            valid=len(errors) == 0,
            issues=[i.message for i in issues],
            warnings=[i.message for i in warnings],
            suggestions=self._generate_suggestions(code, issues)
        )
    
    def _static_analysis(self, code: str) -> List[_ValidationIssue]:
        """Perform static analysis on code."""
        issues = []
        lines = code.split('\n')
        
        # Check required patterns
        for pattern, message in self.REQUIRED_PATTERNS:
            if not re.search(pattern, code, re.IGNORECASE):
                issues.append(_ValidationIssue(
                    severity='error',
                    message=message,
                    rule='required-pattern'
                ))
        
        # Check recommended patterns
        for pattern, message in self.RECOMMENDED_PATTERNS:
            if not re.search(pattern, code, re.IGNORECASE):
                issues.append(_ValidationIssue(
                    severity='info',
                    message=message,
                    rule='recommended-pattern'
                ))
        
        # Check problematic patterns
        for pattern, message, severity in self.PROBLEMATIC_PATTERNS:
            matches = list(re.finditer(pattern, code))
            for match in matches:
                # Find line number
                line_num = code[:match.start()].count('\n') + 1
                issues.append(_ValidationIssue(
                    severity=severity,
                    message=message,
                    line=line_num,
                    rule='deprecated-pattern'
                ))
        
        # Check for balanced braces
        brace_balance = self._check_brace_balance(code)
        if brace_balance != 0:
            issues.append(_ValidationIssue(
                severity='error',
                message=f'Unbalanced braces: {abs(brace_balance)} {"extra opening" if brace_balance > 0 else "extra closing"}',
                rule='syntax'
            ))
        
        # Check process structure
        process_issues = self._validate_processes(code)
        issues.extend(process_issues)
        
        # Check channel usage
        channel_issues = self._validate_channels(code)
        issues.extend(channel_issues)
        
        return issues
    
    def _check_brace_balance(self, code: str) -> int:
        """Check for balanced braces, accounting for strings and comments."""
        # Remove strings and comments
        cleaned = re.sub(r'"[^"]*"', '""', code)
        cleaned = re.sub(r"'[^']*'", "''", cleaned)
        cleaned = re.sub(r'//.*$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        
        return cleaned.count('{') - cleaned.count('}')
    
    def _validate_processes(self, code: str) -> List[_ValidationIssue]:
        """Validate process definitions."""
        issues = []
        
        # Find all processes
        process_pattern = r'process\s+(\w+)\s*\{'
        processes = re.findall(process_pattern, code)
        
        for proc_name in processes:
            # Find the process block
            proc_match = re.search(
                rf'process\s+{proc_name}\s*\{{([\s\S]*?)^\}}',
                code,
                re.MULTILINE
            )
            
            if not proc_match:
                continue
            
            proc_body = proc_match.group(1)
            
            # Check for input block
            if 'input:' not in proc_body:
                issues.append(_ValidationIssue(
                    severity='warning',
                    message=f'Process {proc_name} has no input block',
                    rule='process-structure'
                ))
            
            # Check for output block
            if 'output:' not in proc_body:
                issues.append(_ValidationIssue(
                    severity='warning',
                    message=f'Process {proc_name} has no output block',
                    rule='process-structure'
                ))
            
            # Check for script/shell block
            if 'script:' not in proc_body and 'shell:' not in proc_body and 'exec:' not in proc_body:
                issues.append(_ValidationIssue(
                    severity='error',
                    message=f'Process {proc_name} has no script/shell/exec block',
                    rule='process-structure'
                ))
            
            # Check for container
            if 'container' not in proc_body and 'label' not in proc_body:
                issues.append(_ValidationIssue(
                    severity='warning',
                    message=f'Process {proc_name} should specify container or label',
                    rule='reproducibility'
                ))
        
        return issues
    
    def _validate_channels(self, code: str) -> List[_ValidationIssue]:
        """Validate channel usage."""
        issues = []
        
        # Check for undefined channel references in workflow
        workflow_match = re.search(r'workflow\s*\{([\s\S]*?)^\}', code, re.MULTILINE)
        if workflow_match:
            workflow_body = workflow_match.group(1)
            
            # Find all channel variable assignments
            assigned_channels = set(re.findall(r'(\w+_ch)\s*=', code))
            assigned_channels.update(re.findall(r'\.set\s*\{\s*(\w+)', code))
            
            # Find channel usages in workflow
            used_channels = set(re.findall(r'(\w+_ch)\s*[,)]', workflow_body))
            
            # Check for undefined channels
            for ch in used_channels:
                if ch not in assigned_channels and not re.search(rf'Channel\.[^)]+\.set\s*\{{\s*{ch}', code):
                    issues.append(_ValidationIssue(
                        severity='warning',
                        message=f'Channel {ch} may not be defined',
                        rule='channel-definition'
                    ))
        
        return issues
    
    async def _llm_review(self, code: str) -> List[_ValidationIssue]:
        """Use LLM to review code semantically."""
        issues = []
        
        prompt = f"""{self.SYSTEM_PROMPT}

Code to review:
```nextflow
{code}
```"""
        
        try:
            response = await self.router.route_async(prompt)
            
            # Parse JSON response
            import json
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                review = json.loads(json_match.group())
                
                for issue in review.get('issues', []):
                    issues.append(_ValidationIssue(
                        severity=issue.get('severity', 'info'),
                        message=issue.get('message', ''),
                        line=issue.get('line'),
                        rule='llm-review'
                    ))
        except Exception as e:
            logger.debug(f"Could not parse LLM review: {e}")
        
        return issues
    
    def _generate_suggestions(self, code: str, issues: List[_ValidationIssue]) -> List[str]:
        """Generate suggestions based on issues."""
        suggestions = []
        
        for issue in issues:
            if issue.severity == 'error':
                if 'DSL2' in issue.message:
                    suggestions.append("Add 'nextflow.enable.dsl = 2' at the top of the file")
                elif 'process' in issue.message.lower():
                    suggestions.append("Define at least one process with 'process NAME { ... }'")
                elif 'workflow' in issue.message.lower():
                    suggestions.append("Add a workflow block: 'workflow { ... }'")
                elif 'brace' in issue.message.lower():
                    suggestions.append("Check for matching { } braces in all blocks")
            
            elif issue.severity == 'warning':
                if 'container' in issue.message.lower():
                    suggestions.append("Add container directives for reproducibility")
                elif 'input' in issue.message.lower():
                    suggestions.append("Add input blocks to processes")
        
        # Add general suggestions
        if not any('publishDir' in str(i.message) for i in issues):
            if 'publishDir' not in code:
                suggestions.append("Consider adding publishDir to save outputs")
        
        return list(set(suggestions))  # Remove duplicates
    
    def check_compatibility(self, code: str, nextflow_version: str = "23.04") -> List[_ValidationIssue]:
        """Check code compatibility with specific Nextflow version."""
        issues = []
        
        version_parts = [int(p) for p in nextflow_version.split('.')]
        major, minor = version_parts[0], version_parts[1] if len(version_parts) > 1 else 0
        
        # DSL2 became default in 22.04
        if major >= 22 and minor >= 4:
            if 'nextflow.enable.dsl = 2' not in code and 'nextflow.enable.dsl=2' not in code:
                issues.append(_ValidationIssue(
                    severity='info',
                    message=f'DSL2 is default in Nextflow {nextflow_version}, explicit declaration optional',
                    rule='compatibility'
                ))
        
        # Check for deprecated features
        if major >= 23:
            if 'Channel.from' in code:
                issues.append(_ValidationIssue(
                    severity='warning',
                    message='Channel.from is deprecated in Nextflow 23.x, use Channel.of instead',
                    rule='compatibility'
                ))
        
        return issues
