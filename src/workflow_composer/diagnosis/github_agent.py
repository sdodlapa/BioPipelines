"""
GitHub Copilot integration for code-level fixes.

Integrates with GitHub Copilot Coding Agent (via MCP) for
creating pull requests that fix workflow issues.
"""

import os
import logging
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from .categories import ErrorDiagnosis
from .prompts import build_code_fix_prompt

logger = logging.getLogger(__name__)


@dataclass
class PullRequestResult:
    """Result of creating a fix PR."""
    success: bool
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    issue_number: Optional[int] = None
    issue_url: Optional[str] = None
    branch_name: Optional[str] = None
    message: str = ""


class GitHubCopilotAgent:
    """
    Integration with GitHub Copilot Coding Agent.
    
    Uses GitHub's Copilot Coding Agent to create pull requests
    that fix workflow code issues. Requires:
    - GitHub Copilot Pro+ subscription
    - Repository with GitHub Copilot enabled
    - GITHUB_TOKEN environment variable
    
    Example:
        agent = GitHubCopilotAgent(owner="user", repo="BioPipelines")
        result = await agent.create_fix_pr(diagnosis, workflow_content)
    """
    
    API_BASE = "https://api.github.com"
    
    def __init__(
        self,
        owner: str,
        repo: str,
        github_token: Optional[str] = None,
    ):
        """
        Initialize GitHub Copilot agent.
        
        Args:
            owner: Repository owner
            repo: Repository name
            github_token: GitHub token (or from GITHUB_TOKEN env)
        """
        self.owner = owner
        self.repo = repo
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        
        if not self.github_token:
            logger.warning("GITHUB_TOKEN not set - GitHub Copilot will be unavailable")
    
    def is_available(self) -> bool:
        """Check if GitHub Copilot integration is available."""
        return bool(self.github_token)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
    ) -> Dict:
        """Make a GitHub API request."""
        import requests
        
        url = f"{self.API_BASE}{endpoint}"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data,
            timeout=30,
        )
        
        response.raise_for_status()
        return response.json() if response.text else {}
    
    def create_issue(
        self,
        diagnosis: ErrorDiagnosis,
        title: Optional[str] = None,
    ) -> PullRequestResult:
        """
        Create a GitHub issue from a diagnosis.
        
        Args:
            diagnosis: ErrorDiagnosis
            title: Custom title (optional)
            
        Returns:
            PullRequestResult with issue details
        """
        if not self.is_available():
            return PullRequestResult(
                success=False,
                message="GitHub token not configured"
            )
        
        if not title:
            title = f"🔧 Fix: {diagnosis.category.value.replace('_', ' ').title()} - {diagnosis.root_cause[:50]}"
        
        body = self.format_issue_body(diagnosis)
        
        try:
            result = self._request(
                "POST",
                f"/repos/{self.owner}/{self.repo}/issues",
                data={
                    "title": title,
                    "body": body,
                    "labels": ["bug", "auto-generated", diagnosis.category.value],
                }
            )
            
            return PullRequestResult(
                success=True,
                issue_number=result.get("number"),
                issue_url=result.get("html_url"),
                message=f"Issue #{result.get('number')} created successfully!",
            )
            
        except Exception as e:
            logger.error(f"Failed to create issue: {e}")
            return PullRequestResult(
                success=False,
                message=f"Failed to create issue: {str(e)}"
            )
    
    async def create_fix_pr(
        self,
        diagnosis: ErrorDiagnosis,
        workflow_content: str,
        workflow_file: str = "main.nf",
        base_branch: str = "main",
    ) -> PullRequestResult:
        """
        Create a pull request with a fix for the diagnosed error.
        
        This delegates to GitHub Copilot Coding Agent which will:
        1. Create a new branch
        2. Implement the fix
        3. Open a pull request
        
        Args:
            diagnosis: ErrorDiagnosis with error details
            workflow_content: Content of the workflow file
            workflow_file: Name of the workflow file
            base_branch: Branch to base the fix on
            
        Returns:
            PullRequestResult with PR details
        """
        if not self.is_available():
            return PullRequestResult(
                success=False,
                message="GitHub token not configured"
            )
        
        # Build the problem statement for Copilot
        problem_statement = build_code_fix_prompt(
            diagnosis=diagnosis,
            workflow_content=workflow_content,
            workflow_file=workflow_file,
        )
        
        # Title for the PR
        title = f"Fix: {diagnosis.root_cause[:60]}"
        if len(diagnosis.root_cause) > 60:
            title += "..."
        
        try:
            # For now, create an issue that Copilot can work on
            # The MCP integration will be done through the chat interface
            
            logger.info(
                f"Creating issue for Copilot to fix:\n"
                f"  Owner: {self.owner}\n"
                f"  Repo: {self.repo}\n"
                f"  Title: {title}\n"
                f"  Base: {base_branch}"
            )
            
            # Create an issue with the fix request
            result = self.create_issue(diagnosis, title=f"🤖 [Copilot] {title}")
            
            if result.success:
                result.message = (
                    f"Issue #{result.issue_number} created! "
                    f"GitHub Copilot can now be assigned to fix it.\n"
                    f"URL: {result.issue_url}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create fix PR: {e}")
            return PullRequestResult(
                success=False,
                message=f"Failed to create PR: {str(e)}"
            )
    
    async def assign_to_issue(
        self,
        issue_number: int,
        diagnosis: Optional[ErrorDiagnosis] = None,
    ) -> PullRequestResult:
        """
        Assign Copilot to fix an existing issue.
        
        Args:
            issue_number: GitHub issue number
            diagnosis: Optional error diagnosis for context
            
        Returns:
            PullRequestResult
        """
        if not self.is_available():
            return PullRequestResult(
                success=False,
                message="GitHub token not configured"
            )
        
        try:
            # Add a comment with diagnosis context
            if diagnosis:
                comment = f"""## 🔍 AI Diagnosis Added

**Error Type:** `{diagnosis.category.value}`
**Root Cause:** {diagnosis.root_cause}
**Confidence:** {diagnosis.confidence:.0%}

### Suggested Fixes
"""
                for i, fix in enumerate(diagnosis.suggested_fixes[:3], 1):
                    comment += f"{i}. {fix.description}\n"
                    if fix.command:
                        comment += f"   ```bash\n   {fix.command}\n   ```\n"
                
                comment += "\n*@copilot please fix this issue based on the diagnosis above.*"
                
                self._request(
                    "POST",
                    f"/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments",
                    data={"body": comment}
                )
            
            logger.info(
                f"Added Copilot comment to issue #{issue_number} "
                f"in {self.owner}/{self.repo}"
            )
            
            return PullRequestResult(
                success=True,
                issue_number=issue_number,
                message=f"Copilot has been requested to fix issue #{issue_number}",
            )
            
        except Exception as e:
            logger.error(f"Failed to assign Copilot: {e}")
            return PullRequestResult(
                success=False,
                message=f"Failed to assign Copilot: {str(e)}"
            )
    
    def format_issue_body(self, diagnosis: ErrorDiagnosis) -> str:
        """
        Format an error diagnosis as a GitHub issue body.
        
        Args:
            diagnosis: ErrorDiagnosis
            
        Returns:
            Markdown-formatted issue body
        """
        risk_icons = {
            "safe": "🟢",
            "low": "🟡",
            "medium": "🟠",
            "high": "🔴",
        }
        
        fixes_md = ""
        for i, fix in enumerate(diagnosis.suggested_fixes, 1):
            icon = risk_icons.get(fix.risk_level.value, "⚪")
            fixes_md += f"{i}. {icon} {fix.description}\n"
            if fix.command:
                fixes_md += f"   ```\n   {fix.command}\n   ```\n"
        
        return f"""## Bug Report: {diagnosis.category.value.replace('_', ' ').title()}

### Description
{diagnosis.user_explanation}

### Root Cause
{diagnosis.root_cause}

### Error Log
```
{diagnosis.log_excerpt[:1000]}
```

### Suggested Fixes
{fixes_md}

### Context
- Failed Process: {diagnosis.failed_process or 'Unknown'}
- Work Directory: {diagnosis.work_directory or 'Unknown'}
- Confidence: {diagnosis.confidence:.0%}
- Diagnosed by: {diagnosis.llm_provider_used or 'Pattern Matching'}

---
*This issue was automatically generated by BioPipelines Error Diagnosis Agent*
"""


def get_github_copilot_agent(
    owner: Optional[str] = None,
    repo: Optional[str] = None,
) -> Optional[GitHubCopilotAgent]:
    """
    Get a GitHub Copilot agent if available.
    
    Will try to infer owner/repo from git config if not provided.
    
    Args:
        owner: Repository owner
        repo: Repository name
        
    Returns:
        GitHubCopilotAgent or None
    """
    # Try to get from environment or git config
    if not owner or not repo:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Parse github.com:owner/repo or github.com/owner/repo
                if "github.com" in url:
                    parts = url.split("github.com")[-1]
                    parts = parts.strip(":/").rstrip(".git").split("/")
                    if len(parts) >= 2:
                        owner = owner or parts[0]
                        repo = repo or parts[1]
        except Exception:
            pass
    
    # Use defaults if still not set
    owner = owner or os.getenv("GITHUB_OWNER", "sdodlapa")
    repo = repo or os.getenv("GITHUB_REPO", "BioPipelines")
    
    agent = GitHubCopilotAgent(owner=owner, repo=repo)
    
    if agent.is_available():
        return agent
    
    return None
