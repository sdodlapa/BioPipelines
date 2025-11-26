"""
GitHub Copilot integration for code-level fixes.

Integrates with GitHub Copilot Coding Agent (via MCP) for
creating pull requests that fix workflow issues.
"""

import os
import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

from .categories import ErrorDiagnosis
from .prompts import build_code_fix_prompt

logger = logging.getLogger(__name__)


def load_github_token(token_file: str = "github_token") -> Optional[str]:
    """
    Load GitHub token from .secrets directory or environment.
    
    Priority:
    1. .secrets/github_token file (project directory)
    2. .secrets/github_token file (relative to module)
    3. ~/.secrets/github_token file (home directory)
    4. GITHUB_TOKEN environment variable (fallback)
    
    Note: File takes precedence over environment to allow explicit
    account configuration via .secrets files.
    """
    # Check project .secrets directory first
    secrets_paths = [
        Path.cwd() / ".secrets" / token_file,
        Path(__file__).parent.parent.parent.parent.parent / ".secrets" / token_file,
        Path.home() / ".secrets" / token_file,
    ]
    
    for path in secrets_paths:
        if path.exists():
            try:
                token = path.read_text().strip()
                if token:
                    logger.debug(f"Loaded GitHub token from {path}")
                    return token
            except Exception as e:
                logger.warning(f"Failed to read token from {path}: {e}")
    
    # Fallback to environment variable (only for default token)
    if token_file == "github_token" and os.getenv("GITHUB_TOKEN"):
        logger.debug("Using GITHUB_TOKEN from environment")
        return os.getenv("GITHUB_TOKEN")
    
    return None


@dataclass
class GitHubAccount:
    """GitHub account configuration."""
    username: str
    token: str
    is_primary: bool = False
    is_pro: bool = False


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
    - GITHUB_TOKEN environment variable or .secrets/github_token
    
    Supports multiple GitHub accounts:
    - Primary: sdodlapati3 (Pro+) - .secrets/github_token
    - Secondary: sdodlapa (Pro+) - .secrets/github_token_sdodlapa
    
    Example:
        agent = GitHubCopilotAgent(owner="sdodlapati3", repo="BioPipelines")
        result = await agent.create_fix_pr(diagnosis, workflow_content)
    """
    
    API_BASE = "https://api.github.com"
    
    # Registered accounts
    ACCOUNTS = {
        "sdodlapati3": {
            "token_file": "github_token",
            "is_pro": True,
            "is_primary": True,
        },
        "sdodlapa": {
            "token_file": "github_token_sdodlapa",
            "is_pro": True,
            "is_primary": False,
        },
    }
    
    def __init__(
        self,
        owner: str,
        repo: str,
        github_token: Optional[str] = None,
        account: Optional[str] = None,
    ):
        """
        Initialize GitHub Copilot agent.
        
        Args:
            owner: Repository owner
            repo: Repository name
            github_token: GitHub token (or auto-loaded from .secrets)
            account: Specific account to use ('sdodlapa' or 'sdodlapati3')
        """
        self.owner = owner
        self.repo = repo
        self._accounts: Dict[str, GitHubAccount] = {}
        
        # Load token
        if github_token:
            self.github_token = github_token
        elif account and account in self.ACCOUNTS:
            # Load specific account
            self.github_token = load_github_token(self.ACCOUNTS[account]["token_file"])
        else:
            # Try primary account first
            self.github_token = load_github_token("github_token")
            if not self.github_token:
                # Try secondary
                self.github_token = load_github_token("github_token_secondary")
        
        if not self.github_token:
            logger.warning(
                "GitHub token not found. Check:\n"
                "  - GITHUB_TOKEN environment variable\n"
                "  - .secrets/github_token file\n"
                "  - .secrets/github_token_secondary file"
            )
        else:
            # Validate and identify account
            self._validate_token()
    
    def _validate_token(self) -> bool:
        """Validate token and identify account."""
        try:
            import requests
            resp = requests.get(
                f"{self.API_BASE}/user",
                headers={"Authorization": f"token {self.github_token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                username = data.get("login", "unknown")
                logger.info(f"GitHub authenticated as: {username}")
                return True
            else:
                logger.warning(f"GitHub token validation failed: {resp.status_code}")
                return False
        except Exception as e:
            logger.warning(f"GitHub token validation error: {e}")
            return False
    
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
    
    # ==================== Direct PR Creation via GitHub API ====================
    
    def get_branch_sha(self, branch: str = "main") -> Optional[str]:
        """
        Get the SHA of a branch's HEAD commit.
        
        Args:
            branch: Branch name
            
        Returns:
            SHA string or None if failed
        """
        try:
            response = self._request(
                "GET",
                f"/repos/{self.owner}/{self.repo}/git/refs/heads/{branch}"
            )
            return response.get("object", {}).get("sha")
        except Exception as e:
            logger.error(f"Failed to get branch SHA: {e}")
            return None
    
    def create_branch(self, branch_name: str, base_branch: str = "main") -> bool:
        """
        Create a new branch from a base branch.
        
        Args:
            branch_name: Name for the new branch
            base_branch: Branch to create from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the SHA of the base branch
            sha = self.get_branch_sha(base_branch)
            if not sha:
                logger.error(f"Could not get SHA for branch {base_branch}")
                return False
            
            # Create the new branch
            self._request(
                "POST",
                f"/repos/{self.owner}/{self.repo}/git/refs",
                data={
                    "ref": f"refs/heads/{branch_name}",
                    "sha": sha
                }
            )
            logger.info(f"Created branch {branch_name} from {base_branch}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create branch {branch_name}: {e}")
            return False
    
    def get_file_sha(self, file_path: str, branch: str = "main") -> Optional[str]:
        """
        Get the SHA of an existing file (needed for updates).
        
        Args:
            file_path: Path to file in repo
            branch: Branch to check
            
        Returns:
            File SHA or None if not found
        """
        try:
            response = self._request(
                "GET",
                f"/repos/{self.owner}/{self.repo}/contents/{file_path}",
                params={"ref": branch}
            )
            return response.get("sha")
        except Exception:
            return None  # File doesn't exist
    
    def update_file(
        self,
        file_path: str,
        content: str,
        message: str,
        branch: str,
        create_new: bool = False
    ) -> bool:
        """
        Create or update a file in the repository.
        
        Args:
            file_path: Path to file in repo
            content: New file content
            message: Commit message
            branch: Branch to commit to
            create_new: If True, creates new file; if False, updates existing
            
        Returns:
            True if successful, False otherwise
        """
        import base64
        
        try:
            # Encode content as base64
            content_b64 = base64.b64encode(content.encode()).decode()
            
            data = {
                "message": message,
                "content": content_b64,
                "branch": branch
            }
            
            # If updating, need the existing file's SHA
            if not create_new:
                file_sha = self.get_file_sha(file_path, branch)
                if file_sha:
                    data["sha"] = file_sha
            
            self._request(
                "PUT",
                f"/repos/{self.owner}/{self.repo}/contents/{file_path}",
                data=data
            )
            
            action = "Created" if create_new else "Updated"
            logger.info(f"{action} file {file_path} on branch {branch}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update file {file_path}: {e}")
            return False
    
    def create_direct_pr(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main"
    ) -> PullRequestResult:
        """
        Create a pull request directly via GitHub API.
        
        Args:
            title: PR title
            body: PR description
            head_branch: Branch with changes
            base_branch: Branch to merge into
            
        Returns:
            PullRequestResult with PR details
        """
        try:
            response = self._request(
                "POST",
                f"/repos/{self.owner}/{self.repo}/pulls",
                data={
                    "title": title,
                    "body": body,
                    "head": head_branch,
                    "base": base_branch
                }
            )
            
            pr_number = response.get("number")
            pr_url = response.get("html_url")
            
            logger.info(f"Created PR #{pr_number}: {pr_url}")
            
            return PullRequestResult(
                success=True,
                pr_number=pr_number,
                pr_url=pr_url,
                message=f"Pull request #{pr_number} created successfully",
            )
            
        except Exception as e:
            logger.error(f"Failed to create PR: {e}")
            return PullRequestResult(
                success=False,
                message=f"Failed to create PR: {str(e)}"
            )
    
    async def create_fix_pr_direct(
        self,
        diagnosis: ErrorDiagnosis,
        fixed_content: str,
        file_path: str,
        base_branch: str = "main",
    ) -> PullRequestResult:
        """
        Create a PR directly with fixed file content.
        
        This creates an actual pull request with the fix, rather than
        delegating to Copilot. Use this when you have the fixed content ready.
        
        Args:
            diagnosis: ErrorDiagnosis with error details
            fixed_content: The fixed file content
            file_path: Path to the file being fixed
            base_branch: Branch to create PR against
            
        Returns:
            PullRequestResult with PR details
        """
        if not self.is_available():
            return PullRequestResult(
                success=False,
                message="GitHub token not configured"
            )
        
        import time
        
        # Generate a unique branch name
        timestamp = int(time.time())
        branch_name = f"fix/{diagnosis.category.value.replace('_', '-')}-{timestamp}"
        
        try:
            # Step 1: Create the fix branch
            logger.info(f"Creating branch {branch_name} from {base_branch}")
            if not self.create_branch(branch_name, base_branch):
                return PullRequestResult(
                    success=False,
                    message=f"Failed to create branch {branch_name}"
                )
            
            # Step 2: Commit the fixed content
            commit_message = f"fix: {diagnosis.root_cause[:50]}\n\n" \
                           f"Auto-fix for {diagnosis.category.value} error.\n" \
                           f"Confidence: {diagnosis.confidence:.0%}"
            
            logger.info(f"Committing fix to {file_path}")
            if not self.update_file(
                file_path=file_path,
                content=fixed_content,
                message=commit_message,
                branch=branch_name
            ):
                return PullRequestResult(
                    success=False,
                    message=f"Failed to commit fix to {file_path}"
                )
            
            # Step 3: Create the pull request
            pr_title = f"🔧 Auto-fix: {diagnosis.root_cause[:60]}"
            if len(diagnosis.root_cause) > 60:
                pr_title += "..."
            
            pr_body = self._format_pr_body(diagnosis, file_path)
            
            logger.info(f"Creating pull request: {pr_title}")
            result = self.create_direct_pr(
                title=pr_title,
                body=pr_body,
                head_branch=branch_name,
                base_branch=base_branch
            )
            
            if result.success:
                result.message = (
                    f"Pull request #{result.pr_number} created!\n"
                    f"Branch: {branch_name}\n"
                    f"URL: {result.pr_url}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create fix PR: {e}")
            return PullRequestResult(
                success=False,
                message=f"Failed to create fix PR: {str(e)}"
            )
    
    def _format_pr_body(self, diagnosis: ErrorDiagnosis, file_path: str) -> str:
        """Format a PR body for an auto-fix."""
        risk_icons = {
            "safe": "🟢",
            "low": "🟡", 
            "medium": "🟠",
            "high": "🔴",
        }
        
        fixes_md = ""
        for i, fix in enumerate(diagnosis.suggested_fixes[:3], 1):
            icon = risk_icons.get(fix.risk_level.value, "⚪")
            fixes_md += f"{i}. {icon} {fix.description}\n"
        
        return f"""## 🔧 Automated Fix

### Problem
**Category:** `{diagnosis.category.value}`
**Root Cause:** {diagnosis.root_cause}

### Changes
This PR modifies `{file_path}` to fix the issue.

### Diagnosis Details
- **Confidence:** {diagnosis.confidence:.0%}
- **Failed Process:** {diagnosis.failed_process or 'Unknown'}
- **Diagnosed by:** {diagnosis.llm_provider_used or 'Pattern Matching'}

### Suggested Fixes Applied
{fixes_md}

### Error Excerpt
```
{diagnosis.log_excerpt[:500]}
```

---
*This PR was automatically generated by BioPipelines Error Diagnosis Agent*
"""

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
    owner = owner or os.getenv("GITHUB_OWNER", "sdodlapati3")
    repo = repo or os.getenv("GITHUB_REPO", "BioPipelines")
    
    agent = GitHubCopilotAgent(owner=owner, repo=repo)
    
    if agent.is_available():
        return agent
    
    return None
