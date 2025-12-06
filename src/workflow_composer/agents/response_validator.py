"""
Response Validator for BioPipelines Chat Agent
===============================================

Validates and refines agent responses before showing to user.
Uses LLM to check relevance, completeness, and correctness.

Key Features:
1. Validates response is relevant to user's query
2. Checks if response is complete or needs follow-up
3. Refines unclear/incomplete responses
4. Maintains conversation context for coherent multi-turn dialogue
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    is_complete: bool
    is_relevant: bool
    refined_response: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0


class ResponseValidator:
    """
    Validates and refines agent responses using LLM.
    
    This addresses the key issue: the agent was returning raw tool output
    without checking if it actually answered the user's question.
    """
    
    def __init__(self, llm_client=None, context_memory=None):
        """
        Initialize validator.
        
        Args:
            llm_client: LLM client for validation (optional, uses rules if None)
            context_memory: Session memory for context awareness
        """
        self.llm_client = llm_client
        self.context = context_memory
        self._validation_cache = {}
        
    def validate_response(
        self,
        query: str,
        response: str,
        tool_used: Optional[str] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict]] = None,
    ) -> ValidationResult:
        """
        Validate if a response properly answers the user's query.
        
        Args:
            query: User's original question
            response: Agent's generated response
            tool_used: Which tool was executed
            tool_result: Raw result from tool execution
            history: Conversation history for context
            
        Returns:
            ValidationResult with validation status and refined response
        """
        issues = []
        suggestions = []
        
        # Quick rule-based checks first
        rule_result = self._rule_based_validation(query, response, tool_used, tool_result)
        if rule_result:
            return rule_result
        
        # Check for common problems
        
        # 1. Response asks for info that was already in the query
        if self._asks_unnecessary_question(query, response):
            issues.append("Response asks for information already provided by user")
            refined = self._refine_unnecessary_question(query, response, tool_result)
            if refined:
                return ValidationResult(
                    is_valid=False,
                    is_complete=False,
                    is_relevant=True,
                    refined_response=refined,
                    issues=issues,
                    confidence=0.7
                )
        
        # 2. Response is irrelevant to the query type
        if not self._is_relevant(query, response, tool_used):
            issues.append("Response does not address the user's question")
            suggestions.append("Rephrase your question or try a more specific query")
            return ValidationResult(
                is_valid=False,
                is_complete=False,
                is_relevant=False,
                issues=issues,
                suggestions=suggestions,
                confidence=0.5
            )
        
        # 3. Response is incomplete (e.g., just says "scanning..." with no results)
        if self._is_incomplete(response):
            issues.append("Response appears incomplete")
            return ValidationResult(
                is_valid=False,
                is_complete=False,
                is_relevant=True,
                issues=issues,
                confidence=0.6
            )
        
        # 4. Use LLM for deeper validation if available
        if self.llm_client:
            return self._llm_validation(query, response, tool_used, tool_result, history)
        
        # Default: assume valid if passed rule checks
        return ValidationResult(
            is_valid=True,
            is_complete=True,
            is_relevant=True,
            confidence=0.8
        )
    
    def _rule_based_validation(
        self,
        query: str,
        response: str,
        tool_used: Optional[str],
        tool_result: Optional[Dict]
    ) -> Optional[ValidationResult]:
        """Fast rule-based validation for common patterns."""
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Pattern 1: User explicitly mentions a dataset ID but response asks for it
        dataset_patterns = [
            r'\b(GSE\d+)\b',
            r'\b(ENCSR[A-Z0-9]+)\b',
            r'\b(TCGA-[A-Z]+)\b',
            r'\b(SRR\d+)\b',
        ]
        
        mentioned_dataset = None
        for pattern in dataset_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                mentioned_dataset = match.group(1)
                break
        
        if mentioned_dataset:
            # User mentioned a dataset, but response asks "where is your data?"
            if any(phrase in response_lower for phrase in [
                "where is your", 
                "what is your", 
                "could you specify",
                "please provide",
                "what would you like"
            ]):
                # Auto-fix: return what they should have done
                return ValidationResult(
                    is_valid=False,
                    is_complete=False,
                    is_relevant=False,
                    refined_response=f"📥 Preparing to download **{mentioned_dataset}**...\n\n_Initiating download from online database._",
                    issues=["Response incorrectly asked for dataset ID that was already provided"],
                    confidence=0.9
                )
        
        # Pattern 2: Download intent but scan_data was used
        if tool_used == "scan_data" and "download" in query_lower:
            # Wrong tool was used
            return ValidationResult(
                is_valid=False,
                is_complete=False,
                is_relevant=False,
                issues=["Used scan_data tool instead of download_dataset"],
                suggestions=["Try: 'download dataset GSE12345'"],
                confidence=0.9
            )
        
        # Pattern 3: User asks about online sources but got local scan
        if tool_used == "scan_data" and any(phrase in query_lower for phrase in [
            "online", "from web", "from internet", "download from", "find online"
        ]):
            return ValidationResult(
                is_valid=False,
                is_complete=False,
                is_relevant=False,
                issues=["Scanned local data when user asked about online sources"],
                refined_response="🔍 I'll search online databases for datasets matching your criteria...",
                confidence=0.85
            )
        
        return None
    
    def _asks_unnecessary_question(self, query: str, response: str) -> bool:
        """Check if response asks for info already in query."""
        response_lower = response.lower()
        
        # Common unnecessary question patterns
        unnecessary_patterns = [
            "where is your input data",
            "where is your data located",
            "could you please be more specific",
            "i'm not sure what you'd like",
            "what would you like me to do",
        ]
        
        return any(pattern in response_lower for pattern in unnecessary_patterns)
    
    def _refine_unnecessary_question(
        self, 
        query: str, 
        response: str, 
        tool_result: Optional[Dict]
    ) -> Optional[str]:
        """Generate a better response when agent asked unnecessary question."""
        
        query_lower = query.lower()
        
        # If user asked to download something they found earlier
        if "download" in query_lower:
            # Check for dataset ID in query
            for pattern in [r'\b(GSE\d+)\b', r'\b(ENCSR[A-Z0-9]+)\b', r'\b(TCGA-[A-Z]+)\b']:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    dataset_id = match.group(1).upper()
                    return f"📥 **Downloading {dataset_id}**\n\nInitiating download from database. This may take a few minutes depending on dataset size.\n\n_Use `check job status` to monitor progress._"
            
            # User said "download" without specific ID - check context
            if "the" in query_lower or "those" in query_lower or "them" in query_lower:
                return "📥 **Downloading datasets from search results**\n\nI'll download the datasets from your previous search. Starting download...\n\n_Use `check job status` to monitor progress._"
        
        return None
    
    def _is_relevant(self, query: str, response: str, tool_used: Optional[str]) -> bool:
        """Check if response is relevant to query intent."""
        
        query_lower = query.lower()
        
        # Map intents to expected response types
        intent_response_map = {
            "download": ["download", "📥", "initiating", "started", "progress"],
            "search": ["found", "results", "datasets", "matches", "🔍"],
            "scan": ["found", "samples", "files", "📁", "data"],
            "workflow": ["workflow", "pipeline", "generated", "created"],
            "run": ["submitted", "running", "job", "slurm"],
            "status": ["status", "running", "completed", "failed", "queued"],
        }
        
        # Find user intent
        user_intent = None
        for intent in intent_response_map:
            if intent in query_lower:
                user_intent = intent
                break
        
        if not user_intent:
            return True  # Can't determine intent, assume relevant
        
        # Check if response contains expected keywords
        response_lower = response.lower()
        expected_words = intent_response_map[user_intent]
        
        return any(word in response_lower for word in expected_words)
    
    def _is_incomplete(self, response: str) -> bool:
        """Check if response appears incomplete."""
        
        incomplete_indicators = [
            "scanning...",
            "searching...",
            "loading...",
            "processing...",
        ]
        
        response_lower = response.lower().strip()
        
        # Very short responses are suspicious
        if len(response) < 50 and not any(emoji in response for emoji in ["✅", "❌", "📁", "🔍"]):
            return True
        
        # Ends with incomplete indicator
        return any(response_lower.endswith(ind) for ind in incomplete_indicators)
    
    def _llm_validation(
        self,
        query: str,
        response: str,
        tool_used: Optional[str],
        tool_result: Optional[Dict],
        history: Optional[List[Dict]]
    ) -> ValidationResult:
        """Use LLM for deep validation."""
        
        prompt = f"""Evaluate if this assistant response properly answers the user's question.

User Question: {query}
Assistant Response: {response}
Tool Used: {tool_used or 'None'}

Evaluation criteria:
1. Does the response directly address what the user asked?
2. Is the response complete or does it leave the user hanging?
3. Is the response accurate based on the tool results?
4. Does the response unnecessarily ask for information already provided?

Respond with:
VALID: yes/no
COMPLETE: yes/no  
RELEVANT: yes/no
ISSUES: (list any problems, one per line)
REFINED_RESPONSE: (optional - provide a better response if current one has issues)
"""
        
        try:
            result = self.llm_client.complete(prompt)
            return self._parse_llm_validation(result)
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return ValidationResult(
                is_valid=True,
                is_complete=True,
                is_relevant=True,
                confidence=0.6
            )
    
    def _parse_llm_validation(self, llm_response: str) -> ValidationResult:
        """Parse LLM validation response."""
        
        lines = llm_response.strip().split("\n")
        result = ValidationResult(
            is_valid=True,
            is_complete=True,
            is_relevant=True,
            confidence=0.85
        )
        
        current_section = None
        refined_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("VALID:"):
                result.is_valid = "yes" in line.lower()
            elif line.startswith("COMPLETE:"):
                result.is_complete = "yes" in line.lower()
            elif line.startswith("RELEVANT:"):
                result.is_relevant = "yes" in line.lower()
            elif line.startswith("ISSUES:"):
                current_section = "issues"
            elif line.startswith("REFINED_RESPONSE:"):
                current_section = "refined"
            elif current_section == "issues" and line.startswith("- "):
                result.issues.append(line[2:])
            elif current_section == "refined" and line:
                refined_lines.append(line)
        
        if refined_lines:
            result.refined_response = "\n".join(refined_lines)
        
        return result


class ContextAwareResponseBuilder:
    """
    Builds context-aware responses using conversation history.
    
    This ensures the agent:
    1. Remembers what was discussed earlier
    2. Uses entities from previous turns
    3. Provides coherent multi-turn dialogue
    """
    
    def __init__(self, session_memory=None, llm_client=None):
        self.memory = session_memory
        self.llm = llm_client
        
    def build_response(
        self,
        tool_result: Dict[str, Any],
        query: str,
        history: List[Dict],
        tool_used: str,
    ) -> str:
        """
        Build a context-aware response from tool results.
        
        Instead of just formatting tool output, this:
        1. Considers conversation history
        2. References previous entities/topics
        3. Provides helpful follow-up suggestions
        """
        
        # Get context from history
        context = self._extract_context(history)
        
        # Base response from tool
        base_response = self._format_tool_result(tool_result, tool_used)
        
        # Enhance with context
        enhanced = self._enhance_with_context(base_response, context, query)
        
        # Add relevant follow-ups
        follow_ups = self._suggest_follow_ups(tool_result, tool_used, context)
        
        if follow_ups:
            enhanced += "\n\n💡 **Suggested next steps:**\n"
            for step in follow_ups:
                enhanced += f"• {step}\n"
        
        return enhanced
    
    def _extract_context(self, history: List[Dict]) -> Dict[str, Any]:
        """Extract relevant context from history."""
        context = {
            "mentioned_datasets": [],
            "last_search_query": None,
            "discussed_topics": [],
            "pending_actions": [],
        }
        
        if not history:
            return context
        
        for msg in history[-10:]:  # Last 10 messages
            content = msg.get("content", "")
            
            # Extract dataset IDs
            for pattern in [r'\b(GSE\d+)\b', r'\b(ENCSR[A-Z0-9]+)\b']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                context["mentioned_datasets"].extend(matches)
            
            # Track search queries
            if msg.get("role") == "user" and "search" in content.lower():
                context["last_search_query"] = content
        
        # Deduplicate
        context["mentioned_datasets"] = list(set(context["mentioned_datasets"]))
        
        return context
    
    def _format_tool_result(self, result: Dict[str, Any], tool_used: str) -> str:
        """Format raw tool result into readable response."""
        
        if not result:
            return "No results found."
        
        # Tool-specific formatting
        if tool_used == "search_databases":
            return self._format_search_results(result)
        elif tool_used == "scan_data":
            return self._format_scan_results(result)
        elif tool_used == "download_dataset":
            return self._format_download_result(result)
        
        # Generic formatting
        if isinstance(result, dict):
            message = result.get("message", "")
            if message:
                return message
        
        return str(result)
    
    def _format_search_results(self, result: Dict) -> str:
        """Format search results nicely."""
        results = result.get("results", result.get("data", {}).get("results", []))
        total = len(results)
        
        if total == 0:
            return "🔍 No matching datasets found. Try different search terms."
        
        response = f"🔍 **Found {total} matching datasets:**\n\n"
        for r in results[:5]:  # Show top 5
            name = r.get("accession", r.get("id", "Unknown"))
            desc = r.get("title", r.get("description", ""))[:80]
            source = r.get("source", "")
            response += f"• **{name}** ({source}): {desc}...\n"
        
        if total > 5:
            response += f"\n_...and {total - 5} more results._"
        
        return response
    
    def _format_scan_results(self, result: Dict) -> str:
        """Format data scan results."""
        samples = result.get("samples", result.get("data", {}).get("samples", []))
        total = len(samples)
        
        if total == 0:
            return "📁 No data files found in the specified location."
        
        return result.get("message", f"📁 Found {total} samples in the data directory.")
    
    def _format_download_result(self, result: Dict) -> str:
        """Format download result."""
        if result.get("success"):
            return f"📥 Download initiated: {result.get('message', 'Started')}"
        return f"❌ Download failed: {result.get('error', 'Unknown error')}"
    
    def _enhance_with_context(self, response: str, context: Dict, query: str) -> str:
        """Add contextual references to response."""
        
        # Reference earlier datasets if relevant
        if context["mentioned_datasets"] and "download" in query.lower():
            datasets = context["mentioned_datasets"][:3]
            if len(datasets) > 0 and "which" not in response.lower():
                response = response.replace(
                    "📥",
                    f"📥 (Based on earlier discussion: {', '.join(datasets)})"
                )
        
        return response
    
    def _suggest_follow_ups(
        self, 
        result: Dict, 
        tool_used: str, 
        context: Dict
    ) -> List[str]:
        """Generate relevant follow-up suggestions."""
        
        suggestions = []
        
        if tool_used == "search_databases":
            results = result.get("results", result.get("data", {}).get("results", []))
            if results:
                suggestions.append(f'"download {results[0].get("accession", "dataset")}" - Download the first result')
                suggestions.append('"download all" - Download all found datasets')
                suggestions.append('"show details for <ID>" - View more information')
        
        elif tool_used == "scan_data":
            suggestions.append('"create workflow" - Generate analysis pipeline')
            suggestions.append('"search online for more data" - Find additional datasets')
        
        elif tool_used == "download_dataset":
            suggestions.append('"check job status" - Monitor download progress')
            suggestions.append('"create workflow" - Start analysis after download')
        
        return suggestions[:3]  # Max 3 suggestions
