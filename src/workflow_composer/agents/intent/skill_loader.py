"""
Skill Loader Module
===================

Loads and manages skill documentation files for intelligent tool selection.
Inspired by Claude Scientific Skills project's documentation-first approach.

Usage:
    from workflow_composer.agents.intent.skill_loader import SkillLoader, get_skill_loader
    
    loader = get_skill_loader()
    
    # Find skills matching a query
    matches = loader.find_skills_for_query("align my RNA-seq reads")
    
    # Get skill details
    skill = loader.get_skill("star")
    
    # Get skills by category
    qc_skills = loader.get_skills_by_category("quality_control")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml
import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class SkillMatch:
    """Represents a skill that matches a query."""
    name: str
    display_name: str
    score: float  # Match confidence 0-1
    match_reasons: list[str]  # Why this skill matched
    skill_data: dict[str, Any]  # Full skill definition


@dataclass
class Skill:
    """Represents a loaded skill definition."""
    name: str
    display_name: str
    version: str
    category: str
    description: str
    capabilities: list[str]
    aliases: list[str] = field(default_factory=list)
    trigger_phrases: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    related_skills: list[dict[str, str]] = field(default_factory=list)
    prerequisites: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    best_practices: list[str] = field(default_factory=list)
    interpretation: dict[str, Any] = field(default_factory=dict)
    references: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        """Create a Skill from a dictionary (loaded from YAML)."""
        return cls(
            name=data.get("name", ""),
            display_name=data.get("display_name", ""),
            version=data.get("version", "1.0.0"),
            category=data.get("category", ""),
            description=data.get("description", ""),
            capabilities=data.get("capabilities", []),
            aliases=data.get("aliases", []),
            trigger_phrases=data.get("trigger_phrases", []),
            examples=data.get("examples", []),
            parameters=data.get("parameters", []),
            outputs=data.get("outputs", []),
            related_skills=data.get("related_skills", []),
            prerequisites=data.get("prerequisites", {}),
            limitations=data.get("limitations", []),
            best_practices=data.get("best_practices", []),
            interpretation=data.get("interpretation", {}),
            references=data.get("references", {}),
            metadata=data.get("metadata", {}),
            raw_data=data,
        )
    
    def get_all_trigger_terms(self) -> set[str]:
        """Get all terms that should trigger this skill."""
        terms = set()
        
        # Add name and display name
        terms.add(self.name.lower())
        terms.add(self.display_name.lower())
        
        # Add aliases
        for alias in self.aliases:
            terms.add(alias.lower())
        
        # Add trigger phrases
        for phrase in self.trigger_phrases:
            terms.add(phrase.lower())
        
        return terms
    
    def get_required_parameters(self) -> list[dict[str, Any]]:
        """Get list of required parameters."""
        return [p for p in self.parameters if p.get("required", False)]
    
    def get_optional_parameters(self) -> list[dict[str, Any]]:
        """Get list of optional parameters."""
        return [p for p in self.parameters if not p.get("required", False)]


class SkillLoader:
    """
    Loads and manages skill documentation files.
    
    Provides intelligent skill discovery based on:
    - Direct name/alias matching
    - Trigger phrase matching
    - Keyword extraction from capabilities
    - Category-based filtering
    """
    
    def __init__(self, skills_dir: Optional[Path] = None):
        """
        Initialize the skill loader.
        
        Args:
            skills_dir: Directory containing skill YAML files.
                       Defaults to config/skills/ relative to project root.
        """
        if skills_dir is None:
            # Find default skills directory
            project_root = self._find_project_root()
            skills_dir = project_root / "config" / "skills"
        
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}
        self._categories: dict[str, list[str]] = {}  # category -> skill names
        self._trigger_index: dict[str, list[str]] = {}  # term -> skill names
        self._loaded = False
        
        # Load skills on initialization
        self._load_skills()
    
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        # Start from current file and go up
        current = Path(__file__).resolve()
        
        # Look for indicators of project root
        indicators = ["pyproject.toml", "setup.py", "config"]
        
        for parent in current.parents:
            if any((parent / indicator).exists() for indicator in indicators):
                return parent
        
        # Fallback to cwd
        return Path.cwd()
    
    def _load_skills(self) -> None:
        """Load all skill files from the skills directory."""
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            self._loaded = True
            return
        
        # Load each YAML file (except schema.yaml)
        for skill_file in self.skills_dir.glob("*.yaml"):
            if skill_file.name == "schema.yaml":
                continue
            
            try:
                with open(skill_file, "r") as f:
                    data = yaml.safe_load(f)
                
                if data and "name" in data:
                    skill = Skill.from_dict(data)
                    self._skills[skill.name] = skill
                    
                    # Index by category
                    category = skill.category
                    if category not in self._categories:
                        self._categories[category] = []
                    self._categories[category].append(skill.name)
                    
                    # Build trigger index
                    for term in skill.get_all_trigger_terms():
                        if term not in self._trigger_index:
                            self._trigger_index[term] = []
                        self._trigger_index[term].append(skill.name)
                    
                    logger.debug(f"Loaded skill: {skill.name}")
            except Exception as e:
                logger.error(f"Error loading skill file {skill_file}: {e}")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._skills)} skills from {self.skills_dir}")
    
    def reload(self) -> None:
        """Reload all skills from disk."""
        self._skills.clear()
        self._categories.clear()
        self._trigger_index.clear()
        self._loaded = False
        self._load_skills()
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.
        
        Args:
            name: Skill name (e.g., "fastqc", "star")
            
        Returns:
            Skill object if found, None otherwise.
        """
        return self._skills.get(name)
    
    def get_all_skills(self) -> list[Skill]:
        """Get all loaded skills."""
        return list(self._skills.values())
    
    def get_skills_by_category(self, category: str) -> list[Skill]:
        """
        Get all skills in a category.
        
        Args:
            category: Category name (e.g., "quality_control", "alignment")
            
        Returns:
            List of skills in the category.
        """
        skill_names = self._categories.get(category, [])
        return [self._skills[name] for name in skill_names if name in self._skills]
    
    def get_categories(self) -> list[str]:
        """Get list of all categories."""
        return list(self._categories.keys())
    
    def find_skills_for_query(
        self, 
        query: str, 
        limit: int = 5,
        min_score: float = 0.1
    ) -> list[SkillMatch]:
        """
        Find skills that match a user query.
        
        Uses multiple matching strategies:
        1. Direct name/alias match (highest score)
        2. Trigger phrase match
        3. Keyword matching in capabilities
        4. Fuzzy matching on description
        
        Args:
            query: User's natural language query
            limit: Maximum number of results
            min_score: Minimum score threshold
            
        Returns:
            List of SkillMatch objects, sorted by score (highest first).
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        matches: dict[str, SkillMatch] = {}
        
        for skill in self._skills.values():
            score = 0.0
            reasons = []
            
            # 1. Direct name match (highest priority)
            if skill.name.lower() in query_lower:
                score += 1.0
                reasons.append(f"Direct match on tool name '{skill.name}'")
            
            if skill.display_name.lower() in query_lower:
                score += 0.9
                reasons.append(f"Match on display name '{skill.display_name}'")
            
            # 2. Alias match
            for alias in skill.aliases:
                if alias.lower() in query_lower:
                    score += 0.8
                    reasons.append(f"Match on alias '{alias}'")
                    break
            
            # 3. Trigger phrase match
            for phrase in skill.trigger_phrases:
                phrase_lower = phrase.lower()
                phrase_words = set(re.findall(r'\w+', phrase_lower))
                
                # Exact phrase match
                if phrase_lower in query_lower:
                    score += 0.7
                    reasons.append(f"Trigger phrase match: '{phrase}'")
                    break
                
                # Partial word overlap
                overlap = len(query_words & phrase_words)
                if overlap >= 2:
                    partial_score = 0.3 * (overlap / len(phrase_words))
                    if partial_score > 0:
                        score += partial_score
                        reasons.append(f"Partial trigger match ({overlap} words)")
            
            # 4. Capability keyword match
            for capability in skill.capabilities:
                cap_words = set(re.findall(r'\w+', capability.lower()))
                overlap = len(query_words & cap_words)
                if overlap >= 2:
                    cap_score = 0.2 * (overlap / len(cap_words))
                    score += cap_score
                    if cap_score > 0.1:
                        reasons.append(f"Capability match: '{capability[:50]}...'")
            
            # 5. Description match
            desc_words = set(re.findall(r'\w+', skill.description.lower()))
            overlap = len(query_words & desc_words)
            if overlap >= 3:
                score += 0.1 * (overlap / len(desc_words))
                reasons.append("Description keyword match")
            
            # 6. Category relevance
            if skill.category.lower() in query_lower:
                score += 0.3
                reasons.append(f"Category match: {skill.category}")
            
            # Add to matches if above threshold
            if score >= min_score and reasons:
                # Cap score at 1.0
                score = min(score, 1.0)
                matches[skill.name] = SkillMatch(
                    name=skill.name,
                    display_name=skill.display_name,
                    score=score,
                    match_reasons=reasons,
                    skill_data=skill.raw_data,
                )
        
        # Sort by score and return top matches
        sorted_matches = sorted(matches.values(), key=lambda m: m.score, reverse=True)
        return sorted_matches[:limit]
    
    def get_skill_help(self, skill_name: str) -> Optional[str]:
        """
        Generate user-friendly help text for a skill.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Formatted help text, or None if skill not found.
        """
        skill = self.get_skill(skill_name)
        if not skill:
            return None
        
        lines = [
            f"**{skill.display_name}**",
            "",
            skill.description.strip(),
            "",
            "**Capabilities:**",
        ]
        
        for cap in skill.capabilities[:5]:
            lines.append(f"  • {cap}")
        
        if len(skill.capabilities) > 5:
            lines.append(f"  • ... and {len(skill.capabilities) - 5} more")
        
        # Required parameters
        required = skill.get_required_parameters()
        if required:
            lines.extend(["", "**Required Parameters:**"])
            for param in required:
                lines.append(f"  • `{param['name']}`: {param.get('description', 'No description')}")
        
        # Usage examples
        if skill.examples:
            lines.extend(["", "**Example Usage:**"])
            for i, example in enumerate(skill.examples[:2], 1):
                lines.append(f"  {i}. \"{example.get('query', '')}\"")
        
        # Best practices
        if skill.best_practices:
            lines.extend(["", "**Best Practices:**"])
            for practice in skill.best_practices[:3]:
                lines.append(f"  • {practice}")
        
        return "\n".join(lines)
    
    def suggest_workflow(self, skill_name: str) -> list[dict[str, Any]]:
        """
        Suggest a workflow based on skill's related skills.
        
        Args:
            skill_name: Starting skill name
            
        Returns:
            List of workflow steps with related skills.
        """
        skill = self.get_skill(skill_name)
        if not skill:
            return []
        
        workflow = []
        
        # Get preceding skills
        for related in skill.related_skills:
            if isinstance(related, dict) and related.get("relationship") == "precedes":
                related_skill = self.get_skill(related.get("name", ""))
                if related_skill:
                    workflow.append({
                        "step": "before",
                        "skill": related_skill.name,
                        "display_name": related_skill.display_name,
                        "description": related.get("description", ""),
                    })
        
        # Current skill
        workflow.append({
            "step": "main",
            "skill": skill.name,
            "display_name": skill.display_name,
            "description": skill.description[:100],
        })
        
        # Get following skills
        for related in skill.related_skills:
            if isinstance(related, dict) and related.get("relationship") == "follows":
                related_skill = self.get_skill(related.get("name", ""))
                if related_skill:
                    workflow.append({
                        "step": "after",
                        "skill": related_skill.name,
                        "display_name": related_skill.display_name,
                        "description": related.get("description", ""),
                    })
        
        return workflow
    
    def get_skill_parameters_for_prompt(self, skill_name: str) -> str:
        """
        Get skill parameters formatted for LLM prompts.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Formatted string describing parameters.
        """
        skill = self.get_skill(skill_name)
        if not skill:
            return ""
        
        lines = [f"Parameters for {skill.display_name}:"]
        
        for param in skill.parameters:
            name = param.get("name", "unknown")
            ptype = param.get("type", "any")
            required = "required" if param.get("required") else "optional"
            desc = param.get("description", "")
            default = param.get("default", "")
            
            line = f"  - {name} ({ptype}, {required}): {desc}"
            if default:
                line += f" [default: {default}]"
            lines.append(line)
        
        return "\n".join(lines)


# Global singleton instance
_skill_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """
    Get the global SkillLoader instance.
    
    Returns:
        SkillLoader singleton.
    """
    global _skill_loader
    if _skill_loader is None:
        _skill_loader = SkillLoader()
    return _skill_loader


def reset_skill_loader() -> None:
    """Reset the global SkillLoader (for testing)."""
    global _skill_loader
    _skill_loader = None


# Convenience functions
def find_skills(query: str, limit: int = 5) -> list[SkillMatch]:
    """Find skills matching a query."""
    return get_skill_loader().find_skills_for_query(query, limit=limit)


def get_skill_help_text(skill_name: str) -> Optional[str]:
    """Get help text for a skill."""
    return get_skill_loader().get_skill_help(skill_name)
