"""
Secrets Management
==================

Loads API keys and secrets from .secrets/ directory into environment variables.

This module automatically loads secrets on import, making API keys available
to all LLM adapters without manual environment variable setup.

Secrets are stored in:
    .secrets/openai_key       -> OPENAI_API_KEY
    .secrets/lightning_key    -> LIGHTNING_API_KEY
    .secrets/anthropic_key    -> ANTHROPIC_API_KEY
    
Usage:
    # Just import to load secrets
    from workflow_composer import secrets
    
    # Or call explicitly
    secrets.load_secrets()
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def find_secrets_dir() -> Optional[Path]:
    """
    Find the .secrets directory.
    
    Looks in:
    1. Current working directory
    2. Project root (where this package is installed)
    3. User home directory
    
    Returns:
        Path to .secrets directory, or None if not found
    """
    # Try current directory
    cwd_secrets = Path.cwd() / ".secrets"
    if cwd_secrets.exists():
        return cwd_secrets
    
    # Try package root (go up from src/workflow_composer)
    try:
        package_root = Path(__file__).parent.parent.parent
        root_secrets = package_root / ".secrets"
        if root_secrets.exists():
            return root_secrets
    except:
        pass
    
    # Try home directory
    home_secrets = Path.home() / ".secrets"
    if home_secrets.exists():
        return home_secrets
    
    return None


def load_secrets(secrets_dir: Optional[Path] = None, override: bool = False) -> Dict[str, str]:
    """
    Load secrets from .secrets/ directory into environment variables.
    
    Args:
        secrets_dir: Path to secrets directory (auto-detected if None)
        override: Whether to override existing environment variables
        
    Returns:
        Dict of loaded secrets (name -> value)
    """
    if secrets_dir is None:
        secrets_dir = find_secrets_dir()
    
    if secrets_dir is None:
        logger.debug("No .secrets directory found")
        return {}
    
    # Mapping of secret file names to environment variable names
    secret_mappings = {
        "openai_key": "OPENAI_API_KEY",
        "lightning_key": "LIGHTNING_API_KEY",
        "anthropic_key": "ANTHROPIC_API_KEY",
        "huggingface_token": "HUGGINGFACE_TOKEN",
        "hf_token": "HF_TOKEN",
    }
    
    loaded = {}
    
    for filename, env_var in secret_mappings.items():
        secret_file = secrets_dir / filename
        
        if not secret_file.exists():
            continue
        
        # Skip if already set (unless override=True)
        if env_var in os.environ and not override:
            logger.debug(f"{env_var} already set, skipping")
            continue
        
        try:
            # Read secret (strip whitespace)
            secret_value = secret_file.read_text().strip()
            
            if secret_value:
                os.environ[env_var] = secret_value
                loaded[env_var] = secret_value
                logger.info(f"Loaded {env_var} from {filename}")
            else:
                logger.warning(f"{secret_file} is empty")
                
        except Exception as e:
            logger.error(f"Failed to load {secret_file}: {e}")
    
    return loaded


def get_secret(name: str) -> Optional[str]:
    """
    Get a secret by environment variable name.
    
    Args:
        name: Environment variable name (e.g., "OPENAI_API_KEY")
        
    Returns:
        Secret value, or None if not set
    """
    return os.environ.get(name)


def check_secrets() -> Dict[str, bool]:
    """
    Check which secrets are available.
    
    Returns:
        Dict mapping secret names to availability (True/False)
    """
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "LIGHTNING_API_KEY": bool(os.environ.get("LIGHTNING_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "HUGGINGFACE_TOKEN": bool(os.environ.get("HUGGINGFACE_TOKEN")),
    }


def save_secret(name: str, value: str, secrets_dir: Optional[Path] = None) -> bool:
    """
    Save a secret to .secrets/ directory.
    
    Args:
        name: Environment variable name (e.g., "OPENAI_API_KEY")
        value: Secret value
        secrets_dir: Path to secrets directory (auto-detected if None)
        
    Returns:
        True if saved successfully, False otherwise
    """
    if secrets_dir is None:
        secrets_dir = find_secrets_dir()
    
    if secrets_dir is None:
        # Create in current directory
        secrets_dir = Path.cwd() / ".secrets"
        secrets_dir.mkdir(mode=0o700, exist_ok=True)
    
    # Reverse mapping
    reverse_mappings = {
        "OPENAI_API_KEY": "openai_key",
        "LIGHTNING_API_KEY": "lightning_key",
        "ANTHROPIC_API_KEY": "anthropic_key",
        "HUGGINGFACE_TOKEN": "huggingface_token",
        "HF_TOKEN": "hf_token",
    }
    
    filename = reverse_mappings.get(name)
    if not filename:
        logger.error(f"Unknown secret name: {name}")
        return False
    
    secret_file = secrets_dir / filename
    
    try:
        secret_file.write_text(value)
        secret_file.chmod(0o600)  # Read/write for owner only
        os.environ[name] = value
        logger.info(f"Saved {name} to {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save secret: {e}")
        return False


# Auto-load secrets on import
try:
    loaded = load_secrets()
    if loaded:
        logger.info(f"Auto-loaded {len(loaded)} secrets from .secrets/")
except Exception as e:
    logger.warning(f"Failed to auto-load secrets: {e}")
