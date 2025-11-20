"""
BioPipelines Core Module
========================

Core utilities for file I/O, configuration, and logging.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional
import logging
import yaml
import json

__version__ = "0.1.0"


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level (default: INFO)
    log_file : Path, optional
        Path to log file
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
        
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If file format is not supported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    
    with open(config_path, 'r') as f:
        if suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json"
            )


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_path : str or Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = output_path.suffix.lower()
    
    with open(output_path, 'w') as f:
        if suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(
                f"Unsupported format: {suffix}. Use .yaml, .yml, or .json"
            )


def validate_file_exists(file_path: Union[str, Path], file_type: str = "file") -> Path:
    """
    Validate that a file exists and return Path object.
    
    Parameters
    ----------
    file_path : str or Path
        Path to validate
    file_type : str
        Description of file type for error message
        
    Returns
    -------
    Path
        Validated Path object
        
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_type} not found: {path}")
    return path


def create_output_dir(output_dir: Union[str, Path], exist_ok: bool = True) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Parameters
    ----------
    output_dir : str or Path
        Output directory path
    exist_ok : bool
        If True, don't raise error if directory exists
        
    Returns
    -------
    Path
        Created directory path
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path
