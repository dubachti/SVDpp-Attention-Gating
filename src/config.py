"""
Configuration loader for YAML config files.
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def get_config_dir() -> Path:
    """Get the path to the configs directory."""
    return Path(__file__).parent.parent / "configs"


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_name: Name of the config file (with or without .yaml extension)
    
    Returns:
        Dictionary containing the configuration
    """
    config_dir = get_config_dir()
    
    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"
    
    config_path = config_dir / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_als_config() -> Dict[str, Any]:
    """Load ALS model configuration."""
    return load_config("als")


def load_svdpp_config() -> Dict[str, Any]:
    """Load SVD++ model configuration."""
    return load_config("svdpp")


def load_svdppag_config() -> Dict[str, Any]:
    """Load SVD++ AG model configuration."""
    return load_config("svdppag")


def load_grid_search_als_config() -> Dict[str, Any]:
    """Load ALS grid search configuration."""
    return load_config("grid_search_als")


def load_grid_search_svdppag_config() -> Dict[str, Any]:
    """Load SVD++ AG grid search configuration."""
    return load_config("grid_search_svdppag")
