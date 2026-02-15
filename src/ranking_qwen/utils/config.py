"""Configuration management utilities."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {config_path.suffix}. "
                "Use .yaml, .yml, or .json"
            )


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    
    Raises:
        ValueError: If file format is not supported
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif config_path.suffix == ".json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(
                f"Unsupported config format: {config_path.suffix}. "
                "Use .yaml, .yml, or .json"
            )
