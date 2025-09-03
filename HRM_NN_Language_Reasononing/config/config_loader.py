# config/config_loader.py
import yaml
import os
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """Handles loading and merging of YAML configuration files.

    This class supports configuration inheritance, where a child configuration
    file can extend a parent configuration file using the 'extends' keyword.
    It also caches loaded configurations to avoid redundant file I/O.
    """

    def __init__(self, base_config_dir: str = "config/"):
        """Initializes the ConfigLoader.

        Args:
            base_config_dir (str): The base directory where configuration
                                   files are located. Defaults to "config/".
        """
        self.base_config_dir = Path(base_config_dir)
        self.loaded_configs = {}

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads a YAML configuration file with support for inheritance.

        If a configuration file contains an 'extends' key, it will first load
        the parent configuration and then recursively merge the child's
        configuration into the parent's.

        Args:
            config_path (str): The path to the configuration file.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded and merged
                            configuration.
        """
        config_path = Path(config_path)

        # Check if already loaded
        if config_path in self.loaded_configs:
            return self.loaded_configs[config_path]

        # Load the config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Handle inheritance with 'extends' keyword
        if 'extends' in config:
            parent_config_path = self.base_config_dir / config['extends']
            parent_config = self.load_config(parent_config_path)

            # Merge configurations (child overrides parent)
            merged_config = self._deep_merge(parent_config, config)
            config = merged_config

        # Remove the extends key as it's no longer needed
        config.pop('extends', None)

        # Cache the loaded config
        self.loaded_configs[config_path] = config

        return config

    def _deep_merge(self, parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        """Deeply merges two dictionaries.

        The child's values will override the parent's values. If a key exists
        in both dictionaries and its value is a dictionary, the method will
        recursively merge them.

        Args:
            parent (Dict[str, Any]): The parent dictionary.
            child (Dict[str, Any]): The child dictionary.

        Returns:
            Dict[str, Any]: The merged dictionary.
        """
        result = parent.copy()

        for key, value in child.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Performs a basic validation of the configuration.

        Checks for the presence of required top-level sections in the
        configuration dictionary.

        Args:
            config (Dict[str, Any]): The configuration dictionary to validate.

        Returns:
            bool: True if the configuration is valid.

        Raises:
            ValueError: If a required section is missing.
        """
        required_sections = [
            'system', 'model', 'training', 'data_sources',
            'inference', 'paths', 'logging'
        ]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        return True

# Global config loader instance
config_loader = ConfigLoader()

def load_config(config_path: str) -> Dict[str, Any]:
    """A convenience function to load a configuration file.

    This function uses a global instance of the ConfigLoader.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the loaded configuration.
    """
    return config_loader.load_config(config_path)
