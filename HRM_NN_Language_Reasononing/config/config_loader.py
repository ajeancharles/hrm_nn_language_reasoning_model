# config/config_loader.py
import yaml
import os
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """Handles loading and merging configuration files"""
    
    def __init__(self, base_config_dir: str = "config/"):
        self.base_config_dir = Path(base_config_dir)
        self.loaded_configs = {}
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with inheritance support"""
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
        """Deep merge two dictionaries"""
        result = parent.copy()
        
        for key, value in child.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration completeness"""
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
    """Convenience function to load configuration"""
    return config_loader.load_config(config_path)