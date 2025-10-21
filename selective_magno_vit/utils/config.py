import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration handler with environment variable support."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._resolve_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _resolve_paths(self):
        """Resolve relative paths and environment variables."""
        # Allow environment variable overrides
        for section in ['data', 'output']:
            if section in self.config:
                for key, value in self.config[section].items():
                    if isinstance(value, str):
                        # Resolve environment variables
                        value = os.path.expandvars(value)
                        # Resolve ${key} references within config
                        while '${' in value:
                            for replace_key, replace_val in self.config[section].items():
                                value = value.replace(f'${{{section}.{replace_key}}}', str(replace_val))
                        self.config[section][key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot-notation key."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __repr__(self):
        return f"Config({self.config_path})"