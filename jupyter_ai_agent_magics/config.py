import json
import os
from dataclasses import dataclass, asdict, fields
from enum import StrEnum
from pathlib import Path
from typing import Optional, Dict, Any
from jupyter_core.paths import jupyter_data_dir


class ModelProvider(StrEnum):
    Anthropic = "anthropic"
    AnthropicVertex = "vertex_anthropic"
    Google = "google"
    OpenAI = "openai"


@dataclass
class ModelProviderSettings:
    """Container for model-related settings."""
    model_name: str = "claude-3-opus-20240229"
    api_key: Optional[str] = None
    project_id: Optional[str] = None
    location: Optional[str] = "us-central1"
    temperature: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelProviderSettings':
        """Create instance from dictionary."""
        # Only include fields that are defined in the dataclass
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


@dataclass
class Settings:
    model_provider: ModelProvider
    model_provider_settings: Dict[ModelProvider, ModelProviderSettings]


class ConfigManager:
    """Singleton configuration manager with JSON persistence."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Setup config file path
        data_dir = jupyter_data_dir()
        self.config_dir = Path(data_dir) / 'ai_extension'
        self.config_file = self.config_dir / 'config.json'
        
        # Initialize settings
        self._model_provider = ModelProvider.Anthropic
        self._model_provider_settings = self._create_default_settings()
        
        # Load from file
        self._load_config()
        self._initialized = True
    
    @classmethod
    @property
    def instance(cls) -> 'ConfigManager':
        """Get the singleton instance."""
        return cls()
    
    def _create_default_settings(self) -> Dict[ModelProvider, ModelProviderSettings]:
        """Create default settings for all providers."""
        defaults = {
            ModelProvider.Anthropic: ModelProviderSettings(
                model_name="claude-3-opus-20240229"
            ),
            ModelProvider.AnthropicVertex: ModelProviderSettings(
                model_name="claude-3-opus-20240229",
                location="us-central1"
            ),
            ModelProvider.Google: ModelProviderSettings(
                model_name="gemini-pro",
                location="us-central1"
            ),
            ModelProvider.OpenAI: ModelProviderSettings(
                model_name="gpt-4"
            ),
        }
        return defaults
    
    def _load_config(self):
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            return
            
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            # Load model provider
            if 'model_provider' in data:
                self._model_provider = ModelProvider(data['model_provider'])
            
            # Load provider settings
            if 'model_provider_settings' in data:
                for provider_str, settings_dict in data['model_provider_settings'].items():
                    try:
                        provider = ModelProvider(provider_str)
                        self._model_provider_settings[provider] = ModelProviderSettings.from_dict(settings_dict)
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not load settings for provider {provider_str}: {e}")
                        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config from {self.config_file}: {e}")
    
    def _save_config(self):
        """Save current configuration to JSON file."""
        # Ensure directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        data = {
            'model_provider': self._model_provider.value,
            'model_provider_settings': {
                provider.value: asdict(settings)
                for provider, settings in self._model_provider_settings.items()
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save config to {self.config_file}: {e}")
    
    @property
    def model_provider(self) -> ModelProvider:
        """Get current model provider."""
        return self._model_provider
    
    @model_provider.setter
    def model_provider(self, value: ModelProvider):
        """Set current model provider."""
        if not isinstance(value, ModelProvider):
            raise ValueError(f"Expected ModelProvider, got {type(value)}")
        self._model_provider = value
        self._save_config()
    
    @property
    def model_provider_settings(self) -> ModelProviderSettings:
        """Get settings for the current model provider."""
        return self._model_provider_settings[self._model_provider]
    
    def get_settings_for_provider(self, provider: Optional[ModelProvider] = None) -> ModelProviderSettings:
        """Get settings for a specific provider (defaults to current)."""
        if provider is None:
            provider = self._model_provider
        return self._model_provider_settings[provider]
    
    def save_settings_for_provider(self, provider: ModelProvider, settings: ModelProviderSettings):
        """Save settings for a specific provider."""
        if not isinstance(provider, ModelProvider):
            raise ValueError(f"Expected ModelProvider, got {type(provider)}")
        if not isinstance(settings, ModelProviderSettings):
            raise ValueError(f"Expected ModelProviderSettings, got {type(settings)}")
            
        self._model_provider_settings[provider] = settings
        self._save_config()
    
    # Convenience properties for current provider settings
    @property
    def model_name(self) -> str:
        """Get model name for current provider."""
        return self.model_provider_settings.model_name
    
    @model_name.setter
    def model_name(self, value: str):
        """Set model name for current provider."""
        self.model_provider_settings.model_name = value
        self._save_config()
    
    @property
    def api_key(self) -> Optional[str]:
        """Get API key for current provider."""
        return self.model_provider_settings.api_key
    
    @api_key.setter
    def api_key(self, value: Optional[str]):
        """Set API key for current provider."""
        self.model_provider_settings.api_key = value
        self._save_config()
    
    @property
    def project_id(self) -> Optional[str]:
        """Get project ID for current provider."""
        return self.model_provider_settings.project_id
    
    @project_id.setter
    def project_id(self, value: Optional[str]):
        """Set project ID for current provider."""
        self.model_provider_settings.project_id = value
        self._save_config()
    
    @property
    def location(self) -> Optional[str]:
        """Get location for current provider."""
        return self.model_provider_settings.location
    
    @location.setter
    def location(self, value: Optional[str]):
        """Set location for current provider."""
        self.model_provider_settings.location = value
        self._save_config()
    
    @property
    def temperature(self) -> Optional[float]:
        """Get temperature for current provider."""
        return self.model_provider_settings.temperature
    
    @temperature.setter
    def temperature(self, value: Optional[float]):
        """Set temperature for current provider."""
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"Temperature must be a number, got {type(value)}")
        self.model_provider_settings.temperature = value
        self._save_config()
