"""
Configuration handler for unified config parsing.

Supports both legacy (config["unet"]) and new (config["model"]) formats.
"""

from typing import Any, Dict, List, Optional, Union


class InferenceConfig:
    """
    Unified config parser for legacy and new config formats.
    
    This class handles:
    - Legacy configs with config["unet"] keys
    - New configs with config["model"] keys  
    - Context window parameters for transformers
    - LA vector conditioning parameters
    - Data preprocessing parameters
    
    Example usage:
        >>> raw_config = yaml.load(config_path)
        >>> config = InferenceConfig(raw_config)
        >>> print(config.model_name)
        'UNet_Transformer'
    """
    
    def __init__(self, raw_config: Dict[str, Any]):
        """
        Initialize config parser.
        
        Args:
            raw_config: Raw configuration dictionary from YAML
        """
        # Unwrap WandB config format if needed (everything in "value" keys)
        self.raw_config = self._unwrap_wandb_config(raw_config)
        self._parse()
    
    def _unwrap_wandb_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unwrap WandB config format where all values are in nested 'value' keys.
        
        Args:
            config: Raw config that might have WandB structure
            
        Returns:
            Unwrapped config dict
        """
        # Check if this looks like a WandB config (has top-level keys with "value" dicts)
        is_wandb_format = False
        if isinstance(config, dict):
            # Sample some top-level keys to check format
            for key in list(config.keys())[:5]:
                if isinstance(config.get(key), dict) and "value" in config[key]:
                    is_wandb_format = True
                    break
        
        if not is_wandb_format:
            return config
        
        # Unwrap the WandB format recursively
        unwrapped = {}
        for key, value in config.items():
            if isinstance(value, dict) and "value" in value:
                unwrapped[key] = value["value"]
            else:
                unwrapped[key] = value
        
        return unwrapped
    
    def _parse(self):
        """Parse all configuration parameters."""
        # Model architecture
        self.model_name = self._get_model_name()
        self.nr_image_channels = self._get_field(
            ["model.nr_image_channels", "unet.nr_image_channels"], 
            default=1
        )
        self.channels_list = self._get_field(
            ["model.channels_list", "unet.channels_list"],
            required=True
        )
        self.nr_classes = self._get_field(
            ["model.nr_output_classes", "unet.nr_output_classes"],
            required=True
        )
        self.nr_output_scales = self._get_field(
            ["model.nr_output_scales", "unet.nr_output_scales"],
            default=-1
        )
        
        # Transformer-specific parameters
        self.transformer_params = self.raw_config.get("model", {}).get("transformer_params")
        
        # Context window configuration
        context_config = self.raw_config.get("data", {}).get("context_window", {})
        self.context_window = context_config if context_config else None
        self.spatial_window = context_config.get("spatial", 0) if context_config else 0
        self.temporal_window = context_config.get("temporal", 0) if context_config else 0
        self.spatial_stride = context_config.get("spatial_stride", 1) if context_config else 1
        self.temporal_stride = context_config.get("temporal_stride", 1) if context_config else 1
        
        # LA vector conditioning
        self.la_vector_dim = self.raw_config.get("model", {}).get("la_vector_dim")
        self.la_vector_integration = self.raw_config.get("model", {}).get("la_vector_integration")
        
        # Data preprocessing parameters
        self.target_size = self._get_field(["data.target_size"], required=True)
        self.target_pixdim = self._get_field(["data.target_pixdim"], required=True)
        self.image_grid_sample_mode = self._get_field(
            ["data.image_grid_sample_mode"], 
            default="bilinear"
        )
        
        # Special transformer settings
        self.decode_all_context = False
        if self.transformer_params:
            self.decode_all_context = self.transformer_params.get("decode_all_context", False)
        
        # Dimensionality
        self.dimensionality = self.raw_config.get("data", {}).get("dimensionality", "2D")
    
    def _get_model_name(self) -> str:
        """
        Infer model name from config.
        
        Priority:
        1. config["model"]["name"]
        2. config["unet"] -> "UNet"
        
        Returns:
            Model name string
            
        Raises:
            ValueError: If model name cannot be determined
        """
        if "model" in self.raw_config and "name" in self.raw_config["model"]:
            return self.raw_config["model"]["name"]
        elif "unet" in self.raw_config:
            return "UNet"
        else:
            raise ValueError("Cannot determine model name from config. "
                           "Expected config['model']['name'] or config['unet']")
    
    def _get_field(self, paths: List[str], default: Any = None, required: bool = False) -> Any:
        """
        Get field from config using fallback paths.
        
        Args:
            paths: List of paths to try (in order of priority)
            default: Default value if not found
            required: If True, raises error when not found
            
        Returns:
            Field value or default
            
        Raises:
            ValueError: If required=True and field not found
        """
        for path in paths:
            value = self._navigate_path(path)
            if value is not None:
                return value
        
        if required:
            raise ValueError(f"Required config field not found. Tried paths: {paths}")
        return default
    
    def _navigate_path(self, path: str) -> Optional[Any]:
        """
        Navigate nested dict using dot notation.
        
        Args:
            path: Dot-separated path (e.g., "data.target_size")
            
        Returns:
            Value at path or None if not found
        """
        keys = path.split(".")
        current = self.raw_config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def is_transformer(self) -> bool:
        """
        Check if model is a transformer.
        
        Returns:
            True if transformer-based model
        """
        model_name_lower = self.model_name.lower()
        return any(token in model_name_lower for token in ["trans", "att", "tc", "sc"])
    
    def needs_context(self) -> bool:
        """
        Check if model needs context window.
        
        Returns:
            True if context window is configured
        """
        return (self.context_window is not None and 
                (self.spatial_window > 0 or self.temporal_window > 0))
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"InferenceConfig(model={self.model_name}, "
                f"is_transformer={self.is_transformer()}, "
                f"needs_context={self.needs_context()}, "
                f"spatial_window={self.spatial_window}, "
                f"temporal_window={self.temporal_window})")
