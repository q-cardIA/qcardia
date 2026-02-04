"""
Model loader for dynamic model instantiation.

Supports loading UNet2d and UNet_Transformer models based on config,
with optional LA vector conditioning.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

# Add qcardia_models to path
models_path = Path(__file__).parent.parent.parent.parent.parent / "qcardia-models-dev" / "src"
if models_path.exists():
    sys.path.insert(0, str(models_path))

# Import models from qcardia_models
try:
    from qcardia_models.models import UNet2d, UNet_Transformer
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Warning: qcardia_models not available. Model loading will fail.")


def determine_model_type(config: Dict) -> str:
    """
    Determine model type from config.
    
    Args:
        config: Raw config dictionary or InferenceConfig object
        
    Returns:
        Model type: "unet" or "transformer"
        
    Example:
        >>> model_type = determine_model_type(config)
        >>> print(model_type)  # "transformer"
    """
    # Unwrap WandB config if needed
    config = _unwrap_wandb_config(config)
    
    # Handle InferenceConfig objects
    if hasattr(config, 'model_name'):
        model_name = config.model_name.lower()
    else:
        # Try different config structures
        if "model" in config and "name" in config["model"]:
            model_name = config["model"]["name"].lower()
        elif "unet" in config:
            return "unet"
        else:
            raise ValueError("Cannot determine model type from config")
    
    # Check for transformer indicators
    if any(token in model_name for token in ["trans", "att", "tc", "sc"]):
        return "transformer"
    elif "unet" in model_name or "baseline" in model_name:
        return "unet"
    else:
        # Default to transformer if unsure (safer for new models)
        return "transformer"


def _unwrap_wandb_config(config: Dict) -> Dict:
    """
    Unwrap WandB config format where all values are in nested 'value' keys.
    
    Args:
        config: Raw config that might have WandB structure
        
    Returns:
        Unwrapped config dict
    """
    if not isinstance(config, dict):
        return config
        
    # Check if this looks like a WandB config (has top-level keys with "value" dicts)
    is_wandb_format = False
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


def load_model_from_config(config: Dict, wandb_run_path: Path, 
                          device: str = "cpu") -> nn.Module:
    """
    Load model from config and weights.
    
    Args:
        config: Configuration dictionary (raw or InferenceConfig)
        wandb_run_path: Path to WandB run directory containing model weights
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
        
    Raises:
        FileNotFoundError: If weights file not found
        ValueError: If model type cannot be determined
        
    Example:
        >>> model = load_model_from_config(config, wandb_path)
        >>> prediction = model(input_tensor)
    """
    if not MODELS_AVAILABLE:
        raise ImportError("qcardia_models package not available. Cannot load models.")
    
    # Unwrap WandB config if needed
    config = _unwrap_wandb_config(config)
    
    # Extract parameters from config
    if hasattr(config, 'model_name'):
        # InferenceConfig object
        model_name = config.model_name
        nr_image_channels = config.nr_image_channels
        channels_list = config.channels_list
        nr_output_classes = config.nr_classes
        nr_output_scales = config.nr_output_scales
        transformer_params = config.transformer_params
        context_window = config.context_window
        la_vector_dim = config.la_vector_dim
        la_vector_integration = config.la_vector_integration
    else:
        # Raw config dict
        model_name = config.get("model", {}).get("name", "UNet")
        nr_image_channels = config.get("model", config.get("unet", {})).get("nr_image_channels", 1)
        channels_list = config.get("model", config.get("unet", {})).get("channels_list")
        nr_output_classes = config.get("model", config.get("unet", {})).get("nr_output_classes")
        nr_output_scales = config.get("model", config.get("unet", {})).get("nr_output_scales", -1)
        transformer_params = config.get("model", {}).get("transformer_params")
        context_window = config.get("data", {}).get("context_window")
        la_vector_dim = config.get("model", {}).get("la_vector_dim")
        la_vector_integration = config.get("model", {}).get("la_vector_integration")
    
    # Determine model type
    model_type = determine_model_type(config)
    
    # Create model
    if model_type == "unet":
        model = UNet2d(
            nr_input_channels=nr_image_channels,
            channels_list=channels_list,
            nr_output_classes=nr_output_classes,
            nr_output_scales=nr_output_scales,
            la_vector_dim=la_vector_dim,
            la_vector_integration=la_vector_integration
        )
    elif model_type == "transformer":
        model = UNet_Transformer(
            nr_input_channels=nr_image_channels,
            channels_list=channels_list,
            nr_output_classes=nr_output_classes,
            nr_output_scales=nr_output_scales,
            transformer_params=transformer_params,
            context_window=context_window,
            la_vector_dim=la_vector_dim,
            la_vector_integration=la_vector_integration
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    weights_path = wandb_run_path / "files" / "last_model.pt"
    if not weights_path.exists():
        # Try alternative filename
        weights_path = wandb_run_path / "files" / "best_model.pt"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {wandb_run_path / 'files'}. "
                              f"Expected 'last_model.pt' or 'best_model.pt'")
    
    model_weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(model_weights)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Ensure all modules in eval mode (disable dropout/batchnorm)
    for module in model.modules():
        if hasattr(module, 'training'):
            module.training = False
    
    return model


def get_model_info(model: nn.Module) -> Dict:
    """
    Get information about a loaded model.
    
    Args:
        model: Loaded PyTorch model
        
    Returns:
        Dictionary with model information
    """
    info = {
        "type": model.__class__.__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": next(model.parameters()).device if len(list(model.parameters())) > 0 else "unknown",
        "is_eval": not model.training
    }
    
    # Check for transformer-specific attributes
    if hasattr(model, 'transformer'):
        info["has_transformer"] = True
        if hasattr(model.transformer, 'depth'):
            info["transformer_depth"] = model.transformer.depth
    else:
        info["has_transformer"] = False
    
    # Check for LA conditioning
    if hasattr(model, 'la_vector_dim'):
        info["la_vector_dim"] = model.la_vector_dim
        info["has_la_conditioning"] = model.la_vector_dim is not None
    else:
        info["has_la_conditioning"] = False
    
    return info
