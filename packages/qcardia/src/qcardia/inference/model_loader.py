"""Model instantiation from a run config."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from qcardia_models.models import UNet2d

# The public qcardia-models has no context-aware model. Plain UNets still load.
try:
    from qcardia_models.models import ContextUNet2d
except ImportError:
    ContextUNet2d = None


def is_transformer_model(model_config: Dict) -> bool:
    """Same rule as qcardia_models.training_utils.is_transformer_model."""
    return bool(model_config.get("transformer_params"))


from .config_handler import InferenceConfig, unwrap_wandb_config

WEIGHTS_NAMES = ("last_model.pt", "best_model.pt")


def determine_model_type(config: Dict) -> str:
    """Return "unet" or "transformer" for a run config.

    Decided by the presence of transformer_params, matching
    qcardia_models.training_utils.is_transformer_model, which is what training
    used. Matching on the model name instead misroutes any name that does not
    happen to contain the expected substring.
    """
    if isinstance(config, InferenceConfig):
        return "transformer" if config.transformer_params else "unet"

    config = unwrap_wandb_config(config)
    if "model" in config:
        return "transformer" if is_transformer_model(config["model"]) else "unet"
    if "unet" in config:
        return "unet"
    raise ValueError("Cannot determine model type from config")


def load_model_from_config(
    config: Dict, wandb_run_path: Path, device: str = "cpu"
) -> nn.Module:
    """Build the model a config describes and load its weights.

    Args:
        config: Run config, raw or already parsed into an InferenceConfig.
        wandb_run_path: Run directory holding the weights, either directly or
            in a files/ subdirectory.
        device: Device to load onto.

    Returns:
        The model, in eval mode.
    """
    parsed = config if isinstance(config, InferenceConfig) else InferenceConfig(config)

    if determine_model_type(config) == "transformer":
        if ContextUNet2d is None:
            raise ImportError(
                "This qcardia-models has no ContextUNet2d. Context-aware models "
                "need the qcardia-models-dev context-unet branch."
            )
        model = ContextUNet2d(
            nr_input_channels=parsed.nr_image_channels,
            channels_list=parsed.channels_list,
            nr_output_classes=parsed.nr_classes,
            nr_output_scales=parsed.nr_output_scales,
            transformer_params=parsed.transformer_params,
            context_window=parsed.context_window,
            la_vector_dim=parsed.la_vector_dim,
            la_vector_integration=parsed.la_vector_integration,
        )
    else:
        # The public UNet2d takes no LA-vector arguments, so pass them only for
        # an LA-conditioned model (which then needs qcardia-models-dev).
        la_kwargs = {}
        if parsed.la_vector_dim:
            la_kwargs = {
                "la_vector_dim": parsed.la_vector_dim,
                "la_vector_integration": parsed.la_vector_integration,
            }
        model = UNet2d(
            nr_input_channels=parsed.nr_image_channels,
            channels_list=parsed.channels_list,
            nr_output_classes=parsed.nr_classes,
            nr_output_scales=parsed.nr_output_scales,
            **la_kwargs,
        )

    candidates = [wandb_run_path / name for name in WEIGHTS_NAMES]
    candidates += [wandb_run_path / "files" / name for name in WEIGHTS_NAMES]
    weights_path = next((p for p in candidates if p.exists()), None)
    if weights_path is None:
        raise FileNotFoundError(
            f"No {' or '.join(WEIGHTS_NAMES)} in {wandb_run_path} or its files/ "
            f"subdirectory."
        )

    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=False)
    )
    return model.to(device).eval()


def resolve_lax_model_path(
    lax_model_path: Optional[Path], raw_config: Dict[str, Any], wandb_run_path: Path
) -> Path:
    """Resolve the weights used to pre-segment the long-axis view.

    Falls back to the conditioned model's own `weights_path`, the
    unconditioned checkpoint it was initialised from. Configs written by a
    run that started from scratch record that field as the string "none",
    so an unset value arrives here as text rather than as None.
    """
    if lax_model_path is not None:
        return Path(lax_model_path)

    model_config = raw_config.get("model", {})
    if isinstance(model_config, dict) and "value" in model_config:
        model_config = model_config["value"]
    weights_path = (
        model_config.get("weights_path") if isinstance(model_config, dict) else None
    )
    if weights_path and str(weights_path).lower() not in ("none", "null", ""):
        return Path(weights_path)
    raise ValueError(
        f"No long-axis segmentation model given, and the model at "
        f"{wandb_run_path} records model.weights_path as {weights_path!r}, so "
        f"there is nothing to fall back on. Pass lax_model_path explicitly."
    )
