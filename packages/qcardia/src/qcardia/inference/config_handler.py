"""Config parsing for inference.

Reads both the older config["unet"] layout and the current config["model"]
layout, and unwraps the nesting WandB adds when it exports a run config.
"""

from typing import Any, Dict


class InferenceConfig:
    """The subset of a run config that inference needs.

    Example:
        >>> config = InferenceConfig(yaml.safe_load(config_path.read_text()))
        >>> config.needs_context()
        True
    """

    def __init__(self, raw_config: Dict[str, Any]):
        config = unwrap_wandb_config(raw_config)
        model = config.get("model") or config.get("unet") or {}
        data = config.get("data", {})

        self.nr_image_channels = model.get("nr_image_channels", 1)
        self.channels_list = _required(model, "channels_list")
        self.nr_classes = _required(model, "nr_output_classes")
        self.nr_output_scales = model.get("nr_output_scales", -1)
        self.transformer_params = model.get("transformer_params")

        context = data.get("context_window") or {}
        self.context_window = context or None
        self.spatial_window = context.get("spatial", 0)
        self.temporal_window = context.get("temporal", 0)
        self.spatial_stride = context.get("spatial_stride", 1)
        self.temporal_stride = context.get("temporal_stride", 1)

        self.la_vector_dim = model.get("la_vector_dim")
        self.la_vector_integration = model.get("la_vector_integration")

        self.target_size = _required(data, "target_size")
        self.target_pixdim = _required(data, "target_pixdim")
        self.image_grid_sample_mode = data.get("image_grid_sample_mode", "bilinear")

        self.decode_all_context = bool(
            (self.transformer_params or {}).get("decode_all_context", False)
        )

    def needs_context(self) -> bool:
        """Whether the model consumes neighbouring slices or frames."""
        return self.spatial_window > 0 or self.temporal_window > 0


def _required(section: Dict[str, Any], key: str) -> Any:
    if key not in section:
        raise ValueError(f"Required config field missing: {key}")
    return section[key]


def unwrap_wandb_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the {"value": ...} wrapper WandB puts around every top-level key."""
    if not isinstance(config, dict):
        return config
    if not any(
        isinstance(v, dict) and "value" in v for v in list(config.values())[:5]
    ):
        return config
    return {
        key: value["value"] if isinstance(value, dict) and "value" in value else value
        for key, value in config.items()
    }
