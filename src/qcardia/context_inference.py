"""
Config-driven model dispatch for CineSeries/LGESeries inference.

Chooses between a plain UNet2d and a spatial/temporal context-integrating
ContextUNet2d based on the model config. For context models, builds
per-(slice, frame) context windows and runs inference, optionally with
per-position LA-vector conditioning (see la_conditioning.py).

Kept as its own module so BaseSeries stays focused on DICOM loading and
pre/post-processing; this module owns "given a loaded series and a model
directory, build the right model and run it".
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

import qcardia.utils as utils
from qcardia_models.models import UNet2d

try:
    from qcardia_models.models import ContextUNet2d
except ImportError:
    # Only available on qcardia-models-dev's context-unet branch. A
    # config that requests a context model raises a clear error at
    # model-build time instead; plain UNet2d models are unaffected.
    ContextUNet2d = None

# Tokens in a config's model name that indicate a context-integrating
# architecture (ContextUNet2d) rather than a plain UNet2d.
_CONTEXT_MODEL_NAME_TOKENS = ("trans", "att", "tc", "sc")


def run_model(
    series,
    wandb_run_path: Path,
    image_type: str = "pixel",
    lax_dicom_dir: Optional[Path] = None,
    lax_model_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Run model inference for `series` (plain UNet2d or context-aware
    ContextUNet2d, chosen from the model config) and return the segmentation
    in the series' original image shape.

    Args:
        series: A BaseSeries (or subclass) instance, already loaded.
        wandb_run_path: Path to the model/WandB run directory.
        image_type: Type of image array to use ('pixel' or other).
        lax_dicom_dir: DICOM folder of a 4CH view, for LA-vector
            conditioning (only used if the config requests it).
        lax_model_path: Model directory for the plain UNet2d used to segment
            the 4CH view for LA-vector sampling.
    """
    preprocessed_slices = series._preproccess_slices(
        series._get_array(image_type=image_type)
    )
    raw_config = series._get_config(wandb_run_path)
    parsed = parse_model_config(raw_config)

    series.inference_dict["target_pixdim"] = torch.tensor(parsed["target_pixdim"])
    series.inference_dict["target_size"] = torch.tensor(parsed["target_size"])
    series.inference_dict["grid_sample_modes"] = [parsed["image_grid_sample_mode"]]
    series.inference_dict["nr_output_classes"] = parsed["nr_output_classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = series._resolve_weights_path(wandb_run_path)
    the_model = build_model(parsed, weights_path, device)

    series.inference_dict["dimension_scale_factor"], rescaled_tensor = (
        series._rescale_tensor(preprocessed_slices)
    )
    standardised_tensor = utils.standardise(rescaled_tensor)

    la_vectors = _maybe_compute_la_vectors(
        series, parsed, lax_dicom_dir, lax_model_path, device
    )

    if parsed["needs_context"]:
        tensor_6d = _to_6d(
            standardised_tensor,
            series.number_of_slices,
            series.number_of_temporal_positions,
        )
        model_output = run_context_aware(the_model, tensor_6d, parsed, la_vectors, device)
    else:
        model_output = series._forward_model(the_model, standardised_tensor)

    rescale_model_output = series._invert_rescale_tensor(model_output)
    model_prediction = torch.argmax(rescale_model_output, dim=1, keepdim=True).float()
    return series._postprocess_output(model_prediction)


def _maybe_compute_la_vectors(
    series, parsed: dict, lax_dicom_dir, lax_model_path, device: str
) -> Optional[torch.Tensor]:
    if not parsed["la_vector_integration"]:
        return None
    if lax_dicom_dir is None or lax_model_path is None:
        print(
            f"  WARNING: Model has la_vector_integration="
            f"{parsed['la_vector_integration']!r} but no "
            f"lax_dicom_dir/lax_model_path provided — running "
            f"without LA conditioning."
        )
        return None
    try:
        from qcardia.la_conditioning import compute_la_vectors

        return compute_la_vectors(
            sax_dicom_dir=series.folder,
            lax_dicom_dir=Path(lax_dicom_dir),
            lax_model_path=Path(lax_model_path),
            n_samples=parsed["la_vector_dim"],
            device=device,
        )
    except Exception as exc:
        print(
            f"  WARNING: LA vector computation failed ({exc}); "
            f"running without LA conditioning"
        )
        return None


def _to_6d(standardised_tensor: torch.Tensor, n_slices: int, n_frames: int) -> torch.Tensor:
    """(Z*T, C, H, W) -> (1, C, H, W, Z, T)."""
    _, C, H, W = standardised_tensor.shape
    return (
        standardised_tensor.view(n_slices, n_frames, C, H, W)
        .permute(2, 3, 4, 0, 1)
        .unsqueeze(0)
    )


def parse_model_config(raw_config: dict) -> dict:
    """
    Parse a model config into the fields needed to build and run a model.

    Supports both the legacy format (`config["unet"]`, plain UNet2d) and the
    new format (`config["model"]`, which may add context-window/attention and
    LA-vector-conditioning fields). Any config missing the new fields parses
    to a plain UNet2d with no context window and no LA conditioning, so
    existing model configs are unaffected.
    """
    model_cfg = raw_config.get("model", raw_config.get("unet", {}))
    model_name = raw_config.get("model", {}).get("name", "UNet")
    is_context_model = any(
        token in model_name.lower() for token in _CONTEXT_MODEL_NAME_TOKENS
    )

    context_cfg = raw_config.get("data", {}).get("context_window") or {}
    spatial_window = context_cfg.get("spatial", 0)
    temporal_window = context_cfg.get("temporal", 0)

    return {
        "model_name": model_name,
        "is_context_model": is_context_model,
        "nr_image_channels": model_cfg.get("nr_image_channels", 1),
        "channels_list": model_cfg["channels_list"],
        "nr_output_classes": model_cfg["nr_output_classes"],
        "nr_output_scales": model_cfg.get("nr_output_scales", 1),
        "transformer_params": raw_config.get("model", {}).get("transformer_params"),
        "context_window": context_cfg or None,
        "spatial_window": spatial_window,
        "spatial_stride": context_cfg.get("spatial_stride", 1),
        "temporal_window": temporal_window,
        "temporal_stride": context_cfg.get("temporal_stride", 1),
        "la_vector_dim": raw_config.get("model", {}).get("la_vector_dim"),
        "la_vector_integration": raw_config.get("model", {}).get(
            "la_vector_integration"
        ),
        "needs_context": bool(context_cfg)
        and (spatial_window > 0 or temporal_window > 0),
        "target_size": raw_config["data"]["target_size"],
        "target_pixdim": raw_config["data"]["target_pixdim"],
        "image_grid_sample_mode": raw_config["data"].get(
            "image_grid_sample_mode", "bilinear"
        ),
    }


def build_model(parsed: dict, weights_path: Path, device: str) -> torch.nn.Module:
    """Build and load a model (UNet2d or ContextUNet2d) from a parsed config."""
    common_kwargs = dict(
        nr_input_channels=parsed["nr_image_channels"],
        channels_list=parsed["channels_list"],
        nr_output_classes=parsed["nr_output_classes"],
        nr_output_scales=parsed["nr_output_scales"],
    )
    la_kwargs = {}
    if parsed["la_vector_dim"] is not None:
        la_kwargs = dict(
            la_vector_dim=parsed["la_vector_dim"],
            la_vector_integration=parsed["la_vector_integration"],
        )

    if parsed["is_context_model"]:
        if ContextUNet2d is None:
            raise ImportError(
                f"Model config requests a context-integrating architecture "
                f"({parsed['model_name']!r}) but ContextUNet2d is not "
                f"available. Install qcardia-models-dev on the "
                f"context-unet branch."
            )
        the_model = ContextUNet2d(
            **common_kwargs,
            transformer_params=parsed["transformer_params"],
            context_window=parsed["context_window"],
            **la_kwargs,
        )
    else:
        try:
            the_model = UNet2d(**common_kwargs, **la_kwargs)
        except TypeError:
            # Installed qcardia_models predates LA-vector support.
            the_model = UNet2d(**common_kwargs)

    model_weights = torch.load(weights_path, map_location=device, weights_only=False)
    the_model.load_state_dict(model_weights)

    the_model = the_model.to(device)
    the_model.eval()
    return the_model


def _get_context_neighbor_indices(
    center_idx: int, window: int, stride: int, max_range: int
) -> List[int]:
    """Get `window` neighbor indices around `center_idx`, wrapping at `max_range`."""
    indices = []
    start = center_idx - (window * stride) // 2
    i = 0
    while len(indices) < window:
        idx = (start + i * stride) % max_range
        if idx != center_idx:
            indices.append(idx)
        i += 1
    return indices


def _build_context_window(
    tensor_6d: torch.Tensor,
    z: int,
    t: int,
    spatial_window: int,
    spatial_stride: int,
    temporal_window: int,
    temporal_stride: int,
) -> torch.Tensor:
    """
    Build a context tensor for position (z, t): [target] + [spatial
    neighbors] + [temporal neighbors], concatenated along a new last
    dimension Q.

    Args:
        tensor_6d: Shape (B, C, H, W, Z, T).

    Returns:
        torch.Tensor: Shape (B, C, H, W, Q).
    """
    _, _, _, _, Z, T = tensor_6d.shape
    parts = [tensor_6d[..., z, t].unsqueeze(-1)]

    if spatial_window > 0:
        neighbor_z = _get_context_neighbor_indices(z, spatial_window, spatial_stride, Z)
        if neighbor_z:
            parts.append(tensor_6d[..., neighbor_z, t])

    if temporal_window > 0:
        neighbor_t = _get_context_neighbor_indices(t, temporal_window, temporal_stride, T)
        if neighbor_t:
            parts.append(tensor_6d[..., z, neighbor_t])

    return torch.cat(parts, dim=-1)


def run_context_aware(
    model: torch.nn.Module,
    tensor_6d: torch.Tensor,
    parsed: dict,
    la_vectors: Optional[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """
    Run a context-aware (ContextUNet2d) model over every (slice, frame)
    position, optionally with per-position LA-vector conditioning.

    Args:
        tensor_6d: Shape (B, C, H, W, Z, T).

    Returns:
        torch.Tensor: Shape (Z*T, num_classes, H_target, W_target).
    """
    B, _, _, _, Z, T = tensor_6d.shape
    nr_classes = parsed["nr_output_classes"]
    target_h, target_w = int(parsed["target_size"][0]), int(parsed["target_size"][1])

    output = torch.zeros(B, nr_classes, target_h, target_w, Z, T)
    supports_la = hasattr(model, "la_vector_dim")

    model.eval()
    with torch.no_grad():
        for z in range(Z):
            for t in range(T):
                context = _build_context_window(
                    tensor_6d,
                    z,
                    t,
                    parsed["spatial_window"],
                    parsed["spatial_stride"],
                    parsed["temporal_window"],
                    parsed["temporal_stride"],
                ).to(device)

                la_vec_zt = None
                if (
                    la_vectors is not None
                    and z < la_vectors.shape[0]
                    and t < la_vectors.shape[1]
                ):
                    la_vec_zt = la_vectors[z, t].unsqueeze(0).float().to(device)

                batch_preds = []
                for b_idx in range(B):
                    if la_vec_zt is not None and supports_la:
                        pred = model(context[b_idx : b_idx + 1], la_vectors=la_vec_zt)
                    else:
                        pred = model(context[b_idx : b_idx + 1])

                    # ContextUNet2d returns (decoder_outputs, attention);
                    # decoder_outputs is itself a per-scale list. UNet2d
                    # returns the per-scale list directly.
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                        if isinstance(pred, (tuple, list)):
                            pred = pred[0]

                    batch_preds.append(pred.cpu())

                output[..., z, t] = torch.cat(batch_preds, dim=0)

    # (B, num_classes, H, W, Z, T) -> (Z*T, num_classes, H, W)
    output_permuted = output[0].permute(3, 4, 0, 1, 2)
    return output_permuted.reshape(Z * T, nr_classes, target_h, target_w)
