"""Inference driver for both the plain and the context-aware models."""

from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .config_handler import InferenceConfig
from .context_builder import ContextBuilder


class InferencePredictor:
    """Runs a loaded model over a preprocessed volume.

    Models without a context window are run in batches over the flattened
    (slice, frame) axis. Models with one are run position by position, since
    each needs its own neighbourhood gathered first.
    """

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        device: str = None,
        batch_size: int = 50,
    ):
        self.device = torch.device(device) if device else _default_device()
        self.model = model.to(self.device)
        self.config = config
        self.batch_size = batch_size
        self.needs_context = config.needs_context()
        self.context_builder = (
            ContextBuilder(
                spatial_window=config.spatial_window,
                temporal_window=config.temporal_window,
                spatial_stride=config.spatial_stride,
                temporal_stride=config.temporal_stride,
            )
            if self.needs_context
            else None
        )

    def predict(
        self,
        preprocessed_tensor: torch.Tensor,
        la_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Segment a preprocessed volume.

        Args:
            preprocessed_tensor: (N, C, H, W) without a context window,
                (B, C, H, W, Z, T) with one.
            la_vectors: (Z, T, la_vector_dim) for LA-conditioned models.

        Returns:
            (N, nr_classes, H, W), or (B, nr_classes, H, W, Z, T) with context.
        """
        if self.needs_context:
            return self._predict_context_aware(preprocessed_tensor, la_vectors)
        return self._predict_simple(preprocessed_tensor, la_vectors)

    def _predict_simple(
        self, tensor: torch.Tensor, la_vectors: Optional[torch.Tensor]
    ) -> torch.Tensor:
        n_positions = tensor.shape[0]

        la_flat = None
        if la_vectors is not None:
            la_flat = la_vectors.reshape(-1, la_vectors.shape[-1])
            if la_flat.shape[0] != n_positions:
                raise ValueError(
                    f"la_vectors covers {la_flat.shape[0]} (slice, frame) positions "
                    f"but the image tensor has {n_positions}. They must match."
                )

        output = torch.zeros(
            n_positions, self.config.nr_classes, *self.config.target_size
        )
        with torch.no_grad():
            for start in tqdm(
                range(0, n_positions, self.batch_size), desc="Inference"
            ):
                end = min(start + self.batch_size, n_positions)
                la_batch = (
                    None
                    if la_flat is None
                    else la_flat[start:end].float().to(self.device)
                )
                output[start:end] = _first(
                    self.model(tensor[start:end].to(self.device), la_vectors=la_batch)
                ).cpu()
        return output

    def _predict_context_aware(
        self, tensor: torch.Tensor, la_vectors: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch, _, _, _, n_slices, n_frames = tensor.shape
        output = torch.zeros(
            batch,
            self.config.nr_classes,
            *self.config.target_size,
            n_slices,
            n_frames,
        )

        with torch.no_grad():
            with tqdm(total=n_slices * n_frames, desc="Inference") as pbar:
                for z in range(n_slices):
                    for t in range(n_frames):
                        context = self.context_builder.build_context(tensor, z, t)
                        la_vec = None
                        if la_vectors is not None:
                            la_vec = (
                                la_vectors[z, t].unsqueeze(0).float().to(self.device)
                            )

                        prediction = _first(
                            self.model(context.to(self.device), la_vectors=la_vec)
                        )
                        if self.config.decode_all_context and prediction.dim() > 4:
                            prediction = prediction[:, 0, ...]
                        output[..., z, t] = prediction.cpu()
                        pbar.update(1)
        return output


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _first(model_output):
    """Unwrap however many list/tuple layers a model wraps its logits in."""
    while isinstance(model_output, (tuple, list)):
        model_output = model_output[0]
    return model_output
