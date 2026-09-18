"""Context window extraction for the spatiotemporal-context models."""

from typing import List

import torch


class ContextBuilder:
    """Collects the neighbouring slices and frames a context model expects.

    Mirrors what qcardia_data's DataModule assembled at training time: one
    context is [target] + [spatial neighbours] + [temporal neighbours], in that
    order, and always exactly 1 + spatial_window + temporal_window long.
    """

    def __init__(
        self,
        spatial_window: int = 0,
        temporal_window: int = 0,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
    ):
        self.spatial_window = spatial_window
        self.temporal_window = temporal_window
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride

    def build_context(self, image_tensor: torch.Tensor, z: int, t: int) -> torch.Tensor:
        """Build the context around slice z, frame t.

        Args:
            image_tensor: Shape (B, C, H, W, Z, T).
            z: Target slice index.
            t: Target frame index.

        Returns:
            Shape (B, C, H, W, Q), Q = 1 + spatial_window + temporal_window.
        """
        _, _, _, _, n_slices, n_frames = image_tensor.shape
        parts = [image_tensor[..., z, t].unsqueeze(-1)]

        if self.spatial_window > 0:
            neighbours = self.get_neighbor_indices(
                z, self.spatial_window, self.spatial_stride, n_slices
            )
            parts.append(image_tensor[..., neighbours, t])

        if self.temporal_window > 0:
            neighbours = self.get_neighbor_indices(
                t, self.temporal_window, self.temporal_stride, n_frames
            )
            parts.append(image_tensor[..., z, neighbours])

        return torch.cat(parts, dim=-1)

    def get_neighbor_indices(
        self, center_idx: int, window: int, stride: int, max_range: int
    ) -> List[int]:
        """Neighbours of center_idx, wrapping around max_range.

        Reproduces DataModule._get_neighbor_indices, including its bounds. The
        loop bound and the max_range <= 1 case matter: when every candidate
        collides with the centre, which a small max_range and a stride sharing a
        factor with it can cause, an unbounded loop never terminates. Both pad
        with the centre index so the result is always `window` long.
        """
        if max_range <= 1:
            return [center_idx] * window

        indices = []
        start = center_idx - (window * stride) // 2
        i = 0
        while len(indices) < window and i < max_range * window * 2:
            idx = (start + i * stride) % max_range
            if idx != center_idx:
                indices.append(idx)
            i += 1
        while len(indices) < window:
            indices.append(center_idx)
        return indices
