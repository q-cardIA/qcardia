"""Tests for the preprocessing -> inference path.

The invariants here are the ones that fail silently rather than loudly: a model
routed to the wrong class, conditioning vectors that never reach the model, a
context window that disagrees with the one training used, or a reshape that
scrambles (slice, frame) ordering. Each produces a plausible-looking
segmentation, so none of them show up without an explicit check.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from qcardia.inference.config_handler import InferenceConfig
from qcardia.inference.context_builder import (
    ContextBuilder,
    from_slice_major_layout,
    to_slice_major_layout,
)
from qcardia.inference.model_loader import determine_model_type, resolve_lax_model_path
from qcardia.inference.predictor import InferencePredictor

from qcardia_data.pipeline.data_module import DataModule
from qcardia_models.training_utils import is_transformer_model


TARGET_SIZE = [8, 8]


def make_config(transformer=False, la=False, spatial=0, temporal=0, temporal_stride=1):
    config = {
        "model": {
            "name": "test-model",
            "nr_image_channels": 1,
            "channels_list": [4, 8],
            "nr_output_classes": 4,
            "nr_output_scales": -1,
        },
        "data": {
            "target_size": TARGET_SIZE,
            "target_pixdim": [1.25, 1.25],
            "image_grid_sample_mode": "bicubic",
            "context_window": {
                "spatial": spatial,
                "temporal": temporal,
                "temporal_stride": temporal_stride,
            },
        },
    }
    if transformer:
        config["model"]["transformer_params"] = {"depth": 1, "heads": 1}
    if la:
        config["model"]["la_vector_dim"] = 16
        config["model"]["la_vector_integration"] = "film"
    return config


class RecordingModel(torch.nn.Module):
    """Stands in for a real network and records what it was called with."""

    def __init__(self, nr_classes=4):
        super().__init__()
        self.nr_classes = nr_classes
        self.la_calls = []

    def forward(self, x, la_vectors=None, la_present=None):
        self.la_calls.append(la_vectors)
        if x.dim() == 5:                       # (B, C, H, W, Q)
            batch = x.shape[0]
        else:                                  # (N, C, H, W)
            batch = x.shape[0]
        return [torch.zeros(batch, self.nr_classes, *TARGET_SIZE)]


# --- model routing ---------------------------------------------------------

@pytest.mark.parametrize("transformer", [False, True])
def test_model_routing_matches_training(transformer):
    """Routing must key on transformer_params, exactly as training did."""
    config = make_config(transformer=transformer)
    expected = "transformer" if is_transformer_model(config["model"]) else "unet"
    assert determine_model_type(config) == expected


def test_model_routing_ignores_name():
    """A name containing 'att'/'tc'/'sc' must not promote a plain UNet."""
    config = make_config(transformer=False)
    config["model"]["name"] = "CA_nnUNet-att-STC-scan"
    assert determine_model_type(config) == "unet"


# --- LA conditioning delivery ----------------------------------------------

@pytest.mark.parametrize(
    "spatial,temporal", [(0, 0), (2, 2)], ids=["no-context", "with-context"]
)
def test_la_vectors_reach_the_model(spatial, temporal):
    """A model declaring la_vector_integration must actually be handed vectors.

    UNet2d/ContextUNet2d skip FiLM when la_vectors is None, so dropping them
    turns a conditioned model into its unconditioned twin without any error.
    """
    n_slices, n_frames, la_dim = 3, 4, 16
    config = InferenceConfig(
        make_config(transformer=bool(spatial or temporal), la=True,
                    spatial=spatial, temporal=temporal)
    )
    model = RecordingModel()
    predictor = InferencePredictor(model, config, device="cpu", batch_size=4)

    la_vectors = torch.randn(n_slices, n_frames, la_dim)
    if config.needs_context():
        tensor = torch.randn(1, 1, *TARGET_SIZE, n_slices, n_frames)
    else:
        tensor = torch.randn(n_slices * n_frames, 1, *TARGET_SIZE)

    predictor.predict(tensor, la_vectors=la_vectors)

    assert model.la_calls, "model was never called"
    assert all(c is not None for c in model.la_calls), (
        "la_vectors were dropped before reaching forward()"
    )


def test_la_vector_count_mismatch_is_rejected():
    config = InferenceConfig(make_config(la=True))
    predictor = InferencePredictor(RecordingModel(), config, device="cpu")
    with pytest.raises(ValueError, match="must match"):
        predictor.predict(torch.randn(12, 1, *TARGET_SIZE),
                          la_vectors=torch.randn(3, 3, 16))


# --- context window parity with training -----------------------------------

@pytest.mark.parametrize("window", [2, 4, 6])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("max_range", [2, 5, 12, 25, 30])
def test_neighbour_indices_match_training(window, stride, max_range):
    """Inference must pick the same context positions the DataModule picked.

    max_range=1 is deliberately excluded: DataModule._get_neighbor_indices has
    no bound on that case, and with max_range=1 every candidate index is 0 (%
    1), always equal to a center_idx of 0 — so calling it here would hang the
    test suite rather than fail it. See
    test_neighbour_indices_terminate_when_every_candidate_is_the_centre and
    ContextBuilder.get_neighbor_indices's own max_range <= 1 special case.
    """
    builder = ContextBuilder()
    data_module = DataModule.__new__(DataModule)
    for center in range(max_range):
        assert builder.get_neighbor_indices(center, window, stride, max_range) == \
            data_module._get_neighbor_indices(center, window, stride, max_range)


@pytest.mark.parametrize("window", [2, 4, 6])
def test_neighbour_indices_max_range_one_returns_center_padding(window):
    """DataModule hangs on max_range=1 (see above); assert our own contract."""
    builder = ContextBuilder()
    assert builder.get_neighbor_indices(0, window, stride=1, max_range=1) == [0] * window


def test_neighbour_indices_terminate_when_every_candidate_is_the_centre():
    """max_range=2 with stride=2 makes every candidate collide with the centre."""
    builder = ContextBuilder()
    assert len(builder.get_neighbor_indices(0, 4, 2, 2)) == 4


def test_context_order_is_target_then_spatial_then_temporal():
    """DataModule builds [current] + spatial + temporal; inference must agree."""
    n_slices, n_frames, spatial, temporal = 5, 6, 2, 2
    builder = ContextBuilder(spatial_window=spatial, temporal_window=temporal)
    # Encode each (z, t) as a unique constant so positions are identifiable.
    tensor = torch.zeros(1, 1, 1, 1, n_slices, n_frames)
    for z in range(n_slices):
        for t in range(n_frames):
            tensor[0, 0, 0, 0, z, t] = z * 100 + t

    z, t = 2, 3
    context = builder.build_context(tensor, z, t).flatten().tolist()
    expected = [z * 100 + t]
    expected += [nz * 100 + t
                 for nz in builder.get_neighbor_indices(z, spatial, 1, n_slices)]
    expected += [z * 100 + nt
                 for nt in builder.get_neighbor_indices(t, temporal, 1, n_frames)]
    assert context == expected


# --- (slice, frame) ordering through the reshape ----------------------------

def test_reshape_round_trip_preserves_slice_major_order():
    """series._run_model's to_/from_slice_major_layout must invert exactly.

    Calls the real functions rather than reimplementing the reshape math, so a
    change to either one is actually caught here.
    """
    n_slices, n_frames, channels, height, width = 5, 7, 1, 4, 3
    flat = torch.arange(
        n_slices * n_frames * channels * height * width, dtype=torch.float32
    ).reshape(n_slices * n_frames, channels, height, width)

    six = to_slice_major_layout(flat, n_slices, n_frames)
    assert six.shape == (1, channels, height, width, n_slices, n_frames)
    for z in range(n_slices):
        for t in range(n_frames):
            assert torch.equal(six[0, :, :, :, z, t], flat[z * n_frames + t])

    back, back_n_slices, back_n_frames = from_slice_major_layout(six)
    assert (back_n_slices, back_n_frames) == (n_slices, n_frames)
    assert torch.equal(back, flat)


def test_reshape_agrees_with_numpy_flattening_in_series():
    """_reshape_array does pa.reshape(-1, 1, H, W) on a (Z, T, H, W) array."""
    n_slices, n_frames, height, width = 4, 5, 3, 2
    volume = np.arange(n_slices * n_frames * height * width, dtype=np.float32).reshape(
        n_slices, n_frames, height, width)
    flat = volume.reshape(-1, 1, height, width)
    for z in range(n_slices):
        for t in range(n_frames):
            assert np.array_equal(flat[z * n_frames + t, 0], volume[z, t])


# --- long-axis model resolution --------------------------------------------

@pytest.mark.parametrize("recorded", ["none", "None", "null", "", None])
def test_unset_lax_weights_path_is_rejected(recorded):
    """A run started from scratch records weights_path as the string "none".

    Treating that as a path makes the long-axis segmentation fail deeper in,
    where the original code swallowed it and ran without conditioning.
    """
    config = {"model": {"weights_path": recorded}}
    with pytest.raises(ValueError, match="nothing to fall back on"):
        resolve_lax_model_path(None, config, Path("run"))


def test_lax_weights_path_is_used_when_set():
    config = {"model": {"value": {"weights_path": "/weights/best_model.pt"}}}
    assert resolve_lax_model_path(None, config, Path("run")) == \
        Path("/weights/best_model.pt")
