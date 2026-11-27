"""InferencePredictor must work with models that have no la_vectors argument."""

import torch
import torch.nn as nn

from qcardia.inference.predictor import InferencePredictor


class _Config:
    nr_classes = 2
    target_size = (8, 8)
    decode_all_context = False

    def needs_context(self):
        return False


class _PlainModel(nn.Module):
    """Like the public UNet2d: forward(x) only."""

    def forward(self, x):
        return x[:, :1].repeat(1, 2, 1, 1)


def test_plain_model_without_la_vectors():
    predictor = InferencePredictor(_PlainModel(), _Config(), device="cpu")
    out = predictor.predict(torch.zeros(3, 1, 8, 8))
    assert out.shape == (3, 2, 8, 8)
