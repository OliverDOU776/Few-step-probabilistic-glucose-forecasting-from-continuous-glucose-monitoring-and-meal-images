from __future__ import annotations

import torch

from glucoflow.models import build_flow_model
from glucoflow.models.meal_encoder import MealEncoder


def tiny_model():
    return build_flow_model(
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        patch_len=4,
        history_len=24,
        prediction_len=24,
        n_time_features=4,
        d_meal=16,
        dropout=0.0,
    )


def test_flow_loss_and_sampling() -> None:
    torch.manual_seed(3)
    model = tiny_model()
    history = torch.randn(2, 24)
    future = torch.randn(2, 24)
    time_features = torch.randn(2, 24, 4)
    meal_embed = torch.randn(2, 16)

    loss = model.compute_loss(history, future, time_features, meal_embed)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    model.eval()
    for nfe in (1, 2, 4):
        samples = model.sample(
            history,
            nfe=nfe,
            num_samples=3,
            time_features=time_features,
            meal_embed=meal_embed,
        )
        assert samples.shape == (2, 3, 24)
        assert torch.isfinite(samples).all()


def test_meal_encoder_fusion_alignment_and_null_fallback() -> None:
    encoder = MealEncoder(d_meal=16, n_nutrients=5, clip_dim=8)
    nutrients = torch.tensor(
        [[30.0, 10.0, 8.0, 5.0, 400.0], [float("nan")] * 5]
    )
    images = torch.tensor([[0.1] * 8, [0.0] * 8])

    output = encoder(nutrients=nutrients, clip_embed=images)
    assert output["meal_embed"].shape == (2, 16)
    assert torch.isfinite(output["meal_embed"]).all()
    assert output["alignment_loss"] is not None
    assert torch.isfinite(output["alignment_loss"])
    assert torch.allclose(output["meal_embed"][1], encoder.null_token[0])
