"""
Flow Matching model wrapper: training loss + Euler ODE sampling.

Wraps FlowTransformer backbone with:
- compute_loss(): velocity-space flow matching loss (x₁-prediction)
- sample(): Euler ODE solver with NFE=1/2/4
- sample_with_latency(): timing wrapper

Core math (from core_math_spec.md):
  x_t = (1-t)*z + t*y          (interpolant)
  v* = (y - x_t) / (1 - t)     (target velocity)
  ŷ = f_θ(x_t, t, cond)        (model output, x₁-prediction)
  v̂ = (ŷ - x_t) / (1 - t)     (implied velocity)
  L = ||v̂ - v*||²              (velocity MSE loss)
"""

import time

import torch
import torch.nn as nn

from .flow_transformer import FlowTransformer


class FlowMatchingModel(nn.Module):
    """Conditional Rectified Flow model for glucose trajectory forecasting."""

    def __init__(self, backbone: FlowTransformer, eps: float = 1e-3):
        super().__init__()
        self.backbone = backbone
        self.eps = eps

    @property
    def prediction_len(self):
        return self.backbone.prediction_len

    def compute_loss(
        self,
        past_glucose: torch.Tensor,
        future_glucose: torch.Tensor,
        time_features: torch.Tensor = None,
        meal_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute velocity-space flow matching loss.

        Args:
            past_glucose: [B, H] past glucose (normalized)
            future_glucose: [B, T] ground-truth future (normalized)
            time_features: [B, H, F] optional time features
            meal_embed: [B, D_meal] optional meal embedding

        Returns:
            Scalar loss tensor.
        """
        B = past_glucose.shape[0]
        device = past_glucose.device

        # Sample t ~ U[eps, 1 - eps]
        t = torch.rand(B, device=device) * (1 - 2 * self.eps) + self.eps

        # Sample noise z ~ N(0, I)
        z = torch.randn_like(future_glucose)

        # Construct x_t = (1 - t) * z + t * y
        t_expand = t[:, None]  # [B, 1]
        x_t = (1 - t_expand) * z + t_expand * future_glucose

        # Model predicts denoised trajectory ŷ
        y_hat = self.backbone(past_glucose, x_t, t, time_features, meal_embed)

        # Velocity-space loss: ||(ŷ - y) / (1 - t)||²
        denom = (1 - t_expand).clamp(min=self.eps)
        v_hat = (y_hat - x_t) / denom
        v_star = (future_glucose - x_t) / denom
        loss = ((v_hat - v_star) ** 2).mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        past_glucose: torch.Tensor,
        nfe: int = 4,
        num_samples: int = 50,
        time_features: torch.Tensor = None,
        meal_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate future trajectory samples via Euler ODE solver.

        Args:
            past_glucose: [B, H] past glucose
            nfe: Number of Function Evaluations (Euler steps)
            num_samples: Number of independent trajectory samples
            time_features: [B, H, F] optional
            meal_embed: [B, D_meal] optional

        Returns:
            samples: [B, num_samples, T] sampled future trajectories
        """
        B = past_glucose.shape[0]
        T = self.prediction_len
        device = past_glucose.device

        # Expand inputs for S samples: [B, ...] → [B*S, ...]
        BS = B * num_samples
        past_exp = past_glucose.unsqueeze(1).expand(B, num_samples, -1).reshape(BS, -1)

        tf_exp = None
        if time_features is not None:
            tf_exp = (
                time_features.unsqueeze(1)
                .expand(B, num_samples, -1, -1)
                .reshape(BS, *time_features.shape[1:])
            )

        meal_exp = None
        if meal_embed is not None:
            meal_exp = (
                meal_embed.unsqueeze(1)
                .expand(B, num_samples, -1)
                .reshape(BS, -1)
            )

        # Start from noise z ~ N(0, I)
        x = torch.randn(BS, T, device=device)

        # Euler ODE solver: K steps from t=0 to t=1
        dt = 1.0 / nfe
        for k in range(nfe):
            t_k = k * dt
            t_tensor = torch.full((BS,), t_k, device=device)

            # Model predicts denoised trajectory
            y_hat = self.backbone(past_exp, x, t_tensor, tf_exp, meal_exp)

            # Implied velocity
            denom = max(1.0 - t_k, self.eps)
            v = (y_hat - x) / denom

            # Euler step
            x = x + dt * v

        return x.reshape(B, num_samples, T)

    @torch.no_grad()
    def sample_with_latency(
        self,
        past_glucose: torch.Tensor,
        nfe: int = 4,
        num_samples: int = 50,
        **kwargs,
    ) -> dict:
        """Sample with GPU-synchronized latency measurement."""
        if past_glucose.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        samples = self.sample(past_glucose, nfe, num_samples, **kwargs)
        if past_glucose.is_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"samples": samples, "latency_ms": elapsed_ms}


def build_flow_model(
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 256,
    patch_len: int = 4,
    history_len: int = 24,
    prediction_len: int = 24,
    n_time_features: int = 4,
    d_meal: int = 512,
    dropout: float = 0.1,
    eps: float = 1e-3,
) -> FlowMatchingModel:
    """Factory function to build a FlowMatchingModel with default hyperparameters."""
    backbone = FlowTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        patch_len=patch_len,
        history_len=history_len,
        prediction_len=prediction_len,
        n_time_features=n_time_features,
        d_meal=d_meal,
        dropout=dropout,
    )
    return FlowMatchingModel(backbone, eps=eps)
