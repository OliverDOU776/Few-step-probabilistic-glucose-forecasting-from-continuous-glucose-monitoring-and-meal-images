"""History-conditioned few-step flow model for GlucoBench-style forecasting."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .flow_transformer import SinusoidalTimeEmbedding


class HistoryResidualBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class HistoryFlowBackbone(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        prediction_len: int,
        d_model: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.prediction_len = prediction_len
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.input_proj = nn.Linear(cond_dim + prediction_len + d_model, d_model)
        self.blocks = nn.ModuleList(
            [HistoryResidualBlock(d_model=d_model, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(d_model, prediction_len)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_proj(self.time_embed(t))
        h = self.input_proj(torch.cat([x_t, cond, t_embed], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class HistoryFlowModel(nn.Module):
    """Conditional rectified flow for future residual generation."""

    def __init__(
        self,
        cond_dim: int,
        prediction_len: int,
        d_model: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.prediction_len = prediction_len
        self.eps = eps
        self.backbone = HistoryFlowBackbone(
            cond_dim=cond_dim,
            prediction_len=prediction_len,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
        )

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch_size = target.shape[0]
        device = target.device
        t = torch.rand(batch_size, device=device) * (1 - 2 * self.eps) + self.eps
        z = torch.randn_like(target)
        t_expand = t[:, None]
        x_t = (1 - t_expand) * z + t_expand * target
        y_hat = self.backbone(x_t=x_t, t=t, cond=cond)
        denom = (1 - t_expand).clamp(min=self.eps)
        v_hat = (y_hat - x_t) / denom
        v_star = (target - x_t) / denom
        return ((v_hat - v_star) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, nfe: int = 4, num_samples: int = 50) -> torch.Tensor:
        batch_size = cond.shape[0]
        device = cond.device
        bs = batch_size * num_samples
        cond_exp = cond.unsqueeze(1).expand(batch_size, num_samples, -1).reshape(bs, -1)
        x = torch.randn(bs, self.prediction_len, device=device)
        dt = 1.0 / nfe
        for k in range(nfe):
            t_k = k * dt
            t_tensor = torch.full((bs,), t_k, device=device)
            y_hat = self.backbone(x_t=x, t=t_tensor, cond=cond_exp)
            denom = max(1.0 - t_k, self.eps)
            v = (y_hat - x) / denom
            x = x + dt * v
        return x.reshape(batch_size, num_samples, self.prediction_len)

    @torch.no_grad()
    def sample_with_latency(self, cond: torch.Tensor, nfe: int = 4, num_samples: int = 50) -> dict:
        if cond.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        samples = self.sample(cond=cond, nfe=nfe, num_samples=num_samples)
        if cond.is_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"samples": samples, "latency_ms": elapsed_ms}


def build_history_flow_model(
    cond_dim: int,
    prediction_len: int,
    d_model: int = 256,
    n_blocks: int = 4,
    dropout: float = 0.1,
    eps: float = 1e-3,
) -> HistoryFlowModel:
    return HistoryFlowModel(
        cond_dim=cond_dim,
        prediction_len=prediction_len,
        d_model=d_model,
        n_blocks=n_blocks,
        dropout=dropout,
        eps=eps,
    )
