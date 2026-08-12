"""Direct conditional flow model for meal-level PPGR generation.

This module keeps the repo's core generative framework intact for the
CGMacros Protocol P1 benchmark:
  - flow matching loss in x1-prediction parameterization
  - Euler sampling with NFE=1/2/4
  - conditioning on subject health + meal information
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .flow_transformer import SinusoidalTimeEmbedding
from .meal_encoder import MealEncoder


class ResidualMLPBlock(nn.Module):
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


class DirectConditionEncoder(nn.Module):
    """Fuse subject health features with meal information."""

    def __init__(
        self,
        health_dim: int,
        d_meal: int = 512,
        d_cond: int = 256,
        clip_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.health_encoder = nn.Sequential(
            nn.Linear(health_dim, d_cond),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_cond, d_cond),
        )
        self.meal_encoder = MealEncoder(d_meal=d_meal, clip_dim=clip_dim)
        self.meal_proj = nn.Sequential(
            nn.Linear(d_meal, d_cond),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_cond, d_cond),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_cond * 2, d_cond),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_cond, d_cond),
        )

    def forward(
        self,
        health: torch.Tensor,
        nutrients: torch.Tensor,
        clip_embed: torch.Tensor | None = None,
        compute_alignment: bool = True,
    ) -> dict:
        health_embed = self.health_encoder(health)
        meal_out = self.meal_encoder(
            nutrients=nutrients,
            clip_embed=clip_embed,
            compute_alignment=compute_alignment,
        )
        meal_embed = self.meal_proj(meal_out["meal_embed"])
        cond = self.fusion(torch.cat([health_embed, meal_embed], dim=-1))
        return {"cond": cond, "alignment_loss": meal_out["alignment_loss"]}


class DirectFlowBackbone(nn.Module):
    """Small MLP backbone for direct PPGR generation."""

    def __init__(
        self,
        prediction_len: int = 13,
        d_cond: int = 256,
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
        self.input_proj = nn.Linear(prediction_len + d_cond + d_model, d_model)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model=d_model, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(d_model, prediction_len)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_proj(self.time_embed(t))
        h = torch.cat([x_t, cond, t_embed], dim=-1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class DirectFlowModel(nn.Module):
    """Conditional rectified flow for direct meal-level PPGR generation."""

    def __init__(
        self,
        health_dim: int,
        prediction_len: int = 13,
        d_cond: int = 256,
        d_model: int = 256,
        n_blocks: int = 4,
        d_meal: int = 512,
        clip_dim: int = 512,
        dropout: float = 0.1,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.prediction_len = prediction_len
        self.eps = eps
        self.condition_encoder = DirectConditionEncoder(
            health_dim=health_dim,
            d_meal=d_meal,
            d_cond=d_cond,
            clip_dim=clip_dim,
            dropout=dropout,
        )
        self.backbone = DirectFlowBackbone(
            prediction_len=prediction_len,
            d_cond=d_cond,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
        )

    def encode_condition(
        self,
        health: torch.Tensor,
        nutrients: torch.Tensor,
        clip_embed: torch.Tensor | None = None,
        compute_alignment: bool = True,
    ) -> dict:
        return self.condition_encoder(
            health=health,
            nutrients=nutrients,
            clip_embed=clip_embed,
            compute_alignment=compute_alignment,
        )

    def compute_loss(
        self,
        target_ppgr: torch.Tensor,
        health: torch.Tensor,
        nutrients: torch.Tensor,
        clip_embed: torch.Tensor | None = None,
        alignment_weight: float = 0.0,
    ) -> dict:
        batch_size = target_ppgr.shape[0]
        device = target_ppgr.device
        cond_out = self.encode_condition(
            health=health,
            nutrients=nutrients,
            clip_embed=clip_embed,
            compute_alignment=alignment_weight > 0.0,
        )

        t = torch.rand(batch_size, device=device) * (1 - 2 * self.eps) + self.eps
        z = torch.randn_like(target_ppgr)
        t_expand = t[:, None]
        x_t = (1 - t_expand) * z + t_expand * target_ppgr
        y_hat = self.backbone(x_t=x_t, t=t, cond=cond_out["cond"])

        denom = (1 - t_expand).clamp(min=self.eps)
        v_hat = (y_hat - x_t) / denom
        v_star = (target_ppgr - x_t) / denom
        flow_loss = ((v_hat - v_star) ** 2).mean()

        alignment_loss = cond_out["alignment_loss"]
        if alignment_loss is None:
            alignment_loss = torch.tensor(0.0, device=device)
        total_loss = flow_loss + alignment_weight * alignment_loss
        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "alignment_loss": alignment_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        health: torch.Tensor,
        nutrients: torch.Tensor,
        clip_embed: torch.Tensor | None = None,
        nfe: int = 4,
        num_samples: int = 50,
    ) -> torch.Tensor:
        batch_size = health.shape[0]
        device = health.device
        cond_out = self.encode_condition(
            health=health,
            nutrients=nutrients,
            clip_embed=clip_embed,
            compute_alignment=False,
        )
        cond = cond_out["cond"]

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
    def sample_with_latency(
        self,
        health: torch.Tensor,
        nutrients: torch.Tensor,
        clip_embed: torch.Tensor | None = None,
        nfe: int = 4,
        num_samples: int = 50,
    ) -> dict:
        if health.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        samples = self.sample(
            health=health,
            nutrients=nutrients,
            clip_embed=clip_embed,
            nfe=nfe,
            num_samples=num_samples,
        )
        if health.is_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"samples": samples, "latency_ms": elapsed_ms}


def build_direct_flow_model(
    health_dim: int,
    prediction_len: int = 13,
    d_cond: int = 256,
    d_model: int = 256,
    n_blocks: int = 4,
    d_meal: int = 512,
    clip_dim: int = 512,
    dropout: float = 0.1,
    eps: float = 1e-3,
) -> DirectFlowModel:
    return DirectFlowModel(
        health_dim=health_dim,
        prediction_len=prediction_len,
        d_cond=d_cond,
        d_model=d_model,
        n_blocks=n_blocks,
        d_meal=d_meal,
        clip_dim=clip_dim,
        dropout=dropout,
        eps=eps,
    )
