"""
Unified Transformer backbone for Conditional Rectified Flow.

A single Transformer processes [time_token, meal_token, past_patches, future_patches]
through self-attention and outputs a denoised future trajectory ŷ.

This is the NEW architecture (x₁-prediction parameterization):
  - Input: past glucose + noisy future x_t + flow time t + optional meal embed
  - Output: ŷ (denoised future trajectory)
  - Velocity derived as: v̂ = (ŷ - x_t) / (1 - t)
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous flow time t ∈ [0, 1]."""

    def __init__(self, d_embed: int):
        super().__init__()
        self.d_embed = d_embed

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or [B, 1] flow time values
        Returns:
            [B, d_embed] sinusoidal embedding
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        half = self.d_embed // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowTransformer(nn.Module):
    """Unified Transformer for conditional rectified flow glucose forecasting.

    Token sequence: [t_token, meal_token, past_patch_1..P, future_patch_1..F]
    Output: denoised future trajectory ŷ extracted from future token positions.
    """

    def __init__(
        self,
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
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.history_len = history_len
        self.prediction_len = prediction_len
        self.n_time_features = n_time_features

        n_past_patches = history_len // patch_len
        n_future_patches = prediction_len // patch_len
        self.n_past_patches = n_past_patches
        self.n_future_patches = n_future_patches

        # --- Input projections ---
        # Past patches: glucose + time features concatenated
        past_input_dim = patch_len * (1 + n_time_features)
        self.past_proj = nn.Linear(past_input_dim, d_model)

        # Future patches: noisy glucose only
        self.future_proj = nn.Linear(patch_len, d_model)

        # --- Flow time embedding ---
        self.time_sinusoidal = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Meal embedding adapter ---
        self.meal_adapter = nn.Sequential(
            nn.Linear(d_meal, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.meal_null = nn.Parameter(torch.zeros(1, d_model))

        # --- Token type embeddings ---
        self.past_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.future_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # --- Positional embeddings ---
        total_tokens = 2 + n_past_patches + n_future_patches  # time + meal + past + future
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, d_model) * 0.02)

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Output head: future tokens → denoised patches ---
        self.output_proj = nn.Linear(d_model, patch_len)

    def forward(
        self,
        past_glucose: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        time_features: torch.Tensor = None,
        meal_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass: predict denoised future trajectory ŷ.

        Args:
            past_glucose: [B, history_len] normalized past CGM
            x_t: [B, prediction_len] noisy future (interpolated state)
            t: [B] flow time in [0, 1]
            time_features: [B, history_len, n_time_features] optional
            meal_embed: [B, d_meal] optional CLIP or nutrient embedding

        Returns:
            y_hat: [B, prediction_len] denoised future trajectory
        """
        B = past_glucose.shape[0]
        device = past_glucose.device

        # --- Patchify past glucose ---
        # [B, history_len] → [B, n_past, patch_len]
        past_patches = past_glucose.unfold(-1, self.patch_len, self.patch_len)

        if time_features is not None:
            # [B, history_len, F] → unfold on dim 1 → [B, n_past, F, patch_len]
            tf_patches = time_features.unfold(1, self.patch_len, self.patch_len)
            # → [B, n_past, patch_len, F] → [B, n_past, patch_len * F]
            tf_patches = tf_patches.permute(0, 1, 3, 2).reshape(
                B, self.n_past_patches, self.patch_len * self.n_time_features
            )
            past_input = torch.cat([past_patches, tf_patches], dim=-1)
        else:
            zeros = torch.zeros(
                B, self.n_past_patches, self.patch_len * self.n_time_features,
                device=device,
            )
            past_input = torch.cat([past_patches, zeros], dim=-1)

        past_tokens = self.past_proj(past_input) + self.past_type

        # --- Patchify noisy future ---
        future_patches = x_t.unfold(-1, self.patch_len, self.patch_len)
        future_tokens = self.future_proj(future_patches) + self.future_type

        # --- Time token ---
        t_embed = self.time_sinusoidal(t)
        t_token = self.time_mlp(t_embed).unsqueeze(1)  # [B, 1, d_model]

        # --- Meal token ---
        if meal_embed is not None:
            m_token = self.meal_adapter(meal_embed).unsqueeze(1)
        else:
            m_token = self.meal_null.expand(B, 1, -1)

        # --- Assemble: [time, meal, past_1..P, future_1..F] ---
        tokens = torch.cat([t_token, m_token, past_tokens, future_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]

        # --- Transformer ---
        out = self.transformer(tokens)

        # --- Extract future tokens → output ---
        future_start = 2 + self.n_past_patches
        future_out = out[:, future_start:, :]  # [B, n_future, d_model]
        y_patches = self.output_proj(future_out)  # [B, n_future, patch_len]
        y_hat = y_patches.reshape(B, self.prediction_len)

        return y_hat
