"""
Meal embedding module: nutrients MLP + CLIP adapter + alignment loss.

Two pathways map to a common d_meal space:
  - Nutrient path: (carbs, protein, fat, fiber, calories) -> MLP -> e_nut [d_meal]
  - Image path: CLIP embedding [clip_dim] -> adapter MLP -> e_img [d_meal]

Fusion rule:
  - both present: average(e_img, e_nut)
  - image only: e_img
  - nutrient only: e_nut
  - neither: learned NULL token

Alignment loss: cosine similarity between e_img and e_nut (when both exist).

Modality dropout: during training, randomly drops image or nutrient embeddings
to improve robustness to missing modalities at test time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MealEncoder(nn.Module):
    """Encode meal information from nutrients and/or CLIP image embeddings."""

    def __init__(
        self,
        d_meal: int = 512,
        n_nutrients: int = 5,
        clip_dim: int = 512,
        image_drop_prob: float = 0.0,
        nutrient_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.d_meal = d_meal
        self.image_drop_prob = image_drop_prob
        self.nutrient_drop_prob = nutrient_drop_prob

        # Nutrient encoder: (carbs, protein, fat, fiber, calories) → d_meal
        self.nutrient_mlp = nn.Sequential(
            nn.Linear(n_nutrients, d_meal // 2),
            nn.GELU(),
            nn.Linear(d_meal // 2, d_meal),
        )

        # CLIP adapter: frozen CLIP embedding → d_meal
        self.clip_adapter = nn.Sequential(
            nn.Linear(clip_dim, d_meal),
            nn.GELU(),
            nn.Linear(d_meal, d_meal),
        )

        # Learned NULL token for meals without any info
        self.null_token = nn.Parameter(torch.zeros(1, d_meal))

    def forward(
        self,
        nutrients: torch.Tensor = None,
        clip_embed: torch.Tensor = None,
        compute_alignment: bool = True,
        image_drop_override: float | None = None,
        nutrient_drop_override: float | None = None,
    ) -> dict:
        """Encode meal embedding with multimodal fusion and null fallback.

        Args:
            nutrients: [B, 5] macro nutrients (carbs, protein, fat, fiber, calories),
                       or None. May contain NaN rows for meals without macros.
            clip_embed: [B, clip_dim] CLIP image embedding, or None.
                        May contain zero rows for meals without photos.
            compute_alignment: whether to compute alignment loss
            image_drop_override: if set, overrides self.image_drop_prob (for test-time sweeps)
            nutrient_drop_override: if set, overrides self.nutrient_drop_prob (for test-time sweeps)

        Returns:
            dict with keys:
                "meal_embed": [B, d_meal] meal embedding
                "alignment_loss": scalar tensor or None
        """
        B = (clip_embed.shape[0] if clip_embed is not None
             else nutrients.shape[0] if nutrients is not None
             else 1)
        device = (clip_embed.device if clip_embed is not None
                  else nutrients.device if nutrients is not None
                  else self.null_token.device)

        # Default: null token
        meal_embed = self.null_token.expand(B, -1).clone()
        alignment_loss = None

        # Per-sample masks: which samples have image / nutrients
        if clip_embed is not None:
            has_img = clip_embed.abs().sum(dim=-1) > 0  # [B] bool
        else:
            has_img = torch.zeros(B, dtype=torch.bool, device=device)

        if nutrients is not None:
            has_nut = ~torch.isnan(nutrients).any(dim=-1)  # [B] bool
        else:
            has_nut = torch.zeros(B, dtype=torch.bool, device=device)

        # Modality dropout: zero out modalities randomly
        img_drop_p = image_drop_override if image_drop_override is not None else self.image_drop_prob
        nut_drop_p = nutrient_drop_override if nutrient_drop_override is not None else self.nutrient_drop_prob

        if img_drop_p > 0 and (self.training or image_drop_override is not None):
            img_drop_mask = torch.rand(B, device=device) < img_drop_p
            has_img = has_img & ~img_drop_mask

        if nut_drop_p > 0 and (self.training or nutrient_drop_override is not None):
            nut_drop_mask = torch.rand(B, device=device) < nut_drop_p
            has_nut = has_nut & ~nut_drop_mask

        e_img = None
        if clip_embed is not None:
            e_img = self.clip_adapter(clip_embed)  # [B, d_meal]
            meal_embed = torch.where(has_img.unsqueeze(-1), e_img, meal_embed)

        e_nut = None
        if nutrients is not None:
            nutrients_clean = torch.nan_to_num(nutrients, nan=0.0)
            e_nut = self.nutrient_mlp(nutrients_clean)  # [B, d_meal]
            meal_embed = torch.where(has_nut.unsqueeze(-1), e_nut, meal_embed)

        both_mask = has_img & has_nut
        if e_img is not None and e_nut is not None and both_mask.any():
            fused = 0.5 * (e_img + e_nut)
            meal_embed = torch.where(both_mask.unsqueeze(-1), fused, meal_embed)

        # Alignment loss: cosine between e_img and e_nut where both exist
        # (computed on original presence masks, not dropout-masked ones,
        #  so dropout doesn't suppress alignment signal)
        if compute_alignment and clip_embed is not None and nutrients is not None:
            orig_has_img = clip_embed.abs().sum(dim=-1) > 0
            orig_has_nut = ~torch.isnan(nutrients).any(dim=-1)
            orig_both = orig_has_img & orig_has_nut
            if orig_both.any() and e_img is not None and e_nut is not None:
                cos_sim = F.cosine_similarity(e_img[orig_both], e_nut[orig_both], dim=-1)
                alignment_loss = (1 - cos_sim).mean()

        return {"meal_embed": meal_embed, "alignment_loss": alignment_loss}
