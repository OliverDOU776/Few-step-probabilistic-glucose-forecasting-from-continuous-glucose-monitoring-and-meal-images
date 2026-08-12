"""Per-dataset z-score normalization using train-split statistics only."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ZScoreStats:
    """Mean and std for z-score normalization."""
    mean: float
    std: float

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-6)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + 1e-6) + self.mean


def compute_zscore_stats(cgm_df: pd.DataFrame) -> ZScoreStats:
    """Compute z-score stats from a CGM DataFrame (train split only).

    Parameters
    ----------
    cgm_df : pd.DataFrame
        Must have ``glucose_mgdl`` column. Should be train split only.

    Returns
    -------
    ZScoreStats
    """
    values = cgm_df["glucose_mgdl"].dropna().values.astype(np.float64)
    return ZScoreStats(mean=float(np.mean(values)), std=float(np.std(values)))
