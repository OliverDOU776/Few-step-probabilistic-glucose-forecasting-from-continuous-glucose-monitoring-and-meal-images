"""PyTorch Dataset for CGM forecasting windows."""

from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalization import ZScoreStats


def compute_time_features(timestamps: np.ndarray) -> np.ndarray:
    """Compute cyclic time features from timestamps.

    Returns array of shape [N, 4]: hour_sin, hour_cos, min_sin, min_cos.
    """
    ts = pd.to_datetime(timestamps)
    hours = ts.hour + ts.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    mins = ts.minute
    min_sin = np.sin(2 * np.pi * mins / 60)
    min_cos = np.cos(2 * np.pi * mins / 60)
    return np.stack([hour_sin, hour_cos, min_sin, min_cos], axis=-1).astype(np.float32)


def resample_cgm_5min(cgm_df: pd.DataFrame) -> pd.DataFrame:
    """Resample CGM data to 5-min intervals by decimation (take every 5th row).

    Assumes input is at 1-min intervals, sorted by (subject_id, timestamp).
    """
    resampled = []
    for _, group in cgm_df.groupby("subject_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        resampled.append(group.iloc[::5])
    return pd.concat(resampled, ignore_index=True)


class CGMWindowDataset(Dataset):
    """PyTorch Dataset wrapping extracted CGM windows.

    Each item returns:
        history: [H] normalized glucose history
        future: [T] normalized glucose future (training target)
        time_features: [H, 4] cyclic time features for history
        future_raw: [T] raw (unnormalized) glucose future (for evaluation)
    """

    def __init__(
        self,
        windows: List[Dict],
        stats: ZScoreStats,
        prediction_length: int = 24,
        horizon_key: str = "future_120",
    ):
        self.stats = stats
        self.prediction_length = prediction_length
        self.horizon_key = horizon_key

        # Filter windows that have the required horizon
        self.windows = [w for w in windows if horizon_key in w]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]

        history_raw = w["history"].astype(np.float32)
        future_raw = w[self.horizon_key].astype(np.float32)

        # Normalize
        history = self.stats.normalize(history_raw).astype(np.float32)
        future = self.stats.normalize(future_raw).astype(np.float32)

        # Time features from history timestamps
        time_feats = compute_time_features(w["timestamps_history"])

        return {
            "history": torch.from_numpy(history),
            "future": torch.from_numpy(future),
            "time_features": torch.from_numpy(time_feats),
            "future_raw": torch.from_numpy(future_raw),
        }
