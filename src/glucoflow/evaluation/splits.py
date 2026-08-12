"""
Subject-wise splitting for CGMacros evaluation protocols.

P2 protocol: subject-disjoint train / val / test split.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def subject_disjoint_split(
    cgm_df: pd.DataFrame,
    meals_df: Optional[pd.DataFrame] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    """Split CGM (and optionally meals) data by subject into train/val/test.

    Parameters
    ----------
    cgm_df : pd.DataFrame
        CGM DataFrame with at least ``subject_id`` column.
    meals_df : pd.DataFrame or None
        Meals DataFrame with ``subject_id`` column, or None.
    train_frac : float
        Fraction of subjects for training (default 0.70).
    val_frac : float
        Fraction of subjects for validation (default 0.15).
        Test fraction is ``1 - train_frac - val_frac``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"train": {"cgm": df, "meals": df|None},
          "val":   {"cgm": df, "meals": df|None},
          "test":  {"cgm": df, "meals": df|None}}``
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) must be < 1.0"
        )

    subjects = np.array(sorted(cgm_df["subject_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_subs = set(subjects[:n_train])
    val_subs = set(subjects[n_train : n_train + n_val])
    test_subs = set(subjects[n_train + n_val :])

    def _filter(df: Optional[pd.DataFrame], subs: set) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        return df[df["subject_id"].isin(subs)].reset_index(drop=True)

    result = {}
    for split_name, subs in [("train", train_subs), ("val", val_subs), ("test", test_subs)]:
        result[split_name] = {
            "cgm": _filter(cgm_df, subs),
            "meals": _filter(meals_df, subs),
        }

    return result
