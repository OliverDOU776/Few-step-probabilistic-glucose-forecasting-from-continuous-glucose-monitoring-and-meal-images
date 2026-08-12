"""
Sliding window extraction for CGM forecasting tasks.

Handles gap detection: windows straddling data gaps are skipped.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _check_contiguous(
    timestamps: np.ndarray, max_gap_sec: float
) -> bool:
    """Return True if all consecutive time deltas are <= max_gap_sec."""
    if len(timestamps) < 2:
        return True
    deltas = np.diff(timestamps.astype("datetime64[s]").astype(np.int64))
    return bool(np.all(deltas <= max_gap_sec))


def extract_windows(
    cgm_df: pd.DataFrame,
    history_minutes: int = 120,
    forecast_minutes_list: Optional[List[int]] = None,
    sampling_interval_sec: int = 60,
) -> List[Dict]:
    """Extract sliding windows from CGM data for multi-horizon forecasting.

    Parameters
    ----------
    cgm_df : pd.DataFrame
        CGM data with columns: subject_id, timestamp, glucose_mgdl.
        Must be sorted by (subject_id, timestamp).
    history_minutes : int
        Length of the input history window in minutes.
    forecast_minutes_list : list of int
        Forecast horizons in minutes. Default: [30, 60, 120].
    sampling_interval_sec : int
        Expected sampling interval in seconds (CGMacros = 60).

    Returns
    -------
    list of dict
        Each dict contains:
        - subject_id: str
        - history: np.ndarray of glucose values (history_steps,)
        - timestamps_history: np.ndarray of timestamps
        - future_30, future_60, future_120: np.ndarray of glucose values
          (only those horizons in forecast_minutes_list)
        - timestamps_future_{max_horizon}: np.ndarray
    """
    if forecast_minutes_list is None:
        forecast_minutes_list = [30, 60, 120]

    history_steps = history_minutes * 60 // sampling_interval_sec
    max_forecast_minutes = max(forecast_minutes_list)
    max_forecast_steps = max_forecast_minutes * 60 // sampling_interval_sec
    total_steps = history_steps + max_forecast_steps

    # Gap threshold: 2x the sampling interval
    max_gap_sec = 2.0 * sampling_interval_sec

    windows = []

    for subject_id, group in cgm_df.groupby("subject_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        glucose = group["glucose_mgdl"].values
        timestamps = group["timestamp"].values

        n = len(group)
        if n < total_steps:
            continue

        for start in range(0, n - total_steps + 1):
            end = start + total_steps
            ts_window = timestamps[start:end]

            # Check contiguity
            if not _check_contiguous(ts_window, max_gap_sec):
                continue

            g_window = glucose[start:end]

            w = {
                "subject_id": subject_id,
                "history": g_window[:history_steps].copy(),
                "timestamps_history": ts_window[:history_steps].copy(),
            }

            # Extract each forecast horizon
            for h_min in forecast_minutes_list:
                h_steps = h_min * 60 // sampling_interval_sec
                key = f"future_{h_min}"
                w[key] = g_window[history_steps : history_steps + h_steps].copy()

            # Full future timestamps (up to max horizon)
            w[f"timestamps_future_{max_forecast_minutes}"] = ts_window[
                history_steps:
            ].copy()

            windows.append(w)

    return windows


def compute_iauc_2h(
    glucose_future: np.ndarray,
    baseline_glucose: float,
    sampling_interval_sec: int = 60,
    iauc_minutes: int = 120,
) -> float:
    """Compute incremental area under the curve (iAUC) for post-meal glucose.

    Uses the trapezoidal rule on the incremental glucose response
    (glucose - baseline), only counting positive excursions.

    Parameters
    ----------
    glucose_future : np.ndarray
        Post-meal glucose values.
    baseline_glucose : float
        Pre-meal baseline glucose (typically the last value before the meal).
    sampling_interval_sec : int
        Interval between samples in seconds.
    iauc_minutes : int
        Duration over which to compute iAUC (default 120 min = 2h).

    Returns
    -------
    float
        iAUC in (mg/dL) * minutes.
    """
    # Only use the first iauc_minutes of post-meal data
    iauc_steps = iauc_minutes * 60 // sampling_interval_sec
    segment = glucose_future[:iauc_steps]
    incremental = np.maximum(segment - baseline_glucose, 0.0)
    dt_min = sampling_interval_sec / 60.0
    return float(np.trapz(incremental, dx=dt_min))


def extract_meal_windows(
    cgm_df: pd.DataFrame,
    meals_df: pd.DataFrame,
    pre_minutes: int = 120,
    post_minutes: int = 180,
    sampling_interval_sec: int = 60,
) -> List[Dict]:
    """Extract meal-anchored windows for PPGR (postprandial glucose response) task.

    For each meal event, extract:
    - pre_minutes of CGM history before the meal
    - post_minutes of CGM future after the meal (default 180 min = 3h PPGR)
    - iAUC_2h computed from the first 2h of post-meal trajectory

    Parameters
    ----------
    cgm_df : pd.DataFrame
        CGM data with columns: subject_id, timestamp, glucose_mgdl.
    meals_df : pd.DataFrame
        Meal events with columns: subject_id, timestamp, plus meal metadata.
    pre_minutes : int
        Minutes of history before the meal.
    post_minutes : int
        Minutes of forecast after the meal (default 180 for 3h PPGR).
    sampling_interval_sec : int
        Expected sampling interval in seconds.

    Returns
    -------
    list of dict
        Each dict contains:
        - subject_id: str
        - meal_timestamp: pd.Timestamp
        - history: np.ndarray of glucose (pre_steps,)
        - future: np.ndarray of glucose (post_steps,)
        - timestamps_history: np.ndarray
        - timestamps_future: np.ndarray
        - meal_info: dict with meal metadata columns
        - iauc_2h: float (incremental AUC, 2h post-meal)
    """
    pre_steps = pre_minutes * 60 // sampling_interval_sec
    post_steps = post_minutes * 60 // sampling_interval_sec
    max_gap_sec = 2.0 * sampling_interval_sec

    # Meal metadata columns (beyond subject_id and timestamp)
    meta_cols = [
        c for c in meals_df.columns if c not in ("subject_id", "timestamp")
    ]

    windows = []

    for subject_id, cgm_group in cgm_df.groupby("subject_id"):
        cgm_group = cgm_group.sort_values("timestamp").reset_index(drop=True)
        glucose = cgm_group["glucose_mgdl"].values
        timestamps = cgm_group["timestamp"].values

        # Get meals for this subject
        sub_meals = meals_df[meals_df["subject_id"] == subject_id]
        if sub_meals.empty:
            continue

        for _, meal_row in sub_meals.iterrows():
            meal_ts = pd.Timestamp(meal_row["timestamp"])

            # Find the CGM index closest to meal timestamp
            ts_diffs = np.abs(
                (timestamps - np.datetime64(meal_ts)).astype("timedelta64[s]").astype(np.int64)
            )
            anchor_idx = int(np.argmin(ts_diffs))

            # Check the anchor is close enough to the meal time (within 2x interval)
            if ts_diffs[anchor_idx] > max_gap_sec:
                continue

            start_idx = anchor_idx - pre_steps
            end_idx = anchor_idx + post_steps

            if start_idx < 0 or end_idx > len(glucose):
                continue

            ts_window = timestamps[start_idx:end_idx]

            # Check contiguity of the full window
            if not _check_contiguous(ts_window, max_gap_sec):
                continue

            g_window = glucose[start_idx:end_idx]

            history = g_window[:pre_steps].copy()
            future = g_window[pre_steps:].copy()
            baseline = float(history[-1])  # last pre-meal glucose

            w = {
                "subject_id": subject_id,
                "meal_timestamp": meal_ts,
                "history": history,
                "future": future,
                "timestamps_history": ts_window[:pre_steps].copy(),
                "timestamps_future": ts_window[pre_steps:].copy(),
                "meal_info": {c: meal_row[c] for c in meta_cols},
                "iauc_2h": compute_iauc_2h(
                    future, baseline, sampling_interval_sec, iauc_minutes=120
                ),
            }
            windows.append(w)

    return windows
