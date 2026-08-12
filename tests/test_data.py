from __future__ import annotations

import numpy as np
import pandas as pd

from glucoflow.evaluation.metrics import coverage_90, mae, rmse
from glucoflow.evaluation.splits import subject_disjoint_split
from glucoflow.evaluation.windows import extract_windows


def test_subject_split_has_no_overlap() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat([f"s{i}" for i in range(20)], 2),
            "timestamp": pd.date_range("2025-01-01", periods=40, freq="5min"),
            "glucose_mgdl": np.arange(40, dtype=float) + 100,
        }
    )
    split = subject_disjoint_split(frame, seed=9)
    ids = {name: set(part["cgm"]["subject_id"]) for name, part in split.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert set.union(*ids.values()) == set(frame["subject_id"])


def test_window_extraction_rejects_gaps() -> None:
    contiguous = pd.DataFrame(
        {
            "subject_id": "s1",
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="5min"),
            "glucose_mgdl": np.arange(8, dtype=float) + 100,
        }
    )
    windows = extract_windows(
        contiguous,
        history_minutes=20,
        forecast_minutes_list=[20],
        sampling_interval_sec=300,
    )
    assert len(windows) == 1

    with_gap = contiguous.copy()
    with_gap.loc[4:, "timestamp"] += pd.Timedelta(minutes=20)
    assert not extract_windows(
        with_gap,
        history_minutes=20,
        forecast_minutes_list=[20],
        sampling_interval_sec=300,
    )


def test_basic_metrics() -> None:
    truth = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.0, 3.0, 2.0])
    assert mae(truth, prediction) == 2 / 3
    assert np.isclose(rmse(truth, prediction), np.sqrt(2 / 3))
    assert coverage_90(truth, truth - 0.1, truth + 0.1) == 1.0
