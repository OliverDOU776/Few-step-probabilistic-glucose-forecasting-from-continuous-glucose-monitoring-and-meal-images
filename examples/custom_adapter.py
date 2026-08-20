#!/usr/bin/env python3
"""Example adapter for bringing a new longitudinal dataset into GlucoFlow.

Expected input
--------------
<raw_dir>/cgm.csv with at least:
    subject_id,timestamp,glucose_mgdl

Optional files
--------------
<raw_dir>/meals.csv
<raw_dir>/subjects.csv

The column names follow the original glucose application. For another domain,
you may map the target series into ``glucose_mgdl`` or fork the shared schema
with a more appropriate name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from glucoflow.data.adapters.base import BaseAdapter, DatasetOutput


class CustomCSVAdapter(BaseAdapter):
    """Load a user-supplied CSV dataset into GlucoFlow's shared schema."""

    @property
    def name(self) -> str:
        return "custom_csv"

    def load(self) -> DatasetOutput:
        cgm_path = self.raw_dir / "cgm.csv"
        if not cgm_path.is_file():
            raise FileNotFoundError(
                f"Expected {cgm_path}. Create a CSV with subject_id, timestamp, "
                "and glucose_mgdl columns."
            )

        cgm = pd.read_csv(cgm_path)
        required = {"subject_id", "timestamp", "glucose_mgdl"}
        missing = sorted(required.difference(cgm.columns))
        if missing:
            raise ValueError(f"cgm.csv is missing required columns: {missing}")

        cgm = cgm.copy()
        cgm["subject_id"] = cgm["subject_id"].astype(str)
        cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="raise")
        cgm["glucose_mgdl"] = pd.to_numeric(cgm["glucose_mgdl"], errors="raise")
        cgm = cgm.dropna(subset=["subject_id", "timestamp", "glucose_mgdl"])
        cgm["glucose_mgdl"] = self.clip_glucose(cgm["glucose_mgdl"])
        cgm = cgm.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

        meals = self._read_optional_table("meals.csv")
        subjects = self._read_optional_table("subjects.csv", parse_timestamp=False)
        sampling_interval_sec = self._estimate_sampling_interval(cgm)

        return DatasetOutput(
            name=self.name,
            cgm=cgm,
            meals=meals,
            subjects=subjects,
            sampling_interval_sec=sampling_interval_sec,
        )

    def _read_optional_table(
        self,
        filename: str,
        *,
        parse_timestamp: bool = True,
    ) -> pd.DataFrame | None:
        path = self.raw_dir / filename
        if not path.is_file():
            return None

        frame = pd.read_csv(path)
        if "subject_id" in frame.columns:
            frame["subject_id"] = frame["subject_id"].astype(str)
        if parse_timestamp and "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        return frame

    @staticmethod
    def _estimate_sampling_interval(cgm: pd.DataFrame) -> int:
        deltas = (
            cgm.groupby("subject_id", sort=False)["timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
        )
        positive = deltas[deltas > 0]
        if positive.empty:
            raise ValueError(
                "Could not estimate a sampling interval. Each subject needs at "
                "least two chronologically distinct observations."
            )
        return int(round(float(positive.median())))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_dir",
        type=Path,
        help="Directory containing cgm.csv and optional meals.csv/subjects.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CustomCSVAdapter(args.raw_dir).load()

    print(f"dataset={dataset.name}")
    print(f"subjects={dataset.cgm['subject_id'].nunique()}")
    print(f"observations={len(dataset.cgm)}")
    print(f"sampling_interval_sec={dataset.sampling_interval_sec}")
    print(f"meals={'none' if dataset.meals is None else len(dataset.meals)}")
    print(f"subject_rows={'none' if dataset.subjects is None else len(dataset.subjects)}")


if __name__ == "__main__":
    main()
