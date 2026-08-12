"""
Adapter for GlucoBench datasets (Weinstock, Colas, Dubosson, Hall, IGLU).

Reads the raw CSV files from the GlucoBench submodule and converts to our
shared schema: subject_id, timestamp, glucose_mgdl.
"""

from pathlib import Path

import pandas as pd

from .base import BaseAdapter, DatasetOutput


class GlucoBenchAdapter(BaseAdapter):
    """Load a single GlucoBench dataset into the shared schema.

    GlucoBench CSVs have columns: id, gl, time (+ optional covariates).
    All glucose values are in mg/dL with 5-minute sampling.
    """

    def __init__(self, raw_dir: Path, dataset_name: str = "weinstock"):
        self.dataset_name = dataset_name
        csv_path = Path(raw_dir) / f"{dataset_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"GlucoBench CSV not found: {csv_path}")
        self._csv_path = csv_path
        # Pass the parent dir as raw_dir for BaseAdapter
        super().__init__(csv_path.parent)

    @property
    def name(self) -> str:
        return f"glucobench_{self.dataset_name}"

    def load(self) -> DatasetOutput:
        df = pd.read_csv(self._csv_path)

        # Rename to shared schema
        cgm = pd.DataFrame()
        cgm["subject_id"] = "glucobench_" + self.dataset_name + "_" + df["id"].astype(str)
        cgm["timestamp"] = pd.to_datetime(df["time"])
        cgm["glucose_mgdl"] = self.clip_glucose(df["gl"].astype(float))

        # Drop rows with missing glucose
        cgm = cgm.dropna(subset=["glucose_mgdl"])

        # Sort by subject and time
        cgm = cgm.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

        return DatasetOutput(
            name=self.name,
            cgm=cgm,
            meals=None,
            subjects=None,
            sampling_interval_sec=300,  # 5-minute intervals
        )
