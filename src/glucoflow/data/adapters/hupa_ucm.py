"""
HUPA-UCM Diabetes dataset adapter.

Converts raw semicolon-separated per-patient CSVs into shared schema.
Pretraining dataset: CGM-only (has carb_input but no images).
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BaseAdapter, DatasetOutput


class HupaUcmAdapter(BaseAdapter):

    @property
    def name(self) -> str:
        return "hupa_ucm"

    def load(self) -> DatasetOutput:
        # Find consolidated per-patient files (HUPA####P.csv)
        patient_files = sorted(self.raw_dir.rglob("HUPA*P.csv"))
        if not patient_files:
            raise FileNotFoundError(f"No HUPA patient files found in {self.raw_dir}")

        cgm_frames = []

        for fpath in patient_files:
            # Extract subject ID: HUPA0001P.csv -> hupa_0001
            stem = fpath.stem  # e.g. "HUPA0001P"
            pid = stem.replace("HUPA", "").replace("P", "")
            subject_id = f"hupa_{pid}"

            df = pd.read_csv(fpath, sep=";")

            # Parse timestamps
            df["timestamp"] = pd.to_datetime(df["time"])

            # Glucose
            glucose = pd.to_numeric(df["glucose"], errors="coerce")

            cgm = pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "timestamp": df["timestamp"].values,
                    "glucose_mgdl": self.clip_glucose(glucose).values,
                }
            )

            # Optional covariates
            covariate_map = {
                "heart_rate": "heart_rate",
                "steps": "steps",
                "calories": "calories_activity",
                "basal_rate": "basal_rate",
                "bolus_volume_delivered": "bolus_volume",
                "carb_input": "carb_input",
            }
            for raw_col, schema_col in covariate_map.items():
                if raw_col in df.columns:
                    cgm[schema_col] = pd.to_numeric(df[raw_col], errors="coerce").values

            cgm_frames.append(cgm)

        cgm_all = pd.concat(cgm_frames, ignore_index=True)

        return DatasetOutput(
            name=self.name,
            cgm=cgm_all,
            meals=None,  # No meal event table (carb_input is a covariate)
            subjects=None,
            sampling_interval_sec=300,
        )
