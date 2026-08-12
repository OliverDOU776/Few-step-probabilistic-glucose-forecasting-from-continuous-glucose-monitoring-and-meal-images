"""
BIG IDEAs Lab dataset adapter.

Converts raw Dexcom G6 CGM CSVs into shared schema.
Pretraining dataset: CGM-only, no meal images.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BaseAdapter, DatasetOutput


class BigIdeasAdapter(BaseAdapter):

    @property
    def name(self) -> str:
        return "big_ideas"

    def load(self) -> DatasetOutput:
        # PhysioNet distributes one subject directory per participant.  A
        # recursive search also supports the flattened layout used by older
        # local mirrors without asking users to rearrange the release.
        cgm_files = sorted(self.raw_dir.rglob("Dexcom_*.csv"))
        if not cgm_files:
            raise FileNotFoundError(f"No Dexcom CGM files found in {self.raw_dir}")

        cgm_frames = []

        for fpath in cgm_files:
            # Extract subject ID from filename: Dexcom_001.csv -> 001
            sid = fpath.stem.split("_")[1]
            subject_id = f"big_ideas_{sid}"

            df = pd.read_csv(fpath, low_memory=False)

            # Filter for EGV (estimated glucose value) rows only
            if "Event Type" in df.columns:
                df = df[df["Event Type"] == "EGV"].copy()

            # Parse timestamps
            ts_col = "Timestamp (YYYY-MM-DDThh:mm:ss)"
            if ts_col not in df.columns:
                # Fallback: try other common names
                for c in df.columns:
                    if "timestamp" in c.lower():
                        ts_col = c
                        break

            df["timestamp"] = pd.to_datetime(df[ts_col])

            # Glucose values
            gl_col = "Glucose Value (mg/dL)"
            if gl_col not in df.columns:
                for c in df.columns:
                    if "glucose" in c.lower() and "mg" in c.lower():
                        gl_col = c
                        break

            glucose = pd.to_numeric(df[gl_col], errors="coerce")

            cgm = pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "timestamp": df["timestamp"].values,
                    "glucose_mgdl": self.clip_glucose(glucose).values,
                }
            )
            cgm_frames.append(cgm)

        cgm_all = pd.concat(cgm_frames, ignore_index=True)

        # Subject metadata from Demographics.csv
        subjects = self._load_demographics()

        return DatasetOutput(
            name=self.name,
            cgm=cgm_all,
            meals=None,  # No meal images
            subjects=subjects,
            sampling_interval_sec=300,
        )

    def _load_demographics(self) -> Optional[pd.DataFrame]:
        demo_paths = sorted(self.raw_dir.rglob("Demographics.csv"))
        if not demo_paths:
            return None
        demo_path = demo_paths[0]

        demo = pd.read_csv(demo_path)
        result = pd.DataFrame()
        result["subject_id"] = demo["ID"].apply(lambda x: f"big_ideas_{x:03d}")
        if "Gender" in demo.columns:
            result["gender"] = demo["Gender"].str.lower()
        return result
