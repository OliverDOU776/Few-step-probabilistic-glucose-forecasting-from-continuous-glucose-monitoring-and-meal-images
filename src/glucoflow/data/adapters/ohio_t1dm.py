"""
OhioT1DM dataset adapter.

Loads the six-subject XML release into the shared schema. The raw XML includes
CGM, meal, bolus, and basal events; this adapter exposes CGM plus a meal table.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseAdapter, DatasetOutput


TS_FORMAT = "%d-%m-%Y %H:%M:%S"


class OhioT1DMAdapter(BaseAdapter):
    def __init__(self, raw_dir: Path):
        super().__init__(raw_dir)
        self._xml_files = sorted(self.raw_dir.glob("*-ws-*.xml"))
        if not self._xml_files:
            raise FileNotFoundError(f"No OhioT1DM XML files found in {self.raw_dir}")

    @property
    def name(self) -> str:
        return "ohio_t1dm"

    def load(self) -> DatasetOutput:
        return self._load_from_files(self._xml_files)

    def load_split(self, split_name: str) -> DatasetOutput:
        split_suffix = f"-ws-{split_name}.xml"
        files = [path for path in self._xml_files if path.name.endswith(split_suffix)]
        if not files:
            raise FileNotFoundError(f"No OhioT1DM XML files found for split={split_name!r} in {self.raw_dir}")
        return self._load_from_files(files)

    def _load_from_files(self, xml_files: list[Path]) -> DatasetOutput:
        cgm_frames = []
        meal_frames = []

        for xml_path in xml_files:
            subject_token = xml_path.name.split("-")[0]
            subject_id = f"ohio_{subject_token}"
            root = ET.parse(xml_path).getroot()

            cgm_frames.append(self._load_glucose(root, subject_id))
            meal_df = self._load_meals(root, subject_id)
            if not meal_df.empty:
                meal_frames.append(meal_df)

        cgm = pd.concat(cgm_frames, ignore_index=True)
        cgm = cgm.dropna(subset=["timestamp", "glucose_mgdl"])
        cgm["glucose_mgdl"] = self.clip_glucose(cgm["glucose_mgdl"])
        cgm = cgm.drop_duplicates(subset=["subject_id", "timestamp"]).sort_values(
            ["subject_id", "timestamp"]
        )
        cgm = cgm.reset_index(drop=True)

        meals = None
        if meal_frames:
            meals = pd.concat(meal_frames, ignore_index=True)
            meals = meals.dropna(subset=["timestamp"]).drop_duplicates(
                subset=["subject_id", "timestamp", "carbs_g", "meal_type"]
            )
            meals = meals.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

        return DatasetOutput(
            name=self.name,
            cgm=cgm,
            meals=meals,
            subjects=None,
            sampling_interval_sec=300,
        )

    def _load_glucose(self, root: ET.Element, subject_id: str) -> pd.DataFrame:
        glucose_level = root.find("glucose_level")
        rows = []
        if glucose_level is None:
            return pd.DataFrame(columns=["subject_id", "timestamp", "glucose_mgdl"])
        for event in glucose_level.findall("event"):
            rows.append(
                {
                    "subject_id": subject_id,
                    "timestamp": pd.to_datetime(event.attrib.get("ts"), format=TS_FORMAT, errors="coerce"),
                    "glucose_mgdl": pd.to_numeric(event.attrib.get("value"), errors="coerce"),
                }
            )
        return pd.DataFrame(rows)

    def _load_meals(self, root: ET.Element, subject_id: str) -> pd.DataFrame:
        meal_tag = root.find("meal")
        rows = []
        if meal_tag is None:
            return pd.DataFrame(columns=["subject_id", "timestamp"])
        for event in meal_tag.findall("event"):
            carbs = pd.to_numeric(event.attrib.get("carbs"), errors="coerce")
            rows.append(
                {
                    "subject_id": subject_id,
                    "timestamp": pd.to_datetime(event.attrib.get("ts"), format=TS_FORMAT, errors="coerce"),
                    "meal_type": event.attrib.get("type", "") or "",
                    "calories": np.nan,
                    "carbs_g": carbs,
                    "protein_g": np.nan,
                    "fat_g": np.nan,
                    "fiber_g": np.nan,
                    "image_path": "",
                    "amount_consumed_pct": 100.0,
                }
            )
        return pd.DataFrame(rows)
