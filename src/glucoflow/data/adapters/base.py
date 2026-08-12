"""
Shared schema definitions for unified CGM dataset representation.

All dataset adapters convert raw data into these standardized DataFrames.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# --- Shared Schema Column Names ---

CGM_COLUMNS = {
    "subject_id": "str",
    "timestamp": "datetime64[ns]",
    "glucose_mgdl": "float64",
}

# Optional covariate columns (adapters include whichever are available)
OPTIONAL_CGM_COVARIATES = [
    "heart_rate",
    "calories_activity",
    "mets",
    "steps",
    "basal_rate",
    "bolus_volume",
    "carb_input",
]

MEAL_COLUMNS = {
    "subject_id": "str",
    "timestamp": "datetime64[ns]",
    "meal_type": "str",
    "calories": "float64",
    "carbs_g": "float64",
    "protein_g": "float64",
    "fat_g": "float64",
    "fiber_g": "float64",
    "image_path": "str",  # absolute path or empty string
    "amount_consumed_pct": "float64",
}

SUBJECT_META_COLUMNS = {
    "subject_id": "str",
    "age": "float64",
    "gender": "str",
    "bmi": "float64",
}


@dataclass
class DatasetOutput:
    """Standard output from a dataset adapter."""

    name: str
    cgm: pd.DataFrame  # Required: subject_id, timestamp, glucose_mgdl, + optional covariates
    meals: Optional[pd.DataFrame]  # Optional: meal event table
    subjects: Optional[pd.DataFrame]  # Optional: subject metadata
    sampling_interval_sec: int  # Expected interval in seconds


class BaseAdapter(ABC):
    """Base class for dataset adapters."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")

    @abstractmethod
    def load(self) -> DatasetOutput:
        """Load raw data and convert to shared schema."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        ...

    @staticmethod
    def clip_glucose(series: pd.Series, lo: float = 40.0, hi: float = 400.0) -> pd.Series:
        """Clip glucose values to clinical range [40, 400] mg/dL."""
        return series.clip(lower=lo, upper=hi)
