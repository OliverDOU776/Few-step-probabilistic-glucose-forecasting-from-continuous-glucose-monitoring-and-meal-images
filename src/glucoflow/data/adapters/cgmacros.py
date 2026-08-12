"""CGMacros dataset adapter.

Converts raw CGMacros per-subject CSVs into shared schema.
Primary multimodal dataset: CGM + meal photos + macronutrients.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseAdapter, DatasetOutput


# Path from extraction root to subject folders
_INNER_PATH = (
    "cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0"
    "/CGMacros_dateshifted365/CGMacros"
)


class CGMacrosAdapter(BaseAdapter):

    @property
    def name(self) -> str:
        return "cgmacros"

    def _data_root(self) -> Path:
        return self.raw_dir / _INNER_PATH

    def load(self) -> DatasetOutput:
        root = self._data_root()
        if not root.exists():
            raise FileNotFoundError(
                f"CGMacros data root not found: {root}. "
                "Ensure CGMacros_dateshifted365.zip has been extracted."
            )

        # Discover subject folders
        subject_dirs = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("CGMacros-")]
        )

        cgm_frames = []
        meal_frames = []

        for sdir in subject_dirs:
            sid = sdir.name  # e.g. "CGMacros-001"
            csv_path = sdir / f"{sid}.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()  # Normalize column names

            # Parse timestamps
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])

            # --- CGM rows ---
            # Use Libre GL as primary (lower missingness), fallback to Dexcom GL
            glucose = df["Libre GL"].fillna(df["Dexcom GL"])
            cgm = pd.DataFrame(
                {
                    "subject_id": sid,
                    "timestamp": df["Timestamp"],
                    "glucose_mgdl": self.clip_glucose(glucose),
                }
            )

            # Optional covariates
            covariate_map = {
                "HR": "heart_rate",
                "Calories (Activity)": "calories_activity",
                "METs": "mets",
            }
            for raw_col, schema_col in covariate_map.items():
                if raw_col in df.columns:
                    cgm[schema_col] = df[raw_col]

            cgm_frames.append(cgm)

            # --- Meal rows ---
            meal_mask = df["Meal Type"].notna() & (df["Meal Type"].str.strip() != "")
            if meal_mask.any():
                mdf = df[meal_mask].copy()

                # Normalize meal types
                mdf["Meal Type"] = mdf["Meal Type"].str.strip().str.lower()

                # Resolve image paths
                photo_dir = sdir / "photos"
                image_paths = []
                for _, row in mdf.iterrows():
                    img_val = row.get("Image path", "")
                    if pd.notna(img_val) and str(img_val).strip():
                        # Image path in CSV is relative; resolve against photo_dir
                        img_name = Path(str(img_val).strip()).name
                        abs_path = photo_dir / img_name
                        if abs_path.exists():
                            image_paths.append(str(abs_path))
                        else:
                            # Try the raw path relative to subject dir
                            alt_path = sdir / str(img_val).strip()
                            image_paths.append(str(alt_path) if alt_path.exists() else "")
                    else:
                        image_paths.append("")

                meal = pd.DataFrame(
                    {
                        "subject_id": sid,
                        "timestamp": mdf["Timestamp"].values,
                        "meal_type": mdf["Meal Type"].values,
                        "calories": pd.to_numeric(mdf["Calories"], errors="coerce").values,
                        "carbs_g": pd.to_numeric(mdf["Carbs"], errors="coerce").values,
                        "protein_g": pd.to_numeric(mdf["Protein"], errors="coerce").values,
                        "fat_g": pd.to_numeric(mdf["Fat"], errors="coerce").values,
                        "fiber_g": pd.to_numeric(mdf["Fiber"], errors="coerce").values,
                        "image_path": image_paths,
                        "amount_consumed_pct": (
                            pd.to_numeric(mdf["Amount Consumed"], errors="coerce").values
                            if "Amount Consumed" in mdf.columns
                            else np.nan
                        ),
                    }
                )
                meal_frames.append(meal)

        cgm_all = pd.concat(cgm_frames, ignore_index=True)
        meals_all = pd.concat(meal_frames, ignore_index=True) if meal_frames else None

        # Subject metadata from bio.csv
        subjects = self._load_bio(root)

        return DatasetOutput(
            name=self.name,
            cgm=cgm_all,
            meals=meals_all,
            subjects=subjects,
            sampling_interval_sec=60,
        )

    def _load_bio(self, root: Path) -> Optional[pd.DataFrame]:
        bio_path = root / "bio.csv"
        if not bio_path.exists():
            return None

        bio = pd.read_csv(bio_path)
        result = self._normalize_subject_table(bio, prefix="bio")
        if result is None:
            return None

        gut_path = root / "gut_health_test.csv"
        if gut_path.exists():
            gut = pd.read_csv(gut_path)
            gut_norm = self._normalize_subject_table(gut, prefix="gut")
            if gut_norm is not None:
                # Use the bio subject set as the canonical list because it matches the sensor folders.
                result = result.merge(gut_norm, on="subject_id", how="left")

        # Preserve the shared-schema fields explicitly.
        if "bio_age" in result.columns:
            result["age"] = result["bio_age"]
        if "bio_gender" in result.columns:
            result["gender"] = result["bio_gender"].astype(str).str.lower()
        if "bio_bmi" in result.columns:
            result["bmi"] = result["bio_bmi"]

        return result

    @staticmethod
    def _subject_id_from_value(value) -> str:
        try:
            return f"CGMacros-{int(float(value)):03d}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _normalize_feature_name(name: str) -> str:
        name = name.strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name

    def _normalize_subject_table(self, df: pd.DataFrame, prefix: str) -> Optional[pd.DataFrame]:
        subject_cols = [c for c in df.columns if c.strip().lower() == "subject"]
        if not subject_cols:
            return None

        subject_col = subject_cols[0]
        result = pd.DataFrame({"subject_id": df[subject_col].map(self._subject_id_from_value)})
        for col in df.columns:
            if col == subject_col:
                continue
            norm_name = self._normalize_feature_name(col)
            if not norm_name:
                continue
            result[f"{prefix}_{norm_name}"] = df[col].values
        return result
