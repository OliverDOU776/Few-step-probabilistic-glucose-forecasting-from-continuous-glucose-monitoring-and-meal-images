"""
Simple deterministic baselines for glucose forecasting.

Each baseline implements .fit(windows) and .predict(windows) for a given horizon.
"""

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression


class LastValueBaseline:
    """Persistence baseline: repeat the last observed glucose value."""

    def __init__(self):
        pass

    def fit(self, windows: List[Dict], horizon_key: str = "future_30") -> "LastValueBaseline":
        """No fitting required for persistence baseline."""
        # Store horizon key for reference
        self.horizon_key = horizon_key
        return self

    def predict(self, windows: List[Dict], horizon_key: str = "future_30") -> np.ndarray:
        """Predict by repeating the last history value for the full horizon.

        Parameters
        ----------
        windows : list of dict
            Each dict must have 'history' key (np.ndarray).
        horizon_key : str
            Key for the future target (used to determine output length).

        Returns
        -------
        np.ndarray, shape (n_windows, horizon_steps)
            Predicted glucose trajectories.
        """
        preds = []
        for w in windows:
            last_val = w["history"][-1]
            horizon_len = len(w[horizon_key])
            preds.append(np.full(horizon_len, last_val))
        return np.array(preds)


class LinearRegressionBaseline:
    """Linear regression baseline: flatten history as features, predict full future sequence.

    Fits a separate model per horizon.
    """

    def __init__(self):
        self.model = None
        self.horizon_key = None

    def fit(
        self, windows: List[Dict], horizon_key: str = "future_30"
    ) -> "LinearRegressionBaseline":
        """Fit linear regression: X = flattened history, Y = flattened future.

        Parameters
        ----------
        windows : list of dict
            Each dict must have 'history' and ``horizon_key`` arrays.
        horizon_key : str
            Key for the future target array.

        Returns
        -------
        self
        """
        self.horizon_key = horizon_key

        X = np.array([w["history"] for w in windows], dtype=np.float64)
        Y = np.array([w[horizon_key] for w in windows], dtype=np.float64)

        self.model = LinearRegression()
        self.model.fit(X, Y)
        return self

    def predict(self, windows: List[Dict], horizon_key: str = "future_30") -> np.ndarray:
        """Predict future glucose trajectory using fitted linear regression.

        Parameters
        ----------
        windows : list of dict
            Each dict must have 'history'.
        horizon_key : str
            Key for horizon (used for consistency, model was fit for specific horizon).

        Returns
        -------
        np.ndarray, shape (n_windows, horizon_steps)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X = np.array([w["history"] for w in windows], dtype=np.float64)
        return self.model.predict(X)
