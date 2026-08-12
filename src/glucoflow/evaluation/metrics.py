"""
Evaluation metrics for glucose forecasting.

All functions operate on numpy arrays.
Convention: lower is better for MAE, RMSE, CRPS, ECE.
Coverage-90: closer to 0.90 is better.
"""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def crps_gaussian(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> float:
    """Continuous Ranked Probability Score for Gaussian predictive distribution.

    CRPS_gauss = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma, Phi = standard normal CDF, phi = PDF.

    Parameters
    ----------
    y_true : array-like
        True values.
    mu : array-like
        Predicted means.
    sigma : array-like
        Predicted standard deviations (must be > 0).

    Returns
    -------
    float
        Mean CRPS over all elements.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-6)

    z = (y_true - mu) / sigma

    # Standard normal PDF and CDF
    from scipy.stats import norm

    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)

    crps_vals = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps_vals))


def coverage_90(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of true values within 90% prediction interval.

    Parameters
    ----------
    y_true : array-like
        True values.
    lower : array-like
        Lower bound of 90% prediction interval.
    upper : array-like
        Upper bound of 90% prediction interval.

    Returns
    -------
    float
        Coverage fraction in [0, 1]. Ideal = 0.90.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def ece(
    y_true: np.ndarray,
    predicted_quantiles: np.ndarray,
    quantile_levels: np.ndarray,
) -> float:
    """Expected Calibration Error for quantile forecasts.

    For each quantile level q, compute the observed frequency of
    ``y_true <= predicted_quantile(q)`` and compare to q.
    ECE = mean(|observed_freq - q|) over all quantile levels.

    Parameters
    ----------
    y_true : array-like, shape (N,)
        True values.
    predicted_quantiles : array-like, shape (N, Q)
        Predicted quantiles. Column j corresponds to ``quantile_levels[j]``.
    quantile_levels : array-like, shape (Q,)
        Quantile probability levels, e.g. [0.1, 0.2, ..., 0.9].

    Returns
    -------
    float
        Mean absolute calibration error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    predicted_quantiles = np.asarray(predicted_quantiles, dtype=np.float64)
    quantile_levels = np.asarray(quantile_levels, dtype=np.float64)

    if predicted_quantiles.ndim == 1:
        predicted_quantiles = predicted_quantiles.reshape(-1, 1)

    n = len(y_true)
    errors = []
    for j, q in enumerate(quantile_levels):
        observed_freq = np.mean(y_true <= predicted_quantiles[:, j])
        errors.append(abs(observed_freq - q))

    return float(np.mean(errors))
