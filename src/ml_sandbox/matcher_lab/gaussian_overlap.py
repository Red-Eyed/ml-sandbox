"""Bhattacharyya, Hellinger, and ProbIoU from one shared Gaussian representation.

Reference: Llerena et al., https://arxiv.org/abs/2106.06072.
"""

from math import log

import numpy as np

from .gaussian import Gaussian2D, nonnegative_distance, relative_rotation


def gaussian_bhattacharyya(a: Gaussian2D, b: Gaussian2D) -> float:
    """Return Bhattacharyya distance using QR of concatenated covariance factors.

    A local principal-axis frame and common units reduce conditioning problems.
    QR computes a factor of the mean covariance without explicitly forming it.
    No jitter is added; unrepresentable factors or distances raise ValueError.
    """
    scale = max(*a.standard_deviations, *b.standard_deviations)
    std_a = np.array(a.standard_deviations) / scale
    std_b = np.array(b.standard_deviations) / scale
    factor_a = np.diag(std_a)
    factor_b = relative_rotation(a, b) * std_b
    _, root = np.linalg.qr(np.concatenate([factor_a, factor_b], axis=1).T / np.sqrt(2))
    diagonal = np.abs(np.diag(root))
    if np.any(diagonal == 0) or not np.isfinite(diagonal).all():
        raise ValueError("Mean Gaussian covariance is outside float64 resolution")
    delta = (np.array(b.mean) - a.mean) / scale
    c, s = np.cos(a.angle), np.sin(a.angle)
    local = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
    try:
        whitened = np.linalg.solve(root.T, local)
    except np.linalg.LinAlgError as error:
        raise ValueError("Mean Gaussian covariance is outside float64 resolution") from error
    logdet_mean = 2 * float(np.log(diagonal).sum())
    logdet_a = 2 * sum(log(v) for v in std_a)
    logdet_b = 2 * sum(log(v) for v in std_b)
    shape = (logdet_mean - (logdet_a + logdet_b) / 2) / 2
    value = float(whitened @ whitened) / 8 + shape
    return nonnegative_distance(
        value,
        name="Bhattacharyya distance",
        roundoff_scale=abs(logdet_mean) + abs(logdet_a) + abs(logdet_b),
    )


def log_probiou(distance: float) -> float:
    """Return log(1-Hellinger) without cancellation for separated distributions."""
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("Bhattacharyya distance must be finite and nonnegative")
    return -distance - float(np.log1p(np.sqrt(-np.expm1(-distance))))


def gaussian_hellinger(a: Gaussian2D, b: Gaussian2D) -> float:
    """Return Hellinger distance in [0,1], not its square."""
    return float(np.sqrt(-np.expm1(-gaussian_bhattacharyya(a, b))))


def gaussian_probiou(a: Gaussian2D, b: Gaussian2D) -> float:
    """Return 1-Hellinger, retaining small similarities until exponential underflow."""
    return float(np.exp(log_probiou(gaussian_bhattacharyya(a, b))))
