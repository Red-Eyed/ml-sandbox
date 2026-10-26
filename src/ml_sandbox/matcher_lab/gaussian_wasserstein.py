"""Gaussian 2-Wasserstein distance using a stable covariance-factor alignment.

Reference: Yang et al., ICML 2021, https://proceedings.mlr.press/v139/yang21l.html.
This is raw W2; the detection paper's additional loss transforms are not applied.
"""

import numpy as np

from .gaussian import Gaussian2D, nonnegative_distance, relative_rotation


def gaussian_wasserstein(a: Gaussian2D, b: Gaussian2D) -> float:
    """Return W2, in scene-coordinate units, rather than its square.

    Orthogonal Procrustes alignment of covariance factors avoids subtraction
    of nearly equal covariance traces. A common scale protects the small SVD.
    No clipping, logarithmic transform, or distance-to-similarity mapping is used.
    """
    scale = max(*a.standard_deviations, *b.standard_deviations)
    factor_a = np.diag(np.array(a.standard_deviations) / scale)
    factor_b = relative_rotation(a, b) * (np.array(b.standard_deviations) / scale)
    left, _, right = np.linalg.svd(factor_b.T @ factor_a)
    residual = factor_a - factor_b @ (left @ right)
    shape_distance = scale * float(np.hypot.reduce(residual.ravel()))
    delta = np.array(a.mean) - b.mean
    distance = float(np.hypot(np.hypot(delta[0], delta[1]), shape_distance))
    return nonnegative_distance(distance, name="Gaussian W2")
