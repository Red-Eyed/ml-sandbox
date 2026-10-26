"""Directional and mean-symmetric KL divergence between factored 2-D Gaussians.

Gaussian KL reference: Yang et al., NeurIPS 2021, arXiv:2106.01883.
The symmetric mean is an explicit lab comparison, not the full paper's loss.
"""

from math import cos, log, sin

import numpy as np

from .gaussian import Gaussian2D, nonnegative_distance


def gaussian_kl(prediction: Gaussian2D, target: Gaussian2D) -> float:
    """Return KL(prediction || target), dimensionless, without covariance inversion.

    Major-axis ordering makes the orientation contribution nonnegative. Axis
    ratios and expm1 avoid cancellation in the covariance term near identity.
    Positive Gaussian factors are required; nonfinite distances raise ValueError.
    """
    p = np.array(prediction.standard_deviations)
    t = np.array(target.standard_deviations)
    ratios = p / t
    log_ratios = np.array([log(a) - log(b) for a, b in zip(p, t, strict=True)])
    diagonal = float(np.sum(np.expm1(2 * log_ratios) - 2 * log_ratios))
    difference = prediction.angle - target.angle
    orientation = (
        sin(difference) ** 2
        * ((p[0] - p[1]) / t[0])
        * ((p[0] + p[1]) / t[1])
        * ((t[0] - t[1]) / t[1])
        * ((t[0] + t[1]) / t[0])
    )
    delta = np.array(prediction.mean) - target.mean
    c, s = cos(target.angle), sin(target.angle)
    local = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]) / t
    translation = float(local @ local)
    value = (diagonal + orientation + translation) / 2
    return nonnegative_distance(value, name="Gaussian KL", roundoff_scale=float(np.sum(ratios**2)))


def symmetric_kl(a: Gaussian2D, b: Gaussian2D) -> float:
    """Return the arithmetic mean of both KL directions, not their unhalved sum.

    This is symmetric and dimensionless but does not satisfy the metric triangle
    inequality. Changing the mean convention changes its weight in mixed costs.
    """
    return gaussian_kl(a, b) / 2 + gaussian_kl(b, a) / 2
