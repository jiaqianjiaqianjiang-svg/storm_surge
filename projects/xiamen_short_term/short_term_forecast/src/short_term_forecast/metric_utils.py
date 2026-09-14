"""Numerically small metric helpers that avoid platform-specific BLAS calls."""

from __future__ import annotations

import numpy as np


def safe_pearson_r(left: object, right: object) -> float:
    """Return Pearson r without calling ``numpy.corrcoef``/``numpy.cov``.

    Some Windows MKL and PyTorch combinations abort inside ``np.cov`` even for
    tiny arrays. The centered-sum definition is equivalent and avoids that
    native-code path.
    """
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return float("nan")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(np.clip(np.sum(x * y) / denominator, -1.0, 1.0))
