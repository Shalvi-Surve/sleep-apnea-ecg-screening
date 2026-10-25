from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a signal.
    """
    mean = np.mean(x)
    std = np.std(x)

    if std <= 1e-8:
        return x - mean

    return (x - mean) / std