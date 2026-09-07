from __future__ import annotations

import numpy as np


def compute_hrv_features(
    rr: np.ndarray,
) -> dict[str, float]:
    """
    Compute short-epoch time-domain HRV features.

    Frequency-domain LF/HF features are intentionally excluded
    from the core pipeline because a one-minute epoch provides
    limited frequency resolution for conventional LF/HF analysis.
    """

    feature_names = [
        "mean_rr",
        "sdnn",
        "rmssd",
        "pnn50",
        "cvrr",
        "mean_hr",
        "min_rr",
        "max_rr",
        "rr_range",
    ]

    if len(rr) < 2:
        return {
            name: np.nan
            for name in feature_names
        }

    differences = np.diff(rr)

    mean_rr = np.mean(rr)

    sdnn = (
        np.std(rr, ddof=1)
        if len(rr) > 1
        else np.nan
    )

    rmssd = (
        np.sqrt(np.mean(differences ** 2))
        if len(differences)
        else np.nan
    )

    pnn50 = (
        np.mean(
            np.abs(differences) > 0.05
        ) * 100.0
        if len(differences)
        else np.nan
    )

    cvrr = (
        sdnn / mean_rr
        if mean_rr > 0
        else np.nan
    )

    mean_hr = (
        60.0 / mean_rr
        if mean_rr > 0
        else np.nan
    )

    return {
        "mean_rr": float(mean_rr),
        "sdnn": float(sdnn),
        "rmssd": float(rmssd),
        "pnn50": float(pnn50),
        "cvrr": float(cvrr),
        "mean_hr": float(mean_hr),
        "min_rr": float(np.min(rr)),
        "max_rr": float(np.max(rr)),
        "rr_range": float(
            np.max(rr) - np.min(rr)
        ),
    }