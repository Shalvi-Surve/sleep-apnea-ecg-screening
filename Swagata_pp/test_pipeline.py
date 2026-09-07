"""
Unit tests for preprocess_apnea_ecg.py

Run with:
    pytest tests/ -v

These don't touch PhysioNet or the network at all — everything is exercised
against synthetic signals so the suite runs anywhere, including CI.
"""
import numpy as np
import pandas as pd
import pytest

from preprocess_apnea_ecg import (
    RR_MAX_SEC,
    RR_MIN_SEC,
    _make_synthetic_record,
    assemble_dataset,
    bandpass_filter,
    build_epoch_features,
    hrv_freq_domain_features,
    hrv_time_domain_features,
    impute_and_scale,
    notch_filter,
    rr_intervals_from_peaks,
    zscore,
)

# --------------------------------------------------------------------------- #
# Signal cleaning
# --------------------------------------------------------------------------- #

def test_bandpass_filter_removes_low_frequency_drift():
    fs = 100
    t = np.arange(fs * 10) / fs
    drift = np.sin(2 * np.pi * 0.05 * t)  # well below the 0.5 Hz cutoff
    signal_component = np.sin(2 * np.pi * 5 * t)  # inside the passband
    x = drift + signal_component

    filtered = bandpass_filter(x, fs)
    # The slow drift should be substantially attenuated relative to the
    # in-band component's amplitude.
    assert np.std(filtered) < np.std(x)
    assert np.std(filtered) == pytest.approx(np.std(signal_component), rel=0.3)


def test_notch_filter_attenuates_powerline_frequency():
    fs = 250  # needs to be well above 2x50Hz for a meaningful notch
    t = np.arange(fs * 5) / fs
    hum = np.sin(2 * np.pi * 50 * t)
    x = hum + np.sin(2 * np.pi * 5 * t)

    filtered = notch_filter(x, fs, freq=50.0)
    assert np.std(filtered) < np.std(x)


def test_notch_filter_noop_when_freq_exceeds_nyquist():
    fs = 100  # Nyquist = 50Hz, so a 50Hz notch is right at the edge -> should be a no-op
    x = np.random.default_rng(0).normal(size=fs * 2)
    out = notch_filter(x, fs, freq=50.0)
    np.testing.assert_array_equal(out, x)


def test_zscore_output_has_zero_mean_unit_std():
    x = np.random.default_rng(1).normal(loc=5, scale=3, size=1000)
    z = zscore(x)
    assert z.mean() == pytest.approx(0, abs=1e-8)
    assert z.std() == pytest.approx(1, abs=1e-8)


def test_zscore_handles_constant_signal_without_dividing_by_zero():
    x = np.full(100, 7.0)
    z = zscore(x)
    assert np.all(np.isfinite(z))
    assert np.allclose(z, 0.0)


# --------------------------------------------------------------------------- #
# RR interval artifact rejection
# --------------------------------------------------------------------------- #

def test_rr_intervals_rejects_implausible_beats():
    fs = 100
    # Peaks include one absurdly close pair (0.1s apart -> 600bpm, invalid)
    # and one absurdly far pair (3s apart -> 20bpm, invalid).
    peaks = np.array([0, 80, 90, 170, 470, 550])  # samples
    rr, rr_t = rr_intervals_from_peaks(peaks, fs)
    assert np.all(rr >= RR_MIN_SEC)
    assert np.all(rr <= RR_MAX_SEC)
    assert len(rr) == len(rr_t)


def test_rr_intervals_empty_for_fewer_than_two_peaks():
    rr, rr_t = rr_intervals_from_peaks(np.array([42]), fs=100)
    assert len(rr) == 0
    assert len(rr_t) == 0


# --------------------------------------------------------------------------- #
# HRV feature extraction
# --------------------------------------------------------------------------- #

def test_hrv_time_domain_features_reasonable_values():
    rr = np.array([0.8, 0.82, 0.79, 0.81, 0.80, 0.83])
    feats = hrv_time_domain_features(rr)
    assert feats["mean_rr"] == pytest.approx(np.mean(rr))
    assert feats["mean_hr"] == pytest.approx(60.0 / np.mean(rr))
    assert feats["min_rr"] <= feats["mean_rr"] <= feats["max_rr"]
    assert all(np.isfinite(v) for v in feats.values())


def test_hrv_time_domain_features_nan_for_insufficient_beats():
    feats = hrv_time_domain_features(np.array([0.8]))
    assert all(np.isnan(v) for v in feats.values())


def test_hrv_freq_domain_features_returns_finite_values_for_enough_beats():
    rng = np.random.default_rng(0)
    rr = 0.8 + rng.normal(0, 0.02, size=60)
    rr_t = np.cumsum(rr)
    feats = hrv_freq_domain_features(rr, rr_t)
    assert np.isfinite(feats["lf_power"])
    assert np.isfinite(feats["hf_power"])


def test_hrv_freq_domain_features_nan_for_too_few_beats():
    feats = hrv_freq_domain_features(np.array([0.8, 0.81]), np.array([0.8, 1.61]))
    assert np.isnan(feats["lf_power"])
    assert np.isnan(feats["hf_power"])
    assert np.isnan(feats["lf_hf_ratio"])


# --------------------------------------------------------------------------- #
# End-to-end epoch feature building (synthetic record)
# --------------------------------------------------------------------------- #

def test_build_epoch_features_shape_and_labels():
    rec = _make_synthetic_record("test_rec", minutes=10, apnea_rate=0.4, seed=42)
    df = build_epoch_features(rec)

    assert len(df) == 10
    assert set(df["label"].unique()) <= {0, 1}
    assert (df["n_beats"] > 0).all()
    # HR should land in a physiologically sane range given the synthetic
    # generator's 55-95bpm design.
    valid_hr = df["mean_hr"].dropna()
    assert (valid_hr > 30).all() and (valid_hr < 150).all()


def test_build_epoch_features_no_nans_in_time_domain_columns():
    rec = _make_synthetic_record("test_rec2", minutes=8, seed=7)
    df = build_epoch_features(rec)
    time_domain_cols = ["mean_rr", "sdnn", "rmssd", "mean_hr"]
    assert not df[time_domain_cols].isna().any().any()


# --------------------------------------------------------------------------- #
# Dataset assembly (sequential vs parallel paths agree)
# --------------------------------------------------------------------------- #

def test_assemble_dataset_parallel_matches_sequential(monkeypatch, tmp_path):
    import preprocess_apnea_ecg as m

    records = {f"r{i}": _make_synthetic_record(f"r{i}", minutes=5, seed=i) for i in range(4)}
    monkeypatch.setattr(m, "load_record", lambda name, data_dir: records[name])

    seq_df = assemble_dataset(list(records.keys()), tmp_path, n_jobs=1)
    par_df = assemble_dataset(list(records.keys()), tmp_path, n_jobs=4)

    seq_sorted = seq_df.sort_values(["record", "epoch_idx"]).reset_index(drop=True)
    par_sorted = par_df.sort_values(["record", "epoch_idx"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(seq_sorted, par_sorted)


# --------------------------------------------------------------------------- #
# Imputation / scaling (no leakage, stable columns)
# --------------------------------------------------------------------------- #

def test_impute_and_scale_handles_all_nan_column():
    train_df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, np.nan],
        "f2": [np.nan, np.nan, np.nan, np.nan],  # entirely NaN in training
    })
    test_df = pd.DataFrame({"f1": [1.5, np.nan], "f2": [np.nan, np.nan]})

    train_X, test_X, imputer, scaler = impute_and_scale(train_df, test_df, ["f1", "f2"])

    assert np.isfinite(train_X).all()
    assert np.isfinite(test_X).all()
    assert train_X.shape == (4, 2)
    assert test_X.shape == (2, 2)


def test_impute_and_scale_scaler_fit_only_on_train():
    train_df = pd.DataFrame({"f1": [10.0, 20.0, 30.0]})
    test_df = pd.DataFrame({"f1": [1000.0]})  # wildly out of training distribution

    train_X, test_X, imputer, scaler = impute_and_scale(train_df, test_df, ["f1"])

    # Scaler params should come from train only, so the test point maps to
    # a large z-score rather than being folded into a re-fit distribution.
    assert abs(test_X[0, 0]) > 3
