# Apnea-ECG Preprocessing Pipeline

Source dataset: [PhysioNet Apnea-ECG Database v1.0.0](https://physionet.org/content/apnea-ecg/1.0.0/)

## Dataset at a glance

| | |
|---|---|
| Records | 70 total: 35 learning-set (labeled), 35 test-set (x01–x35) |
| Learning set | a01–a20, b01–b05, c01–c10 |
| Signal | Single-lead ECG, 100 Hz, ~7–10 hours per record |
| Labels | One `A` (apnea) / `N` (normal) annotation per 1-minute epoch |
| Extras | 8 records also have respiration + SpO2 channels (not required here) |
| Known issue | `c05` and `c06` are the same underlying recording (c05 starts ~80s later) |

## Why this preprocessing approach

Raw 100 Hz ECG is rarely fed straight into a classifier for apnea detection — the
signal-to-label ratio (6000 raw samples → 1 label) is too sparse, and almost all
published Apnea-ECG work (starting with the original CinC 2000 challenge entries)
instead derives **per-minute HRV features** from the RR-interval series, since
sleep apnea produces measurable heart-rate variability signatures (cyclic
variation of heart rate around apneic events). This pipeline follows that
approach:

1. **Clean** — Butterworth bandpass (0.5–40 Hz) + 50 Hz notch + z-score.
2. **Detect beats** — uses the dataset's own `.qrs` annotations where present
   (fast, and audited by nothing but consistent across the whole corpus);
   falls back to NeuroKit2 when they're missing.
3. **Reject artifacts** — RR intervals outside 0.3–1.8s (33–200 bpm) are dropped
   before any feature is computed from them.
4. **Segment** — into the same 1-minute epochs the `.apn` labels use.
5. **Feature-extract per epoch**:
   - Time-domain HRV: mean RR, SDNN, RMSSD, pNN50, CVRR, mean HR, min/max RR
   - Frequency-domain HRV: LF power, HF power, LF/HF ratio (Lomb-Scargle
     periodogram — chosen because it handles the *unevenly*-sampled RR series
     directly, without needing to resample/interpolate first)
   - A cheap ECG-derived-respiration proxy (R-peak amplitude modulation),
     since true respiration signals only exist for 8/70 records
6. **Quality-gate** — epochs with too few detected beats (signal dropout /
   lead-off) are dropped rather than imputed, since HRV computed from a
   handful of beats is not trustworthy.
7. **Split-safe scaling** — median imputation + StandardScaler are fit on the
   training split only, then applied to test, to avoid leakage.
8. **De-duplication** — `c05` is dropped by default (see known issue above).
9. **Parallelized** — `--n-jobs N` processes multiple recordings concurrently
   (thread pool; the work is I/O + numpy/scipy bound, so threads help
   without process-pickling overhead). Default is 4.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline — downloads records from PhysioNet on first run and caches
# them locally, then builds train/test feature tables.
python preprocess_apnea_ecg.py \
    --data-dir ./physionet_cache \
    --out-dir ./processed

# Process only a subset of records (useful for a quick trial run)
python preprocess_apnea_ecg.py --records a01 a02 b01 c01 --out-dir ./processed_subset

# No internet needed — sanity-checks the whole pipeline on synthetic data
python preprocess_apnea_ecg.py --self-test --out-dir ./processed_selftest
```

**Note on this environment:** this pipeline was written and validated here
using the `--self-test` mode (synthetic ECG-like signal, verified against a
known ground truth) because this sandbox's outbound network is restricted to
package-index domains and can't reach `physionet.org` directly. The self-test
run completed cleanly — 120/150 synthetic epochs processed, zero NaNs in the
final feature matrix, sensible HR/HRV values, correct class balance reporting.
Run the commands above on a machine with normal internet access (or Colab) to
pull the real data; no code changes are needed.

## Outputs

- `train_features.csv` / `test_features.csv` — one row per 1-minute epoch:
  13 scaled HRV/EDR features + `label` (1=apnea, 0=normal, NaN if unlabeled)
  + `record` (source recording, for grouped CV / leakage checks).
- `preprocessing_pipeline.pkl` — fitted `SimpleImputer` + `StandardScaler` +
  the feature column order, via `joblib`, so new records can be transformed
  identically at inference time.

## Model training & evaluation (`train_model.py`)

Consumes the feature CSVs above and trains/evaluates classifiers.

```bash
python train_model.py \
    --train-csv ./processed/train_features.csv \
    --test-csv ./processed/test_features.csv \
    --out-dir ./model_output

# On the preprocessing script's self-test output:
python train_model.py \
    --train-csv ./processed_selftest/train_features_selftest.csv \
    --test-csv ./processed_selftest/test_features_selftest.csv \
    --out-dir ./model_output_selftest
```

What it does:

1. **Group-aware CV** (`GroupKFold` on `record`, not row-level k-fold) compares
   Logistic Regression, Random Forest, Gradient Boosting, and XGBoost (if
   installed) — because row-level CV would leak correlated epochs from the
   same recording across folds and inflate scores.
2. **Class imbalance** is handled via `class_weight="balanced"` (or
   `scale_pos_weight` for XGBoost) inside each fold, not by oversampling
   before splitting.
3. Picks the best model by mean CV ROC-AUC, then **calibrates its
   probabilities** with cross-validated isotonic regression
   (`CalibratedClassifierCV`, reusing the same group-aware folds) — raw
   tree-ensemble scores in particular are not well-calibrated probabilities
   out of the box, and calibration matters if you ever want the output to
   mean "X% likely apnea" rather than just a ranking.
4. **Picks a decision threshold from out-of-fold CV predictions** (Youden's
   J statistic) instead of assuming the default 0.5 — under class
   imbalance, 0.5 is an arbitrary cutoff, and this picks the point that
   best trades off sensitivity and specificity based on the training data
   itself, not the held-out test set (avoiding threshold leakage).
5. Evaluates the calibrated model at that threshold on the held-out test
   set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix + ROC
   plot.
6. Also reports **per-recording apnea-minute correlation** — aggregating
   per-minute predictions into total apnea minutes per recording and
   correlating with ground truth. This is closer to how the original CinC
   2000 challenge and clinical AHI scoring actually evaluate apnea
   detection than per-minute accuracy alone.
7. Saves the fitted (calibrated) model + its chosen threshold in
   `best_model.pkl`, a feature-importance plot, and a
   `results_summary.json` with every metric.

**Validated here via self-test**: `example_model_output/` shows a full run
against the synthetic self-test features — all four models included,
`logistic_regression` selected as best, full evaluation artifacts produced.
Scores are near-perfect there only because the synthetic apnea/normal HR
patterns are far more separable than real physiology; expect materially
noisier, more realistic metrics on the actual PhysioNet data.

## Development

```bash
make install    # deps + pytest + ruff + xgboost
make test       # pytest, fully synthetic - no network needed
make lint       # ruff
make selftest   # preprocessing -> training, end-to-end, no network needed
make clean      # remove generated artifacts
```

CI (`.github/workflows/ci.yml`) runs the test suite across Python 3.9-3.12
plus the two self-test smoke commands on every push/PR, so a break in the
pipeline mechanics gets caught before it reaches a real PhysioNet run.

A `Dockerfile` is included for a reproducible environment (`docker build -t
apnea-ecg . && docker run apnea-ecg` runs the self-test by default) — note
this hasn't been build-tested in the environment this repo was assembled in
(no Docker daemon available there), so treat it as a starting point and
sanity-check the build once before relying on it.

See `CONTRIBUTING.md` before opening a PR.

## Repository layout

```
preprocess_apnea_ecg.py   # signal cleaning -> HRV feature extraction
train_model.py             # group-aware CV, model comparison, calibration, evaluation
tests/test_pipeline.py     # unit tests, fully synthetic
.github/workflows/ci.yml   # CI: tests + lint + self-test smoke runs
Makefile
Dockerfile
CONTRIBUTING.md
pyproject.toml             # deps + ruff config
requirements.txt
LICENSE                    # MIT
```


- **Group your CV and train/test splits by `record`**, always — this is
  already handled in `train_model.py`, but keep it in mind for any custom
  analysis you build on top of the feature tables.
- **Sequential structure**: apnea events are autocorrelated in time (an
  apneic minute is likely followed by another). To exploit this, window
  multiple consecutive epochs as sequence input to an RNN/1D-CNN instead of
  treating epochs independently — the current pipeline treats each epoch
  i.i.d., which is a reasonable baseline but leaves this signal on the table.
- **Threshold tuning**: the default 0.5 decision threshold may not be optimal
  given class imbalance — consider tuning it against the CV ROC curve
  (e.g. via Youden's J statistic) if recall on the apnea class matters most.
