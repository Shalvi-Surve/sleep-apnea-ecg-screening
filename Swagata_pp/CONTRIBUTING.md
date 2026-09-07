# Contributing

Thanks for considering a contribution. This is a small, focused pipeline, so
the bar is: does it work, is it tested, does it stay reproducible?

## Setup

```bash
git clone <your-fork-url>
cd apnea-ecg-pipeline
make install
```

## Before opening a PR

```bash
make test      # pytest, all synthetic - no PhysioNet/network access needed
make lint      # ruff
make selftest  # full pipeline dry-run: preprocessing -> training -> evaluation
```

All three run in CI automatically, but running them locally first saves a
review round-trip.

## Guidelines

- **No network calls in tests.** Everything under `tests/` runs against
  synthetic signals (`_make_synthetic_record` in `preprocess_apnea_ecg.py`).
  If you add a feature, add a synthetic-data test for it rather than one
  that needs to hit PhysioNet.
- **Group-aware everything.** Any new CV, split, or aggregation logic that
  touches epoch-level rows must respect the `record` column — epochs from
  the same recording are correlated, and row-level shuffling leaks
  information across folds. See `cross_validate_group_aware` in
  `train_model.py` for the existing pattern.
- **Don't silently swallow exceptions except at the per-record boundary.**
  The `except Exception` blocks in `load_record` are intentional (one bad
  record shouldn't crash a 70-record batch run) — that's the only place
  blind excepts belong. New code elsewhere should raise or fail loudly.
- **New features get docstring rationale, not just code.** HRV/ECG feature
  functions include a short "why this metric" note — keep that pattern so
  the feature set stays legible to people who aren't already deep in sleep
  medicine literature.

## Reporting issues

Open a GitHub issue with:
- what you ran (exact command)
- what you expected vs what happened
- if it's a numerical/output discrepancy, the record name(s) involved
