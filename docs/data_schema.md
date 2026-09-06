# Data Schema

This document defines the expected data flow and structure used in the ECG-based sleep apnea screening pipeline.

## Data Flow

Raw PhysioNet ECG
→ ECG Preprocessing
→ R-Peak Detection
→ RR Interval Extraction
→ Heart Rate / HRV Features
→ Windowing & Label Assignment
→ ML-Ready Dataset
→ Model Training / Evaluation

## Raw Data

Source:

PhysioNet Apnea-ECG Database

The original dataset is stored locally under:

data/raw/

The dataset must not be committed to GitHub.

## Expected ML-Ready Fields

The processed dataset should contain, where applicable:

- `subject_id`
- `record_id`
- `window_id`
- `window_start`
- `window_end`
- `rr_mean`
- `rr_std`
- `hr_mean`
- `hr_std`
- `rmssd`
- `sdnn`
- `label`

## Label

The target variable represents binary apnea/non-apnea classification for the corresponding window.

## Data Handling Rules

- Do not modify the original raw dataset.
- Keep raw, intermediate and processed data separate.
- Do not commit patient/record data to GitHub.
- Keep preprocessing and feature-generation steps reproducible.
- Any changes to the ML-ready schema should be documented here.