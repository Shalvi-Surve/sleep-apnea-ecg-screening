# Lightweight ECG-Based Screening System for Early Sleep Apnea Detection

A lightweight and interpretable machine-learning and deep-learning pipeline for screening apnea-related events from single-lead ECG signals using ECG preprocessing, R-peak detection, RR-interval and HRV representations, temporal modeling, explainable AI, and computational-efficiency analysis.

---

## Project Overview

Sleep apnea is a sleep-related breathing disorder characterized by repeated episodes of interrupted or reduced breathing during sleep. Conventional diagnosis generally relies on sleep studies involving specialized monitoring and multiple physiological signals.

This project investigates the feasibility of using a **single-lead ECG signal** as a reduced-sensor input for computational screening of apnea-related events.

Rather than attempting to directly measure airway obstruction, the proposed system uses ECG-derived physiological information. Changes associated with apnea-related physiological responses can influence cardiac activity and may therefore be reflected indirectly in ECG-derived measures such as RR intervals, heart rate, and heart-rate variability (HRV).

The project develops and evaluates three modeling approaches:

1. **Random Forest** — conventional machine-learning baseline.
2. **1D Convolutional Neural Network (1D CNN)** — deep-learning baseline.
3. **Lightweight Transformer** — proposed temporal modeling approach.

The models are evaluated using **subject-wise/grouped validation** to reduce the risk of patient-level data leakage. In addition to predictive performance, the project considers **interpretability and computational efficiency**.

A Streamlit-based prototype is being developed to demonstrate the complete screening pipeline.

> **Important:** This project is an academic research prototype for screening research. It is not a medical diagnostic system and does not replace polysomnography, clinical evaluation, or professional medical advice.

---

# 1. Project Information

| Field | Details |
|---|---|
| **Project Title** | Lightweight ECG-Based Screening System for Early Sleep Apnea Detection |
| **Project Type** | B.Tech Project-I |
| **Institution** | VIT Vellore |
| **Academic Year** | 2026 |
| **Project Guide** | Dr. Divya Meena S. |
| **Dataset** | PhysioNet Apnea-ECG Database |
| **Primary Signal** | Single-lead ECG |
| **Primary Task** | Binary apnea/non-apnea screening |
| **Main Models** | Random Forest, 1D CNN, Lightweight Transformer |
| **Validation** | Subject-wise / Grouped Cross-Validation |
| **Interface** | Streamlit |
| **Status** | Under Development |

---

# 2. Team

| Team Member | Registration Number | Primary Technical Responsibility |
|---|---:|---|
| **Shalvi Atul Surve** | 23BCE0088 | Model development, evaluation, integration and technical coordination |
| **Swagata Banerjee** | 23BCE0143 | ECG preprocessing, R-peak detection, RR/HRV and data pipeline |
| **Narhari Sharma** | 23BCE0171 | Validation, testing, result verification, error analysis and evidence |


The repository is structured so that the three members can work in parallel while maintaining clear interfaces between data processing, model development, evaluation, and application integration.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the detailed responsibility and collaboration plan.

---

# 3. Problem Statement

Conventional sleep-apnea assessment may require specialized sleep studies and multiple physiological measurements. Although ECG does not directly measure airway obstruction, apnea-related physiological changes can influence cardiac activity and can therefore be investigated through ECG-derived information.

The problem addressed in this project is:

> **To develop a lightweight, interpretable, and computationally efficient ECG-based screening pipeline capable of classifying temporal segments as apnea or non-apnea while maintaining subject-wise evaluation and providing a practical prototype for demonstration.**

The project focuses on computational screening rather than clinical diagnosis.

---

# 4. Objectives

## Objective 1 — Develop the ECG Processing Pipeline

Develop a reproducible pipeline that:

- loads single-lead ECG recordings,
- preprocesses the ECG signal,
- detects R-peaks,
- extracts RR intervals,
- derives heart-rate and HRV representations,
- creates temporal windows, and
- produces machine-learning-ready data.

---

## Objective 2 — Develop and Compare Screening Models

Develop and compare:

- Random Forest,
- 1D CNN, and
- Lightweight Transformer

for binary apnea/non-apnea screening.

The models will be evaluated using subject-wise/grouped validation and appropriate classification metrics.

---

## Objective 3 — Evaluate Interpretability and Efficiency

Investigate:

- model performance,
- explainability,
- model complexity,
- parameter count,
- model size,
- inference latency, and
- practical deployment considerations.

The final pipeline will be integrated into a Streamlit prototype for demonstration.

---

# 5. Research Focus

The project investigates the following broader research question:

> **Can a lightweight ECG-based temporal modeling pipeline provide effective apnea/non-apnea screening while reducing computational complexity and maintaining interpretable predictions?**

The project particularly focuses on the following aspects.

### 5.1 ECG as a Reduced-Sensor Signal

Investigating whether information derived from a single ECG lead can support apnea-related screening without requiring the full set of signals used in a clinical sleep study.

### 5.2 Temporal Representation

Investigating RR intervals, heart rate, HRV, and temporal windows as representations of cardiac dynamics associated with apnea-related events.

### 5.3 Lightweight Temporal Modeling

Investigating whether a compact Transformer can model temporal dependencies while maintaining a lower computational burden than larger deep-learning architectures.

### 5.4 Subject-Wise Generalization

Ensuring that evaluation does not incorrectly benefit from having windows from the same individual in both training and testing data.

### 5.5 Explainability

Investigating which input features or temporal regions contribute to model predictions.

### 5.6 Performance versus Complexity

Evaluating model performance together with computational requirements rather than optimizing only for predictive accuracy.

---

# 6. Proposed System

The overall pipeline is:

```text
                  PHYSIONET APNEA-ECG
                         |
                         v
                Single-Lead ECG Data
                         |
                         v
                ECG Preprocessing
                         |
                         v
                  R-Peak Detection
                         |
                         v
                 RR Interval Extraction
                         |
                         v
                 Heart Rate / HRV
                         |
                         v
                Temporal Windowing
                         |
                         v
                Label Assignment
                         |
              +----------+----------+
              |          |          |
              v          v          v
        Random Forest  1D CNN   Lightweight
                                Transformer
              |          |          |
              +----------+----------+
                         |
                         v
              Apnea / Non-Apnea
                  Screening
                         |
              +----------+----------+
              |                     |
              v                     v
       Explainability        Evaluation &
                              Efficiency
              |                     |
              +----------+----------+
                         |
                         v
                  Streamlit Demo
                         |
                         v
                       User
```

---

# 7. System Architecture

The system is divided into the following major stages.

## 7.1 ECG Data Acquisition

Single-lead ECG records are obtained from the PhysioNet Apnea-ECG Database.

The current supervised development pipeline uses the labelled learning records.

---

## 7.2 ECG Preprocessing

The raw ECG signal is processed to obtain a suitable representation for downstream physiological feature extraction.

The preprocessing stage is designed to reduce signal artifacts and improve the reliability of subsequent R-peak detection.

---

## 7.3 R-Peak Detection

R-peaks corresponding to individual cardiac cycles are detected from the preprocessed ECG signal.

The detected R-peak locations provide the basis for calculating consecutive inter-beat intervals.

---

## 7.4 RR-Interval Extraction

The time difference between consecutive R-peaks is used to calculate RR intervals.

RR intervals represent beat-to-beat cardiac timing information.

---

## 7.5 Heart Rate and HRV Representation

RR intervals are used to derive heart-rate and heart-rate-variability representations.

These features capture variations in cardiac cycle dynamics that may contain indirect physiological information associated with apnea-related events.

---

## 7.6 Temporal Windowing and Label Assignment

The processed physiological information is divided into temporal windows.

Each labelled window is assigned an apnea/non-apnea target according to the available annotations.

---

## 7.7 Model Development

Three model families are considered.

### Random Forest

Used as the conventional machine-learning baseline.

### 1D CNN

Used as a deep-learning baseline capable of learning local patterns from temporal input representations.

### Lightweight Transformer

Used as the proposed temporal model.

The Transformer uses self-attention to model temporal dependencies while keeping the architecture deliberately compact.

---

## 7.8 Screening Output

Each model produces a binary screening prediction:

```text
0 → Non-Apnea
1 → Apnea
```

The exact prediction interpretation must remain consistent with the annotation definition used by the dataset.

---

## 7.9 Explainability

Explainability methods are used to investigate influential features and/or temporal information contributing to predictions.

Only explainability techniques actually implemented and evaluated will be reported in the final project results.

---

## 7.10 Evaluation and Computational Analysis

The system evaluates both predictive and computational characteristics.

Performance metrics may include:

- Accuracy
- Sensitivity
- Specificity
- Precision
- F1-score
- ROC-AUC
- PR-AUC
- Balanced Accuracy
- Matthews Correlation Coefficient where appropriate

Computational analysis may include:

- Number of parameters
- Model size
- Inference latency
- Memory requirements where practical
- FLOPs where practical

---

# 8. Dataset

## PhysioNet Apnea-ECG Database

The project uses the **Apnea-ECG Database** available through PhysioNet.

Official dataset:

<https://physionet.org/content/apnea-ecg/1.0.0/>

The database contains overnight ECG recordings and associated annotations for sleep-apnea research.

For the current supervised development pipeline, the labelled learning set is used.

### Learning Records

The learning records comprise:

```text
a01 – a20
b01 – b05
c01 – c10
```

Total:

```text
35 labelled records
```

The recordings contain single-lead ECG signals sampled at:

```text
100 Hz
```

The learning records contain minute-level apnea annotations.

The `x` records are treated separately because the ordinary public apnea labels required for supervised training are not provided for those records.

---

# 9. Dataset File Types

The downloaded dataset contains several WFDB-related file types.

| Extension | Description |
|---|---|
| `.dat` | ECG signal data |
| `.hea` | WFDB header information |
| `.apn` | Apnea annotations for labelled learning records |
| `.qrs` | QRS annotations |
| `.xws` | Associated annotation/window information |

Additional metadata and documentation files may also be present in the downloaded dataset.

---

# 10. Dataset Storage

The dataset is **not committed to this GitHub repository**.

The expected local directory is:

```text
data/
└── raw/
    ├── a01.dat
    ├── a01.hea
    ├── a01.apn
    ├── ...
    ├── c10.dat
    ├── c10.hea
    ├── c10.apn
    └── ...
```

The contents of `data/raw/` are ignored by Git.

Only the directory placeholder:

```text
data/raw/.gitkeep
```

is maintained in the repository.

This keeps the repository lightweight and avoids redistributing the external dataset through the project repository.

---

# 11. Data Processing Pipeline

The data pipeline follows:

```text
Raw ECG
   |
   v
Signal Cleaning
   |
   v
Normalization
   |
   v
R-Peak Detection
   |
   v
RR Intervals
   |
   v
Heart Rate
   |
   v
HRV Features
   |
   v
Temporal Windowing
   |
   v
Apnea / Non-Apnea Labels
   |
   v
ML-Ready Dataset
```

The reusable implementation is maintained under:

```text
src/
├── data/
├── preprocessing/
└── features/
```

---

# 12. Model Pipeline

The model-development stage contains three approaches.

```text
ML-Ready Data
      |
      +------------------+
      |                  |
      v                  v
Feature-Based        Temporal Input
Representation       Representation
      |                  |
      v            +-----+-----+
Random Forest      |           |
                   v           v
                 1D CNN   Transformer
                   |           |
                   +-----+-----+
                         |
                         v
                 Model Predictions
```

---

# 13. Random Forest Baseline

Random Forest provides a conventional machine-learning reference point.

It is useful for evaluating how well engineered ECG-derived physiological features perform without using a deep neural architecture.

Implementation:

```text
src/models/random_forest.py
```

---

# 14. 1D CNN Baseline

The 1D CNN provides a deep-learning baseline for learning local temporal patterns from the prepared input representation.

Implementation:

```text
src/models/cnn_1d.py
```

---

# 15. Lightweight Transformer

The proposed Transformer is intentionally designed to remain compact.

The architecture is expected to include components such as:

```text
Input Representation
        |
        v
Input Projection
        |
        v
Positional Encoding
        |
        v
Multi-Head Self-Attention
        |
        v
Transformer Encoder Blocks
        |
        v
Pooling
        |
        v
Classification Layer
        |
        v
Apnea / Non-Apnea
```

The exact number of layers, embedding dimensions, attention heads, dropout, and other hyperparameters are determined experimentally and documented in the final implementation.

Implementation:

```text
src/models/lightweight_transformer.py
```

---

# 16. Validation Strategy

A major methodological consideration is subject-level data leakage.

Multiple temporal windows may originate from the same individual. Randomly splitting individual windows can therefore cause windows from one subject to appear in both training and testing sets.

The primary evaluation strategy therefore uses:

```text
5-Fold Grouped / Subject-Wise Cross-Validation
```

where all windows belonging to a particular subject remain within the same fold.

Results should be reported using aggregate statistics such as:

```text
Mean ± Standard Deviation
```

across validation folds where appropriate.

---

# 17. Evaluation Metrics

The project evaluates classification performance using multiple metrics.

## Accuracy

Overall proportion of correctly classified samples.

## Sensitivity

Ability to identify positive apnea-related samples.

## Specificity

Ability to identify non-apnea samples.

## Precision

Proportion of predicted positive samples that are actually positive.

## F1-Score

Harmonic mean of precision and recall.

## ROC-AUC

Measures discrimination across classification thresholds.

## PR-AUC

Useful for evaluating performance under class imbalance.

## Balanced Accuracy

Provides a balanced view of positive and negative class performance.

## Confusion Matrix

Provides detailed counts of:

- True Positives
- True Negatives
- False Positives
- False Negatives

---

# 18. Computational Efficiency

The project does not define "lightweight" only through model architecture.

Computational efficiency is evaluated using measurable characteristics such as:

- Parameter count
- Model file size
- Inference latency
- Memory usage where practical
- Computational operations where practical

This enables comparison between predictive performance and computational requirements.

---

# 19. Explainable AI

The project investigates model interpretability to improve understanding of the predictions.

Depending on the final implementation, explainability may include:

- Feature importance
- SHAP-based analysis
- Temporal importance
- Saliency-based explanations

The final report will only claim explainability techniques that have actually been implemented and evaluated.

---

# 20. Repository Structure

```text
sleep-apnea-ecg-screening/
│
├── .gitignore
├── .python-version
├── environment.yml
├── LICENSE
├── README.md
├── requirements.txt
├── CONTRIBUTING.md
│
├── app/
│   ├── streamlit_app.py
│   ├── assets/
│   │   └── .gitkeep
│   └── components/
│       ├── ecg_visualization.py
│       ├── explanation_display.py
│       └── prediction_display.py
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── docs/
│   ├── data_schema.md
│   ├── experiments.md
│   ├── meeting_notes.md
│   ├── methodology.md
│   └── project_architecture.md
│
├── models/
│   ├── checkpoints/
│   │   └── .gitkeep
│   └── trained/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_ecg_preprocessing.ipynb
│   ├── 03_rpeak_rr_hrv.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_lightweight_transformer.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_explainability.ipynb
│
├── reports/
│   ├── figures/
│   ├── outcome_paper/
│   └── project_report/
│
├── results/
│   ├── experiments/
│   │   └── .gitkeep
│   ├── figures/
│   │   └── .gitkeep
│   ├── metrics/
│   │   └── .gitkeep
│   └── predictions/
│       └── .gitkeep
│
├── src/
│   ├── data/
│   │   ├── dataset_utils.py
│   │   ├── download_dataset.py
│   │   ├── load_ecg.py
│   │   └── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── signal_cleaning.py
│   │   ├── normalization.py
│   │   ├── preprocessing_pipeline.py
│   │   └── __init__.py
│   │
│   ├── features/
│   │   ├── rpeak_detection.py
│   │   ├── rr_intervals.py
│   │   ├── heart_rate.py
│   │   ├── hrv_features.py
│   │   ├── windowing.py
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── cnn_1d.py
│   │   ├── lightweight_transformer.py
│   │   ├── model_utils.py
│   │   └── __init__.py
│   │
│   ├── evaluation/
│   │   ├── cross_validation.py
│   │   ├── metrics.py
│   │   ├── model_comparison.py
│   │   ├── complexity.py
│   │   ├── error_analysis.py
│   │   └── __init__.py
│   │
│   ├── explainability/
│   │   ├── feature_importance.py
│   │   ├── temporal_explanations.py
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logging_utils.py
│       ├── reproducibility.py
│       └── __init__.py
│
└── tests/
    ├── test_data_loading.py
    ├── test_features.py
    ├── test_models.py
    ├── test_pipeline.py
    └── test_preprocessing.py
```

---

# 21. Directory Responsibilities

## `app/`

Contains the Streamlit demonstration application.

This directory should contain presentation and application-layer code rather than core model-training logic.

---

## `data/`

Contains local dataset-related files.

### `data/raw/`

Raw external dataset.

### `data/interim/`

Temporary/intermediate processing outputs.

### `data/processed/`

ML-ready or cleaned datasets generated by the pipeline.

Raw and generated data are excluded from Git where appropriate.

---

## `docs/`

Contains project documentation, experimental decisions, architecture descriptions, methodology, and data schemas.

---

## `models/`

Contains locally generated model checkpoints and trained model artifacts.

Large trained models should not be committed to the repository unless specifically required.

---

## `notebooks/`

Contains exploratory analysis and experimental notebooks.

Notebooks should primarily be used for exploration, visualization, and experimentation.

Reusable implementation should be moved into `src/`.

---

## `reports/`

Contains the project report, outcome paper, and associated figures.

---

## `results/`

Contains generated experimental outputs including:

- metrics,
- predictions,
- figures,
- experiment records.

---

## `src/`

Contains the reusable project implementation.

This is the primary software-development directory.

---

## `tests/`

Contains automated tests for the project's reusable modules.

---

# 22. Installation

## Option A — Python Virtual Environment

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd sleep-apnea-ecg-screening
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Option B — Conda

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate sleep-apnea-ecg
```

---

# 23. Dataset Setup

Download the Apnea-ECG Database from:

<https://physionet.org/content/apnea-ecg/1.0.0/>

Place the extracted files under:

```text
data/raw/
```

The raw dataset is intentionally excluded from Git tracking.

Refer to:

```text
docs/data_schema.md
```

for additional information about the dataset and expected data structure.

---

# 24. Running the Pipeline

The exact execution commands will evolve as the implementation is completed.

The conceptual execution order is:

```text
1. Dataset loading
2. ECG preprocessing
3. R-peak detection
4. RR interval extraction
5. Heart rate / HRV extraction
6. Window generation
7. Model training
8. Subject-wise validation
9. Model comparison
10. Explainability
11. Computational analysis
12. Streamlit integration
```

Corresponding source modules are organized under:

```text
src/data/
src/preprocessing/
src/features/
src/models/
src/evaluation/
src/explainability/
```

---

# 25. Running the Streamlit Application

Once the required trained model and preprocessing artifacts are available:

```bash
streamlit run app/streamlit_app.py
```

The prototype is intended to demonstrate:

- ECG input/selection,
- ECG visualization,
- preprocessing/analysis,
- screening prediction,
- prediction explanation, and
- relevant model information.

---

# 26. Notebooks

The notebooks are organized chronologically.

| Notebook | Purpose |
|---|---|
| `01_dataset_exploration.ipynb` | Explore dataset structure, records and annotations |
| `02_ecg_preprocessing.ipynb` | Investigate ECG preprocessing |
| `03_rpeak_rr_hrv.ipynb` | R-peaks, RR intervals, heart rate and HRV |
| `04_baseline_models.ipynb` | Random Forest and 1D CNN experiments |
| `05_lightweight_transformer.ipynb` | Lightweight Transformer experiments |
| `06_model_evaluation.ipynb` | Comparative and subject-wise evaluation |
| `07_explainability.ipynb` | Explainability experiments |

---

# 27. Testing

Run the test suite using:

```bash
pytest
```

The test suite is intended to cover:

- dataset loading,
- preprocessing,
- feature extraction,
- model components, and
- pipeline functionality.

Before submitting a Pull Request, contributors should run the tests relevant to their changes.

---

# 28. Reproducibility

The project emphasizes reproducible experimentation.

Important experimental settings should be centralized where practical through:

```text
src/utils/config.py
```

Examples include:

- random seed,
- ECG sampling frequency,
- validation configuration,
- model configuration,
- window configuration.

The final implementation should avoid duplicating important experimental constants across multiple files.

---

# 29. Experiment Documentation

Major experimental decisions are documented in:

```text
docs/experiments.md
```

The document records:

- dataset configuration,
- input representation,
- models,
- validation strategy,
- performance metrics,
- computational metrics,
- explainability approach, and
- interpretation rules.

---

# 30. Scientific and Medical Scope

This project is an academic research prototype.

Important interpretation rules:

1. ECG is treated as an indirect physiological signal associated with apnea-related events.

2. ECG does not directly measure airway obstruction.

3. The system performs computational screening and should not be described as a clinical diagnostic system.

4. A model prediction of an apnea-related window must not be presented as a medical diagnosis.

5. The percentage of predicted positive windows must not be referred to as the clinical Apnea-Hypopnea Index (AHI).

6. External validation should only be claimed if an appropriate external labelled dataset is actually evaluated.

7. Experimental results must only contain values obtained from executed experiments.

---

# 31. Project Status

## Repository

- [x] Repository structure
- [x] Dataset directory structure
- [x] Source-code directory structure
- [x] Documentation structure
- [ ] Initial GitHub release

## Data Pipeline

- [x] Dataset acquisition
- [ ] Dataset exploration
- [ ] ECG preprocessing
- [ ] R-peak detection
- [ ] RR interval extraction
- [ ] HRV feature extraction
- [ ] Temporal windowing
- [ ] ML-ready dataset generation

## Models

- [ ] Random Forest baseline
- [ ] 1D CNN baseline
- [ ] Lightweight Transformer
- [ ] Model comparison

## Evaluation

- [ ] Subject-wise cross-validation
- [ ] Classification metrics
- [ ] Confusion matrices
- [ ] ROC/PR analysis
- [ ] Error analysis
- [ ] Computational complexity analysis
- [ ] Inference latency analysis

## Explainability

- [ ] Feature-level explanation
- [ ] Temporal explanation

## Application

- [ ] ECG visualization
- [ ] Prediction display
- [ ] Explanation display
- [ ] Streamlit integration
- [ ] End-to-end demonstration

## Documentation

- [x] Methodology documentation
- [x] Architecture documentation
- [x] Data schema documentation
- [x] Experimental protocol
- [ ] Final project report
- [ ] Outcome paper
- [ ] Final presentation

---

# 32. Planned Project Workflow

The project follows the following development sequence:

```text
Problem Identification
        ↓
Literature Survey
        ↓
Research Gap Identification
        ↓
Dataset Selection
        ↓
Dataset Preparation
        ↓
ECG Preprocessing
        ↓
R-Peak / RR / HRV Pipeline
        ↓
ML-Ready Dataset
        ↓
Baseline Models
        ↓
Lightweight Transformer
        ↓
Subject-Wise Evaluation
        ↓
Explainability
        ↓
Computational Analysis
        ↓
Streamlit Integration
        ↓
Final Demonstration
```

Report and outcome-paper preparation are carried out in parallel with the technical development.

---

# 33. Contribution Guidelines

This is a collaborative academic repository.

Before modifying the project, contributors should read:

```text
CONTRIBUTING.md
```

The contribution guide defines:

- individual responsibilities,
- branch structure,
- files owned by each member,
- development workflow,
- testing requirements,
- commit conventions,
- Pull Request workflow,
- code ownership,
- data handling,
- experiment handling, and
- integration rules.

---

# 34. License

The source code of this project is released under the MIT License.

The external PhysioNet Apnea-ECG Database is not covered by this repository's MIT License. Users must comply with the dataset provider's licensing, attribution, and usage requirements.

See:

```text
LICENSE
```

for the project's source-code license.

---

# 35. Dataset Attribution

Dataset:

**Apnea-ECG Database**

PhysioNet:

<https://physionet.org/content/apnea-ecg/1.0.0/>

The original dataset should be appropriately cited in any academic work, publication, presentation, or derivative research using the dataset.

---

# 36. Academic Use

This repository is developed as part of an academic B.Tech Project-I at VIT Vellore.

The repository is intended to support:

- reproducible research,
- collaborative development,
- experimental evaluation,
- academic documentation, and
- demonstration of the proposed screening pipeline.

---

# 37. Acknowledgement

The project team acknowledges the maintainers and contributors of the PhysioNet Apnea-ECG Database for making the dataset available for research.

The project is conducted under the guidance of:

**Dr. Divya Meena S.**

VIT Vellore.

---

# Quick Start

```bash
git clone <REPOSITORY_URL>
cd sleep-apnea-ecg-screening

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

pytest

streamlit run app/streamlit_app.py
```

For contribution instructions:

```text
CONTRIBUTING.md
```

For experimental configuration:

```text
docs/experiments.md
```

For dataset structure:

```text
docs/data_schema.md
```

For architecture:

```text
docs/project_architecture.md
```

---

## Disclaimer

This software is provided for academic and research purposes only.

It is not a medical device and must not be used as a substitute for professional medical evaluation, polysomnography, or clinical diagnosis.