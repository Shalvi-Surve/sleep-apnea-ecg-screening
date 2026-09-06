# Contributing Guide

Thank you for contributing to **Lightweight ECG-Based Screening System for Early Sleep Apnea Detection**.

This guide defines how the team should work with the GitHub repository, branches, code, experiments, and documentation.

---

## 1. Team Workflow

We use a simple **feature branch → Pull Request → review → merge** workflow.

### Branches

- `main` → Stable, review-ready version. **Do not push directly.**
- `develop` → Integration branch for completed work.
- Feature branches → Individual tasks/features.

### Feature Branch Naming

Use:

```text
feature/<name>-<task>
```

Examples:

```text
feature/swagata-ecg-pipeline
feature/shalvi-model-pipeline
feature/narhari-validation
```

---

## 2. How to Work on a Task

Before starting:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/<your-name>-<task>
```

Work only on the assigned task.

After completing the work:

```bash
git status
git add .
git commit -m "Add <short description>"
git push -u origin feature/<your-name>-<task>
```

Then create a **Pull Request (PR)** to:

```text
feature branch → develop
```

A teammate should review the PR before it is merged.

---

## 3. Team Responsibilities & Folder/File Ownership

Each member has primary ownership over specific parts of the repository. Ownership means being responsible for development, testing and maintenance of that component — **not exclusive access**. Everyone should review the overall project.

| Team Member | Primary Technical Responsibility |
|---|---|
| **Swagata** | ECG preprocessing, R-peak detection, RR intervals, HRV features, labelled windows and ML-ready dataset |
| **Shalvi** | Random Forest, 1D CNN, Lightweight Transformer, model evaluation and system integration |
| **Narhari** | Model/output validation, result verification, master results, confusion matrices, plots and demo testing |

Everyone should understand the **complete pipeline**, not only their assigned component.


### Swagata — ECG Processing & Feature Pipeline

**Primary folders/files:**

```text
src/
├── data/
│   ├── dataset_utils.py
│   ├── load_ecg.py
│   └── download_dataset.py
│
├── preprocessing/
│   ├── signal_cleaning.py
│   ├── normalization.py
│   └── preprocessing_pipeline.py
│
└── features/
    ├── rpeak_detection.py
    ├── rr_intervals.py
    ├── heart_rate.py
    └── hrv_features.py

notebooks/
├── 01_dataset_exploration.ipynb
├── 02_ecg_preprocessing.ipynb
└── 03_rpeak_rr_hrv.ipynb

docs/
└── data_schema.md
```

**Main responsibilities:**

- PhysioNet dataset loading and preparation
- ECG preprocessing and signal cleaning
- R-peak detection
- RR interval extraction
- Heart-rate and HRV feature extraction
- Windowing and label preparation
- Validation of the ML-ready dataset
- Documentation of the data pipeline

---

### Shalvi — Models, Evaluation & System Integration

**Primary folders/files:**

```text
src/
├── models/
│   ├── random_forest.py
│   ├── cnn_1d.py
│   ├── lightweight_transformer.py
│   └── model_utils.py
│
└── evaluation/
    ├── cross_validation.py
    ├── metrics.py
    ├── model_comparison.py
    ├── complexity.py
    └── error_analysis.py

notebooks/
├── 04_baseline_models.ipynb
├── 05_lightweight_transformer.ipynb
└── 06_model_evaluation.ipynb

app/
└── streamlit_app.py

docs/
├── methodology.md
├── project_architecture.md
└── experiments.md

app/components/
├── ecg_visualization.py
├── explanation_display.py
└── prediction_display.py
```

**Main responsibilities:**

- Random Forest baseline
- 1D CNN baseline
- Lightweight Transformer
- Subject-wise/grouped validation
- Performance comparison
- Computational efficiency analysis
- Error analysis
- Model integration
- Streamlit application integration
- Experiment and methodology documentation

---

### Narhari — Validation, Results & Explainability

**Primary folders/files:**

```text
src/
├── evaluation/
│   └── error_analysis.py
│
└── explainability/
    ├── feature_importance.py
    └── temporal_explanations.py

notebooks/
└── 07_explainability.ipynb

results/
├── experiments/
├── figures/
├── metrics/
└── predictions/

```

**Main responsibilities:**

- Validate model outputs and predictions
- Maintain the master results
- Check metrics and confusion matrices
- Organise experiment figures and tables
- Error and failure-case analysis
- Feature-level and temporal explainability
- ECG/model visualisations
- Streamlit result and explanation displays
- Demo testing and final output verification

---

### Shared Files — Everyone

The following files are maintained collaboratively:

```text
README.md
requirements.txt
environment.yml
.gitignore
LICENSE

docs/
├── meeting_notes.md
├── methodology.md
├── project_architecture.md
├── data_schema.md
└── experiments.md

tests/
├── test_data_loading.py
├── test_features.py
├── test_models.py
├── test_pipeline.py
└── test_preprocessing.py

src/utils/
├── config.py
├── logging_utils.py
└── reproducibility.py
```

Everyone should:

- Keep shared documentation up to date.
- Add or update tests when modifying functionality.
- Review changes affecting shared utilities.
- Ensure configuration changes do not break another member's work.
- Keep the README consistent with the actual implementation.

---

### Repository Ownership Summary

| Area | Primary Owner | Others |
|---|---|---|
| Dataset & data loading | Swagata | Shalvi validates |
| ECG preprocessing | Swagata | Shalvi validates |
| R-peaks / RR / HRV | Swagata | Shalvi validates |
| ML-ready dataset | Swagata | Shalvi validates |
| Random Forest | Shalvi | Narhari validates |
| 1D CNN | Shalvi | Narhari validates |
| Lightweight Transformer | Shalvi | Narhari validates |
| Model evaluation | Shalvi | Narhari validates |
| Complexity analysis | Shalvi | Narhari validates |
| Error analysis | Narhari | Shalvi reviews |
| Explainability | Narhari | Shalvi reviews |
| Results & figures | Narhari | Everyone verifies |
| Streamlit integration | Shalvi | Narhari tests |
| Streamlit visualisation | Shalvi | Narhari tests |
| Documentation | Shared | Everyone reviews |
| Testing | Shared | Everyone |
| Final report | Shared | Everyone |
| Final presentation/demo | Shared | Everyone |

> **Important:** Folder ownership does not mean that other members cannot edit those files. If a change affects another member's pipeline, discuss it with the owner first and ensure the change is tested.

---

## 4. Code Guidelines

- Keep code modular and readable.
- Follow the existing project structure.
- Reuse existing utilities instead of creating duplicate functions.
- Use meaningful variable, function and file names.
- Add comments only where they improve understanding.
- Do not commit temporary files, notebooks outputs, datasets or unnecessary generated files.
- Do not hard-code paths specific to one computer.
- Keep configuration values in the shared configuration file where appropriate.
- Test your code before creating a PR.

---

## 5. Data Guidelines

The PhysioNet Apnea-ECG dataset is used for this project.

**Do not commit the dataset to GitHub.**

Keep the dataset locally under:

```text
data/raw/
```

The repository should contain only the required folder structure/placeholders.

Do not modify the original dataset. Generated/processed data should be stored separately under:

```text
data/interim/
data/processed/
```

---

## 6. Experiment & Model Guidelines

All experiments must be reproducible and properly documented.

Our core models are:

- Random Forest — traditional ML baseline
- 1D CNN — deep learning baseline
- Lightweight Transformer — proposed model

Evaluation should use **subject-wise/grouped validation** to reduce patient-level data leakage.

Record important experiment details such as:

- Model configuration
- Dataset/version used
- Validation strategy
- Performance metrics
- Model parameters
- Model size
- Inference time
- Important observations

Do **not** change the validation strategy just to obtain better results.

---

## 7. Results & Evidence

Every reported result must come from an actual experiment.

Do **not**:

- Invent metrics
- Copy results from papers and present them as project results
- Modify results to improve performance
- Claim diagnosis when the system performs screening
- Report unverified outputs

Keep generated results organised under:

```text
results/
├── experiments/
├── figures/
├── metrics/
└── predictions/
```

Narhari will maintain the **master results and evidence**, but every team member is responsible for checking the correctness of their own outputs.

---

## 8. Testing Before PR

Before opening a Pull Request:

```text
☐ Code runs without errors
☐ Assigned functionality works
☐ Existing functionality is not broken
☐ Paths are portable
☐ No dataset/secrets/temporary files are committed
☐ Relevant tests have been checked
☐ Results are reproducible
☐ Documentation is updated if required
```

---

## 9. Pull Request Guidelines

Each PR should:

- Have a clear title.
- Explain what was changed.
- Mention the relevant task/module.
- Include screenshots or results when useful.
- Be limited to one logical feature/task.
- Be reviewed by at least one teammate.

Example PR title:

```text
Add ECG preprocessing and R-peak detection
```

PR flow:

```text
Create branch
      ↓
Implement task
      ↓
Test locally
      ↓
Commit & Push
      ↓
Create PR → develop
      ↓
Teammate Review
      ↓
Fix requested changes
      ↓
Merge into develop
```

---

## 10. Commit Guidelines

Use short, meaningful commit messages.

Examples:

```text
Add ECG preprocessing pipeline
Implement R-peak detection
Add HRV feature extraction
Add Random Forest baseline
Implement lightweight Transformer
Add subject-wise evaluation
Add model comparison metrics
Update Streamlit prediction display
```

Avoid messages such as:

```text
update
changes
final
final2
new
test
```

---

## 11. Documentation Guidelines

Update documentation whenever a major implementation decision changes.

Important documentation includes:

```text
docs/data_schema.md
docs/experiments.md
docs/methodology.md
docs/project_architecture.md
```

Keep documentation consistent with the actual implementation.

If the code changes, check whether the relevant documentation also needs to be updated.

---

## 12. Report & Presentation

All three members contribute to the final report and presentation.

Initial section ownership:

- **Shalvi:** Problem statement, objectives, methodology, architecture, model architecture, implementation and evaluation.
- **Swagata:** Introduction/background, dataset, ECG preprocessing, R-peak/RR/HRV methodology.
- **Narhari:** Literature survey, research gap, results, discussion, limitations, future scope and references.

These assignments are for coordination only. **Everyone must review the complete report before submission.**

---

## 13. Important Team Rules

1. **Do not push directly to `main`.**
2. **Pull the latest `develop` before starting work.**
3. **Work on your own feature branch.**
4. **Test before creating a PR.**
5. **Get a teammate review before merging.**
6. **Never commit the dataset or sensitive information.**
7. **Never fabricate experimental results.**
8. **Keep code, results and documentation consistent.**
9. **Do not unnecessarily change another member's module without discussion.**
10. **If you break something, communicate it immediately instead of hiding it.**

---

## 14. Recommended Working Pattern

For every technical task, follow:

```text
BUILD
  ↓
TEST
  ↓
VALIDATE
  ↓
DOCUMENT
  ↓
COMMIT
  ↓
PULL REQUEST
  ↓
REVIEW
  ↓
MERGE
```

The goal is to maintain a **clean, reproducible and review-ready repository** throughout the project rather than trying to clean everything immediately before Review-2.

---

## 15. Quick Start for New Contributors

```bash
git clone <repository-url>
cd sleep-apnea-ecg-screening

git checkout develop
git pull origin develop

git checkout -b feature/<your-name>-<task>
```

After completing your task:

```bash
git add .
git commit -m "<meaningful message>"
git push -u origin feature/<your-name>-<task>
```

Create a Pull Request to `develop` and wait for teammate review.

---

**Keep it clean. Keep it reproducible. Keep it reviewable.**