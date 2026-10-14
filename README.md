# Bank Account Fraud Detection - Deep Learning & MLOps

End-to-end fraud-detection capstone using the Bank Account Fraud Dataset Suite.

## Project Goal

Build an employer-presentable fraud-detection system that combines:

- Reproducible data ingestion and preprocessing
- Classical baseline models
- A PyTorch multilayer perceptron
- Imbalanced-class evaluation with PR-AUC as the primary ranking metric
- Threshold and business-cost analysis
- Explainability and responsible-AI review
- MLflow experiment tracking and model registry
- FastAPI model serving
- Docker containerization
- GitHub Actions CI/CD
- Drift monitoring
- Cloud deployment

## Primary Dataset

Initial plan:

- Use `Base.csv` from the Bank Account Fraud Dataset Suite
- Target column: `fraud_bool`
- Verify actual dataset shape and schema after acquisition
- Keep raw data immutable and outside Git

## Development Environment

- Python 3.11
- VS Code
- Windows PowerShell
- Git and GitHub
- Local CPU development
- Google Colab may be used for compute-intensive training

## Repository Structure

```text
app/
configs/
data/
  raw/
  processed/
models/
notebooks/
reports/
  figures/
src/
  data/
  features/
  models/
  training/
  evaluation/
tests/
.github/
  workflows/
```

## Current Status

- **Milestones 01–05:** Completed.
- **Milestone 06 — Model Improvement & Hyperparameter Tuning:** Model experiments, evaluation, and MLflow tracking completed. Final GitHub checkpoint pending.
- **Milestone 07 — Calibration, Business Costs & Threshold Selection:** Not started.
- **Later milestones:** Explainability, model serving, MLOps, and deployment remain planned.

The final month-7 test set remains **SEALED**.

## Milestone 06 — Model Improvement Results

### Experiment Setup

- Dataset: Bank Account Fraud Dataset Suite (`Base.csv`)
- Target: `fraud_bool`
- Training period: months 0–5 (794,989 records)
- Validation period: month 6 (108,168 records)
- Final holdout: month 7 (SEALED)
- Processed features: 55
- Primary model-ranking metric: validation PR-AUC

### Model Comparison

| Model | Validation PR-AUC |
|---|---:|
| HistGradientBoosting | **0.166961** |
| PyTorch MLP — full class weighting | 0.163862 |
| PyTorch MLP — positive weight 25 | 0.161925 |
| PyTorch MLP — unweighted | 0.160393 |
| Milestone 05 CPU MLP | 0.157469 |

HistGradientBoosting remains the leading model by validation PR-AUC.

The full-weighted PyTorch MLP is a promising alternative under an operational alert-budget evaluation.

### Operational Evaluation

At a 5% alert budget on validation month 6:

- Full-weighted PyTorch MLP: 706 fraud cases detected.
- HistGradientBoosting: 680 fraud cases detected.
- Observed MLP advantage: 26 additional fraud cases.

A paired bootstrap analysis with 2,000 resamples estimated a 95% percentile interval of approximately +1 to +48 detected fraud cases for the MLP-minus-HistGradientBoosting difference.

These are exploratory validation findings, not evidence of guaranteed future performance. Model selection, calibration, and the operating threshold are not yet finalized.

### MLflow Experiment Tracking

MLflow 3.17.1 is configured with a local SQLite backend.

Experiment: `bank-account-fraud-m06-historical`

Five completed model experiments were imported and independently verified.

These runs are explicitly identified as historical imports. They do not represent new training executions.

Tracking components:

- Import script: `src/tracking/import_m06_history.py`
- Tracking database: `artifacts/milestone_06/mlflow_tracking.db`
- Artifact directory: `artifacts/milestone_06/mlflow_artifacts/`

The local tracking database and artifacts are excluded from Git.

To preview the historical import from the repository root:

```powershell
python src/tracking/import_m06_history.py
```

To view the existing tracked experiments locally:

```powershell
mlflow ui --backend-store-uri "sqlite:///artifacts/milestone_06/mlflow_tracking.db" --host 127.0.0.1 --port 5050
```

Open `http://127.0.0.1:5050` and select **Model training**.

### Reproducibility

- Python 3.11.9
- PyTorch 2.14.1+cpu (local environment)
- scikit-learn 1.9.1
- MLflow 3.17.1
- `requirements.txt`: Direct dependencies
- `requirements-lock.txt`: 121 pinned installed packages

GPU experiments were originally executed in Google Colab. Their recorded metrics retain that provenance.

### Next Milestone

Milestone 07 will address calibration, business-cost evaluation, and operating-threshold selection.

The month-7 test set must remain sealed until preprocessing, model architecture, hyperparameters, calibration strategy, and threshold policy have been frozen.

## Evaluation Policy

The project will preserve a sealed final test set until:

- preprocessing is frozen
- model architecture is frozen
- hyperparameters are frozen
- calibration strategy is frozen
- threshold policy is frozen

Final-test results will not be used for model or threshold selection.

## Responsible AI Position

This project is intended as a fraud-risk decision-support system.

Model output should not be used as the sole basis for automated denial, punitive action, or adverse treatment. High-risk predictions should support human review and documented escalation procedures.