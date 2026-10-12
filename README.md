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

Milestone 01 - Requirements, Architecture, Repository Setup, and Data Acquisition

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