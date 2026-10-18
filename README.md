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

- **Milestones 01-06:** Completed and merged into `main`.
- **Milestone 07 - Calibration and Final Evaluation:** Completed.
- **Milestone 08 - Explainability and Responsible AI:** Completed and merged.
- **Milestone 09 - Model Serving API:** Implementation, local verification, documentation, and all 40 API regression tests completed; Git merge pending.
- **Milestones 10-12:** Remaining MLOps engineering and deployment work.

### Frozen Model and Evaluation Policy

The selected research candidate is a calibrated PyTorch MLP
with architecture 55 -> 64 -> 32 -> 1.

The serving pipeline uses:
- Frozen fitted preprocessing
- Frozen PyTorch model checkpoint
- Frozen Platt calibration
- Frozen operational threshold of approximately 4.9296%
- Maximum investigation capacity of 5% per complete cohort

The Month-7 final holdout was evaluated after model
selection and policy freezing. It is no longer sealed
and must never be reused for tuning or model selection.

The system produces fraud-risk scores for human review.
It is NOT approved for production use or automated
bank-account rejection.

### Milestone 09 API

The research-only FastAPI application provides:

- `GET /health` - Application liveness
- `GET /ready` - Inference engine readiness
- `POST /predict` - Calibrated fraud-risk scores

Safeguards include strict input validation,
100-application batch limits, 256 KiB request-body
protection, model lifecycle checks, privacy-safe
error logging, and controlled HTTP errors.

Local Uvicorn startup, model readiness, and Swagger
documentation have been verified successfully.

The API does not make automatic banking decisions.
The 5% investigation-capacity policy applies to
complete cohorts, not individual API request batches.

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
- `requirements-lock.txt`: Pinned local dependency environment, including Milestone 09 API dependencies

GPU experiments were originally executed in Google Colab. Their recorded metrics retain that provenance.


### Historical Milestone 06 Handoff

At the conclusion of Milestone 06, calibration,
threshold selection, and final evaluation were pending.

These activities were subsequently completed during
Milestone 07.

The Month-7 holdout was evaluated under the frozen
evaluation protocol and is no longer sealed.

It must not be reused for tuning, calibration,
threshold optimization, or model selection.

The Milestone 06 experimental findings above are
preserved as historical development results.

## Evaluation Policy

The final evaluation followed a frozen-model protocol.

Before the Month-7 holdout was evaluated, the project
froze its preprocessing, model architecture,
hyperparameters, calibration strategy, and
operating-threshold policy.

The Month-7 holdout has been consumed and is reserved
exclusively as a historical final evaluation record.

It must not be used again for:
- Hyperparameter tuning
- Threshold optimization
- Calibration fitting
- Model selection
- Additional exploratory experiments

Operational investigation recommendations must respect
the frozen 5% capacity limit over a complete cohort.

Individual API scoring requests return calibrated
probabilities only; they do not enforce a cohort-level
investigation budget or authorize adverse decisions.

## Responsible AI Position

This project is intended as a fraud-risk decision-support system.

Model output should not be used as the sole basis for automated denial, punitive action, or adverse treatment. High-risk predictions should support human review and documented escalation procedures.

## Milestone 09 - FastAPI Model Serving

### Overview

The project includes a local FastAPI inference service
that serves calibrated fraud probabilities from the
frozen PyTorch MLP.

The application is research-only and is not approved
for production banking decisions.

### Prerequisites

- Python 3.11
- Project dependencies installed in `.venv`
- Frozen model and preprocessing artifacts available locally
- Required model artifact checksums verified

Private model artifacts are excluded from Git.
A fresh repository clone cannot serve the frozen model
until the required artifacts have been provisioned.

### Start the API Locally

From the repository root, activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the ASGI server:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Wait for:

`Application startup complete.`

Stop the server using `Ctrl + C`.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Application liveness |
| GET | `/ready` | Model readiness |
| POST | `/predict` | Calibrated fraud-risk scoring |
| GET | `/docs` | Swagger API documentation |
| GET | `/openapi.json` | OpenAPI specification |

Interactive documentation:

http://127.0.0.1:8000/docs

### Prediction Requests

The `/predict` endpoint accepts a JSON object containing
an `applications` array.

Each application must contain the 29 required raw features
defined in `app/schemas.py`.

The API validates numeric, categorical, and binary fields.
Unknown or excluded features are rejected.

Request limits:
- Minimum batch size: 1 application
- Maximum batch size: 100 applications
- Maximum request body: 256 KiB (262,144 bytes)

Use the interactive Swagger documentation to inspect the
complete input schema before submitting synthetic requests.

Never submit real customer information during development
or demonstrations.

### Prediction Responses

Successful responses include:

- Model version derived from the frozen checkpoint SHA256
- Frozen policy status
- Explicit `production_approved: false`
- `decision_type: risk_score_only`
- Calibrated fraud probability for each application
- Zero-based application index preserving input order

Probabilities are risk scores, not automatic fraud
determinations.

The service does not automatically reject bank-account
applications or select investigations.

### HTTP Error Handling

| Status | Meaning |
|---|---|
| 200 | Request completed successfully |
| 413 | Prediction request body exceeds 256 KiB |
| 422 | Request validation failed |
| 500 | Unexpected inference failure |
| 503 | Inference engine unavailable |

Unexpected inference errors produce sanitized operational
logs without request bodies or exception tracebacks.

### Operational Safeguards

The application implements:

- Model initialization during application startup
- Model readiness and lifecycle checks
- Frozen artifact integrity verification
- Strict Pydantic input and output validation
- Request batch and body-size limits
- Generic client-facing inference errors
- Privacy-conscious inference failure logging
- Localhost-only serving instructions

These safeguards do not replace authentication,
authorization, rate limiting, infrastructure security,
or a formal production approval process.

### Investigation Policy

The frozen operational policy uses an approximately
4.9296% threshold and a maximum 5% investigation
capacity per complete cohort.

The `/predict` API returns probabilities only.

The investigation-capacity policy cannot be enforced
correctly from independent request batches because
the complete cohort is required.

All investigation decisions require authorized human
review.

### Automated Testing

Run the Milestone 09 API tests from the repository root:

```powershell

python -m pytest tests -k api -v

```

The automated API tests use synthetic inference engines
and do not require private model artifacts.

The real frozen-model API has also been verified locally
through its startup, readiness, and Swagger endpoints.

The consumed Month-7 holdout must not be reused for
additional tuning or model selection.
