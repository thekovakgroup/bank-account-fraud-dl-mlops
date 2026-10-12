# Milestone 01 Technical Specification

## Project

Bank Account Fraud Detection - Deep Learning & MLOps

## Objective

Build a production-oriented binary fraud-detection system using the Bank Account Fraud Dataset Suite, with `fraud_bool` as the target and a PyTorch multilayer perceptron as the primary deep-learning model.

## Initial Dataset Scope

Primary dataset:

- `Base.csv`

The complete dataset suite will only be used if later analysis or instructor requirements justify cross-dataset comparison.

## Core Modeling Plan

Baseline models:

- Dummy classifier
- Logistic regression

Primary deep-learning model:

- PyTorch multilayer perceptron

Primary ranking metric:

- PR-AUC

Supporting metrics:

- Recall
- Precision
- F1 score
- ROC AUC
- Confusion matrix
- False-positive count
- False-negative count
- Calibration metrics
- Threshold-dependent business cost
- Cohort and fairness indicators

## Validation Policy

The project will use separate training, validation, and final test partitions.

The final test set will remain sealed until:

- preprocessing is frozen
- feature policy is frozen
- model architecture is frozen
- hyperparameters are frozen
- calibration strategy is frozen
- decision threshold is frozen
- evaluation policy is frozen

Final-test performance will not be used for hyperparameter or threshold selection.

## Split Design

The dataset's `month` field will be audited before selecting the final split.

The audit will evaluate:

- temporal distribution
- fraud rate by month
- row counts by month
- possible chronological split
- train/validation/test distribution shift

A chronological split will only be used if the data structure supports it.

## Leakage Review

The project will explicitly investigate:

- target leakage
- prediction-time feature availability
- duplicate records
- near-duplicate records
- preprocessing leakage
- identity-level leakage
- application-level leakage
- high-cardinality categorical features
- rare categories
- missing-value patterns

All preprocessing will be fit on training data only.

## Imbalance Strategy

Accuracy will not be treated as the primary metric.

Potential methods to evaluate include:

- positive-class weighting
- threshold adjustment
- carefully justified sampling strategies

Techniques will only be retained when they improve the documented evaluation objective.

## PyTorch Training Strategy

Initial components to evaluate:

- `BCEWithLogitsLoss`
- positive-class weighting
- mini-batch training
- Adam optimizer
- learning-rate selection
- early stopping
- dropout
- batch normalization when justified
- gradient monitoring
- gradient clipping when justified
- deterministic random seeds where practical

## Responsible AI

The system will include:

- cohort-level error analysis
- false-positive disparity analysis
- proxy-variable review
- explainability analysis
- fairness assessment
- security review
- human-review recommendations
- model limitations
- monitoring and escalation policy
- model card
- data card

Feature importance and attribution will not be presented as causality.

The model will not be recommended as the sole basis for automated denial, punitive action, or adverse treatment.

## Engineering Architecture

The intended system flow is:

Raw data
-> validation
-> preprocessing
-> train/validation/test datasets
-> baseline models
-> PyTorch MLP
-> MLflow experiment tracking
-> model selection
-> calibration and threshold policy
-> model registry
-> FastAPI inference service
-> Docker container
-> CI/CD
-> cloud deployment
-> drift monitoring

## Artifact Policy

Git will store:

- source code
- notebooks
- tests
- configuration
- lightweight reports
- documentation
- CI/CD definitions
- artifact manifests

Git will not store:

- `.venv`
- raw datasets
- processed datasets
- credentials
- MLflow run directories
- cache files
- large model artifacts
- large checkpoints

The final model artifact strategy will include:

- model version
- preprocessing version
- Git commit SHA
- configuration version
- artifact URI
- checksum
- retrieval instructions

## Compute Strategy

Local development will prioritize CPU-compatible workflows and development subsets.

Google Colab may be used for compute-intensive PyTorch training.

The local GitHub repository remains the canonical source of project code and history.

## Deployment Strategy

The final system will include:

- FastAPI
- Pydantic validation
- automated tests
- Docker
- GitHub Actions
- cloud deployment
- drift monitoring
- rollback and cleanup instructions

Cloud provider selection is deferred until the deployment milestone.

## Milestone Workflow

Each milestone will:

1. use a dedicated Git branch
2. include learner-oriented notebooks where appropriate
3. move reusable logic into `src/`
4. run validation checks and tests
5. generate required report artifacts
6. review `git status`
7. stage only intended files
8. run `git diff --cached --check`
9. commit and push the milestone branch
10. merge to `main` only after verification