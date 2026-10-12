# Milestone 02 - Leakage Review and Split Decision

## Project

Bank Account Fraud Detection - Deep Learning & MLOps

## Objective

Milestone 02 evaluated prediction-time leakage risk, temporal distribution shift, and the train/validation/test strategy for `Base.csv`.

## Leakage Review

No numerical feature displayed an unusually strong linear relationship with `fraud_bool` that would indicate an obvious direct target proxy.

Correlation alone is not sufficient to determine whether a feature is leakage-safe. Feature availability and generation timing remain the primary criteria.

Key review decisions:

- `fraud_bool` is the target and must never enter the feature matrix.
- `device_fraud_count` is constant at zero in `Base.csv` and is a drop candidate.
- `device_fraud_count` also requires semantic caution because its definition references fraudulent applications.
- `credit_risk_score` remains subject to prediction-time availability review because it is a derived application-risk score.
- `month` is used for temporal splitting, analysis, and monitoring rather than as an initial model feature.
- Remaining predictors must represent information available at prediction time.

## Temporal Drift Evidence

The dataset exhibits meaningful temporal distribution shift.

Strong numerical movement was observed in features including:

`velocity_4w`, `velocity_24h`, `velocity_6h`,
`zip_count_4w`, and `date_of_birth_distinct_emails_4w`.

Categorical distributions also changed materially across time, particularly for:

`payment_type`, `housing_status`, `device_os`,
and `employment_status`.

Fraud prevalence increased in the later periods.

Because both the predictors and target distribution change over time, randomly mixing months across partitions could create an evaluation that is less representative of deployment.

## Final Split

The primary split is chronological.

| Partition | Months | Records | Fraud Cases | Fraud Rate |
|---|---|---:|---:|---:|
| Train | 0-5 | 794,989 | 8,151 | 1.0253% |
| Validation | 6 | 108,168 | 1,450 | 1.3405% |
| Test | 7 | 96,843 | 1,428 | 1.4746% |

The partitions have no temporal overlap.

## Final Test Policy

Month 7 is the sealed final test period.

It must not be used for feature selection, preprocessing decisions, hyperparameter tuning, model selection, imbalance-strategy selection, calibration, threshold selection, or business-cost optimization.

Final test evaluation will occur only after the preprocessing pipeline, model configuration, calibration method, threshold policy, and evaluation procedure have been frozen.

## Reproducibility

The split definition is stored in:

`configs/split_config.json`

Reusable implementation is stored in:

`src/data/splitting.py`

Automated tests are stored in:

`tests/test_data_splitting.py`

The split is deterministic and therefore does not require a random seed.

## Next Milestone

Milestone 03 will implement leakage-safe preprocessing and the PyTorch data pipeline.

All learned preprocessing operations must be fit on training data only and then applied unchanged to validation and test data.