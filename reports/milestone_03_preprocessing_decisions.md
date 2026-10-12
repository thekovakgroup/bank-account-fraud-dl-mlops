# Milestone 03 — Preprocessing Decisions

## Purpose

Milestone 03 establishes a leakage-safe, reproducible preprocessing pipeline for both classical baseline models and the later PyTorch neural network.

No neural-network model is trained in this milestone.

## Locked Temporal Split

The chronological split defined in Milestone 02 remains unchanged:

- Train: months 0–5
- Validation: month 6
- Test: month 7

All learned preprocessing parameters are fit on the training partition only.

Validation data may be transformed using the fitted training pipeline.

The month-7 test partition remains sealed for final evaluation in Milestone 07. A small compatibility sample was transformed only to verify schema compatibility; the full test feature matrix was not materialized or used for preprocessing decisions.

## Excluded Model Inputs

The following columns are excluded from the initial predictive feature matrix:

- `fraud_bool` — target variable
- `month` — temporal split, monitoring, and drift field
- `device_fraud_count` — constant feature in the audited dataset

This leaves 29 raw model input features.

## Numeric Features

Eighteen features are treated as numeric and standardized using `StandardScaler`.

Scaler parameters are learned from training data only.

Validation and future inference data use the already-fitted scaler.

## Categorical Features

Five string categorical features are one-hot encoded:

- `payment_type`
- `employment_status`
- `housing_status`
- `source`
- `device_os`

The encoder is fit on training data only.

`OneHotEncoder(handle_unknown="ignore")` is used so unseen production categories do not cause inference failures.

Training data contains 26 learned one-hot output columns across the five categorical variables.

## Binary Features

Six existing binary indicators are passed through without scaling or one-hot encoding:

- `email_is_free`
- `phone_home_valid`
- `phone_mobile_valid`
- `has_other_cards`
- `foreign_request`
- `keep_alive_session`

## Sentinel-Coded Features

Milestone 01 identified five numeric features where `-1` may represent a special unavailable or unknown state:

- `prev_address_months_count`
- `current_address_months_count`
- `bank_months_count`
- `session_length_in_minutes`
- `device_distinct_emails_8w`

The original numeric value is preserved.

An additional binary `__is_minus_one` indicator is created for each feature.

This produces five additional derived model inputs rather than replacing or imputing the original sentinel values.

No generic missing-value imputation is performed because the audited dataset contains no Pandas NaN values.

## Final Feature Matrix

The preprocessing pipeline contains:

- 18 scaled numeric features
- 26 one-hot categorical features
- 6 passthrough binary features
- 5 sentinel indicators

Final transformed width: **55 features**

Development matrices:

- Train: `794,989 × 55`
- Validation: `108,168 × 55`

All transformed model inputs use `float32`.

## PyTorch Data Pipeline

NumPy matrices are converted to PyTorch tensors with `torch.from_numpy()`.

This allows memory sharing where possible and avoids unnecessary copies.

The initial DataLoader configuration is:

- Batch size: 1024
- Training shuffle: enabled
- Validation shuffle: disabled
- Random seed: 42
- Workers: 0

The worker count is intentionally conservative for Windows and VS Code notebook compatibility.

## Reproducibility

Preprocessing configuration is stored in:

`configs/preprocessing_config.json`

Reusable preprocessing code is stored in:

`src/features/preprocessing.py`

Reusable PyTorch dataset utilities are stored in:

`src/data/dataset.py`

The environment now includes:

`torch==2.14.1+cpu`

The exact environment is captured in `requirements-lock.txt`.

## Leakage Controls

The pipeline enforces the following rules:

1. Split data before fitting preprocessing.
2. Fit scalers and encoders on training data only.
3. Never call `fit` or `fit_transform` on validation or test data.
4. Keep `fraud_bool` outside the feature matrix.
5. Keep `month` outside the initial predictive feature matrix.
6. Preserve identical transformed feature order across train, validation, and inference.
7. Keep the full test matrix sealed until Milestone 07.

## Open Semantic Review

`credit_risk_score` remains provisionally included.

Its continued use depends on confirming that the feature is available at prediction time and does not contain information generated after the fraud decision or outcome.

## Milestone 03 Verification

The project test suite currently contains 16 tests.

All 16 tests pass.

The tests cover:

- raw schema validation
- chronological splitting
- unexpected temporal values
- preprocessing configuration
- sentinel indicators
- missing required features
- unseen categorical values
- PyTorch dtype enforcement
- feature/target row alignment
- DataLoader construction