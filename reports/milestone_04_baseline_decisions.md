# Milestone 04 — Baseline Modeling Decisions

## Purpose

Milestone 04 establishes defensible reference models and a reusable evaluation framework before development of the PyTorch neural network.

The objective is not to assume that deep learning will outperform simpler models. Instead, later models must demonstrate measurable value relative to these baselines.

## Data Policy

The chronological split defined in Milestone 02 remains unchanged:

- Train: months 0–5
- Validation: month 6
- Test: month 7

The month-7 test partition remains sealed until Milestone 07.

No test predictions, test metrics, threshold tuning, or model-selection decisions were performed in this milestone.

## Preprocessing

Milestone 04 reuses the preprocessing pipeline established in Milestone 03.

The pipeline is fit on training data only and transforms:

- 18 scaled numeric features
- 5 categorical features expanded through one-hot encoding
- 6 passthrough binary features
- 5 derived sentinel indicators

The resulting model matrix contains 55 `float32` features.

Development matrix sizes:

- Train: 794,989 × 55
- Validation: 108,168 × 55

## Primary Metric

The primary model-selection metric is PR-AUC / Average Precision.

This metric is prioritized because fraud represents only approximately 1.34% of the validation observations.

Accuracy is not used as the primary metric because a classifier predicting almost every observation as legitimate could achieve very high accuracy while detecting little or no fraud.

## Additional Metrics

The evaluation framework also records:

- ROC-AUC
- Log loss
- Precision
- Recall
- F1 score
- True negatives
- False positives
- False negatives
- True positives
- Actual positive rate
- Predicted positive rate

PR-AUC and ROC-AUC evaluate ranking quality.

Precision, recall, F1, and confusion-matrix counts depend on the selected probability threshold.

Log loss evaluates probability quality and penalizes confident incorrect predictions.

## Baseline Models

Three baseline levels were evaluated.

### DummyClassifier

Configuration:

`strategy="prior"`

Purpose:

Establish a no-skill reference that learns only the training class prior.

Validation results:

- PR-AUC: 0.013405
- ROC-AUC: 0.500000
- Log loss: 0.071565
- True positives at threshold 0.50: 0
- False negatives: 1,450

The Dummy classifier has no useful ranking ability.

### Logistic Regression

Configuration:

- Unweighted
- `solver="lbfgs"`
- `max_iter=1000`

Purpose:

Provide an interpretable linear baseline without applying class weighting or resampling.

Validation results:

- PR-AUC: 0.154819
- ROC-AUC: 0.881547
- Log loss: 0.054844
- PR-AUC lift versus Dummy: approximately 11.55×
- True positives at threshold 0.50: 6
- False negatives: 1,444

The model converged successfully in 26 optimization iterations.

Its strong ranking metrics combined with very low recall at threshold 0.50 demonstrate that ranking quality and classification threshold policy are separate concerns.

### HistGradientBoosting

Configuration:

- Unweighted
- Learning rate: 0.10
- Maximum iterations: 100
- Maximum leaf nodes: 31
- Early stopping disabled
- Random state: 42

Purpose:

Provide a nonlinear baseline capable of learning feature interactions while remaining computationally practical for approximately 795,000 training observations.

Validation results:

- PR-AUC: 0.166961
- ROC-AUC: 0.887977
- Log loss: 0.054381
- PR-AUC lift versus Dummy: approximately 12.46×
- True positives at threshold 0.50: 101
- False negatives: 1,349

HistGradientBoosting produced the strongest initial ranking performance, but its advantage over Logistic Regression is modest.

## Initial Baseline Ranking

Based primarily on PR-AUC:

1. HistGradientBoosting — 0.166961
2. Logistic Regression — 0.154819
3. DummyClassifier — 0.013405

HistGradientBoosting also produced the strongest ROC-AUC and lowest log loss.

However, no model is declared the final production model at this stage.

## Threshold 0.50 Findings

The default threshold of 0.50 is used only as a consistent reference point.

At this threshold:

- HistGradientBoosting detects approximately 6.97% of validation fraud.
- Logistic Regression detects approximately 0.41%.
- DummyClassifier detects 0%.

These low detection rates do not contradict the substantially stronger PR-AUC and ROC-AUC values of the real models.

They demonstrate that a model can rank fraud meaningfully while still requiring a much different operating threshold for useful fraud detection.

Threshold optimization is intentionally deferred.

## Imbalance Strategy

The original class distribution was preserved for the initial baseline models.

No:

- class weighting
- oversampling
- undersampling
- SMOTE
- threshold optimization

was applied.

These techniques will be evaluated as explicit experiments rather than mixed into the baseline definition.

## Evaluation API

Reusable evaluation logic is implemented in:

`src/evaluation/metrics.py`

The API calculates both threshold-independent and threshold-dependent metrics from predicted probabilities.

Unit tests validate:

- expected metric calculations
- input length validation
- probability range validation
- binary-target validation
- single-class rejection

The reusable implementation was compared directly against the notebook implementation.

Ten evaluation metrics matched exactly.

## Key Interpretation

The data contains meaningful predictive fraud signal.

Both Logistic Regression and HistGradientBoosting substantially outperform the no-skill baseline.

The nonlinear model currently has the strongest performance, but the relatively small gap between it and Logistic Regression means model complexity must continue to be justified by measurable improvement.

The future PyTorch neural network should therefore be compared against approximately:

- PR-AUC: 0.167
- ROC-AUC: 0.888
- Log loss: 0.0544

rather than simply being judged in isolation.

## Next Step

Milestone 05 will develop the initial PyTorch multilayer perceptron.

The neural network will be evaluated using the same validation partition and reusable evaluation framework established here.