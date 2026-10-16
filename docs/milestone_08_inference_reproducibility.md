
# Milestone 08 — Inference Reproducibility

## Project
Bank Account Fraud Detection — Deep Learning & MLOps

## Status
Development inference pipeline verified.
Provisional model only. Not approved for production.

## Model Architecture
- Raw inputs: 29 application features
- Preprocessed inputs: 55 features
- Model: Weighted PyTorch MLP (55 → 64 → 32 → 1)
- Calibration: Sigmoid / Platt
- Economic probability threshold: 4.9296%
- Maximum investigation capacity: 5%

## Data Protocol
- Training: Months 0–4
- Calibration: Month 5
- Exploratory evaluation: Month 6
- Final holdout: Month 7 — SEALED

Month 6 was previously used during model development.
Its performance must not be reported as independent holdout performance.

## Reusable Inference Module
File: `src/fraud_inference.py`

Methods:
- `predict_proba()` — calibrated fraud probabilities
- `recommend_investigations()` — investigation selections and priorities

The capacity limit applies to the complete input cohort.

## Required Private Artifacts
The following files are required locally:

- `artifacts/milestone_07/restart_bundle_v01/models.joblib`
- `artifacts/milestone_07/restart_bundle_v01/manifest.json`
- `artifacts/milestone_07/m07_weighted_mlp_m0to4_v01.pt`
- `artifacts/milestone_07/m07_weighted_mlp_m0to4_v01.json`
- `artifacts/milestone_07/calibration_policy_v01/mlp_calibrators.joblib`
- `artifacts/milestone_07/calibration_policy_v01/manifest.json`

These artifacts are Git-ignored and must be supplied
through a trusted, secure artifact distribution process.

A fresh GitHub clone alone cannot reproduce the model
predictions without these saved fitted artifacts.

## Verification Tests

From the project root, using the project's Python
virtual environment:

```powershell
python tests/test_m08_development_replay.py
python tests/test_m08_inference_contract.py
```

### Development Replay — PASS
- Month-6 applications: 108,168
- Maximum prediction difference: 0.0000000000
- Investigations: 5,408
- Fraud detected: 705
- No model retraining or refitting

### Synthetic Contract Tests — PASS
- 200 synthetic applications
- Unknown categories supported
- Missing required features rejected
- Excluded features do not affect predictions
- Inference batch-size consistency verified
- Investigation threshold and capacity enforced
- Empty probability input supported

The synthetic test selected zero investigations.
It verifies capacity enforcement but does not, by itself,
exercise positive investigation selections.

## Financial Assumptions
The frozen candidate policy uses hypothetical assumptions:
- Investigation cost: $25
- Additional false-positive cost: $10
- Average fraud loss per case: $1,000
- Loss prevention effectiveness: 70%

These are illustrative, not measured banking costs.

## Remaining Limitations
- Final month-7 holdout evaluation not performed
- No independent final performance estimate yet
- Financial assumptions require business validation
- Application-ID-based tie-breaking not implemented
- Capacity is applied per complete decision cohort
- Fresh-clone reproducibility requires private artifacts
- Production deployment and monitoring not validated

## Conclusion
Milestone 08 demonstrates reproducible standalone
development inference using saved, checksum-verified
artifacts and a frozen provisional investigation policy.

Month 7 remains sealed.
