# Milestone 08 — Responsible AI Audit Report

**Project:** Bank Account Fraud Detection — Deep Learning & MLOps  
**Milestone branch:** `milestone-08-responsible-ai`  
**Status:** Development audit completed; **NOT production approved**  
**Evaluation window:** Month 6 (exploratory development evaluation)  
**Final month-7 holdout:** **SEALED — NOT EVALUATED**  
**Related milestone:** Milestone 07 remains **OPEN** pending the planned one-time final holdout assessment.

## 1. Purpose and scope

Evaluate a frozen PyTorch multilayer perceptron (MLP) fraud-detection model for subgroup performance differences, model explainability, proxy-variable risks, privacy/security considerations, and human oversight. This milestone is an **audit of a frozen candidate**; it does not involve model retraining, recalibration, threshold adjustments, or production approval.

The fitted preprocessing pipeline, model weights, sigmoid calibrator, and provisional decision policy were reused. Saved artifacts were checked for integrity before evaluation.

## 2. Evaluation protocol and frozen policy

| Item | Verified setting |
|---|---|
| Model training window | Months 0–4 |
| Calibration window | Month 5 |
| Audit evaluation window | Month 6 only |
| Applications evaluated | 108,168 |
| Confirmed fraud cases | 1,450 |
| Frozen risk threshold | Approximately 4.9296% calibrated probability |
| Maximum investigation capacity | 5% of evaluation applications |
| Applications selected for human investigation | 5,408 |
| Confirmed fraud cases among selected applications | 705 |
| Month-7 final holdout | Sealed; not touched |
| Policy status | `FROZEN_PROVISIONAL_CANDIDATE`; not production approved |

Saved prediction checksums and label alignment passed. The observed **705 / 1,450 ≈ 48.62% fraud recall** is specific to the frozen investigation decisions on development month 6, and **not an estimate of real-world production performance**. The 5% capacity constraint is applied across the full cohort, not separately within each subgroup.

## 3. Subgroup performance and disparities

The audit used seven available cohort dimensions: **age band, dataset-coded income band, employment status, housing status, device operating system, application source, and payment type**. It computed subgroup confusion matrices, fraud prevalence, selection rate, true-positive rate (TPR/recall), false-positive rate (FPR), and precision.

**Evidence screening rule:** At least 500 applications, 20 confirmed fraud cases, and 20 non-fraud cases per subgroup. Out of **36 reported groups**, **26 met** this minimum screening threshold; **10 were flagged `LOW_SUPPORT`**. These thresholds support basic descriptive screening, not formal statistical inference.

### Observed descriptive disparity gaps

Differences below are **percentage-point (pp)** gaps between maximum and minimum rates among supported groups within each dimension.

| Dimension | Supported groups | Excluded low-support groups | Selection-rate gap (pp) | FPR gap (pp) | TPR/recall gap (pp) |
|---|---:|---:|---:|---:|---:|
| Housing status | 5 | 2 | 19.219 | **17.186** | **59.775** |
| Device OS | 4 | 1 | 12.549 | **11.247** | 33.018 |
| Age band | 5 | 1 | 11.478 | **9.404** | 38.025 |
| Payment type | 4 | 1 | 8.455 | 7.374 | 44.140 |
| Employment status | 3 | 4 | 8.397 | 7.225 | 31.599 |
| Income band | 4 | 0 | 5.217 | 4.571 | 27.365 |
| Application source | 1 | 1 | Not comparable | Not comparable | Not comparable |

For example, legitimate applications in housing category `BA` had an observed **17.614% FPR**, versus **0.428%** in category `BE`; this corresponds to an approximately **17.186 pp** difference. Their observed fraud prevalences were also different (**4.281%** vs. **0.477%**), so these outcomes **cannot be interpreted as proof of biased treatment or causation**.

Similarly, Windows and Linux groups had observed FPRs of approximately **12.668%** and **1.420%**, respectively. For age, the 60+ and 18–24 groups had observed FPRs of approximately **10.858%** and **1.454%**; their observed recall was approximately **70.05%** and **32.03%**. The under-18 age group had only nine fraud cases and was excluded from supported comparisons.

**Interpretation:** These are notable differences in investigation burden and fraud-detection recall that warrant deeper validation. The dataset lacks race and gender variables, so protected-class parity **cannot be established**. Coded categories should not be assigned real-world meanings that the dataset does not document. The current results do not control for feature correlations or prevalence differences, supply confidence intervals, or establish legal fairness findings.

## 4. SHAP explainability

The notebook used **SHAP DeepExplainer** on the frozen MLP with **32 background** and **64 explained** month-6 applications (seed 42). The existing preprocessing pipeline produced **55 transformed features**. A minimal `SHAPLogitAdapter` reshaped the model's output for SHAP compatibility; **it did not alter model parameters**. The expected SHAP array shape **`(64, 55)`** was verified.

The explanations target **raw MLP logits**, not calibrated fraud probabilities or final capped investigation decisions. Global importance was summarized using **mean absolute SHAP values**, in logit units.

| Top transformed feature | Mean absolute SHAP (logit units) |
|---|---:|
| `passthrough__phone_home_valid` | 0.577817 |
| `passthrough__has_other_cards` | 0.487425 |
| `passthrough__prev_address_months_count__is_min...` (display truncated in notebook output) | 0.469955 |
| `categorical__device_os_linux` | 0.381717 |
| `passthrough__keep_alive_session` | 0.357601 |

Housing status, income, and customer age also appeared among the 20 leading transformed features. The overlap between important input-feature families and dimensions showing subgroup disparities **motivates further investigation**, but does not demonstrate that those features caused a particular disparity. SHAP magnitudes are **not percentages**, and averaging absolute values discards contribution direction. Results based on 64 sampled applications are exploratory.

## 5. Proxy-variable and fairness limitations

Potentially sensitive or correlated feature families include **customer age, dataset-coded income, housing and employment status, device operating system, and geographic-related signals**. The audit treats these as **proxy-risk candidates**, not confirmed proxies for any particular protected characteristic. Further assessment should consider data definitions, input-quality issues, legally appropriate protected-group assessments, stability across time, uncertainty in small subgroups, and investigation outcomes.

Do not interpret the encoded `income` feature as a dollar amount. The data does not provide race or gender attributes to support those protected-class comparisons. No automatic adverse action should follow from this exploratory audit.

## 6. Security, privacy, and operational controls

**Verified or maintained in the development workflow:**

- Excluded identified leakage-prone features from the model inputs.
- Reused frozen inference artifacts without fitting or training.
- Verified saved model/prediction artifact checksums and label alignment.
- Kept month 7 sealed during this development audit.
- Saved local aggregate audit outputs under an ignored private-artifact directory.

**Required before production approval:**

- Review trust and provenance of serialized preprocessing/calibration artifacts; SHA-256 detects unexpected modification relative to a trusted digest but **does not independently establish source trust**. Untrusted `joblib` files must not be loaded.
- Implement privacy-appropriate access controls, audit logging, retention limits, and protection of applicant-level records and identifiers.
- Apply authenticated inference access, request validation, rate controls, secrets management, monitoring, and incident-response procedures.
- Establish documented fairness review, escalation, ongoing cohort monitoring, and independent validation.

Controls described as future work have **not** been represented as implemented.

## 7. Human oversight and decision policy

This model provides **investigation recommendations**, not automatic application denials. The draft operating policy requires trained human review of flagged cases, documented evidence and rationale, analyst override ability, escalation for uncertain/high-impact cases, review of potentially disparate burdens, and a process for correction or appeal where applicable. Any threshold or policy change requires separate approval and evaluation; the frozen policy remains unchanged for the planned final test.

**Policy classification:** Draft; **not approved for production**.

## 8. Reproducibility and evidence inventory

**Tracked candidate deliverable:** `notebooks/08_responsible_ai_audit.ipynb` (31 cells after documented learner edits, including 11 code cells; cleared execution outputs for GitHub).

**Private, Git-ignored outputs:** `artifacts/milestone_08_responsible_ai/`

| Local artifact | Verified contents |
|---|---|
| `m08ra_subgroup_metrics_v01.csv` | 36 subgroup rows |
| `m08ra_disparity_screen_v01.csv` | 7 dimensions |
| `m08ra_global_shap_v01.csv` | 55 transformed features |
| `m08ra_audit_manifest_v01.json` | Audit metadata / unchanged model and policy / sealed holdout |
| `08_responsible_ai_audit_executed_backup_v01.ipynb` | Executed local notebook backup before outputs were cleared |

Notebook outputs were cleared after a private executed backup was created; **all 31 notebook cells were preserved**. All listed CSV row counts, manifest checks, and Git exclusion checks passed during the instructor-led walkthrough.

## 9. Audit conclusions and next actions

**Development-audit finding:** The model and capped investigation policy reproduced the expected month-6 decisions, but screening revealed substantial differences in subgroup FPR and TPR, particularly by **housing status, device OS, and age**. SHAP identified influential device, housing, and other input features. Together, the results justify follow-up investigation into data provenance, correlated input distributions, potential proxy effects, and the consequences of additional reviews for legitimate applicants.

**Limitations:** Single exploratory evaluation month; only 64 SHAP-explained applications; no confidence intervals; support-based exclusions; coded categorical features; no race/gender parity evidence; observational, not causal; not production validated.

**To finish Milestone 08:** Preserve this report and the source notebook in Git; inspect staged files for sensitive content; commit on `milestone-08-responsible-ai`; push and merge through a reviewed pull request. Private applicant-level data and locally saved artifacts must not be committed.

**Important cross-milestone dependency:** **Milestone 07 remains OPEN. The month-7 final holdout stays SEALED** until the agreed frozen model and policy undergo the separately planned **one-time** evaluation. Neither Milestone 08 results nor this report authorize retuning on the final holdout.
