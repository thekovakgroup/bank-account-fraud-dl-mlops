# Milestone 07 — Final Holdout Evaluation Report

**Project:** Bank Account Fraud Detection — Deep Learning & MLOps  
**Evaluation type:** One-time, independent chronological holdout  
**Evaluation date (UTC):** 2026-10-16 08:19:21  
**Project status:** Evaluation completed; final documentation and GitHub submission pending  
**Production approval:** **No** — frozen provisional candidate only

## 1. Executive summary

The frozen weighted PyTorch multilayer perceptron (MLP), together with its fitted sigmoid (Platt) calibrator and predefined investigation policy, was evaluated **once** on the previously sealed **month-7** holdout. The final evaluator reported **96,843 applications**, **1,428 confirmed fraud cases**, and **4,842 investigations**, identifying **731 fraud cases** (51.19% recall) at the frozen 5% capacity limit.

Final average precision (PR-AUC) was **0.191745** and ROC-AUC was **0.886352**. Precision among selected investigations was **15.10%**. Under the explicitly hypothetical financial assumptions, the model's investigation policy produced estimated net savings of **$349,540** relative to the stipulated no-review comparison. These are test-set estimates, **not** validated production outcomes.

The final holdout has been **CONSUMED ONCE** and must not be rerun or used to revise this model, its calibration, or its investigation policy.

## 2. Evaluation protocol and safeguards

| Component | Frozen protocol |
|---|---|
| Model | Weighted `FraudMLP`, architecture 55 → 64 → 32 → 1 |
| Training window | Months 0–4 |
| Calibration window | Month 5 |
| Exploratory comparison window | Month 6 |
| Independent final holdout | Month 7 (consumed once) |
| Raw and transformed model inputs | 29 raw features → 55 transformed features |
| Excluded from features | Target `fraud_bool`, temporal field `month`, leakage-risk field `device_fraud_count` |
| Calibration | Saved sigmoid/Platt calibrator |
| Investigation eligibility | Calibrated fraud probability **> 0.04929577464788732** |
| Investigation capacity | At most floor(0.05 × whole-cohort application count) |
| Review priority | Descending calibrated fraud probability; stable input-order tie-breaker (provisional) |
| Candidate status | `FROZEN_PROVISIONAL_CANDIDATE` |
| Production approved | `false` |

### Pre-evaluation checks

- Verified frozen checkpoint and calibration artifact SHA-256 hashes against the saved policy.
- Verified saved development inference results: maximum absolute probability difference **0.0000000000** on month 6, with **5,408** investigations and **705** fraud cases identified.
- Completed a development-only dry run on month 6 before final holdout access; confirmed matching results.
- Confirmed the final access marker and report did not exist before the authorized execution.
- Ran the final evaluator only after explicit authorization. The evaluator created its one-time access marker, reported no model/calibrator fitting and no policy tuning, and saved aggregate metrics.

**Artifact fingerprints:**

- Frozen MLP checkpoint SHA-256: `cea166c0aee86227de2bcf64dce937e81a3c4c6ba60d24577a51026dc1a4ae46`
- Frozen calibrator bundle SHA-256: `ce74364ccef575985bfc938d396f33cf02425a2fb406ad0c8a65cb5c5316be3a`

## 3. Independent final holdout metrics

| Metric | Month 7 (final holdout) |
|---|---:|
| Applications | 96,843 |
| Confirmed fraud cases | 1,428 |
| Fraud prevalence | 1.4746% |
| Average precision / PR-AUC | 0.191745 |
| ROC-AUC | 0.886352 |
| Brier score (lower is better) | 0.013157 |
| Log loss (lower is better) | 0.058571 |
| ECE, 10 equal-width bins | 0.004935 |
| Investigation capacity | 4,842 |
| Investigations selected | 4,842 |
| Actual investigation fraction | 4.9998% |
| True positives (TP) | 731 |
| False positives (FP) | 4,111 |
| False negatives (FN) | 697 |
| True negatives (TN) | 91,304 |
| Recall / TPR at frozen policy | 51.1905% |
| Precision at frozen policy | 15.0971% |
| False-positive rate at frozen policy | 4.3085% |

The confusion-matrix entries reconcile to 96,843 applications: TP + FP + FN + TN = 731 + 4,111 + 697 + 91,304 = **96,843**. The actual fraud total is TP + FN = **1,428**, and investigations equal TP + FP = **4,842**.

### Interpretation

- **Ranking:** PR-AUC of 0.191745 is above final-month fraud prevalence of approximately 0.01475, consistent with useful risk ranking in this imbalanced classification task. This comparison is descriptive, not evidence of deployment readiness.
- **Detection and workload:** The frozen whole-cohort policy selected 4,842 applications, capturing 731 fraud cases and missing 697. There were 4,111 reviews of applications that were not confirmed as fraud.
- **Calibration:** Brier score, log loss and binned ECE describe aspects of probability quality. ECE depends on the binning definition and does not by itself establish adequate calibration across all risk bands or customer groups.
- **Policy scope:** The 5% constraint applies to the entire evaluated cohort, **not** separately to every subgroup or microbatch.

## 4. Exploratory month 6 versus independent month 7

| Metric | Month 6 — exploratory | Month 7 — independent |
|---|---:|---:|
| Applications | 108,168 | 96,843 |
| Fraud cases | 1,450 | 1,428 |
| Investigations | 5,408 | 4,842 |
| Fraud cases caught | 705 | 731 |
| Recall | 48.62% | 51.19% |
| Precision among investigations | 13.04% | 15.10% |
| Average precision / PR-AUC | 0.163322 | 0.191745 |
| Brier score | 0.012077 | 0.013157 |
| Hypothetical net savings | $311,270 | $349,540 |

**Comparison limits:** The months differ in sample size and fraud prevalence. Month 6 had been used in candidate comparison and was **not** an independent final test. Month 7 showed better ranking, recall, and investigation precision under the unchanged policy but a higher (worse) Brier score. These descriptive changes do not establish statistical significance, causal improvements, or future stability.

## 5. Hypothetical business-cost evaluation

These fixed inputs are assumptions for the capstone exercise, **not verified bank economics**:

| Financial assumption | Frozen value |
|---|---:|
| Cost per investigation | $25 |
| Additional cost per false-positive investigation | $10 |
| Fraud loss per case | $1,000 |
| Fraud prevention effectiveness for investigated fraud | 70% |

The evaluator reported:

| Estimated financial component | Month-7 value |
|---|---:|
| Prevented fraud loss: 731 × 0.70 × $1,000 | $511,700 |
| Investigation cost: 4,842 × $25 | −$121,050 |
| Additional false-positive cost: 4,111 × $10 | −$41,110 |
| **Net hypothetical savings versus no review** | **$349,540** |

These savings are conditional on the assumptions above; they exclude potential additional operational, customer, legal, appeals, and intervention costs not represented in the simulation. Detected fraud cases do not necessarily translate into prevented losses in a real financial institution.

## 6. Responsible AI and deployment limitations

The previously completed Milestone 08 audit found descriptive subgroup disparities on development month 6, including variation in false-positive rates across housing status, device OS, and age groups. This independent final report does **not** retune or reselect the model based on those observations or on month-7 labels.

Important remaining limitations:

1. **Not production approved.** The model and decision policy remain a frozen provisional candidate.
2. **Human oversight.** An investigation recommendation should not by itself trigger an automatic adverse application decision.
3. **False-positive burden.** The final month includes 4,111 flagged applications without confirmed fraud; subgroup impact requires continued attention and appropriate review.
4. **Incomplete protected-class evaluation.** Race and gender are unavailable in the dataset, so protected-class parity cannot be established from this evaluation.
5. **Provisional tie-breaker.** Stable input-order tie-breaking should be replaced by an immutable application identifier before production; that change would require a separately versioned policy and validation, not retroactive alteration of this final test.
6. **Financial assumptions.** Costs, loss size, and fraud-prevention effectiveness are hypothetical.
7. **Operational and security readiness.** Inference API controls, authentication, auditing, drift monitoring, artifact security, and production approval remain separate engineering/governance tasks.
8. **Single final test.** Month-7 results can be reported and scrutinized, but must not be used to choose a better threshold, revise calibration, or train a replacement candidate while still being described as untouched holdout evidence.

## 7. Reproducibility and artifact locations

**Source code** (intended to be version controlled):

- `src/m07_final_holdout.py` — separate dry-run and explicitly authorized final evaluation modes.
- `src/fraud_inference.py` — fitted artifact loading, calibrated scoring, and frozen whole-cohort investigation ranking.
- `configs/milestone_07_candidate_policy_v01.json` — frozen provisional candidate and financial assumptions.
- `configs/milestone_07_split_config.json` — chronological split and original holdout safeguards.

**Private local artifacts** (remain Git-ignored):

- `artifacts/milestone_07/final_holdout_dry_run_v01/aggregate_metrics_v01.json` — development month-6 dry-run results.
- `artifacts/milestone_07/final_holdout_once_v01/aggregate_metrics_v01.json` — final month-7 aggregate metrics.
- `artifacts/milestone_07/final_holdout_once_v01/` — one-time access marker and any other private final-evaluation artifacts created by the evaluator.

The final JSON records `final_holdout_consumed_once: true`, `no_model_or_calibrator_fit: true`, `no_policy_tuning: true`, and `production_approved: false`.

**Reproducibility boundary:** The evaluation method is documented, but the irreversible final evaluation must **not** be executed again on month 7. Future validation requires genuinely new, separately reserved data under a prospectively frozen protocol.

## 8. Final milestone conclusion

**Milestone 07's independent final holdout evaluation is complete.** The frozen calibrated MLP demonstrated useful ranking and selected 731 confirmed fraud cases from 4,842 investigations on 96,843 unseen month-7 applications. Observed probability calibration and false-positive burden remain material considerations, and the calculated $349,540 net savings are hypothetical.

**Decision:** Preserve the model and policy as evaluated; do not retune on month 7. Proceed with final documentation review, repository commit/PR/merge, and subsequent inference/API/deployment engineering under the original capstone roadmap. **No production deployment approval is implied.**
