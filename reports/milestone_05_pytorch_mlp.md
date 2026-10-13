# Milestone 05 — PyTorch MLP Fraud Detection

**Project:** Bank Account Fraud Detection — Deep Learning & MLOps
**Repository:** `thekovakgroup/bank-account-fraud-dl-mlops`
**Development branch:** `milestone-05-pytorch-mlp`
**Experiment:** Fixed-budget, five-epoch PyTorch multilayer perceptron (MLP)
**Report date:** October 13, 2026
**Status:** Model training and artifact integrity verified; full uninterrupted end-to-end notebook execution after the final Cell 21 correction remains unverified.

## 1. Executive summary

Milestone 05 implements and evaluates an initial deep-learning classifier for bank account application fraud. The network uses 55 transformed input features, two hidden layers (64 and 32 units), ReLU activation, dropout, and a single output logit. Training used PyTorch, `BCEWithLogitsLoss`, and Adam for a fixed five epochs, with **3,885 total optimizer updates**.

The final, prespecified **epoch-5 validation PR-AUC was 0.157469** (full-precision reference value: `0.15746859566687155`). The MLP surpassed the Logistic Regression baseline's PR-AUC of 0.154819 but did **not** beat the HistGradientBoosting baseline's PR-AUC of 0.166961. The gradient-boosted model therefore remains the strongest reported validation baseline at this milestone. No threshold optimization or test-set-based model selection was performed.

## 2. Data and evaluation protocol

| Property | Value |
|---|---|
| Source data | `data/raw/Base.csv` |
| Observations / raw columns | 1,000,000 / 32 |
| Target | `fraud_bool` (binary fraud indicator) |
| Training window | `month` 0–5: **794,989** records; **8,151** fraud cases |
| Validation window | `month` 6: **108,168** records; **1,450** fraud cases |
| Held-out test window | `month` 7: **96,843** records; **1,428** fraud cases — **SEALED** |
| Modeling features | 29 pre-transform modeling features plus five missing/sentinel indicators in the prepared table; 55 float32 transformed features |
| Leakage control | Temporal split fixed before training; preprocessing fit only on training data and applied to validation |
| Primary model-selection metric | Validation PR-AUC (average precision) |
| Secondary reported metrics | Validation ROC-AUC and validation BCE loss |

Fraud is uncommon, making accuracy and a default probability threshold insufficient to describe detection quality. PR-AUC is the primary metric because it summarizes precision–recall behavior across thresholds in an imbalanced classification setting. The month-7 test set remains held out for a later, explicitly controlled final evaluation; the split's descriptive counts are not model-test results.

## 3. Network architecture and training configuration

**Network:** `55 → Linear(64) → ReLU → Dropout(0.10) → Linear(32) → ReLU → Dropout(0.10) → Linear(1)`; the forward pass returns one raw **logit** per observation, not a thresholded class prediction. The architecture has **5,697 trainable parameters**.

| Setting | Value |
|---|---|
| Framework | PyTorch 2.14.1+cpu |
| Device | CPU |
| Random seed | 42 |
| Batch size | 1,024 |
| Loss | Unweighted `BCEWithLogitsLoss` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 5, fixed in advance |
| Training DataLoader batches/epoch | 777 |
| Total optimizer steps | 777 × 5 = **3,885** |
| Class weighting / oversampling | Not used in this initial benchmark |
| Checkpoint choice | Fixed epoch 5; **not** best-validation-epoch selection |

`BCEWithLogitsLoss` combines a numerically stable sigmoid-based binary cross-entropy calculation with logit inputs. The model's output is converted to a probability using sigmoid for metric evaluation and inference. Dropout is active during training and inactive in evaluation mode.

## 4. Five-epoch results

| Epoch | Train BCE loss | Validation BCE loss | Validation PR-AUC | Validation ROC-AUC |
|---:|---:|---:|---:|---:|
| 1 | 0.066739 | 0.056041 | 0.141584 | 0.874876 |
| 2 | 0.045089 | 0.054853 | 0.153476 | 0.881012 |
| 3 | 0.044481 | 0.054676 | 0.155832 | 0.883217 |
| 4 | 0.044278 | 0.055066 | **0.159173** | **0.884149** |
| 5 | **0.044040** | 0.055352 | **0.157469** | 0.882800 |

The best *observed* validation PR-AUC was 0.159173 at epoch 4. This is a diagnostic observation, **not** a retrospectively chosen final checkpoint. The declared five-epoch experiment reports the final epoch-5 checkpoint and metrics. Training loss decreased throughout; validation loss reached its minimum at epoch 3 and rose slightly afterward. This pattern suggests possible mild overfitting or training fluctuation, which merits study in a future experiment rather than changing this baseline post hoc.

At the initial threshold of 0.5, the first-epoch MLP predicted no positive validation cases; threshold 0.5 is not necessarily appropriate for an imbalanced fraud use case. Threshold choice should be evaluated separately using validation data and explicit operational costs.

## 5. Comparison against Milestone 04 baselines

| Model | Validation PR-AUC | Validation ROC-AUC | Notes |
|---|---:|---:|---|
| DummyClassifier (prior) | 0.013405 | 0.500000 | No-skill reference for PR-AUC |
| Logistic Regression | 0.154819 | 0.881547 | Simpler linear baseline; validation log loss 0.054844 |
| **PyTorch MLP, epoch 5** | **0.157469** | **0.882800** | First deep-learning benchmark; validation BCE loss 0.055352 |
| **HistGradientBoosting** | **0.166961** | **0.887977** | Best PR-AUC among models reported here; validation log loss 0.054381 |

The MLP improved PR-AUC by approximately **0.002650** over Logistic Regression and trailed HistGradientBoosting by approximately **0.009492**. These are absolute PR-AUC-point differences, not percentage-point changes in fraud-detection accuracy. The models' probability losses were computed in their respective evaluation paths; no calibration conclusion should be drawn from the rounded figures alone.

**Decision:** Keep HistGradientBoosting as the leading validation benchmark; retain the PyTorch MLP as a reproducible deep-learning starting point for Milestone 06. Do not claim the MLP is the best-performing model.

## 6. Reusable code and tests

| Component | Purpose |
|---|---|
| `notebooks/05_pytorch_mlp.ipynb` | Guided, executable training and validation workflow |
| `src/models/mlp.py` | Reusable `FraudMLP` architecture |
| `src/training/loops.py` | Reusable `train_one_epoch` and `validate_one_epoch` functions |
| `tests/test_mlp.py` | Architecture shape, parameter count, backpropagation, and synthetic checkpoint round-trip tests |
| `tests/test_training_loops.py` | Parameter-update checks, sample-weighted validation loss, and validation immutability checks |

**Automated tests:** `python -m pytest -q` → **26 passed in 46.67 seconds** on October 13, 2026.
**Notebook source audit:** All **24 code cells** compiled successfully; model checkpoint, preprocessor, history, and metadata saves each have path assignments before overwrite guards. A changed `REPRO_RUN_ID` is required for a new independent artifact-producing run because existing run destinations are intentionally overwrite-protected.

## 7. Artifact integrity and reproducibility

### Original verified reference and reproduced model artifacts

| Artifact | Reference | Reproduction | Evidence |
|---|---|---|---|
| Model weights | `artifacts/milestone_05_reference/milestone_05_mlp_epoch5.pt` | `models/milestone_05_repro_01/milestone_05_mlp_epoch5.pt` | Six state-dictionary tensors **exactly equal** under `torch.equal` |
| Preprocessor | `artifacts/milestone_05_reference/milestone_05_preprocessor.joblib` | `models/milestone_05_repro_01/milestone_05_preprocessor.joblib` | SHA-256 **byte-identical** (`5eac239c3fdf01af6a9c8bb36e9bc74bd32c422b720943a3eab80b89fae536f9`) |
| History CSV | `artifacts/milestone_05_reference/milestone_05_mlp_history.csv` | `artifacts/milestone_05_repro_01/milestone_05_mlp_history.csv` | SHA-256 **byte-identical** (`d986655a2540a7087ca4ac0d0d39fada2085e1483959a67a00fd8e4daa2ac16d`); reproduction file creation provenance not conclusively established |
| Metadata JSON | `artifacts/milestone_05_reference/milestone_05_mlp_metadata.json` | `artifacts/milestone_05_repro_01/milestone_05_mlp_metadata.json` | **Reconstructed** from verified reference metadata, with additional `reproduction_audit` provenance field; saved and reloaded successfully |

The reproduced checkpoint loaded correctly, along with the saved preprocessor. A 128-observation validation inference check reported matching preprocessing outputs, logits, and probabilities. All five reported epoch metrics matched the original experiment to six decimal places. The final four-artifact integrity audit passed.

### Execution and provenance limitation

The fresh execution did **not** yield a verified uninterrupted 24-code-cell Run All after the final notebook fix. Cell 21 initially lacked an explicit `HISTORY_PATH` assignment; that source issue was corrected and syntax-verified. A later kernel did not retain the training-state variables, so the missing records were resolved from available verified artifacts rather than rerunning training. The presence and identical hashes of the reproduction history do **not** independently establish how that file was created. The reproduction metadata explicitly documents its reconstruction.

Therefore, the supported claim is **exact final-model tensor reproducibility, identical preprocessor bytes, matching recorded metrics, and verified recovered records**—not a completely demonstrated clean end-to-end notebook run in its final edited state.

Model binaries, raw data, and generated experiment artifacts are intentionally excluded from Git tracking. They must be regenerated or supplied through a separately governed artifact store when reproducing on another machine.

## 8. Risks, limitations, and next steps

1. **Class imbalance:** A PR-AUC around 0.157 is meaningfully above the validation fraud prevalence (~0.0134) but leaves substantial room for improvement. Compare precision and recall at feasible alert volumes.
2. **Threshold selection:** A default 0.5 threshold is only a reference; selecting a decision threshold requires validation-only precision, recall, false positives, workload constraints, and business costs.
3. **Generalization and drift:** A single month-6 validation period cannot establish stability across all future months. The month-7 test set must stay sealed until the planned final evaluation.
4. **Training choices:** Fixed model width, dropout, and five epochs are baseline settings, not proven optima. Any future tuning must be tracked separately and must not use the test set.
5. **Calibration and fairness:** PR-AUC measures ranking, not probability calibration or subgroup disparities. Both require distinct analysis if relevant to deployment.
6. **Reproducibility scope:** Finish a clean full-notebook execution using a new run ID in a future controlled verification if end-to-end execution is required, rather than overwriting verified outputs.

**Planned Milestone 06 candidates:** validation-only early stopping, regularization/architecture exploration, class weighting or sampling comparisons, calibration and threshold diagnostics, and careful experiment tracking. Maintain a strong non-neural benchmark throughout.

## 9. Interview-ready questions and answers

**Q1. Why use PR-AUC instead of accuracy as the main metric?**
Fraud is rare. A model can achieve high accuracy by predicting the majority class; PR-AUC measures how precision and recall trade off across scores for the positive (fraud) class.

**Q2. Why split the data chronologically?**
A model deployed in the future should be evaluated on later observations. Chronological splitting better approximates that setting and reduces opportunities for temporal leakage compared with a random split.

**Q3. Why fit preprocessing only on the training period?**
Fitting encoders or imputation statistics using validation data leaks information from the evaluation period into training. The validation period should receive only transformations learned from training.

**Q4. Why use `BCEWithLogitsLoss` rather than sigmoid plus BCELoss?**
It combines the relevant operations in a numerically stable loss implementation while letting the model emit raw logits.

**Q5. Why call `model.eval()` during validation?**
Evaluation mode disables dropout's stochastic masking, ensuring the intended inference behavior. Use `torch.no_grad()` to avoid constructing gradients during validation.

**Q6. Why not report epoch 4 as the final model?**
The experiment was defined as five epochs with a fixed final checkpoint. Selecting epoch 4 after inspecting validation results would be a different checkpoint-selection rule and needs explicit documentation and a consistent selection procedure.

**Q7. What does exact checkpoint agreement prove?**
It proves each compared tensor in the saved model state dictionaries is identical. It does not, by itself, prove that all notebook cells executed without interruption or establish the provenance of separately saved records.

**Q8. Did deep learning outperform simpler models?**
The epoch-5 MLP slightly exceeded Logistic Regression PR-AUC (0.157469 vs 0.154819), but HistGradientBoosting performed better (0.166961). The appropriate conclusion is that the neural model is a validated benchmark, not the current winner.

**Q9. Why keep the test set sealed?**
Repeatedly checking test performance while choosing model settings turns the test set into a tuning set and biases estimates of generalization. Use validation for development, then evaluate the sealed test once under a predefined final protocol.

**Q10. How would you extend this experiment in an MLOps setting?**
Version the configuration and source code, run automated tests, track preprocessing and model artifacts with hashes, record validation metrics and run IDs, and document how to reload the model reliably for inference and monitoring.

## 10. Milestone completion checklist

- [x] Temporal split and test-seal maintained
- [x] Fixed five-epoch PyTorch MLP trained and evaluated on validation
- [x] Model architecture and training loops extracted into reusable modules
- [x] Saved checkpoint and preprocessor verified via reload/inference
- [x] Exact tensor comparison and SHA-256 artifact audit passed
- [x] Recovery provenance recorded for reconstructed metadata
- [x] Notebook source syntax and save-guard audit passed
- [x] Automated test suite: 26 passed
- [ ] Verify a clean uninterrupted end-to-end run of the final edited notebook, if required
- [ ] Commit and review Milestone 05 deliverables
- [ ] Push branch, open and merge PR, synchronize `main`

**Git scope note:** Fifteen substantive additional cells in `notebooks/04_baseline_models.ipynb` were preserved separately in Git stash (`Preserve Milestone 04 additional evaluation and report cells`) so they are not accidentally included in the Milestone 05 PR.
