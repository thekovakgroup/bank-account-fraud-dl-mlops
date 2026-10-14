# Milestone 06 — Model Improvement & Hyperparameter Tuning
## Interview Questions and Model Answers

**Project:** Bank Account Fraud Detection — Deep Learning & MLOps\
**Dataset:** Bank Account Fraud Dataset Suite, `Base.csv`\
**Status:** Milestone 06 experiments and MLflow tracking verified; model, calibration, and operating threshold not yet frozen.
**Evaluation boundary:** Training months 0–5; validation month 6; final test month 7 **SEALED**.

Use these answers as interview practice. They describe **observed validation results**, not production or final holdout performance.

## A. Problem Framing, Data, and Leakage

### 1. What business problem are you solving?
**Answer:** I built a fraud-risk ranking system for bank-account applications. The supervised target is `fraud_bool`. Because fraud is relatively rare, the goal is to prioritize suspicious applications for investigation while controlling how many alerts the review team can handle. I treat predictions as decision support, not grounds for automatic adverse action.

### 2. How did you divide the dataset, and why?
**Answer:** I trained on months 0–5 (794,989 records), validated on month 6 (108,168 records), and reserved month 7 as a sealed final test set. A chronological split better reflects a fraud system making predictions on later applications; a random split can overstate performance when patterns shift over time.

### 3. How imbalanced was the validation dataset?
**Answer:** Month 6 contained 1,450 fraud cases among 108,168 applications, approximately 1.34%. This means accuracy alone would be misleading: a classifier could predict nearly everything as non-fraud and still achieve high accuracy while missing the cases the business needs to detect.

### 4. How did you prevent data leakage in preprocessing?
**Answer:** The preprocessing pipeline was fitted on training months only and then used to transform validation data. The same training-derived feature definitions and transformations were used across comparisons. I kept the month-7 final test set inaccessible during feature engineering, tuning, and validation-based selection.

### 5. What features were used by the neural network?
**Answer:** The trained preprocessing pipeline produced 55 numeric `float32` model inputs, combining transformed numerical variables, categorical one-hot encoded variables, and selected passthrough/indicator features. Model experiments used the same feature representation to keep comparisons meaningful.

## B. Neural Networks and Experiment Design

### 6. Describe the PyTorch MLP architecture.
**Answer:** The network has 55 input features, two hidden layers with 64 and 32 units, ReLU activations, dropout rate 0.10, and one raw-logit output. It has 5,697 trainable parameters. The architecture is compact enough for a reproducible baseline while still modeling nonlinear interactions.

### 7. Why does the model output a logit instead of a probability?
**Answer:** The network returns an unrestricted raw score, which can be passed directly to `BCEWithLogitsLoss`. That loss combines sigmoid and binary cross-entropy internally in a numerically stable way. Sigmoid is applied separately when a probability-like score is needed for ranking, evaluation, or later calibration.

### 8. What training configuration did you use for the GPU MLP experiments?
**Answer:** The controlled experiments used five epochs, batch size 1,024, Adam optimizer, learning rate 0.001, and random seed 42. The architecture and data split were held constant while I changed the positive-class weighting strategy. This isolates the effect of weighting more clearly than changing many settings simultaneously.

### 9. Why did you test different class-weighting strategies?
**Answer:** Fraud examples are scarce, so the standard loss can be dominated by the non-fraud class. Positive weighting raises the penalty for missing a fraud example. I compared an unweighted loss, a moderate positive weight of 25, and full inverse-prevalence weighting, approximately 96.53, to measure the ranking and operational tradeoffs.

### 10. What are the limitations of inverse-prevalence weighting?
**Answer:** A large positive weight can help the model focus on fraud examples, but it can also shift raw score distributions, increase false alerts at an unchanged threshold, and harm probability calibration. Weighting is not automatically better: I evaluated both ranking metrics and fraud capture at fixed review budgets rather than assuming a weight formula was optimal.

### 11. What did your weighting experiments show?
**Answer:** On validation month 6, unweighted GPU MLP achieved PR-AUC 0.160393; weight 25 achieved 0.161925; and full positive weighting achieved 0.163862. All three improved on the Milestone 05 CPU MLP PR-AUC of 0.157469. These are validation results, and the changes cannot be attributed solely to weighting when comparing across distinct training environments or historical runs.

### 12. Why did you reproduce the classical HistGradientBoosting baseline?
**Answer:** A deep-learning model should beat or meaningfully complement a strong simpler baseline before being preferred. Reproducing the gradient-boosting model with documented hyperparameters and verifying its artifacts gave me a credible benchmark. It achieved validation PR-AUC 0.166961, ahead of the MLPs on our primary ranking metric.

## C. Evaluation, Operations, and Statistical Uncertainty

### 13. Why did you choose PR-AUC as the primary ranking metric?
**Answer:** Fraud detection is an imbalanced binary-classification problem. Precision and recall directly characterize the quality and coverage of fraud alerts across score thresholds. PR-AUC emphasizes performance on the positive fraud class; ROC-AUC remains useful as a secondary measure, but it can look strong even when false alerts are costly in a low-prevalence setting.

### 14. What was the final Milestone 06 PR-AUC ranking?
**Answer:** HistGradientBoosting led at 0.166961, followed by the full-weighted GPU MLP at 0.163862, weight-25 MLP at 0.161925, unweighted GPU MLP at 0.160393, and the Milestone 05 CPU MLP at 0.157469. The classical baseline is the PR-AUC leader; the full-weighted MLP is the stronger deep-learning candidate.

### 15. What is a fixed alert-budget or top-*k* evaluation?
**Answer:** Instead of applying a default probability threshold, I sort applications by risk score and select the highest-scoring fraction the investigation team can realistically review. A 1% or 5% alert budget compares models at the same operational capacity, making the number of fraud cases detected a concrete business-facing measure.

### 16. How did the full-weighted MLP compare with gradient boosting at a 5% alert budget?
**Answer:** At approximately 5,408 validation alerts, the full-weighted MLP identified 706 fraud cases versus 680 for HistGradientBoosting—a difference of 26 additional fraud cases in that particular validation month. This does not mean the MLP is universally superior or that the improvement will recur in production.

### 17. Did the MLP also outperform gradient boosting at a 1% alert budget?
**Answer:** The full-weighted MLP detected 277 fraud cases compared with 275 for HistGradientBoosting—an observed difference of only two cases. That small difference is not a robust reason to choose a model by itself; uncertainty analysis helps show whether such an advantage is distinguishable from sampling variation.

### 18. Why can one model lead on PR-AUC while another detects more fraud at a specific budget?
**Answer:** PR-AUC summarizes ranking quality across many recall/precision operating points. Top-*k* recall focuses on one narrow point determined by review capacity. The ranking curves can cross, so the gradient-boosting model can lead overall while the weighted MLP places more fraud cases in the highest-scored 5% of applications.

### 19. How did you estimate uncertainty in the operational comparison?
**Answer:** I used a paired bootstrap with 2,000 resamples of the same month-6 validation rows. Both models were evaluated on identical resampled indices in each repetition, preserving the paired nature of the comparison. I recomputed fraud capture at each fixed alert budget and examined the distribution of MLP-minus-gradient-boosting differences.

### 20. What did the bootstrap confidence intervals tell you?
**Answer:** At a 1% alert budget, the observed MLP advantage was +2 fraud cases with a 95% percentile interval of roughly −22 to +22; the interval includes zero. At a 5% budget, the observed advantage was +26 with an interval of roughly +1 to +48. The latter is promising on this validation month but remains exploratory, especially because the same validation set informed multiple comparisons.

### 21. What are the main limitations of the bootstrap findings?
**Answer:** The bootstrap resamples individual validation rows and may not capture time dependence, correlated applications, shifting fraud tactics, or the repeated use of this one validation month for decisions. A percentile interval is not a guarantee of future performance or a substitute for an untouched temporal holdout. I therefore report both effect sizes and limitations.

### 22. Why not use a default threshold of 0.5 for the weighted MLP?
**Answer:** Class weighting changes the loss objective and often shifts the raw score distribution. In our experiments, the heavily weighted MLP showed substantially worse Brier score than the unweighted MLP (about 0.128 versus 0.012), warning against treating its raw sigmoid outputs as calibrated probabilities. Milestone 07 will examine calibration and select a threshold based on review capacity and business costs.

### 23. What is the difference between ranking, calibration, and threshold selection?
**Answer:** **Ranking** asks whether higher-risk cases tend to receive higher scores; PR-AUC measures one aspect of that. **Calibration** asks whether predicted probabilities match observed frequencies. **Threshold selection** chooses which scores trigger action under investigation capacity, false-positive costs, and fraud-loss assumptions. Good performance in one area does not guarantee the others.

### 24. How will you choose the operational model in Milestone 07?
**Answer:** I will compare the PR-AUC-leading HistGradientBoosting model with the weighted MLP using explicit review budgets and a documented business-cost framework. I will assess calibration, estimate false-positive and missed-fraud tradeoffs, and freeze the model, calibration approach, and threshold policy before opening the sealed month-7 test set. No production model or final threshold has been selected yet.

## D. MLflow, Provenance, Reproducibility, and Responsible AI

### 25. How did you use MLflow in Milestone 06, and what does “historical import” mean?
**Answer:** I initialized MLflow 3.17.1 with a local SQLite tracking database and retrospectively imported five completed experiments: the Milestone 05 CPU MLP, three GPU MLP variants, and reproduced HistGradientBoosting. Each MLflow run stores parameters, validation metrics, and provenance tags and is explicitly marked `historical_import=true`. MLflow did **not** execute the original training; it records results from verified prior experiments.

### 26. How did you prevent duplicate MLflow imports and protect sensitive artifacts?
**Answer:** The reusable importer checks each run's historical identifier before writing, validates source records and artifact checksums, and uses preview mode by default. The SQLite database and artifacts live beneath the Git-ignored `artifacts/` directory. The importer does not upload raw transactions, model checkpoints, or validation prediction arrays into MLflow.

### 27. What makes your experiments reproducible?
**Answer:** I recorded the chronological split, feature count, model architecture, training configuration, seed, software versions, source provenance, and SHA-256 checksums for local artifacts. I pinned direct dependencies in `requirements.txt` and all 121 installed packages in `requirements-lock.txt`. The historical GPU metrics retain their Google Colab provenance rather than being presented as local CPU retraining results.

### 28. Is MLflow experiment tracking the same as an MLflow Model Registry?
**Answer:** No. Experiment tracking compares parameters, metrics, and run metadata. A Model Registry manages named model versions and lifecycle information. Milestone 06 completed local experiment tracking; Model Registry integration and deployment-related work are planned for later milestones. No production model has been registered.

### 29. Why is the month-7 test set still sealed?
**Answer:** Once a holdout influences feature design, model selection, calibration, or threshold tuning, it stops being an unbiased final evaluation. Month 7 will be used only after those decisions are frozen. This prevents an overly optimistic estimate caused by repeated testing against the final holdout.

### 30. What would you improve before deploying this fraud model?
**Answer:** I would complete probability calibration and business-cost thresholding; evaluate month 7 once under a frozen protocol; inspect performance across meaningful subgroups and operational slices; review error cases; define human-review escalation; and add serving, tests, logging, monitoring for drift, and retraining governance. The current Milestone 06 results support further selection work, not an autonomous production decision system.

---

## Short Interview Summary (60–90 Seconds)

> “My capstone predicts fraudulent bank-account applications using chronological training data from months zero through five and validation on month six, while keeping month seven sealed. I built a 55-feature PyTorch MLP and compared unweighted, moderately weighted, and inverse-prevalence-weighted training against a reproduced HistGradientBoosting baseline. Gradient boosting led on validation PR-AUC at 0.166961, while the full-weighted MLP reached 0.163862 and detected 26 additional fraud cases at a fixed 5% investigation budget in this validation month. I used a paired bootstrap to quantify uncertainty rather than treating that difference as guaranteed. I also imported five verified historical experiments into MLflow with provenance tags and pinned the environment for reproducibility. My next step is calibration and business-cost-based model and threshold selection before opening the final temporal test set.”

**Important distinction:** Milestone 06 experiment tracking is complete, but MLflow Model Registry, model deployment, final calibration, final threshold selection, and month-7 testing are **not** complete.
