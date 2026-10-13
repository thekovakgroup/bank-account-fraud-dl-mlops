# Milestone 04 — Interview Preparation

## Interview-ready explanation

Practice explaining the milestone in about 60–90 seconds:

> “In Milestone 04, I established baseline models and a reusable evaluation framework before building the deep-learning model. I reused the chronological split and leakage-safe preprocessing pipeline from earlier milestones, so months 0–5 were used for training, month 6 for validation, and month 7 remained sealed. I compared a prior-only Dummy classifier, unweighted Logistic Regression, and unweighted HistGradientBoosting using PR-AUC as the primary metric because fraud represents only about 1.34% of the validation data. HistGradientBoosting performed best with a PR-AUC of about 0.167, compared with 0.155 for Logistic Regression and 0.013 for the Dummy baseline. I also evaluated ROC-AUC, log loss, precision, recall, F1, and confusion-matrix counts. The 0.50 threshold produced very low recall, which demonstrated that model ranking quality and decision-threshold policy are separate problems. I then moved the evaluation logic into a tested reusable Python module so the later PyTorch model can be evaluated with exactly the same framework.”

## Questions you should be ready to answer

1. Why did you use PR-AUC as the primary metric instead of accuracy?
2. Why did you include a Dummy classifier?
3. Why did Logistic Regression have strong ROC-AUC but extremely low recall at a 0.50 threshold?
4. Why did you compare Logistic Regression with HistGradientBoosting before building the neural network?
5. What is the difference between model ranking quality and classification-threshold policy?

## Baseline Modeling Detailed Answers

### 1. Why did you use PR-AUC as the primary metric instead of accuracy?

I used PR-AUC, or Average Precision, as the primary model-comparison metric because this is a highly imbalanced fraud-detection problem.

In the validation period, only approximately 1.34% of the observations are fraudulent.

A model that predicted nearly every observation as legitimate could therefore achieve very high accuracy while detecting almost no fraud.

For example, the prior-only Dummy classifier predicted no validation observations as fraud at a threshold of 0.50. That model would still classify the overwhelming majority of rows correctly because most observations are legitimate, but it detects zero fraud cases.

PR-AUC focuses on the relationship between precision and recall for the positive class, which is fraud in this project.

The no-skill PR-AUC baseline was approximately:

`0.0134`

The trained models achieved:

- Logistic Regression: approximately `0.1548`
- HistGradientBoosting: approximately `0.1670`

This showed that both trained models were substantially better at ranking fraud cases than a no-skill model.

I still tracked ROC-AUC, log loss, precision, recall, F1, and confusion-matrix counts, but PR-AUC remained the primary metric for baseline model comparison.

### 2. Why did you include a Dummy classifier?

I included a Dummy classifier to establish a minimum performance reference.

The Dummy model used:

`strategy="prior"`

This means it learned only the class distribution from the training data rather than learning relationships between features and fraud.

Its validation results were approximately:

- PR-AUC: `0.0134`
- ROC-AUC: `0.5000`
- True positives at threshold 0.50: `0`
- False negatives: `1,450`

Because every validation observation receives essentially the same fraud probability, the model has no meaningful ranking ability.

This baseline is useful because a model score is much easier to interpret when compared against a no-skill reference.

For example, HistGradientBoosting achieved approximately 12.46 times the Dummy PR-AUC, while Logistic Regression achieved approximately 11.55 times the Dummy PR-AUC.

Without a Dummy baseline, I could report that a model achieved a PR-AUC of 0.167, but I would have less context for explaining whether that result represents meaningful predictive improvement.

### 3. Why did Logistic Regression have strong ROC-AUC but extremely low recall at a 0.50 threshold?

Logistic Regression achieved a ROC-AUC of approximately `0.882`, which indicates that it had meaningful ability to rank fraud cases above legitimate cases.

However, at the default classification threshold of `0.50`, its recall was only approximately `0.0041`, or 0.41%.

At that threshold it detected only:

- 6 true fraud cases
- while missing 1,444 fraud cases

This happens because ranking performance and classification decisions are different concepts.

ROC-AUC and PR-AUC evaluate how well predicted probabilities rank observations across many possible thresholds.

Recall at 0.50 evaluates what happens after choosing one specific cutoff.

Because fraud is rare, many useful fraud-risk scores may still be well below 0.50 even when fraud observations tend to receive higher scores than legitimate observations.

Therefore, the model can have strong ranking performance while still being far too conservative at the default threshold.

I intentionally did not optimize the threshold in the baseline milestone because threshold selection should be treated as a separate business and model-policy decision.

### 4. Why did you compare Logistic Regression with HistGradientBoosting before building the neural network?

I wanted the neural network to compete against meaningful simpler alternatives rather than being evaluated in isolation.

Logistic Regression provides a relatively simple and interpretable linear baseline.

HistGradientBoosting provides a nonlinear baseline that can learn interactions and more complex relationships between features.

Their validation PR-AUC results were:

- Logistic Regression: approximately `0.1548`
- HistGradientBoosting: approximately `0.1670`

HistGradientBoosting performed better, but the gap was relatively modest.

This is important because greater model complexity should be justified by measurable performance improvement.

When I build the PyTorch multilayer perceptron, I now have a clear benchmark.

The neural network should not be considered successful simply because it trains correctly. It should demonstrate meaningful value relative to approximately:

- PR-AUC: `0.167`
- ROC-AUC: `0.888`
- Log loss: `0.0544`

This makes the deep-learning comparison much more defensible.

### 5. What is the difference between model ranking quality and classification-threshold policy?

Model ranking quality measures how well the model orders observations from lower risk to higher risk.

Metrics such as:

- PR-AUC
- ROC-AUC

evaluate ranking ability across many possible thresholds.

Classification-threshold policy determines the probability cutoff used to turn a continuous model score into a binary decision.

For example:

`predicted probability >= threshold → fraud`

At a threshold of 0.50, HistGradientBoosting detected only about 6.97% of the validation fraud cases even though its PR-AUC and ROC-AUC showed meaningful ranking ability.

Lowering the threshold would generally increase recall because more observations would be classified as fraud.

However, it would also normally increase false positives and potentially reduce precision.

Therefore, selecting a threshold requires considering operational questions such as:

- How costly is a missed fraud case?
- How costly is a false alert?
- How many cases can investigators review?
- What level of precision is acceptable?
- Is the model well calibrated?

I therefore separated model evaluation from threshold optimization.

Milestone 04 evaluates baseline model quality using a common reference threshold and threshold-independent ranking metrics.

A later milestone will explicitly optimize threshold policy using validation data and business-cost considerations while keeping the final test set sealed.