# Model Risk Management Policy for Audit Analytics

## 1. Scope
This policy governs every machine learning or statistical model used to support audit judgements,
including transaction risk scoring, anomaly detection, vendor segmentation and spend forecasting.

## 2. Model inventory
Every model is recorded in the model inventory with its owner, purpose, training data, version and
validation date. Models are versioned in the model registry and only a registered model version may be
used on a client engagement.

## 3. Independent validation
Before first use, each model is validated by a person who did not build it. Validation covers data
quality, conceptual soundness, performance on a hold-out set, stability over time and an assessment of
whether the model treats customer segments differently. Models are re-validated at least annually, or
sooner if monitoring detects material drift.

## 4. Performance standards
A transaction risk model must achieve a ROC AUC of at least 0.90 and recall of at least 70% at the
operating threshold on the validation data before it can be used. Precision and false-positive rate at
the threshold must also be reported.

## 5. Explainability
Each alert produced by a model must be accompanied by an explanation of the main factors behind the
score, for example SHAP feature contributions. A model whose individual decisions cannot be explained
may not be used to select items for testing.

## 6. Human review
Models support, but never replace, auditor judgement. No transaction is concluded to be fraudulent on
the basis of a model score alone. Generative AI output, including narrative explanations and answers to
policy questions, must be reviewed by an auditor and must cite its sources.

## 7. Monitoring and drift
Production models are monitored monthly for data drift and prediction drift. If the share of drifted
features exceeds 30%, or the alert rate changes by more than 50% compared with the reference period,
the model owner must investigate within 10 business days and decide whether to retrain.

## 8. Retirement
A model that fails validation or is no longer used is marked as retired in the inventory; its
documentation is retained for seven years.
