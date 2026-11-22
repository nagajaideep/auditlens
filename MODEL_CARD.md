# Model card: AuditLens transaction risk model

| | |
|---|---|
| **Model** | Binary classifier (best of logistic regression, Random Forest and XGBoost, with XGBoost tuned by Optuna), chosen by validation PR-AUC |
| **Registry name** | `auditlens-fraud-model` in MLflow (each training run registers a new version) |
| **Companion models** | Isolation Forest (anomaly score), K-Means (vendor segments), SARIMA/Prophet (spend forecast) |
| **Owner** | Project author |
| **Intended users** | Auditors and forensic analysts who choose which transactions to test |
| **Status** | Portfolio / research project. Not validated for use on real client engagements |

## Intended use
- Ranks payment transactions by risk so auditors can choose items for testing and look at the riskiest first.
- Explains each flag with SHAP feature contributions and an optional plain-English note.
- **Out of scope:** automatically concluding that fraud happened, any decision about a person, or
  credit and AML decisions. Every flag needs an auditor's review (see `docs/policies/model_risk_management.md` §6).

## Data
- **Training data:** PaySim (Lopez-Rojas et al., 2016), a *simulated* mobile-money dataset of about 6.3M
  transactions over 30 days (744 hourly steps), 0.13% fraud. CI and the demo use
  `auditlens.data.generate`, a synthetic generator with the PaySim schema (0.4% fraud).
- **Split:** chronological, 64% train / 16% validation / 20% test by `step`, so the model is always
  tested on transactions later than any it was trained on.
- **Validation:** Pandera schemas (`src/auditlens/data/schema.py`) check types, allowed categories,
  non-negative amounts and the account-id format. The cleaning steps are logged in `reports/data_quality_report.md`.

## Features
There are 21 features, all available when a transaction is scored: amount, balances, balance-reconciliation
errors, an account-emptied flag, receiver-shows-zero-balances, amount-to-balance ratio, hour and night
flag, merchant receiver, *causal* prior transaction counts for sender and receiver, and one-hot type.

## Handling class imbalance
Class weights (logistic regression and Random Forest) and `scale_pos_weight` (XGBoost, with the multiplier
tuned by Optuna). The decision threshold is chosen on the validation set to maximise F1, instead of
defaulting to 0.5. The main ranking metric is PR-AUC, because ROC-AUC flatters models on imbalanced data.

## Metrics
`reports/model_comparison.csv` and `reports/training_summary.json` are rewritten on every training run.

**Synthetic demo data** (500k rows, 25 Optuna trials). These numbers check that the pipeline runs end
to end. They are **not** results to quote:

| model | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Random Forest | 1.000 | 0.997 | 0.999 | 1.000 | 1.000 |
| XGBoost (Optuna) | 0.977 | 0.997 | 0.987 | 1.000 | 1.000 |
| XGBoost (default) | 0.990 | 0.995 | 0.992 | 1.000 | 1.000 |
| Logistic regression | 0.950 | 0.388 | 0.551 | 0.992 | 0.538 |

Isolation Forest (no labels): ROC-AUC 0.994, catching 70% of fraud at a 0.8% flag rate.
SARIMA daily spend forecast: 3.2% weighted MAPE over a 7-day holdout.
RAG offline baseline (TF-IDF retrieval + extractive answers): 92% on the 25-question set.

> Simulated fraud data, PaySim included, follows simple mechanics such as accounts being emptied and
> balances that don't reconcile, so tree models score close to perfectly on it. That reflects how the
> simulator works, not how hard real fraud is to detect. Re-run on the real PaySim CSV and report those
> numbers instead, and expect them to fall sharply on real ledgers.

## Explainability and error analysis
- Global SHAP importance: `reports/shap_importance.csv`; summary plot: `reports/figures/shap_summary.png`.
- Typical false positives compared with true positives (median feature values): `reports/error_analysis_outcomes.csv`.
- Slice metrics (recall, precision, false-positive rate, flag rate) by type, amount band, origin-balance band and
  time of day: `reports/slice_metrics.csv`. The false-positive-rate gap between best and worst slice is in `reports/fpr_disparity.csv`.

## Ethical considerations and risks
- **Treating segments differently:** PaySim has no protected attributes, so fairness is checked on
  operational segments instead. Real data may contain proxies for protected groups (region, account
  age, amount bands). Check slice false-positive rates before using the model, and after each retrain.
- **Automation bias:** a high score is not evidence of fraud. The UI and the API always return the reasons, and
  policy requires an auditor to review each flag.
- **Feedback loops:** if only flagged items are ever investigated, the labels will be biased. Keep a
  random control sample in the testing plan.
- **GenAI hallucination:** the RAG prompt tells the model to answer only from the retrieved passages, to
  cite them, and to decline otherwise. The eval set includes an out-of-scope question. Narratives may
  only use SHAP reasons.
- **Privacy:** don't send client-confidential data to an LLM provider that hasn't been approved. The
  `extractive` provider keeps everything local.

## Limitations
- PaySim is simulated and covers only 30 days, so forecasts are *daily* (weekly seasonality). For multi-year
  ledgers set `forecast.freq: MS` to forecast monthly.
- In PaySim, vendors (receiving accounts) look much alike, so cluster separation is weak (silhouette about 0.2).
- The thresholds were tuned on one time window. Monitor drift (PSI above 0.2 on a feature, or more than 30% of
  features drifted) and retrain when it triggers.

## Monitoring
The nightly batch job (`auditlens batch` / K8s CronJob) scores new data and writes an Evidently HTML report
plus a PSI/KS JSON summary. The Streamlit dashboard shows latency, error rate, flag rate and drift.
