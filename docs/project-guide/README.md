# AuditLens: the complete project guide

This is the full record of how AuditLens was built: every step in order, the decision made at each point and
why, what came out of it, and how to run, deploy and explain it. The short overview for recruiters is the
[root README](../../README.md).

**Contents**
1. [The problem](#1-the-problem)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [Build log: step by step](#3-build-log-step-by-step)
4. [Module deep dives: decisions and outcomes](#4-module-deep-dives-decisions-and-outcomes)
5. [Problems hit and how they were fixed](#5-problems-hit-and-how-they-were-fixed)
6. [Full results](#6-full-results)
7. [Running it locally](#7-running-it-locally)
8. [Deployment: what you need to do](#8-deployment-what-you-need-to-do)
9. [Explaining it in an interview](#9-explaining-it-in-an-interview)
10. [Repository map](#10-repository-map)
11. [Limitations and next steps](#11-limitations-and-next-steps)

---

## 1. The problem

An auditor receives a client's transaction ledger, often millions of rows, and has to decide which
transactions to test. Traditional sampling picks a few dozen at random or by rule. AuditLens instead scores
**every** transaction and gives the auditor:

| Auditor question | AuditLens answer | Module |
|---|---|---|
| Which transactions look like fraud? | A risk score per transaction, with the reasons (SHAP) | Classification + explainability |
| What looks unusual even if it doesn't match known fraud? | A label-free anomaly score | Isolation Forest |
| Which vendors behave oddly? | Vendor segments with fraud exposure per segment | K-Means |
| Is this account's spending in line with expectations? | A spend forecast against actuals | SARIMA / Prophet |
| What does our policy say about this? | An answer with citations to the policy documents | RAG with Gemini |
| Why was this one flagged, in plain English? | A 2–4 sentence note built from the SHAP reasons | LLM narrative |

A design principle throughout: **the system supports the auditor's judgement and never replaces it.** Every
flag comes with reasons, every answer comes with citations, and the model card says that no transaction
should be called fraud on a score alone.

---

## 2. Architecture at a glance

```
           ┌──────────── Data ────────────┐
Kaggle ──▶ │ PaySim CSV ──COPY──▶ PostgreSQL│
           └──────────────┬───────────────┘
                          ▼
   Pandera validation → cleaning → data-quality report
                          ▼
                 feature engineering (21 features)
        ┌─────────────┬───┴─────────┬──────────────┐
        ▼             ▼             ▼              ▼
  LogReg / RF /  Isolation     K-Means        SARIMA /
  XGBoost+Optuna  Forest       vendors        Prophet
        ▼
  SHAP + error/slice analysis
        └─────── all logged to MLflow; best model registered ───────┐
                                                                    ▼
Policy docs → chunks → embeddings → Chroma ─▶ RAG (Gemini) ─┐   model bundle
                                                            ▼       ▼
                                FastAPI  /predict  /forecast  /ask  /health
                                   │ JSON request log        ▲
         nightly batch job ────────┼── Evidently/PSI drift   │
                                   ▼                         │
                            Streamlit dashboard     Docker → Azure Container Apps / minikube
```

The detailed Mermaid diagram is in [`docs/architecture.md`](../architecture.md).

---

## 3. Build log: step by step

The plan followed a three-week structure. These are the steps in the order they were done, with the
command for each.

### Week 1: data, validation, classifiers, tracking

| Step | What was done | Command / file |
|---|---|---|
| 1 | Package skeleton (`src/auditlens/`), config with env-var overrides | `configs/config.yaml`, `config.py` |
| 2 | Synthetic generator with the PaySim schema, so everything can be built and tested before the real data arrives | `auditlens generate` |
| 3 | Downloaded real PaySim from Kaggle (6,362,620 rows, 470 MB) | `./scripts/download_paysim.sh` |
| 4 | Created the PostgreSQL schema and bulk-loaded the data with `COPY` in 65 s | `sql/schema.sql`, `auditlens load-db` |
| 5 | Wrote and ran 7 analysis queries in SQL | `sql/eda_queries.sql` → `reports/sql_eda_results.txt` |
| 6 | Pandera data contracts for raw and cleaned data | `data/schema.py` |
| 7 | Cleaning with a log of every change (duplicates, nulls, negatives, derived time columns) | `data/clean.py` |
| 8 | Profiling, EDA figures and the Markdown data-quality report | `data/quality_report.py` → `reports/data_quality_report.md` |
| 9 | Feature engineering (21 features) | `features/build.py` |
| 10 | Logistic regression baseline, then Random Forest, then XGBoost, then XGBoost tuned with Optuna | `models/classify.py` |
| 11 | MLflow tracking of every run and trial, plus the model registry | `tracking.py`, `mlflow.db` |

### Week 2: unsupervised, forecasting, explainability, RAG

| Step | What was done | Command / file |
|---|---|---|
| 12 | Isolation Forest anomaly detection | `models/anomaly.py` |
| 13 | Vendor behaviour profiles and K-Means segmentation with English cluster names | `models/cluster.py` |
| 14 | Spend forecasting with a backtest against a naive baseline | `models/forecast.py` |
| 15 | SHAP global importance and per-transaction reasons | `explain/shap_explain.py` |
| 16 | Error analysis (what false positives and false negatives look like) and slice metrics | `explain/error_analysis.py` |
| 17 | Wrote 6 audit policy documents as the question-answering corpus | `docs/policies/` |
| 18 | Chunking, embeddings, vector store, retrieval, Gemini answers with citations | `genai/` |
| 19 | 25-question test set and evaluation harness | `eval/rag_questions.jsonl`, `auditlens eval-rag` |
| 20 | LLM narrative for each flagged transaction | `genai/narrate.py` |

### Week 3: production

| Step | What was done | Command / file |
|---|---|---|
| 21 | Single training pipeline that runs every module and saves one model bundle | `pipelines/train.py`, `auditlens train` |
| 22 | FastAPI service with request validation | `api/main.py`, `auditlens serve` |
| 23 | Nightly batch scoring job | `pipelines/batch_score.py`, `auditlens batch` |
| 24 | Drift detection (PSI + KS, Evidently HTML) and structured JSON request logs | `monitoring/` |
| 25 | Streamlit monitoring dashboard | `dashboard/app.py` |
| 26 | pytest suite (31 tests) and ruff lint | `tests/` |
| 27 | GitHub Actions CI (tests, quality gate, Docker build), nightly job, Azure DevOps pipeline | `.github/workflows/`, `azure-pipelines.yml` |
| 28 | Dockerfile, docker-compose stack, Kubernetes manifests | `Dockerfile`, `docker-compose.yml`, `k8s/` |
| 29 | Azure Container Apps deploy script and a one-click deploy workflow | `deploy/azure/`, `.github/workflows/deploy-azure.yml` |
| 30 | Model card, architecture diagram, READMEs | `MODEL_CARD.md`, `docs/` |

---

## 4. Module deep dives: decisions and outcomes

### 4.1 Data preparation and SQL

**What PaySim is:** a simulated mobile-money ledger of 744 hourly `step`s (30 days) with columns for type,
amount, sender and receiver, and their balances before and after. `isFraud` is the label. `isFlaggedFraud` is
PaySim's own rule, which flags transfers over 200,000.

**Decisions**
- **PostgreSQL with `COPY`, not `pandas.to_sql`.** Row inserts would take tens of minutes for 6.3M rows; `COPY`
  takes 65 seconds. The schema in `sql/schema.sql` adds indexes on `step`, `type` and the receiver, plus a
  `transaction_scores` table the batch job writes to.
- **Pandera contracts** check types, allowed transaction types, non-negative amounts and account ID formats
  (`C…` for customers, `M…` for merchants). Failures are collected all at once (`lazy=True`) and reported,
  rather than stopping at the first.
- **Balance mismatches are a feature, not dirt.** Around 3.6M rows break the identity
  `oldbalance - amount = newbalance`, including most legitimate payments. Dropping them would throw away
  most of the data, so the mismatch becomes the `err_balance_orig` and `err_balance_dest` features.

**What the SQL showed** (`reports/sql_eda_results.txt`)
- Fraud occurs only in **TRANSFER** (0.77% fraud rate) and **CASH_OUT** (0.18%); never in PAYMENT, CASH_IN or DEBIT.
- Fraud by hour: about **22% of transactions at 3–4am are fraud**, against under 0.1% in daytime. Fraud volume
  is flat across the clock while legitimate volume follows the working day.
- PaySim's built-in rule catches **16 of 8,213 frauds** (0.2% recall). That's the bar to beat.
- Cleaning outcome: 0 duplicates, 0 nulls, 0 negatives in the real data. The cleaning steps exist for messier
  ledgers and are exercised by the tests on the synthetic data.

### 4.2 Feature engineering

21 features (`features/build.py`). The key rule: **use only information available when the transaction
happens.**

| Group | Features | Why |
|---|---|---|
| Amount | `amount`, `log_amount`, `amount_to_balance` | Fraud amounts are large relative to the balance |
| Balances | 4 raw balances | Context |
| Reconciliation | `err_balance_orig`, `err_balance_dest` | The strongest signals (see SHAP) |
| Behaviour flags | `orig_emptied`, `dest_zero_balances`, `dest_is_merchant` | Fraud typically empties the account |
| Time | `hour`, `is_night` | Matches the 3–4am pattern from EDA |
| History | `orig_tx_count`, `dest_tx_count` | Counted in time order, so each row sees only earlier transactions (no leakage) |
| Type | 5 one-hot columns | Fraud only in two types |

### 4.3 Classification

**Decisions**
- **Chronological split** (train 64%, validate 16%, test 20% by `step`), not random. A random split lets the
  model learn from transactions after the ones it is tested on, which flatters it. In production you always
  score the future.
- **Class imbalance** (0.13% fraud): class weights for logistic regression and Random Forest, and
  `scale_pos_weight` for XGBoost. Optuna also tunes a *multiplier* on the weight, because full re-weighting
  often over-flags.
- **Threshold tuned on validation, not left at 0.5.** It's chosen to maximise F1 on the validation set, then
  applied unchanged to the test set.
- **PR-AUC is the selection metric.** With 0.13% positives, ROC-AUC is near 1 for almost any decent model;
  PR-AUC separates them.
- **Optuna** searches 7 XGBoost hyperparameters plus the weight multiplier (TPE sampler, seeded).

**Outcome (test set: 1,272,524 transactions, 4,254 frauds)**

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| **Random Forest (selected)** | 1.000 | **0.997** | 0.999 | 1.000 |
| XGBoost, Optuna-tuned | 1.000 | 0.993 | 0.997 | 1.000 |
| XGBoost, default | 1.000 | 0.993 | 0.996 | 1.000 |
| Logistic regression | 0.958 | 0.808 | 0.877 | 0.919 |

The Random Forest caught 4,242 of 4,254 frauds with **zero false positives** at threshold 0.82. All three
tree models are effectively tied, and Random Forest won on validation PR-AUC.

**How to read it:** PaySim's fraud follows the simulator's simple rules (empty the account, then cash out), so
tree models nearly solve it. The honest claim is that the pipeline beats the rule baseline (0.2% recall) and
the linear baseline (80.8% recall). Real ledgers will be harder.

### 4.4 Anomaly detection (Isolation Forest)

**Why:** a classifier only finds fraud that looks like past fraud. Isolation Forest needs no labels and
catches *unusual* transactions, which is useful for new schemes.

**Decisions:** fitted on training rows only, on 9 features, with signed-log scaling of the error terms so that
extreme values don't dominate. Contamination is set to twice the training fraud rate.

**Outcome:** ROC-AUC **0.995** without seeing a single label. It flags 0.31% of transactions, 68% of the flags
are real fraud, and it catches 63% of all fraud.

### 4.5 Vendor segmentation (K-Means)

**What a vendor is here:** a receiving customer account (`nameDest` starting with `C`) with at least 3
incoming transactions. That gives **382,603 vendors**.

**Profile features:** volume, average and maximum amount, variability (coefficient of variation), share of
transfers, share of cash-outs, share at night, ratio of unique payers, and active days.

**Decisions:** standardised features; k chosen from 3–8 by silhouette score (on a 5,000-vendor sample for
speed); each cluster named after its two most distinctive features so an auditor can read it.

**Outcome:** k = 4, silhouette 0.26.

| Segment | Vendors | Fraud exposure |
|---|---|---|
| big-spikes / erratic-amounts | 49,886 | **3.4%** |
| always-active / high-volume | 111,845 | 1.4% |
| few-cash-outs / sporadic | 95,323 | 0.8% |
| cash-out-heavy / few-transfers | 125,549 | 0.6% |

The erratic-amount segment carries 2.5–6× the fraud exposure of the others. That's where an auditor would
start testing.

### 4.6 Spend forecasting

**Decisions**
- **Daily, not monthly.** PaySim covers only 30 days. The code takes `freq: MS` for multi-year ledgers.
- **Stable window, days 5–17.** In PaySim, legitimate volume collapses by about 95% after day 17 (from about
  70 billion a day to 2–9 billion), which is a simulator artefact. Forecasting across that break means nothing.
- **SARIMA on log spend**, with the order chosen on an inner validation split of the *training* data only, so
  the holdout is never seen. Weekly seasonality is added only when there are at least 2 full weeks.
- **Always compared with a seasonal-naive baseline.**

**Outcome (4-day holdout, MAPE)**

| Group | SARIMA | Naive |
|---|---|---|
| CASH_OUT | **4.0%** | 8.6% |
| CASH_IN | **5.1%** | 5.9% |
| DEBIT | **11.7%** | 21.9% |
| PAYMENT | 22.9% | 19.6% |
| TRANSFER | 160% | 38% |

The median is 11.7%, and SARIMA beats naive on 3 of 5 groups. TRANSFER fails because a single day (day 12)
has a 138-billion spike in the training window. This is reported as is rather than tuned away on the test set.

### 4.7 Explainability and error analysis

- **SHAP (TreeExplainer).** Top global features: `err_balance_orig` (0.18), `orig_emptied` (0.11),
  `amount_to_balance` (0.09), `newbalanceOrig` (0.06). The API returns the top 4 signed reasons for every
  transaction, each with a plain-English description.
- **False negatives.** The 12 missed frauds are large transfers (median 2.9M) that **do not empty** the
  sender's account, the opposite of the typical pattern.
- **Slices** (`reports/slice_metrics.csv`). Recall is above 97% in every slice except senders with a **zero
  opening balance (77%)**. The false-positive rate is 0 in all slices, so there's no disparity at this
  threshold. PaySim has no protected attributes, so fairness is checked on operational segments.

### 4.8 Question answering with citations (RAG) and narratives

**Corpus:** 6 audit policy documents for a sample firm (`docs/policies/`): transaction monitoring, fraud risk
assessment, materiality and sampling, vendor management, model risk management, and audit documentation.
They contain concrete, testable facts (thresholds, days, percentages). Public 10-K filings can be added with
`scripts/fetch_public_docs.py`.

**Pipeline**
1. **Chunking:** split on headings so each chunk keeps its section name for the citation; 900-character
   windows with 150 characters of overlap.
2. **Embeddings:** `all-MiniLM-L6-v2` (384 dimensions), with a TF-IDF + SVD fallback when the model can't be
   downloaded.
3. **Vector store:** Chroma, persistent, cosine similarity. A pgvector table is defined in `sql/pgvector.sql`
   for Postgres-only deployments.
4. **Retrieval:** top 4 chunks, numbered `[1]…[4]` in the prompt.
5. **Generation with Gemini** (`gemini-2.5-flash`, temperature 0.1). The prompt says: answer only from the
   passages, cite `[n]` after each claim, and reply exactly "I could not find this in the provided documents."
   when the answer isn't there.
6. **Citation parsing:** the `[n]` markers are mapped back to file and section and returned in the API response.

**Narratives:** for a flagged transaction, the SHAP reasons and the transaction facts go to Gemini with the
instruction to use only those facts and not to accuse anyone.

**Evaluation** (`eval/rag_questions.jsonl`, 25 questions, including one out-of-scope question). A question
passes when the right document is retrieved **and** cited **and** at least 50% of the expected key facts
appear in the answer. An out-of-scope question passes when the system declines.

**Outcome (offline mode, no Gemini key):** **92%** pass, 100% retrieval hit rate, 96% citation rate, and the
out-of-scope question was declined. To get the Gemini score, set `GEMINI_API_KEY` and run
`auditlens eval-rag`. CI fails the build if the offline score drops below 80%.

### 4.9 Experiment tracking (MLflow)

- Backend `sqlite:///mlflow.db`; the file store is deprecated in current MLflow.
- One parent run per training, with a child run per model, Optuna trials logged as steps, and every report
  file attached as an artifact.
- The best model is registered as **`auditlens-fraud-model`**, with a new version per training run (v2 is
  the current real-data model).
- To view: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### 4.10 Pipelines and API

- **One bundle file** (`artifacts/models/bundle.joblib`, 34 MB) holds the classifier, threshold, feature
  list, Isolation Forest, vendor segmenter, SHAP background sample and spend series. The API and the batch
  job load the same file, so they can't drift apart.
- **Endpoints**

| Endpoint | Input | Output |
|---|---|---|
| `POST /predict` | 1–1,000 transactions, `explain`, `narrate` | fraud score, flag, risk band, anomaly score, SHAP reasons, narrative |
| `POST /forecast` | group, horizon, method | forecast with an 80% interval |
| `POST /ask` | question | answer, citations, retrieved passages |
| `GET /health` | – | model version, threshold, uptime |
| `GET /metrics` | – | the stored evaluation metrics |

- **Validation:** Pydantic rejects unknown types, negative amounts and malformed IDs with HTTP 422.
- **Batch job:** cleans and scores new data, writes Parquet plus `latest_flagged.csv`, optionally writes to the
  `transaction_scores` table, and runs the drift check. It runs as a Kubernetes CronJob at 02:15.
  103,573 transactions take a few seconds.

### 4.11 Monitoring

- **Drift:** PSI and the Kolmogorov–Smirnov test for every feature plus the score, and an Evidently HTML
  report. The rule comes from the model risk policy: more than 30% of features drifted means investigate and
  retrain. The late-day PaySim batch showed 2 of 22 features drifted (9%) and prediction PSI 0.003, so no
  alarm.
- **Request log:** one JSON line per request (request ID, path, status, latency, number flagged).
- **Dashboard:** Streamlit with tabs for service (p50/p95 latency, error rate, flag rate), drift, model
  metrics and flagged transactions.

### 4.12 Testing and CI

- **31 pytest tests**: data contracts and cleaning, causal features, chronological split, each model, the
  forecast backtest, RAG retrieval and declining, the Gemini client (mocked), every API endpoint, the
  request log, drift and the batch job.
- **CI on every push**: ruff, the tests, the RAG quality gate and a Docker build. Both jobs pass.
- **Nightly workflow**: retrains and batch-scores, using real PaySim when the Kaggle secret is set.

---

## 5. Problems hit and how they were fixed

| Problem | Symptom | Fix |
|---|---|---|
| The first synthetic generator made fraud trivially separable | Every model scored 1.000 | Added legitimate look-alike transfers and frauds that leave part of the balance |
| A partial last day in the synthetic data | SARIMA MAPE of 179,000% | Clipped steps to the range and drop a partial trailing period |
| PaySim volume collapses after day 17 | Real-data forecast MAPE of 52–79% | Forecast within the stable window (days 5–17) |
| SARIMA(1,1,1) unstable on 13 points | TRANSFER forecast 3× too high | Order chosen on an inner validation split of the training data |
| Evidently doesn't support pandas 3 | `ImportError: infer_dtype_from_object` | Pinned `pandas<3` |
| MLflow file store deprecated | `MlflowException` at startup | Moved to a SQLite backend |
| MLflow's secure loader rejects sklearn trees | "untrusted types" on log_model | Explicitly trust `sklearn.tree._tree.Tree` (our own model) |
| `to_sql` too slow for 6.3M rows | Long load | PostgreSQL `COPY` (65 s) |
| Vendor profiling with `groupby.apply` | Slow on 380k groups | Vectorised boolean means |
| Generic words matching out-of-scope questions | The out-of-scope question got an answer | Stop-words for domain-generic terms and a higher relevance cut-off in offline mode |

---

## 6. Full results

All generated files are in `reports/`:

| File | Contents |
|---|---|
| `data_quality_report.md` + `figures/` | Profile, schema checks, fraud by type and hour, correlations |
| `sql_eda_results.txt` | Output of the 7 SQL queries |
| `model_comparison.csv` | The four classifiers on the test set |
| `training_summary.json` | Every headline number in one file |
| `shap_importance.csv`, `figures/shap_summary.png` | Explainability |
| `error_analysis_outcomes.csv`, `slice_metrics.csv`, `fpr_disparity.csv` | Error and fairness analysis |
| `vendor_clusters.csv` | Segments |
| `forecast_backtest.csv`, `spend_series.csv` | Forecasting |
| `rag_eval_summary.json` | RAG evaluation |

**Resume bullets with measured numbers**
- Built AuditLens, an end-to-end ML platform for spotting risky financial transactions. A Random Forest
  selected from tuned XGBoost and baseline models reached 1.00 ROC-AUC and 99.7% recall on 6.36M transactions
  stored in PostgreSQL, with SHAP explanations for each flag.
- Added Isolation Forest anomaly detection (0.995 ROC-AUC with no labels), K-Means segmentation of 382K vendors
  and SARIMA spend forecasting (4.0% MAPE on cash-outs), with every experiment tracked and versioned in MLflow.
- Built a cited policy Q&A assistant (embeddings, Chroma/pgvector, Gemini) scoring [A]% on a 25-question
  test set. Fill this in after running `auditlens eval-rag` with your Gemini key; the offline baseline is 92%.
- Served the models through FastAPI on Docker and Azure Container Apps, monitored drift with Evidently, and
  tested with 31 pytest tests in GitHub Actions CI.

---

## 7. Running it locally

```bash
git clone https://github.com/nagajaideep/auditlens && cd auditlens
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[genai,monitoring,dev]"
cp .env.example .env                                     # add GEMINI_API_KEY and KAGGLE_API_TOKEN

# data
./scripts/download_paysim.sh                             # real data (or: auditlens generate --rows 200000)
docker compose up -d db                                  # PostgreSQL 16 + pgvector
auditlens load-db
psql postgresql://auditlens:auditlens@localhost:5432/auditlens -f sql/eda_queries.sql

# train: all 6.3M rows takes about 30 min and 7 GB RAM on 4 cores
AUDITLENS_DATA__SOURCE=postgres AUDITLENS_DATA__SAMPLE_ROWS=null auditlens train --trials 15

auditlens eval-rag                                       # 25-question evaluation
auditlens serve                                          # http://localhost:8000/docs
streamlit run dashboard/app.py                           # http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlflow.db        # http://localhost:5000
pytest
```

For a quicker run, `AUDITLENS_DATA__SAMPLE_ROWS=1000000` trains on the first million rows.

---

## 8. Deployment: what you need to do

Already done: the code, the Dockerfile, the deploy script and the CI. The Docker image build is verified in
GitHub Actions. What remains needs **your accounts**.

### Step 0. Accounts and keys (one time)

| # | What | Where | Used for |
|---|---|---|---|
| 1 | **Revoke the Kaggle tokens shared in chat** and create a new one | kaggle.com → Settings → API → *Create New Token* | Data download |
| 2 | Gemini API key | https://aistudio.google.com/apikey (free tier is enough) | Q&A answers and narratives |
| 3 | Azure for Students | https://azure.microsoft.com/free/students (USD 100 credit, no card) | Cloud hosting |
| 4 | Azure CLI | https://learn.microsoft.com/cli/azure/install-azure-cli | Running the deploy script |

### Option A: one-click deploy from GitHub (recommended)

Nothing runs on your laptop: GitHub downloads the data, trains, builds the image in Azure and deploys.

1. Log in and find your subscription ID:
   ```bash
   az login
   az account show --query id -o tsv
   ```
2. Create the resource group and a deploy identity for GitHub. Use your subscription ID:
   ```bash
   az group create -n auditlens-rg -l centralindia
   az ad sp create-for-rbac --name auditlens-github --role contributor \
     --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/auditlens-rg --json-auth
   ```
   Copy the whole JSON it prints. (Some university tenants block this command. If so, use Option B.)
3. In GitHub, open **Settings → Secrets and variables → Actions**:
   - **Secrets:** `AZURE_CREDENTIALS` (the JSON), `GEMINI_API_KEY`, `KAGGLE_API_TOKEN`
   - **Variables:** `AZURE_ACR`, a globally unique lowercase name such as `auditlensjaideep`
4. Go to **Actions → Deploy to Azure → Run workflow**. It takes about 45 minutes with all the data, and the
   log ends with your URL, `https://auditlens-api.<random>.centralindia.azurecontainerapps.io/docs`.
5. Optional: add `KAGGLE_API_TOKEN` so the nightly workflow uses real data too.

### Option B: deploy from your laptop

```bash
az login
./scripts/download_paysim.sh
AUDITLENS_DATA__SAMPLE_ROWS=null auditlens train --trials 15      # creates artifacts/models/bundle.joblib
ACR=auditlensjaideep GEMINI_API_KEY=<your key> ./deploy/azure/containerapps.sh
```
The image is built in Azure (`az acr build`), so Docker isn't needed. Re-running the script updates the app.

### After deploying: verify

```bash
URL=https://<your-app>.azurecontainerapps.io
curl $URL/health
curl -X POST $URL/ask -H 'content-type: application/json' -d '{"question":"How quickly must alerts be triaged?"}'
```
Open `$URL/docs` to try every endpoint in the browser. This is the page to show in the demo video.

### Cost and teardown
- Container Apps scales to zero when idle (`--min-replicas 0`), so expect a few dollars a month. ACR Basic is
  about USD 5 a month.
- The first request after an idle period takes 30–60 s while the container starts and loads the model.
- Delete everything when you're done: `az group delete -n auditlens-rg --yes --no-wait`.

### Kubernetes demo (minikube, for the video)

```bash
minikube start --cpus 4 --memory 6g
docker build -t auditlens:latest .              # after training, so the bundle is inside
minikube image load auditlens:latest
kubectl apply -f k8s/namespace.yaml && kubectl apply -f k8s/
kubectl -n auditlens create secret generic auditlens-secrets --from-literal=GEMINI_API_KEY=<key>
kubectl -n auditlens get pods,svc,hpa,cronjob
minikube service auditlens-api -n auditlens --url
```

### Demo video checklist (5–7 minutes)
1. The problem in one sentence, then the architecture diagram.
2. The data-quality report and the 3–4am fraud chart.
3. The model comparison table and the MLflow UI with the registered model.
4. `/predict` in Swagger on a fraud-like transfer: show the score, the SHAP reasons and the narrative.
5. `/ask` on a policy question: show the citations; then an out-of-scope question being declined.
6. The Streamlit dashboard: latency and the drift tab.
7. A green GitHub Actions run and the Azure URL.

---

## 9. Explaining it in an interview

**The 30-second version:** "AuditLens scores every transaction in a ledger so auditors can test the riskiest
ones first. It combines a supervised fraud model with SHAP explanations, label-free anomaly detection, vendor
segmentation and spend forecasting, plus a policy assistant that answers with citations. It's served through
FastAPI on Azure, with drift monitoring, MLflow tracking and CI."

**Likely questions**

- *Your recall is 99.7%. Isn't that suspicious?*
  Yes, and I say so in the model card. PaySim's fraud follows the simulator's rules (empty the account, then
  cash out), so balance-reconciliation features nearly give it away; SHAP confirms that. What I'd claim is the
  method: chronological split, validation-tuned threshold, PR-AUC selection, and beating both the rule
  baseline (0.2% recall) and the linear baseline (80.8%). On a real ledger I'd expect much lower numbers.
- *Why a chronological split?* A random split leaks the future into training. Production always scores
  tomorrow's transactions with yesterday's model.
- *How did you handle the imbalance?* Class weights and `scale_pos_weight`, with the weight multiplier tuned,
  plus a threshold tuned on validation instead of 0.5, and PR-AUC as the metric.
- *Why Isolation Forest if the classifier is so good?* The classifier only knows past fraud. The anomaly score
  catches new patterns; at 0.995 ROC-AUC without labels it's a useful second line.
- *How do you stop the LLM making things up?* Retrieval-only prompt, required `[n]` citations mapped back to
  sources, an exact decline phrase, an out-of-scope test question, and a CI gate on accuracy. Narratives can
  only use the SHAP reasons.
- *What would you change for a real client?* Real labels from case outcomes, a random control sample to avoid
  feedback loops, protected-attribute checks if such data exists, retraining driven by drift alerts, and a
  stricter threshold policy agreed with the engagement team.
- *Why is the TRANSFER forecast bad?* One extreme day in a 13-day window. I reported it rather than tuning on
  the test set. More history or a robust or quantile model would fix it.

---

## 10. Repository map

```
auditlens/
├── README.md                    overview for recruiters
├── MODEL_CARD.md                data, assumptions, metrics, risks, limitations
├── docs/
│   ├── project-guide/README.md  this guide
│   ├── architecture.md          Mermaid architecture diagram
│   └── policies/                RAG corpus (6 policy documents)
├── src/auditlens/
│   ├── config.py                YAML config + AUDITLENS_* env overrides
│   ├── cli.py                   `auditlens` command (generate, load-db, train, batch, index, ask, eval-rag, serve)
│   ├── data/                    generate, load (CSV/Postgres COPY), schema (Pandera), clean, quality_report
│   ├── features/build.py        21 causal features + slice segments
│   ├── models/                  classify, anomaly, cluster, forecast
│   ├── explain/                 shap_explain, error_analysis
│   ├── genai/                   ingest, embed, store, llm (Gemini), rag, narrate, eval
│   ├── pipelines/               train, scoring, batch_score, artifacts
│   ├── api/                     FastAPI app + Pydantic schemas
│   ├── monitoring/              drift (PSI/KS/Evidently), request_log
│   └── tracking.py              MLflow helpers
├── sql/                         schema, EDA queries, pgvector table
├── eval/rag_questions.jsonl     25-question test set
├── tests/                       31 pytest tests
├── dashboard/app.py             Streamlit monitoring
├── reports/                     generated results (see section 6)
├── configs/config.yaml
├── scripts/                     download_paysim.sh, fetch_public_docs.py
├── Dockerfile, docker-compose.yml
├── k8s/                         namespace, configmap, postgres, api (+HPA), batch CronJob
├── deploy/azure/                containerapps.sh, azureml/
└── .github/workflows/           ci.yml, nightly.yml, deploy-azure.yml
```

---

## 11. Limitations and next steps

- **Simulated data.** Validate on a real or anonymised ledger before drawing conclusions. The IEEE-CIS fraud
  dataset would be a harder second benchmark.
- **30-day horizon.** Monthly forecasting needs a longer ledger (set `freq: MS`, `day_range: null`).
- **Vendor segments are weakly separated** (silhouette 0.26). Try HDBSCAN or richer features such as network
  features between payers and payees.
- **RAG corpus is small.** Add real standards (PCAOB AS 2401, ISA 240) and 10-K filings, and evaluate with
  an LLM judge as well as key-fact matching.
- **Security for production.** Authentication on the API, secrets in Azure Key Vault, private networking for
  Postgres, and the managed identity in place of ACR admin credentials.
