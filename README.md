# AuditLens

**An ML and generative-AI platform for spotting risky financial transactions, built the way an auditor
would use it.** It flags suspicious transactions and explains each flag, scores anomalies, groups vendors
by behaviour, forecasts spend, and answers audit-policy questions with citations. Everything is served
through a REST API with monitoring, CI and cloud deployment.

```text
PaySim → PostgreSQL → Pandera checks → features → LogReg / RF / XGBoost (+Optuna) → SHAP
                                              ↘ Isolation Forest   ↘ K-Means vendors   ↘ SARIMA/Prophet
Policy docs → chunks → embeddings → Chroma/pgvector → LLM with citations
All runs → MLflow registry → FastAPI (/predict /forecast /ask) → Docker → minikube / Azure Container Apps
Nightly batch + Evidently drift + JSON request logs → Streamlit dashboard
```

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and [`MODEL_CARD.md`](MODEL_CARD.md) for
data, assumptions, limitations and risks.

## Modules

| # | Module | Where | What it does |
|---|---|---|---|
| 1 | Data prep & EDA | `data/`, `sql/` | PostgreSQL loading, Pandera schemas, cleaning log, profiling, 5 EDA figures, `reports/data_quality_report.md`, 7 analysis queries in `sql/eda_queries.sql` |
| 2 | Classification | `models/classify.py` | Logistic-regression baseline → Random Forest → XGBoost tuned with Optuna. Class weights / `scale_pos_weight`, threshold tuned on validation, chronological split, compared on precision, recall, F1, ROC-AUC and PR-AUC |
| 3 | Anomaly & clustering | `models/anomaly.py`, `models/cluster.py` | Isolation Forest (label-free); K-Means over 9 behavioural vendor features, k picked by silhouette, clusters named in plain English |
| 4 | Forecasting | `models/forecast.py` | SARIMA (default) or Prophet per account group; backtest MAPE against a seasonal-naive baseline |
| 5 | Explainability & error analysis | `explain/` | SHAP global and per-transaction reasons; TP/FP/FN/TN profiles; slice metrics and FPR disparity by segment |
| 6 | GenAI Q&A | `genai/` | Chunk → embed (MiniLM, or a TF-IDF fallback) → Chroma / pgvector → LLM answer with `[n]` citations; LLM narratives for flags; 25-question eval |
| 7 | Experiment tracking | `tracking.py` | Every model and Optuna trial logged to MLflow; best model registered as `auditlens-fraud-model` |
| 8 | Pipelines & API | `pipelines/`, `api/`, `cli.py` | Installable package, `auditlens` CLI, FastAPI service, nightly batch scoring job |
| 9 | Deployment | `Dockerfile`, `k8s/`, `deploy/azure/` | Docker image, minikube manifests (Deployment, HPA, Postgres, CronJob), Azure Container Apps script, Azure ML endpoint files |
| 10 | Monitoring | `monitoring/`, `dashboard/` | PSI/KS + Evidently drift, structured JSON request logs, Streamlit dashboard (latency, errors, flag rate, drift) |
| 11 | Testing & CI | `tests/`, `.github/workflows/`, `azure-pipelines.yml` | 29 pytest tests (data checks, models, API, RAG, batch, drift), ruff, RAG quality gate, Docker build |
| 12 | Documentation | `MODEL_CARD.md`, `docs/` | Model card, architecture diagram, this README |

## Quick start (5 minutes, no accounts needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[genai,monitoring,dev]"
auditlens generate --rows 200000          # synthetic PaySim-schema data (or see "Real data" below)
auditlens train --trials 25               # modules 1–5 and 7, writes reports/ and artifacts/models/
auditlens eval-rag                        # 25-question RAG evaluation
auditlens serve                           # http://localhost:8000/docs
streamlit run dashboard/app.py            # http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
pytest
```

Example calls:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{"narrate": true, "transactions": [
  {"step": 3, "type": "TRANSFER", "amount": 950000, "nameOrig": "C1", "oldbalanceOrg": 950000,
   "newbalanceOrig": 0, "nameDest": "C2", "oldbalanceDest": 0, "newbalanceDest": 0}]}'
curl -s localhost:8000/forecast -H 'content-type: application/json' -d '{"group": "CASH_OUT", "horizon": 7}'
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "How quickly must alerts be triaged?"}'
```

### LLM provider
`genai.llm_provider: auto` uses the first provider it finds credentials for:
`ANTHROPIC_API_KEY` (Claude), `AZURE_OPENAI_*` (Azure OpenAI), `OPENAI_API_KEY`. With none set it falls back
to a local **extractive** mode that quotes the best-matching sentences with citations, so CI and offline demos
still work. Copy `.env.example` to `.env`.

### Real data
```bash
export KAGGLE_USERNAME=... KAGGLE_KEY=...
./scripts/download_paysim.sh               # data/raw/paysim.csv (6.3M rows)
docker compose up -d db                    # PostgreSQL 16 + pgvector
auditlens load-db                          # CSV → transactions_raw
AUDITLENS_DATA__SOURCE=postgres auditlens train
```
`data.sample_rows` (default 1M) caps training size on a laptop. Set it to `null` to train on everything.
Add public documents (10-K filings, standards) with `scripts/fetch_public_docs.py` or by dropping PDFs into
`docs/policies/`.

## Results

Numbers from the **synthetic** demo dataset (500k rows, 25 Optuna trials). They only show that the pipeline
runs end to end. Re-run on real PaySim before quoting anything (see the caveat in the model card).

| Metric | Value |
|---|---|
| Best classifier (by validation PR-AUC) | Random Forest: recall 0.997, precision 1.00, ROC-AUC 1.00 |
| Logistic-regression baseline | recall 0.39, PR-AUC 0.54 |
| Isolation Forest (no labels) | ROC-AUC 0.994, catches 70% of fraud while flagging 0.8% of transactions |
| Daily spend forecast (SARIMA, 7-day holdout) | 3.2% weighted MAPE |
| RAG, offline extractive baseline | 92% pass, 100% retrieval hit rate, 96% citation rate on 25 questions |
| Batch scoring | 30k transactions in about 2.5 s on 4 CPU cores |

## Deployment

| Target | How |
|---|---|
| Docker Compose | `docker compose up -d db mlflow && docker compose run --rm trainer && docker compose up -d api dashboard` |
| minikube | see [`k8s/README.md`](k8s/README.md) (2-replica API with HPA, Postgres StatefulSet, nightly CronJob) |
| Azure Container Apps | `az login && ./deploy/azure/containerapps.sh`. Builds in ACR and scales to zero when idle, which suits student credits |
| Azure ML | [`deploy/azure/azureml/`](deploy/azure/azureml/README.md) managed online endpoint using the same image |

Train before you build the image: `artifacts/models/bundle.joblib` is baked in.

## Configuration
`configs/config.yaml`. Any key can be overridden with `AUDITLENS_<SECTION>__<KEY>`, e.g.
`AUDITLENS_FORECAST__METHOD=prophet` or `AUDITLENS_MODEL__OPTUNA_TRIALS=50`.

## Repository layout
```
src/auditlens/   data/ features/ models/ explain/ genai/ pipelines/ api/ monitoring/ cli.py
configs/         config.yaml
sql/             schema.sql, eda_queries.sql, pgvector.sql
docs/            architecture.md, policies/ (RAG corpus)
eval/            rag_questions.jsonl (25 questions)
tests/           pytest suite
dashboard/       Streamlit monitoring app
k8s/  deploy/    Kubernetes manifests, Azure scripts
reports/         generated: data-quality report, model comparison, SHAP, slices, forecasts, RAG eval
```

## License
MIT. PaySim is © its authors (CC BY-SA 4.0 on Kaggle). The policy documents in `docs/policies/` are
original samples written for this project.
