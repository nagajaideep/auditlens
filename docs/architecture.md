# Architecture

```mermaid
flowchart LR
  subgraph Sources
    K[PaySim CSV<br/>Kaggle] --> PG[(PostgreSQL<br/>transactions_raw)]
    G[Synthetic generator<br/>CI / demo] --> PG
    D[Policy docs<br/>.md / .pdf / 10-K]
  end

  subgraph Training["Training pipeline (auditlens train)"]
    PG --> V[Pandera validation<br/>+ cleaning] --> Q[Data-quality report]
    V --> F[Feature engineering]
    F --> C[LogReg / RF / XGBoost<br/>+ Optuna]
    F --> A[Isolation Forest]
    V --> KM[K-Means vendor segments]
    V --> FC[SARIMA / Prophet<br/>spend forecast]
    C --> S[SHAP + error &<br/>slice analysis]
    C & A & KM & FC --> B[[Model bundle]]
    C --> ML[(MLflow tracking<br/>+ model registry)]
  end

  subgraph GenAI
    D --> CH[Chunk + embed<br/>MiniLM / TF-IDF] --> VS[(Chroma / pgvector)]
    VS --> R[RAG with citations]
    LLM[Claude / Azure OpenAI<br/>/ extractive fallback] --> R
    LLM --> N[Flag narratives]
  end

  subgraph Serving
    B --> API[FastAPI<br/>/predict /forecast /ask]
    R --> API
    N --> API
    B --> BJ[Nightly batch job<br/>K8s CronJob] --> PG
    API --> LOG[JSON request log]
    BJ --> DR[Evidently + PSI drift]
    LOG & DR --> DASH[Streamlit dashboard]
  end

  API --> DEP{{Docker → minikube / Azure Container Apps / Azure ML}}
```

## Request flow for `/predict`
1. Pydantic validates the transaction (types, non-negative amounts, account-id format).
2. `build_features` derives the same 21 features used in training.
3. The registered classifier returns a fraud probability, compared with the threshold tuned on validation data.
4. Isolation Forest adds a label-free anomaly score.
5. SHAP TreeExplainer returns the top four signed feature contributions.
6. Optionally, the LLM turns those contributions into a 2–4 sentence note for the auditor. The prompt
   forbids reasons that aren't in the SHAP output.
7. Middleware writes one JSON line per request (latency, status, number flagged), which the dashboard reads.
