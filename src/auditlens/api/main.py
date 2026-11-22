"""FastAPI service: /predict, /forecast, /ask, plus /health and /metrics."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from auditlens import __version__
from auditlens.api.schemas import (
    AskRequest,
    AskResponse,
    ForecastRequest,
    ForecastResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
)
from auditlens.config import get_config, resolve
from auditlens.logging_utils import get_logger, log_event
from auditlens.monitoring.request_log import RequestLogger
from auditlens.pipelines.artifacts import load_bundle
from auditlens.pipelines.scoring import risk_band, score_frame

log = get_logger("auditlens.api")
STATE: dict = {}


@lru_cache(maxsize=1)
def bundle() -> dict:
    return load_bundle()


@lru_cache(maxsize=1)
def explainer():
    from auditlens.explain.shap_explain import Explainer
    b = bundle()
    return Explainer(b["model"], b["background"])


@lru_cache(maxsize=1)
def rag():
    from auditlens.genai.rag import RAGPipeline
    return RAGPipeline.build(persist=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["logger"] = RequestLogger(resolve(get_config()["monitoring"]["request_log"]))
    STATE["started"] = time.time()
    try:
        bundle()
    except FileNotFoundError as e:
        log_event(log, "model bundle missing at startup", error=str(e))
    yield


app = FastAPI(title="AuditLens API", version=__version__, lifespan=lifespan,
              description="Risk scoring, spend forecasting and cited policy Q&A for auditors.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    t0 = time.perf_counter()
    status, err = 500, None
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    except Exception as e:
        err = repr(e)
        raise
    finally:
        ms = (time.perf_counter() - t0) * 1000
        rec = {"request_id": rid, "method": request.method, "path": request.url.path, "status": status,
               "latency_ms": round(ms, 2), "error": err, **getattr(request.state, "extra", {})}
        if "logger" in STATE:
            STATE["logger"].write(**rec)
        log_event(log, "request", **rec)


@app.get("/health")
def health():
    try:
        b = bundle()
        model = {"loaded": True, "version": b["model_version"], "threshold": b["threshold"]}
    except FileNotFoundError:
        model = {"loaded": False}
    return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - STATE.get("started", time.time())),
            "model": model}


@app.get("/metrics")
def metrics():
    try:
        return bundle()["metrics"]
    except FileNotFoundError as err:
        raise HTTPException(503, "model not trained") from err


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    try:
        b = bundle()
    except FileNotFoundError as err:
        raise HTTPException(503, "model not trained - run `auditlens train`") from err
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    scores, X = score_frame(b, df)
    reasons = explainer().reasons(X[b["features"]]) if req.explain else [None] * len(df)
    preds = []
    for i, row in scores.reset_index(drop=True).iterrows():
        narrative = None
        if req.narrate and row.flagged and reasons[i]:
            from auditlens.genai.narrate import narrate
            narrative = narrate(df.iloc[i].to_dict(), float(row.fraud_score), reasons[i])
        preds.append(Prediction(fraud_score=float(row.fraud_score), flagged=bool(row.flagged),
                                risk_band=risk_band(float(row.fraud_score), b["threshold"]),
                                anomaly_score=float(row.anomaly_score), anomaly_flag=bool(row.anomaly_flag),
                                reasons=reasons[i], narrative=narrative))
    request.state.extra = {"n_tx": len(df), "n_flagged": int(scores.flagged.sum()),
                           "mean_score": round(float(scores.fraud_score.mean()), 4)}
    return PredictResponse(model_version=b["model_version"], threshold=b["threshold"], predictions=preds)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    from auditlens.models.forecast import forecast as run_forecast
    try:
        b = bundle()
    except FileNotFoundError as err:
        raise HTTPException(503, "model not trained") from err
    wide, freq = b["spend_series"], b["forecast_cfg"]["freq"]
    if req.group not in wide.columns:
        raise HTTPException(404, f"unknown group '{req.group}'. Available: {list(wide.columns)}")
    fc = run_forecast(wide[req.group], req.horizon, freq, req.method)
    pts = [{"period": str(i.date()), **{k: float(v) for k, v in r.items()}} for i, r in fc.iterrows()]
    return ForecastResponse(group=req.group, freq=freq, method=req.method, forecast=pts)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return rag().ask(req.question).to_dict()
    except ValueError as e:
        raise HTTPException(503, str(e)) from e
