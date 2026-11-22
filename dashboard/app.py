"""AuditLens monitoring dashboard: API latency, errors, traffic, drift and model metrics.
Run: streamlit run dashboard/app.py"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from auditlens.config import get_config, resolve
from auditlens.monitoring.request_log import read_log

cfg = get_config()
st.set_page_config(page_title="AuditLens Monitor", layout="wide")
st.title("AuditLens — model & service monitoring")

log = read_log(resolve(cfg["monitoring"]["request_log"]))
tab_svc, tab_drift, tab_model, tab_flags = st.tabs(["Service", "Drift", "Model", "Flagged transactions"])

with tab_svc:
    if log.empty:
        st.info("No requests logged yet. Call the API to populate logs/requests.jsonl.")
    else:
        hours = st.slider("Window (hours)", 1, 168, 24)
        recent = log[log.ts >= log.ts.max() - pd.Timedelta(hours=hours)]
        c = st.columns(4)
        c[0].metric("Requests", f"{len(recent):,}")
        c[1].metric("p50 latency (ms)", f"{recent.latency_ms.quantile(.5):.1f}")
        c[2].metric("p95 latency (ms)", f"{recent.latency_ms.quantile(.95):.1f}")
        c[3].metric("Error rate", f"{(recent.status >= 500).mean():.2%}")
        per_min = recent.set_index("ts").resample("5min")
        st.subheader("Latency (p95, 5-min buckets)")
        st.line_chart(per_min.latency_ms.quantile(.95).rename("p95_ms"))
        st.subheader("Requests and errors by endpoint")
        st.dataframe(recent.groupby("path").agg(requests=("status", "size"),
                                                 errors=("status", lambda s: int((s >= 400).sum())),
                                                 p95_ms=("latency_ms", lambda s: s.quantile(.95))).round(1))
        if "n_flagged" in recent:
            pr = recent[recent.path == "/predict"].set_index("ts")
            if len(pr):
                st.subheader("Flag rate on /predict traffic")
                st.line_chart((pr.n_flagged.resample("1h").sum() / pr.n_tx.resample("1h").sum()).rename("flag_rate"))

with tab_drift:
    p = resolve(cfg["monitoring"]["drift_report"]).with_suffix(".json")
    if not p.exists():
        st.info("No drift report yet. Run `auditlens batch --input <file>` to create one.")
    else:
        d = json.loads(p.read_text())
        c = st.columns(3)
        c[0].metric("Features drifted", f"{d['n_drifted']} / {d['n_features']}")
        c[1].metric("Share drifted", f"{d['share_drifted']:.0%}", help="Policy threshold: 30%")
        c[2].metric("Prediction PSI", f"{d['prediction_psi']:.3f}" if d.get("prediction_psi") is not None else "n/a")
        if d.get("dataset_drift"):
            st.error("Dataset drift above the 30% policy threshold: investigate and consider retraining.")
        st.dataframe(pd.DataFrame(d["features"]).sort_values("psi", ascending=False), use_container_width=True)
        html = resolve(cfg["monitoring"]["drift_report"])
        if html.exists():
            st.download_button("Download Evidently report", html.read_bytes(), file_name="drift_report.html")

with tab_model:
    s = resolve("reports/training_summary.json")
    if s.exists():
        summ = json.loads(s.read_text())
        st.write(f"**Model:** {summ['model_version']} · trained on {summ['rows']:,} rows")
        st.dataframe(pd.DataFrame(summ["comparison"]))
        st.subheader("Forecast backtest (MAPE %)")
        st.dataframe(pd.DataFrame(summ["forecast"]))
        st.subheader("Top SHAP features")
        st.bar_chart(pd.Series(summ["top_shap_features"]))
        fig = resolve("reports/figures/shap_summary.png")
        if fig.exists():
            st.image(str(fig))
    else:
        st.info("Train the model first: `auditlens train`.")

with tab_flags:
    f = resolve("reports/batch/latest_flagged.csv")
    if f.exists():
        st.dataframe(pd.read_csv(f).head(500), use_container_width=True)
    else:
        st.info("No batch run yet.")
