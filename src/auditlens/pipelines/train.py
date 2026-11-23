"""End-to-end training pipeline (modules 1-5 and 7):
load -> validate -> clean -> quality report -> features -> classifiers (+Optuna) ->
anomaly detection -> vendor clustering -> forecasting -> SHAP + error analysis -> save bundle.
Every stage is logged to MLflow under one parent run; the best classifier is registered."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from auditlens import tracking
from auditlens.config import get_config, resolve
from auditlens.data.clean import clean
from auditlens.data.load import load_raw
from auditlens.data.quality_report import write_report
from auditlens.data.schema import clean_schema
from auditlens.explain.error_analysis import compare_outcomes, disparity, slice_metrics
from auditlens.explain.shap_explain import Explainer
from auditlens.features.build import FEATURES, build_features, segment_columns
from auditlens.logging_utils import get_logger, log_event
from auditlens.models.anomaly import AnomalyDetector
from auditlens.models.classify import comparison_table, time_split, train_all
from auditlens.models.cluster import VendorSegmenter, vendor_profiles
from auditlens.models.forecast import backtest, spend_series
from auditlens.pipelines.artifacts import save_bundle

log = get_logger("auditlens.train")


def run(cfg: dict | None = None, raw: pd.DataFrame | None = None, n_trials: int | None = None,
        reports_dir: str | Path = "reports") -> dict:
    cfg = cfg or get_config()
    reports = resolve(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    seed = cfg["data"]["seed"]
    n_trials = n_trials if n_trials is not None else cfg["model"]["optuna_trials"]
    t0 = time.time()

    with tracking.run("training_pipeline", tags={"stage": "pipeline"}):
        # ---- 1. data prep & quality
        raw = load_raw(cfg) if raw is None else raw
        df, clean_log = clean(raw)
        clean_schema.validate(df, lazy=True)
        write_report(raw, df, clean_log, reports)
        out = resolve(cfg["data"]["processed"]); out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        tracking.log_params({"rows": len(df), "fraud_rate": round(df.isFraud.mean(), 6), "optuna_trials": n_trials})
        log_event(log, "data ready", rows=len(df), fraud=int(df.isFraud.sum()))

        # ---- 2. classification
        X = build_features(df)
        y = df["isFraud"]
        split = time_split(X, y, df["step"])
        results = train_all(split, n_trials=n_trials, seed=seed, threshold_metric=cfg["model"]["threshold_metric"])
        table = comparison_table(results)
        table.to_csv(reports / "model_comparison.csv", index=False)
        best_name = table.iloc[0]["model"]
        best = results[best_name]
        tracking.log_metrics({f"best_test_{k}": v for k, v in best["test"].items() if isinstance(v, float)})
        log_event(log, "classifiers trained", best=best_name, test_auc=best["test"]["roc_auc"],
                  test_recall=best["test"]["recall"])

        # ---- 3a. anomaly detection (fit on training rows only)
        contamination = float(np.clip(split.y_train.mean() * 2, 0.001, 0.05))
        anomaly = AnomalyDetector(contamination=contamination, seed=seed).fit(split.X_train)
        anomaly_metrics = anomaly.evaluate(split.X_test, split.y_test)
        tracking.log_metrics({f"anomaly_{k}": v for k, v in anomaly_metrics.items()})

        # ---- 3b. vendor clustering
        prof = vendor_profiles(df)
        seg = VendorSegmenter(seed=seed).fit(prof)
        clusters = seg.summary(prof)
        clusters.to_csv(reports / "vendor_clusters.csv")
        prof.assign(cluster=seg.km.labels_).to_csv(reports / "vendor_profiles.csv")
        tracking.log_metrics({"kmeans_k": seg.k, "kmeans_silhouette": seg.silhouettes.get(seg.k, np.nan)})

        # ---- 4. forecasting
        fc_cfg = cfg["forecast"]
        wide = spend_series(df, "type", fc_cfg["freq"], fc_cfg.get("day_range"))
        bt = backtest(wide, fc_cfg["horizon"], fc_cfg["freq"], fc_cfg["method"])
        bt.to_csv(reports / "forecast_backtest.csv", index=False)
        wide.to_csv(reports / "spend_series.csv")
        fc_mape = float(np.average(bt["mape"], weights=bt["actual_total"])) if len(bt) else float("nan")
        fc_median = float(bt["mape"].median()) if len(bt) else float("nan")
        tracking.log_metrics({"forecast_weighted_mape": fc_mape, "forecast_median_mape": fc_median})

        # ---- 5. explainability & error analysis
        sample = split.X_test.sample(min(3000, len(split.X_test)), random_state=seed)
        expl = Explainer(best["model"], split.X_train)
        importance = expl.global_importance(sample)
        importance.rename("mean_abs_shap").to_csv(reports / "shap_importance.csv")
        try:
            expl.summary_plot(sample, reports / "figures" / "shap_summary.png")
        except Exception as e:  # plotting is non-critical
            log_event(log, "shap plot skipped", error=str(e))
        proba_test = best["model"].predict_proba(split.X_test)[:, 1]
        pred_test = (proba_test >= best["threshold"]).astype(int)
        outcomes = compare_outcomes(split.X_test, split.y_test, pred_test)
        outcomes.to_csv(reports / "error_analysis_outcomes.csv")
        slices = slice_metrics(segment_columns(df.loc[split.idx_test]), split.y_test, proba_test, best["threshold"])
        slices.to_csv(reports / "slice_metrics.csv", index=False)
        disp = disparity(slices, "fpr")
        disp.to_csv(reports / "fpr_disparity.csv")

        # ---- 7. register best model
        version = tracking.log_and_register_sklearn(best["model"], "fraud_model", cfg["mlflow"]["registered_model"],
                                                    split.X_test.head(5))
        for f in ["model_comparison.csv", "forecast_backtest.csv", "vendor_clusters.csv", "shap_importance.csv",
                  "slice_metrics.csv", "data_quality_report.md"]:
            tracking.log_artifact(str(reports / f))

        # ---- reference data for drift monitoring
        ref = resolve(cfg["monitoring"]["reference"]); ref.parent.mkdir(parents=True, exist_ok=True)
        split.X_train.sample(min(20000, len(split.X_train)), random_state=seed).assign(
            score=lambda d: best["model"].predict_proba(d[FEATURES])[:, 1]).to_parquet(ref)

        model_version = f"{best_name}-v{version}" if version else f"{best_name}-{time.strftime('%Y%m%d%H%M')}"
        bundle = {
            "model": best["model"], "model_name": best_name, "model_version": model_version,
            "threshold": best["threshold"], "features": FEATURES, "anomaly": anomaly,
            "vendor_segmenter": seg, "background": split.X_train.sample(min(500, len(split.X_train)), random_state=0),
            "spend_series": wide, "forecast_cfg": fc_cfg,
            "metrics": {"test": best["test"], "val": best["val"], "anomaly": anomaly_metrics,
                        "forecast_weighted_mape": fc_mape, "forecast_median_mape": fc_median, "kmeans_k": seg.k},
        }
        save_bundle(bundle)

        summary = {
            "rows": len(df), "fraud_cases": int(df.isFraud.sum()), "best_model": best_name,
            "model_version": model_version, "test": _clean(best["test"]), "comparison": table.round(4).to_dict("records"),
            "anomaly": _clean(anomaly_metrics), "forecast_weighted_mape": fc_mape, "forecast_median_mape": fc_median,
            "forecast": bt.round(3).to_dict("records"),
            "vendor_clusters": {"k": seg.k, "silhouette": seg.silhouettes.get(seg.k), "labels": seg.labels_},
            "top_shap_features": importance.head(8).round(4).to_dict(),
            "fpr_disparity": disp.round(4).reset_index().to_dict("records"),
            "seconds": round(time.time() - t0, 1),
        }
        (reports / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log_event(log, "training complete", **{k: summary[k] for k in ["best_model", "model_version", "seconds"]})
        return summary


def _clean(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
