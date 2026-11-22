"""Supervised fraud classifiers: logistic-regression baseline, Random Forest, XGBoost
(Optuna-tuned). Imbalance is handled with class weights / scale_pos_weight, and the
decision threshold is tuned on a validation split rather than left at 0.5."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from auditlens import tracking

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    idx_test: pd.Index = field(default=None)


def time_split(X: pd.DataFrame, y: pd.Series, step: pd.Series, val: float = 0.16, test: float = 0.2) -> Split:
    """Chronological split: train on the past, validate/test on the future (no leakage)."""
    order = step.sort_values(kind="stable").index
    n = len(order)
    a, b = int(n * (1 - val - test)), int(n * (1 - test))
    tr, va, te = order[:a], order[a:b], order[b:]
    return Split(X.loc[tr], y.loc[tr], X.loc[va], y.loc[va], X.loc[te], y.loc[te], te)


def best_threshold(y_true, proba, metric: str = "f1") -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    if metric == "recall@precision50":
        ok = p[:-1] >= 0.5
        return float(t[ok][np.argmax(r[:-1][ok])]) if ok.any() else 0.5
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.nanargmax(f1[:-1])]) if len(t) else 0.5


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    one_class = len(np.unique(y_true)) < 2
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": float("nan") if one_class else roc_auc_score(y_true, proba),
        "pr_auc": float("nan") if one_class else average_precision_score(y_true, proba),
        "threshold": threshold, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def make_logreg() -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])


def make_rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=seed)


def make_xgb(params: dict | None = None, spw: float = 1.0, seed: int = 42) -> XGBClassifier:
    base = dict(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                min_child_weight=1, reg_lambda=1.0, tree_method="hist", eval_metric="aucpr")
    base.update(params or {})
    return XGBClassifier(**base, scale_pos_weight=spw, n_jobs=-1, random_state=seed)


def tune_xgb(s: Split, n_trials: int = 25, seed: int = 42) -> dict:
    spw = float((s.y_train == 0).sum() / max((s.y_train == 1).sum(), 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        spw_mult = trial.suggest_float("spw_mult", 0.1, 1.0, log=True)  # full weighting often over-flags
        m = make_xgb(params, spw * spw_mult, seed).fit(s.X_train, s.y_train)
        score = average_precision_score(s.y_val, m.predict_proba(s.X_val)[:, 1])
        tracking.log_metrics({"trial_val_pr_auc": score}, step=trial.number)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    mult = best.pop("spw_mult")
    return {"params": best, "scale_pos_weight": spw * mult, "best_val_pr_auc": study.best_value}


def train_all(s: Split, n_trials: int = 25, seed: int = 42, threshold_metric: str = "f1") -> dict:
    """Train and evaluate every model; returns {name: {"model", "threshold", "val", "test"}}."""
    spw = float((s.y_train == 0).sum() / max((s.y_train == 1).sum(), 1))
    results = {}
    candidates = {
        "logreg": lambda: make_logreg(),
        "random_forest": lambda: make_rf(seed),
        "xgboost_default": lambda: make_xgb(None, spw, seed),
    }
    for name, ctor in candidates.items():
        with tracking.run(name, tags={"stage": "classification"}, nested=True):
            m = ctor().fit(s.X_train, s.y_train)
            results[name] = _score(name, m, s, threshold_metric)

    with tracking.run("xgboost_optuna", tags={"stage": "classification"}, nested=True):
        tuned = tune_xgb(s, n_trials, seed)
        tracking.log_params({**tuned["params"], "scale_pos_weight": tuned["scale_pos_weight"]})
        m = make_xgb(tuned["params"], tuned["scale_pos_weight"], seed).fit(s.X_train, s.y_train)
        results["xgboost_optuna"] = _score("xgboost_optuna", m, s, threshold_metric)
        results["xgboost_optuna"]["params"] = tuned
    return results


def _score(name, model, s: Split, threshold_metric: str) -> dict:
    pv = model.predict_proba(s.X_val)[:, 1]
    thr = best_threshold(s.y_val, pv, threshold_metric)
    val = evaluate(s.y_val, pv, thr)
    test = evaluate(s.y_test, model.predict_proba(s.X_test)[:, 1], thr)
    tracking.log_metrics({**{f"val_{k}": v for k, v in val.items()}, **{f"test_{k}": v for k, v in test.items()}})
    return {"name": name, "model": model, "threshold": thr, "val": val, "test": test}


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        t = r["test"]
        rows.append({"model": name, "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
                     "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"], "threshold": r["threshold"],
                     "val_pr_auc": r["val"]["pr_auc"]})
    return pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
