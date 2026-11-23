"""Spend forecasting per account group with SARIMA (statsmodels) or Prophet.

PaySim covers 30 days of hourly steps, so the default series is *daily* spend per
transaction type with weekly seasonality. For multi-year ledgers set freq=MS
(monthly) and the same code forecasts monthly spend per account."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

START = pd.Timestamp("2024-01-01")


def spend_series(df: pd.DataFrame, group_col: str = "type", freq: str = "D",
                 day_range: tuple[int, int] | None = None) -> pd.DataFrame:
    """Wide frame: index=period, columns=group, values=total amount.
    `day_range` (inclusive) restricts the series to a stable window: in real PaySim the
    simulated volume collapses after day 17, which would make any holdout there meaningless."""
    if day_range is not None:
        day = (df["step"] - 1) // 24
        df = df[(day >= day_range[0]) & (day <= day_range[1])]
    ts = START + pd.to_timedelta(df["step"] - 1, unit="h")
    wide = (df.assign(ts=ts).groupby([pd.Grouper(key="ts", freq=freq), group_col])["amount"].sum()
            .unstack(fill_value=0.0))
    wide = wide.asfreq(freq, fill_value=0.0)
    counts = df.assign(ts=ts).groupby(pd.Grouper(key="ts", freq=freq)).size()
    if len(counts) > 2 and counts.iloc[-1] < 0.5 * counts.median():  # drop a partial trailing period
        wide = wide.iloc[:-1]
    return wide


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = y_true != 0
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100) if m.any() else float("nan")


def _season(freq: str) -> int:
    return {"D": 7, "W": 52, "MS": 12, "M": 12, "h": 24, "H": 24}.get(freq, 0)


ORDER_GRID = [(1, 0, 0), (2, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 0)]


def _fit(y_log: pd.Series, order, seasonal):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    trend = "c" if order[1] == 0 and seasonal[1] == 0 else "n"
    return SARIMAX(y_log, order=order, seasonal_order=seasonal, trend=trend,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def _select_order(y: pd.Series, horizon: int, seasonal) -> tuple:
    """Pick the order with the lowest MAPE on the last `horizon` points of the training
    series (an inner validation split), falling back to AIC when the series is too short."""
    y_log = np.log1p(y)
    inner = len(y) - horizon >= 6
    best, best_score = ORDER_GRID[0], np.inf
    for order in ORDER_GRID:
        try:
            if inner:
                r = _fit(y_log.iloc[:-horizon], order, seasonal)
                score = mape(y.iloc[-horizon:], np.expm1(r.get_forecast(horizon).predicted_mean))
            else:
                score = _fit(y_log, order, seasonal).aic
        except Exception:  # noqa: BLE001 - some orders fail to converge on short series
            continue
        if score < best_score:
            best, best_score = order, score
    return best


def fit_predict_sarima(y: pd.Series, horizon: int, freq: str = "D") -> pd.DataFrame:
    """SARIMA on log spend. The (p,d,q) order is chosen on an inner validation split of the
    training data only (never the holdout); a seasonal term is added when there are >= 2 full seasons."""
    m = _season(freq)
    seasonal = (1, 0, 1, m) if m and len(y) >= 2 * m else (0, 0, 0, 0)
    y_log = np.log1p(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        order = _select_order(y, horizon, seasonal)
        fc = _fit(y_log, order, seasonal).get_forecast(horizon)
    ci = fc.conf_int(alpha=0.2).clip(upper=float(np.log1p(y.max() * 10)))
    idx = pd.date_range(y.index[-1], periods=horizon + 1, freq=freq)[1:]
    return pd.DataFrame({"yhat": np.expm1(fc.predicted_mean.to_numpy()).clip(min=0),
                         "yhat_lower": np.expm1(ci.iloc[:, 0].to_numpy()).clip(min=0),
                         "yhat_upper": np.expm1(ci.iloc[:, 1].to_numpy())}, index=idx)


def fit_predict_prophet(y: pd.Series, horizon: int, freq: str = "D") -> pd.DataFrame:
    from prophet import Prophet
    d = pd.DataFrame({"ds": y.index, "y": y.to_numpy()})
    m = Prophet(weekly_seasonality=freq == "D", yearly_seasonality=freq in {"MS", "M"}, daily_seasonality=False,
                interval_width=0.8)
    m.fit(d)
    fut = m.make_future_dataframe(horizon, freq=freq, include_history=False)
    out = m.predict(fut).set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
    return out.clip(lower=0)


def forecast(y: pd.Series, horizon: int, freq: str = "D", method: str = "sarima") -> pd.DataFrame:
    if method == "prophet":
        try:
            return fit_predict_prophet(y, horizon, freq)
        except ImportError:
            warnings.warn("prophet not installed, falling back to SARIMA", stacklevel=2)
    return fit_predict_sarima(y, horizon, freq)


def backtest(wide: pd.DataFrame, horizon: int, freq: str = "D", method: str = "sarima") -> pd.DataFrame:
    """Hold out the last `horizon` periods per group and report MAPE vs a seasonal-naive baseline."""
    rows = []
    m = _season(freq) or 1
    for col in wide.columns:
        y = wide[col]
        if (y > 0).sum() < horizon * 2:
            continue
        train, test = y.iloc[:-horizon], y.iloc[-horizon:]
        pred = forecast(train, horizon, freq, method)["yhat"].to_numpy()
        naive = np.resize(train.iloc[-m:].to_numpy(), horizon)
        rows.append({"group": col, "mape": mape(test, pred), "naive_mape": mape(test, naive),
                     "actual_total": float(test.sum()), "forecast_total": float(pred.sum())})
    return pd.DataFrame(rows)
