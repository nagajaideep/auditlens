import numpy as np

from auditlens.features.build import FEATURES, build_features
from auditlens.models.anomaly import AnomalyDetector
from auditlens.models.classify import best_threshold, evaluate, make_logreg, time_split
from auditlens.models.cluster import VendorSegmenter, vendor_profiles
from auditlens.models.forecast import backtest, mape, spend_series


def test_features_shape_and_no_nan(clean_df):
    X = build_features(clean_df)
    assert list(X.columns) == FEATURES and len(X) == len(clean_df)
    assert not X.isna().any().any()


def test_features_history_counts_are_causal(clean_df):
    X = build_features(clean_df)
    first = clean_df.sort_values("step").groupby("nameOrig").head(1).index
    assert (X.loc[first, "orig_tx_count"] == 0).all()


def test_time_split_is_chronological(clean_df):
    X, y = build_features(clean_df), clean_df.isFraud
    s = time_split(X, y, clean_df.step)
    assert clean_df.loc[s.X_train.index, "step"].max() <= clean_df.loc[s.X_test.index, "step"].min()


def test_baseline_beats_random(clean_df):
    X, y = build_features(clean_df), clean_df.isFraud
    s = time_split(X, y, clean_df.step)
    m = make_logreg().fit(s.X_train, s.y_train)
    met = evaluate(s.y_test, m.predict_proba(s.X_test)[:, 1], 0.5)
    assert met["roc_auc"] > 0.8


def test_threshold_and_metrics():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.35, 0.8])
    t = best_threshold(y, p)
    m = evaluate(y, p, t)
    assert 0 <= m["precision"] <= 1 and m["tp"] + m["fn"] == 2


def test_anomaly_detector_ranks_fraud_higher(clean_df):
    X = build_features(clean_df)
    det = AnomalyDetector(0.02).fit(X)
    s = det.score(X)
    assert s[clean_df.isFraud == 1].mean() > s[clean_df.isFraud == 0].mean()


def test_vendor_clusters(clean_df):
    prof = vendor_profiles(clean_df, min_tx=2)
    seg = VendorSegmenter(k_range=range(3, 5)).fit(prof)
    assert seg.k in (3, 4) and len(seg.labels_) == seg.k
    assert seg.summary(prof)["vendors"].sum() == len(prof)


def test_forecast_backtest(clean_df):
    wide = spend_series(clean_df, "type", "D")
    bt = backtest(wide, 5, "D")
    assert len(bt) > 0 and bt["mape"].notna().all()


def test_mape():
    assert mape([100, 200], [110, 180]) == 10.0
    assert np.isnan(mape([0], [1]))
