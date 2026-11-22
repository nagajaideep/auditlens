import numpy as np
import pandas as pd

from auditlens.monitoring.drift import drift_report, psi


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.2


def test_drift_report(tmp_path):
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000), "score": rng.beta(1, 20, 2000)})
    cur = pd.DataFrame({"x": rng.normal(2, 1, 2000), "score": rng.beta(1, 20, 2000)})
    s = drift_report(ref, cur, tmp_path / "d.html")
    assert s["n_drifted"] >= 1 and (tmp_path / "d.json").exists()


def test_batch_job(trained, raw_df, tmp_path):
    from auditlens.pipelines.batch_score import run
    inp = tmp_path / "new.csv"
    raw_df.sample(2000, random_state=3).to_csv(inp, index=False)
    out = run(str(inp), drift=True)
    df = pd.read_parquet(out)
    assert len(df) > 0 and {"fraud_score", "flagged", "model_version"} <= set(df.columns)
