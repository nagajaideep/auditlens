import os

import pytest

os.environ.setdefault("AUDITLENS_MLFLOW__ENABLED", "false")
os.environ.setdefault("AUDITLENS_GENAI__EMBEDDER", "tfidf")
os.environ.setdefault("AUDITLENS_GENAI__LLM_PROVIDER", "extractive")


@pytest.fixture(scope="session")
def raw_df():
    from auditlens.data.generate import generate
    return generate(n_rows=12_000, n_steps=24 * 30, fraud_rate=0.02, seed=7)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    from auditlens.data.clean import clean
    return clean(raw_df)[0]


@pytest.fixture(scope="session")
def trained(tmp_path_factory, raw_df):
    """Train the full pipeline once into a temp dir; shared by API/batch tests."""
    tmp = tmp_path_factory.mktemp("run")
    os.environ["AUDITLENS_MODEL__DIR"] = str(tmp / "models")
    os.environ["AUDITLENS_DATA__PROCESSED"] = str(tmp / "tx.parquet")
    os.environ["AUDITLENS_MONITORING__REFERENCE"] = str(tmp / "ref.parquet")
    os.environ["AUDITLENS_MONITORING__REQUEST_LOG"] = str(tmp / "requests.jsonl")
    os.environ["AUDITLENS_MONITORING__DRIFT_REPORT"] = str(tmp / "drift.html")
    from auditlens.config import get_config
    get_config.cache_clear()
    from auditlens.pipelines.train import run
    summary = run(raw=raw_df, n_trials=2, reports_dir=tmp / "reports")
    return {"summary": summary, "tmp": tmp}
