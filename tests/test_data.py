import numpy as np
import pandas as pd
import pytest

from auditlens.data.clean import clean
from auditlens.data.schema import clean_schema, failure_summary, raw_schema


def test_generator_schema(raw_df):
    raw_schema.validate(raw_df)
    assert set(raw_df.loc[raw_df.isFraud == 1, "type"]) <= {"TRANSFER", "CASH_OUT"}


def test_clean_removes_duplicates_and_nulls(raw_df):
    df, log = clean(raw_df)
    assert log["duplicates_removed"] > 0
    assert df[["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].isna().sum().sum() == 0
    clean_schema.validate(df)
    assert df.tx_id.is_unique and df.hour.between(0, 23).all()


def test_schema_catches_bad_rows(raw_df):
    bad = raw_df.head(50).copy()
    bad.loc[0, "type"] = "WIRE"
    bad.loc[1, "amount"] = -5
    fails = failure_summary(bad)
    assert {"type", "amount"} <= set(fails["column"])


def test_clean_clips_negative():
    df = pd.DataFrame({"step": [1], "type": ["payment "], "amount": [-3.0], "nameOrig": ["C1"],
                       "oldbalanceOrg": [10.0], "newbalanceOrig": [np.nan], "nameDest": ["M2"],
                       "oldbalanceDest": [0.0], "newbalanceDest": [0.0], "isFraud": [0], "isFlaggedFraud": [0]})
    out, log = clean(df)
    assert out.loc[0, "amount"] == 0 and out.loc[0, "type"] == "PAYMENT"
    assert log["negative_values_clipped"] == {"amount": 1}


def test_config_env_override():
    from auditlens.config import apply_env_overrides
    cfg = apply_env_overrides({"data": {"source": "csv"}}, {"AUDITLENS_DATA__SOURCE": "postgres",
                                                            "AUDITLENS_MODEL__OPTUNA_TRIALS": "3"})
    assert cfg["data"]["source"] == "postgres" and cfg["model"]["optuna_trials"] == 3


@pytest.mark.parametrize("col", ["amount", "oldbalanceOrg"])
def test_no_negative_after_clean(clean_df, col):
    assert (clean_df[col] >= 0).all()
