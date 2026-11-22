"""Pandera schemas: the contract raw and cleaned transaction data must satisfy."""
from __future__ import annotations

try:  # pandera >= 0.24
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa

TX_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]

RAW_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]

raw_schema = pa.DataFrameSchema(
    {
        "step": pa.Column(int, pa.Check.ge(1)),
        "type": pa.Column(str, pa.Check.isin(TX_TYPES)),
        "amount": pa.Column(float, pa.Check.ge(0)),
        "nameOrig": pa.Column(str, pa.Check.str_matches(r"^C\d+$")),
        "oldbalanceOrg": pa.Column(float, pa.Check.ge(0), nullable=True),
        "newbalanceOrig": pa.Column(float, pa.Check.ge(0), nullable=True),
        "nameDest": pa.Column(str, pa.Check.str_matches(r"^[CM]\d+$")),
        "oldbalanceDest": pa.Column(float, pa.Check.ge(0), nullable=True),
        "newbalanceDest": pa.Column(float, pa.Check.ge(0), nullable=True),
        "isFraud": pa.Column(int, pa.Check.isin([0, 1])),
        "isFlaggedFraud": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=False,
    coerce=True,
)

clean_schema = raw_schema.update_columns(
    {c: {"nullable": False} for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]}
).add_columns({
    "tx_id": pa.Column(int, unique=True),
    "day": pa.Column(int, pa.Check.ge(0)),
    "hour": pa.Column(int, pa.Check.in_range(0, 23)),
})


def validate(df, schema=raw_schema, lazy: bool = True):
    """Validate and return the (coerced) frame. Raises SchemaErrors listing all failures."""
    return schema.validate(df, lazy=lazy)


def failure_summary(df, schema=raw_schema):
    """Return a DataFrame of failing checks (empty if the data passes)."""
    import pandas as pd
    try:
        schema.validate(df, lazy=True)
        return pd.DataFrame(columns=["column", "check", "failure_case"])
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        return fc[["column", "check", "failure_case"]]
