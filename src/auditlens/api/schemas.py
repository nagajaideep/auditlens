from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    step: int = Field(1, ge=1, description="Hour index since start of the ledger")
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    amount: float = Field(..., ge=0)
    nameOrig: str = Field("C0", pattern=r"^C\d+$")
    oldbalanceOrg: float = Field(0, ge=0)
    newbalanceOrig: float = Field(0, ge=0)
    nameDest: str = Field("C1", pattern=r"^[CM]\d+$")
    oldbalanceDest: float = Field(0, ge=0)
    newbalanceDest: float = Field(0, ge=0)

    model_config = {"json_schema_extra": {"example": {
        "step": 1, "type": "TRANSFER", "amount": 181.0, "nameOrig": "C1305486145", "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0, "nameDest": "C553264065", "oldbalanceDest": 0.0, "newbalanceDest": 0.0}}}


class PredictRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)
    explain: bool = True
    narrate: bool = False


class Reason(BaseModel):
    feature: str
    description: str
    value: float
    shap: float
    direction: str


class Prediction(BaseModel):
    fraud_score: float
    flagged: bool
    risk_band: str
    anomaly_score: float
    anomaly_flag: bool
    reasons: list[Reason] | None = None
    narrative: str | None = None


class PredictResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


class ForecastRequest(BaseModel):
    group: str = Field("CASH_OUT", description="Account group / transaction type to forecast")
    horizon: int = Field(7, ge=1, le=90)
    method: Literal["sarima", "prophet"] = "sarima"


class ForecastPoint(BaseModel):
    period: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    group: str
    freq: str
    method: str
    forecast: list[ForecastPoint]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class Citation(BaseModel):
    n: int
    source: str
    section: str
    page: int | None = None
    score: float
    citation: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[Citation]
    model: str
