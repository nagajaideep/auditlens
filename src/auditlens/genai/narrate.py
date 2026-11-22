"""LLM-written plain-English explanation of why a transaction was flagged,
grounded in its SHAP reasons (the LLM is told not to invent other reasons)."""
from __future__ import annotations

from auditlens.genai.llm import BaseLLM, get_llm

SYSTEM = """You write short explanations of flagged financial transactions for auditors.
Use ONLY the facts and model reasons given. Do not speculate about intent or accuse anyone.
Write 2-4 plain-English sentences: why it was flagged, then what the auditor should check next."""


def narrate(tx: dict, fraud_score: float, reasons: list[dict], llm: BaseLLM | None = None) -> str:
    llm = llm or get_llm()
    lines = [f"- {r['description']} = {r['value']:.2f} ({r['direction']}, shap={r['shap']:+.3f})" for r in reasons]
    user = (f"TRANSACTION: type={tx.get('type')} amount={tx.get('amount')} "
            f"sender_balance_before={tx.get('oldbalanceOrg')} sender_balance_after={tx.get('newbalanceOrig')} "
            f"receiver_balance_before={tx.get('oldbalanceDest')} receiver_balance_after={tx.get('newbalanceDest')}\n"
            f"fraud_score: {fraud_score:.3f}\nMODEL REASONS (most important first):\n" + "\n".join(lines))
    try:
        return llm.complete(SYSTEM, user, max_tokens=400)
    except Exception:
        from auditlens.genai.llm import ExtractiveLLM
        return ExtractiveLLM().complete(SYSTEM, user)
