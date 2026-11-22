"""Evaluate the RAG assistant on a fixed question set (eval/rag_questions.jsonl).

Each question has: expected_source (file that should be retrieved/cited), key_facts
(strings a correct answer must contain), or answerable=false for out-of-scope questions.
A question PASSES when:
  answerable:   expected source retrieved AND cited AND >= 50% of key facts present
  unanswerable: the assistant declines (NOT_FOUND)"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from auditlens.genai.rag import NOT_FOUND, RAGPipeline


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9%$.]+", " ", s.lower()).strip()


def load_questions(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate(rag: RAGPipeline, questions: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for q in questions:
        a = rag.ask(q["question"])
        declined = NOT_FOUND.lower()[:30] in a.answer.lower()
        if q.get("answerable", True):
            retrieved = any(r["source"] == q["expected_source"] for r in a.retrieved)
            cited = any(c["source"] == q["expected_source"] for c in a.citations)
            facts = q.get("key_facts", [])
            hit = [f for f in facts if _norm(f) in _norm(a.answer)]
            coverage = len(hit) / len(facts) if facts else 1.0
            passed = retrieved and cited and coverage >= 0.5
        else:
            retrieved = cited = None
            coverage = None
            passed = declined
        rows.append({"id": q["id"], "question": q["question"], "answerable": q.get("answerable", True),
                     "retrieval_hit": retrieved, "cited_expected": cited, "fact_coverage": coverage,
                     "declined": declined, "passed": passed, "answer": a.answer})
    df = pd.DataFrame(rows)
    ans = df[df.answerable]
    summary = {
        "n": len(df), "accuracy": float(df.passed.mean()),
        "retrieval_hit_rate": float(ans.retrieval_hit.mean()) if len(ans) else None,
        "citation_rate": float(ans.cited_expected.mean()) if len(ans) else None,
        "mean_fact_coverage": float(ans.fact_coverage.mean()) if len(ans) else None,
        "unanswerable_declined": float(df[~df.answerable].declined.mean()) if (~df.answerable).any() else None,
        "model": rag.llm.name, "embedder": rag.embedder.name,
    }
    return df, summary
