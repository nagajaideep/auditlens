"""LLM client abstraction. Providers:
  gemini      - Google Gemini via the google-genai SDK (GEMINI_API_KEY, optional GEMINI_MODEL)
  extractive  - no LLM: returns the most relevant sentences from the context (offline / CI)
'auto' uses Gemini when GEMINI_API_KEY (or GOOGLE_API_KEY) is set, else extractive."""
from __future__ import annotations

import os
import re


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:  # pragma: no cover
        raise NotImplementedError


class GeminiLLM(BaseLLM):
    def __init__(self, model: str | None = None):
        from google import genai
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.name = f"gemini:{self.model}"

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens,
                                               temperature=0.1),
        )
        text = (resp.text or "").strip()
        if not text:
            raise LLMError(f"empty response from {self.model} (finish reason: "
                           f"{resp.candidates[0].finish_reason if resp.candidates else 'unknown'})")
        return text


class ExtractiveLLM(BaseLLM):
    """Deterministic fallback: picks context sentences that overlap the question most."""
    name = "extractive"
    STOP = set(["the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "is", "are", "be", "what", "which", "who", "how", "does", "do", "when", "should", "must", "with", "by", "at", "from", "as", "that", "this", "it", "its", "can", "our", "we", "firm", "firms", "policy", "policies", "client", "clients", "audit"])

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        q = re.search(r"QUESTION:\s*(.*)", user, re.DOTALL)
        question = q.group(1).strip() if q else user
        qterms = {w for w in re.findall(r"[a-z0-9$%]+", question.lower()) if w not in self.STOP}
        if "TRANSACTION:" in user:  # narration request
            return _template_narrative(user)
        best = []
        for m in re.finditer(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\nQUESTION:|\Z)", user, re.DOTALL):
            n, body = m.group(1), m.group(2)
            for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
                words = set(re.findall(r"[a-z0-9$%]+", sent.lower()))
                score = len(words & qterms) / (len(qterms) or 1)
                if len(sent) > 25:
                    best.append((score, sent.strip(), n))
        best.sort(key=lambda t: -t[0])
        if not best or best[0][0] < 0.34:
            return "I could not find this in the provided documents."
        picked = best[:2]
        return " ".join(f"{s} [{n}]" for _, s, n in picked)


def _template_narrative(user: str) -> str:
    reasons = re.findall(r"- (.*?) \((raises risk|lowers risk)", user)
    score = re.search(r"fraud_score:\s*([0-9.]+)", user)
    ups = [r for r, d in reasons if d == "raises risk"][:3]
    s = f"This transaction scored {float(score.group(1)):.0%} risk" if score else "This transaction was flagged"
    if ups:
        s += " mainly because: " + "; ".join(ups) + "."
    return s + " An auditor should verify the counterparty and supporting documents before clearing it."


def get_llm(provider: str = "auto", model: str | None = None) -> BaseLLM:
    if provider == "auto":
        has_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        provider = "gemini" if has_key else "extractive"
    if provider == "gemini":
        return GeminiLLM(model)
    return ExtractiveLLM()
