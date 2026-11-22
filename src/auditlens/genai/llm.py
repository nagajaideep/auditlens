"""LLM client abstraction. Providers:
  anthropic     - Claude via the official Anthropic SDK (ANTHROPIC_API_KEY)
  azure_openai  - Azure OpenAI deployment (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT)
  openai        - OpenAI API (OPENAI_API_KEY, OPENAI_MODEL)
  extractive    - no LLM: returns the most relevant sentences from the context (offline / CI)
'auto' picks the first provider whose credentials are present, else extractive."""
from __future__ import annotations

import os
import re


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:  # pragma: no cover
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str = "claude-opus-5-5"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        # Server-side fallback re-runs the request on another model if a safety
        # classifier declines it; effort=low keeps short Q&A cheap and fast.
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class OpenAICompatLLM(BaseLLM):
    def __init__(self, azure: bool = False):
        if azure:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                      api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                      api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
            self.model = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        else:
            from openai import OpenAI
            self.client = OpenAI()
            self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.name = ("azure_openai:" if azure else "openai:") + self.model

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        r = self.client.chat.completions.create(model=self.model, max_tokens=max_tokens,
                                                messages=[{"role": "system", "content": system},
                                                          {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()


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


def get_llm(provider: str = "auto", anthropic_model: str = "claude-opus-5-5") -> BaseLLM:
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider = "azure_openai"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "extractive"
    if provider == "anthropic":
        return AnthropicLLM(anthropic_model)
    if provider in ("azure_openai", "openai"):
        return OpenAICompatLLM(azure=provider == "azure_openai")
    return ExtractiveLLM()
