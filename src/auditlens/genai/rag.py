"""Retrieval-augmented question answering over audit policy documents, with citations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from auditlens.config import get_config, resolve
from auditlens.genai.embed import get_embedder
from auditlens.genai.ingest import chunk_documents
from auditlens.genai.llm import get_llm
from auditlens.genai.store import MemoryStore

SYSTEM_PROMPT = """You are AuditLens, an assistant for financial auditors.
Answer ONLY from the numbered context passages you are given. After every claim, cite the passage(s)
it came from in square brackets, e.g. [1] or [2][3]. If the context does not contain the answer,
reply exactly: "I could not find this in the provided documents." Do not use outside knowledge.
Be concise: at most 5 sentences. Quote numbers (thresholds, days, percentages) exactly as written."""

NOT_FOUND = "I could not find this in the provided documents."


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


class RAGPipeline:
    def __init__(self, store, embedder, llm, top_k: int = 4, min_score: float = 0.05):
        self.store, self.embedder, self.llm, self.top_k, self.min_score = store, embedder, llm, top_k, min_score

    @classmethod
    def build(cls, docs_dir: str | Path | None = None, cfg: dict | None = None, llm=None, persist: bool = True):
        cfg = (cfg or get_config())["genai"]
        docs_dir = resolve(docs_dir or cfg["docs_dir"])
        chunks = chunk_documents(docs_dir, cfg["chunk_size"], cfg["chunk_overlap"])
        if not chunks:
            raise ValueError(f"no documents found in {docs_dir}")
        emb = get_embedder(cfg["embedder"], cfg["embed_model"]).fit([c.text for c in chunks])
        E = emb.encode([c.text for c in chunks])
        store = _make_store(cfg, persist)
        store.add(chunks, E)
        if persist and isinstance(store, MemoryStore):
            store.save(resolve(cfg["chroma_dir"]) / "memory")
        return cls(store, emb, llm or get_llm(cfg["llm_provider"], cfg["anthropic_model"]), cfg["top_k"])

    def retrieve(self, question: str):
        q = self.embedder.encode([question])[0]
        return [(c, s) for c, s in self.store.search(q, self.top_k) if s >= self.min_score]

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        retrieved = [{"n": i + 1, "source": c.source, "section": c.section, "page": c.page, "score": round(s, 3),
                      "citation": c.citation} for i, (c, s) in enumerate(hits)]
        if not hits:
            return Answer(question, NOT_FOUND, [], [], self.llm.name)
        context = "\n\n".join(f"[{i + 1}] ({c.citation})\n{c.text}" for i, (c, _) in enumerate(hits))
        prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        text = self.llm.complete(SYSTEM_PROMPT, prompt, max_tokens=1200)
        used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
        cites = [retrieved[n - 1] for n in used]
        return Answer(question, text, cites, retrieved, self.llm.name)


def _make_store(cfg: dict, persist: bool):
    if persist:
        try:
            from auditlens.genai.store import ChromaStore
            s = ChromaStore(resolve(cfg["chroma_dir"]), cfg["collection"])
            s.reset()
            return s
        except Exception:
            pass
    return MemoryStore()
