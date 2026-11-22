"""Vector stores: Chroma (persistent, default) and an in-memory NumPy store.
A pgvector table definition is provided in sql/pgvector.sql for Postgres deployments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from auditlens.genai.ingest import Chunk


class MemoryStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.E: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.chunks += chunks
        self.E = embeddings if self.E is None else np.vstack([self.E, embeddings])

    def search(self, q: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        sims = self.E @ q
        idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx]

    def save(self, path: str | Path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.E)
        (path / "chunks.json").write_text(json.dumps([c.__dict__ for c in self.chunks]))

    @classmethod
    def load(cls, path: str | Path) -> MemoryStore:
        path = Path(path)
        s = cls()
        s.E = np.load(path / "embeddings.npy")
        s.chunks = [Chunk(**d) for d in json.loads((path / "chunks.json").read_text())]
        return s


class ChromaStore:
    def __init__(self, path: str | Path, collection: str):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(path))
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def reset(self):
        name = self.col.name
        self.client.delete_collection(name)
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.col.add(ids=[c.id for c in chunks], embeddings=embeddings.tolist(), documents=[c.text for c in chunks],
                     metadatas=[{"source": c.source, "section": c.section, "page": c.page or 0} for c in chunks])

    def search(self, q: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        r = self.col.query(query_embeddings=[q.tolist()], n_results=k)
        out = []
        for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0], strict=False):
            out.append((Chunk(cid, doc, meta["source"], meta["section"], meta["page"] or None), 1 - float(dist)))
        return out
