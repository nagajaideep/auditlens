"""Embedding backends. sentence-transformers is the default; a TF-IDF + SVD
fallback keeps tests and air-gapped environments working without model downloads."""
from __future__ import annotations

import numpy as np


class Embedder:
    name = "base"

    def fit(self, texts: list[str]) -> Embedder:
        return self

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))


class TfidfEmbedder(Embedder):
    name = "tfidf-svd"

    def __init__(self, dim: int = 256):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.dim = dim
        self.svd = None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        M = self.vec.fit_transform(texts)
        k = min(self.dim, M.shape[1] - 1, M.shape[0] - 1)
        if k >= 2:
            self.svd = TruncatedSVD(k, random_state=0).fit(M)
        return self

    def encode(self, texts):
        M = self.vec.transform(texts)
        E = self.svd.transform(M) if self.svd is not None else M.toarray()
        return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)


def get_embedder(kind: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    if kind in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception:  # missing package or no network to fetch weights
            if kind == "sentence-transformers":
                raise
    return TfidfEmbedder()
