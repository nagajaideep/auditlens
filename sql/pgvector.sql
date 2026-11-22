-- Optional: store document chunks in PostgreSQL with pgvector instead of Chroma.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS doc_chunks (
    id        TEXT PRIMARY KEY,
    source    TEXT NOT NULL,
    section   TEXT,
    page      INTEGER,
    content   TEXT NOT NULL,
    embedding vector(384)          -- all-MiniLM-L6-v2 dimension
);
CREATE INDEX IF NOT EXISTS ix_doc_chunks_emb ON doc_chunks USING hnsw (embedding vector_cosine_ops);
-- Query: SELECT id, source, section, content, 1 - (embedding <=> :q) AS score
--        FROM doc_chunks ORDER BY embedding <=> :q LIMIT 4;
