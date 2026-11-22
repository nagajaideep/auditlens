-- AuditLens PostgreSQL schema (PostgreSQL 15+, pgvector optional)
CREATE TABLE IF NOT EXISTS transactions_raw (
    id              BIGSERIAL PRIMARY KEY,
    step            INTEGER NOT NULL,
    type            VARCHAR(16) NOT NULL,
    amount          NUMERIC(18,2) NOT NULL,
    nameorig        VARCHAR(32) NOT NULL,
    oldbalanceorg   NUMERIC(18,2),
    newbalanceorig  NUMERIC(18,2),
    namedest        VARCHAR(32) NOT NULL,
    oldbalancedest  NUMERIC(18,2),
    newbalancedest  NUMERIC(18,2),
    isfraud         SMALLINT NOT NULL,
    isflaggedfraud  SMALLINT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tx_step ON transactions_raw (step);
CREATE INDEX IF NOT EXISTS ix_tx_dest ON transactions_raw (namedest);
CREATE INDEX IF NOT EXISTS ix_tx_type ON transactions_raw (type);

-- Scores written by the nightly batch job
CREATE TABLE IF NOT EXISTS transaction_scores (
    tx_id          BIGINT NOT NULL,
    scored_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version  VARCHAR(64) NOT NULL,
    fraud_score    DOUBLE PRECISION NOT NULL,
    anomaly_score  DOUBLE PRECISION,
    flagged        BOOLEAN NOT NULL,
    PRIMARY KEY (tx_id, model_version)
)
