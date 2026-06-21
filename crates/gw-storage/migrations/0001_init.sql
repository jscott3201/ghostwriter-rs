-- gw-storage v1 schema: the reproducible relational data plane (DATA-SCHEMA §4.2).
--
-- SQLite is the v1 source of truth for run/lifecycle/cache state. This is a minimal
-- runs + records + lifecycle_history + cache + checkpoints schema; the full provenance
-- DAG (nodes/edges, recursive-CTE lineage) is a deferred follow-up (DATA-SCHEMA §4.5).
--
-- All timestamps are RFC 3339 strings for byte-reproducibility (matching the schema
-- contract's `String` timestamps); the harness supplies them, not SQLite.

-- One generation run. `config_json` is a snapshot of the gw_schema::Config in force;
-- `budget_usd` mirrors the budget cap so a run is auditable without re-parsing config.
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY NOT NULL,
    config_json TEXT NOT NULL,
    budget_usd  REAL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- The canonical record store. `record_json` is the full TrainingRecord envelope; the
-- remaining columns are indexed projections for cheap columnar filters (verdict,
-- judge aggregate) and dedup lookups (record_hash / prompt_hash). [INVARIANT d, e]
CREATE TABLE IF NOT EXISTS records (
    record_id       TEXT PRIMARY KEY NOT NULL,
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    lifecycle_state TEXT NOT NULL,
    verdict         TEXT,
    judge_aggregate REAL,
    record_hash     TEXT NOT NULL,
    prompt_hash     TEXT NOT NULL,
    record_json     TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_run_id      ON records (run_id);
CREATE INDEX IF NOT EXISTS idx_records_state       ON records (lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_records_verdict     ON records (verdict);
CREATE INDEX IF NOT EXISTS idx_records_aggregate   ON records (judge_aggregate);
CREATE INDEX IF NOT EXISTS idx_records_record_hash ON records (record_hash);
CREATE INDEX IF NOT EXISTS idx_records_prompt_hash ON records (prompt_hash);

-- Event-sourced lifecycle transitions. Append-only: each `advance_lifecycle` writes one
-- row here AND updates records.lifecycle_state in a single transaction (DATA-SCHEMA §6.1).
CREATE TABLE IF NOT EXISTS lifecycle_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL REFERENCES records(record_id) ON DELETE CASCADE,
    state     TEXT NOT NULL,
    at        TEXT NOT NULL,
    detail    TEXT
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_history_record_id ON lifecycle_history (record_id);

-- The content-addressed "never re-spend" call cache (DATA-SCHEMA §5.2). Key is
-- (content_hash, kind, model, rubric_id); `value_json` is the cached teacher/verify/judge
-- result. `rubric_id` is normalized to '' (not NULL) so the composite key is well-defined.
CREATE TABLE IF NOT EXISTS cache (
    content_hash TEXT NOT NULL,
    kind         TEXT NOT NULL,
    model        TEXT NOT NULL,
    rubric_id    TEXT NOT NULL DEFAULT '',
    value_json   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (content_hash, kind, model, rubric_id)
);

-- Per-(run, shard) crash-recovery checkpoint. `cursor_json` is the opaque resume cursor;
-- `state` is the furthest committed lifecycle state for the shard (DATA-SCHEMA §6.2).
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    shard       INTEGER NOT NULL,
    state       TEXT NOT NULL,
    cursor_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (run_id, shard)
);
