-- Operational policy is separate from immutable generation identity.
CREATE TABLE run_accounting (
    run_id TEXT PRIMARY KEY NOT NULL REFERENCES runs(run_id),
    policy_json TEXT NOT NULL,
    history_complete INTEGER NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1
);
