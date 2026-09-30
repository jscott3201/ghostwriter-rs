-- Physical transmissions and launch-specific cooperative coverage.
CREATE TABLE model_launches (
    launch_id TEXT PRIMARY KEY NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    coverage_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX model_launches_run ON model_launches(run_id);
CREATE TABLE model_attempts (
    attempt_id TEXT PRIMARY KEY NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    launch_id TEXT NOT NULL REFERENCES model_launches(launch_id),
    receipt_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX model_attempts_run ON model_attempts(run_id);
