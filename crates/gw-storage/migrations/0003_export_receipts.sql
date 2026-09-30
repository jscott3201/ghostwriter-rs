-- Durable publication intent, including empty artifacts. Paths remain local to this ledger.
CREATE TABLE export_receipts (
    publication_id TEXT PRIMARY KEY NOT NULL,
    artifact_id TEXT NOT NULL,
    destination TEXT NOT NULL,
    purpose TEXT NOT NULL,
    artifact_json TEXT NOT NULL,
    members_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('prepared', 'acknowledged')),
    prepared_at TEXT NOT NULL,
    acknowledged_at TEXT
);
CREATE INDEX idx_export_receipts_pending ON export_receipts(destination, purpose, state);
