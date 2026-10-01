-- JSON null records the truthful absence of a generation manifest for reference imports.
ALTER TABLE runs ADD COLUMN run_kind TEXT NOT NULL DEFAULT 'generated' CHECK (run_kind IN ('generated', 'reviewed_reference'));
CREATE TABLE reference_registrations (
    registration_id TEXT PRIMARY KEY NOT NULL,
    catalogue_id TEXT NOT NULL UNIQUE,
    capture_json TEXT NOT NULL,
    registered_at TEXT NOT NULL
);
CREATE TABLE reference_batches (
    batch_id TEXT PRIMARY KEY NOT NULL REFERENCES runs(run_id),
    registration_id TEXT NOT NULL UNIQUE REFERENCES reference_registrations(registration_id),
    committed_at TEXT NOT NULL,
    member_count INTEGER NOT NULL CHECK (member_count = 112)
);
CREATE TABLE reference_members (
    batch_id TEXT NOT NULL REFERENCES reference_batches(batch_id),
    ordinal INTEGER NOT NULL,
    member_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train','validation','test')),
    record_id TEXT UNIQUE REFERENCES records(record_id),
    origin_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, ordinal),
    UNIQUE (batch_id, member_id),
    CHECK ((split = 'train' AND record_id IS NOT NULL) OR (split != 'train' AND record_id IS NULL))
);
