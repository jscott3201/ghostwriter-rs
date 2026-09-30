-- Guarded record commands. Historical envelopes and history rows are left byte-for-byte intact.
-- One receipt covers a whole command, including an initial record's existing generation facts.
CREATE TABLE record_mutations (
    mutation_id TEXT PRIMARY KEY NOT NULL,
    record_id TEXT NOT NULL REFERENCES records(record_id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version = 1),
    kind TEXT NOT NULL CHECK (kind IN ('insert', 'transition', 'insert_transition', 'publication')),
    history_start INTEGER NOT NULL CHECK (history_start >= 0),
    history_count INTEGER NOT NULL CHECK (history_count >= 0),
    committed_at TEXT NOT NULL
);
CREATE INDEX idx_record_mutations_record ON record_mutations(record_id);

-- NULL identifies older relational history, whose initial envelope facts were not mirrored.
ALTER TABLE lifecycle_history ADD COLUMN mutation_id TEXT REFERENCES record_mutations(mutation_id);
ALTER TABLE lifecycle_history ADD COLUMN history_ordinal INTEGER CHECK (history_ordinal >= 0);
ALTER TABLE lifecycle_history ADD COLUMN attempt INTEGER CHECK (attempt BETWEEN 0 AND 4294967295);
CREATE UNIQUE INDEX idx_lifecycle_history_ordinal
    ON lifecycle_history(record_id, history_ordinal) WHERE history_ordinal IS NOT NULL;
