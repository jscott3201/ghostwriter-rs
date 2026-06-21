-- Persist the seed partition manifest that makes run resume/replay sound.
--
-- The engine derives work by `(prompt_index % shard_count)` and resumes by a
-- per-(run_id, shard) cursor. These columns pin the effective shard count and
-- ordered prompts hash used for the original launch.
ALTER TABLE runs ADD COLUMN shard_count INTEGER;
ALTER TABLE runs ADD COLUMN prompts_hash TEXT;
