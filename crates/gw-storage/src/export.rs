//! Arrow/Parquet columnar export of admitted records → a [`gw_schema::ExportManifest`].
//!
//! This is a **columnar dump, not template rendering**: each record is projected into a flat
//! set of Arrow columns (ids, hashes, the messages + reasoning as JSON-string columns, scores)
//! and written to Parquet. Full target-template rendering (Gemma-4 / ChatML / ShareGPT byte
//! shapes) belongs to `gw-format`; here we store `messages` + `reasoning` columns and record the
//! [`CotPolicy`] / [`TrlFormat`] in the manifest, so a downstream renderer has everything it
//! needs without re-reading SQLite.
//!
//! The Arrow + Parquet writers are **synchronous** and do file I/O, so [`export_parquet`] runs
//! them inside [`tokio::task::spawn_blocking`] to keep the async runtime unblocked. Tests write
//! to an in-memory `Vec<u8>` buffer (the Parquet `ArrowWriter` accepts any [`std::io::Write`]).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use blake3::Hasher;
use gw_schema::{
    Content, CotPolicy, ExportManifest, Message, MultiTurnLoss, TrainingRecord, TrlFormat, Verdict,
};
use parquet::arrow::ArrowWriter;

use crate::error::{Result, StorageError};

/// The flat columnar projection of one record (one Parquet row). `messages_json` /
/// `reasoning_json` are JSON-string columns to dodge deep-nesting schema churn (DATA-SCHEMA §4.1).
struct Projected {
    record_id: String,
    training_area: String,
    record_hash: String,
    prompt_hash: String,
    verdict: Option<String>,
    judge_aggregate: Option<f64>,
    reasoning_tokens: u32,
    messages_json: String,
    reasoning_json: String,
}

/// Project a record into its export row. `messages_json` is the clean conversation (content
/// only); `reasoning_json` is the per-assistant-turn reasoning, kept in a SEPARATE column so the
/// loss-region policy is applied at render time, never here (INVARIANT a).
///
/// `record_hash` falls back to a freshly computed content hash when the envelope's own hash is
/// empty, so an externally-constructed record can never export `record_hash = ""`.
fn project(rec: &TrainingRecord) -> Result<Projected> {
    let clean: Vec<serde_json::Value> = rec
        .messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": content_text(&m.content) }))
        .collect();
    let reasoning: Vec<Option<String>> = rec.messages.iter().map(|m| m.reasoning.clone()).collect();
    Ok(Projected {
        record_id: rec.record_id.clone(),
        training_area: rec.training_area.clone(),
        record_hash: resolved_record_hash(rec)?,
        prompt_hash: if rec.hashes.prompt_hash.is_empty() {
            crate::cache::prompt_hash(&rec.messages)?
        } else {
            rec.hashes.prompt_hash.clone()
        },
        verdict: rec
            .judging
            .verdict
            .and_then(|v| serde_json::to_value(v).ok())
            .and_then(|v| v.as_str().map(str::to_owned)),
        judge_aggregate: rec.judging.aggregate,
        reasoning_tokens: rec.cost.reasoning_tokens,
        messages_json: serde_json::to_string(&clean)?,
        reasoning_json: serde_json::to_string(&reasoning)?,
    })
}

/// The record's content hash: its envelope `hashes.record_hash` if populated, else a freshly
/// computed one. Used for both the per-row column and the shard `build_inputs_hash` so neither
/// can be content-independent for an externally-constructed record.
fn resolved_record_hash(rec: &TrainingRecord) -> Result<String> {
    if rec.hashes.record_hash.is_empty() {
        crate::cache::record_hash(rec)
    } else {
        Ok(rec.hashes.record_hash.clone())
    }
}

/// Flatten a [`Content`] to plain text (multimodal parts serialize structurally).
fn content_text(c: &Content) -> String {
    match c {
        Content::Text(t) => t.clone(),
        Content::Parts(parts) => serde_json::to_string(parts).unwrap_or_default(),
    }
}

/// The fixed Arrow schema for the export projection.
fn export_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("record_id", DataType::Utf8, false),
        Field::new("training_area", DataType::Utf8, false),
        Field::new("record_hash", DataType::Utf8, false),
        Field::new("prompt_hash", DataType::Utf8, false),
        Field::new("verdict", DataType::Utf8, true),
        Field::new("judge_aggregate", DataType::Float64, true),
        Field::new("reasoning_tokens", DataType::UInt32, false),
        Field::new("messages_json", DataType::Utf8, false),
        Field::new("reasoning_json", DataType::Utf8, false),
    ]))
}

/// Assemble the projected rows into a single Arrow [`RecordBatch`].
fn build_batch(rows: &[Projected]) -> Result<RecordBatch> {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.record_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.training_area.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.record_hash.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.prompt_hash.as_str()),
        )),
        Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.verdict.clone()),
        )),
        Arc::new(Float64Array::from_iter(
            rows.iter().map(|r| r.judge_aggregate),
        )),
        Arc::new(UInt32Array::from_iter_values(
            rows.iter().map(|r| r.reasoning_tokens),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.messages_json.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.reasoning_json.as_str()),
        )),
    ];
    Ok(RecordBatch::try_new(export_schema(), columns)?)
}

/// Encode the rows to an in-memory Parquet byte buffer (sync; called from `spawn_blocking`).
fn encode_parquet(rows: &[Projected]) -> Result<Vec<u8>> {
    let batch = build_batch(rows)?;
    let mut buf: Vec<u8> = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(buf)
}

/// Export `records` to a Parquet shard at `dst`, returning a [`gw_schema::ExportManifest`].
///
/// Only ADMITTED records (`judging.verdict == Admit`) are projected; the manifest's `n_records`
/// counts the input set and `n_admitted` counts what was written. `build_inputs_hash` is the
/// BLAKE3 of the sorted admitted `record_hash`es (a stable content hash of the shard,
/// independent of row order). All projection, hashing, Parquet encoding, AND the file write run
/// under [`tokio::task::spawn_blocking`] so nothing serializes or hashes on the async runtime
/// thread.
///
/// `cot` and `target` are recorded in the manifest only; this function does NOT render templates
/// or move reasoning into a loss region (that is `gw-format`'s job) — it stores the messages and
/// reasoning columns and the policy alongside them.
///
/// # Errors
/// Returns [`StorageError`] on a serialization, Arrow/Parquet encode, or filesystem I/O failure.
pub async fn export_parquet(
    records: &[TrainingRecord],
    target: TrlFormat,
    cot: CotPolicy,
    dst: impl AsRef<Path>,
) -> Result<ExportManifest> {
    let dst_path: PathBuf = dst.as_ref().to_path_buf();
    let (bytes, manifest) = encode_admitted(records, target, cot).await?;
    tokio::task::spawn_blocking(move || std::fs::write(&dst_path, bytes)).await??;
    Ok(manifest)
}

/// Encode the admitted records to a Parquet byte buffer + manifest, entirely off the async
/// runtime. Used by tests (in-memory) and by [`export_parquet`] (then written to disk).
///
/// # Errors
/// Returns [`StorageError`] on a serialization or Arrow/Parquet failure.
pub async fn export_parquet_bytes(
    records: &[TrainingRecord],
    target: TrlFormat,
    cot: CotPolicy,
) -> Result<(Vec<u8>, ExportManifest)> {
    encode_admitted(records, target, cot).await
}

/// Filter to admitted records, then project + shard-hash + Parquet-encode them inside a single
/// `spawn_blocking` (so no serialization/hashing/encoding touches the async runtime thread).
async fn encode_admitted(
    records: &[TrainingRecord],
    target: TrlFormat,
    cot: CotPolicy,
) -> Result<(Vec<u8>, ExportManifest)> {
    let n_records = records.len() as u64;
    // Clone only the admitted subset into the blocking task.
    let admitted: Vec<TrainingRecord> = records
        .iter()
        .filter(|r| r.judging.verdict == Some(Verdict::Admit))
        .cloned()
        .collect();

    let (bytes, build_inputs_hash, n_admitted) =
        tokio::task::spawn_blocking(move || -> Result<(Vec<u8>, String, u64)> {
            let rows: Vec<Projected> = admitted.iter().map(project).collect::<Result<_>>()?;
            let build_inputs_hash = shard_content_hash(&rows);
            let bytes = encode_parquet(&rows)?;
            Ok((bytes, build_inputs_hash, admitted.len() as u64))
        })
        .await
        .map_err(StorageError::from)??;

    let manifest = ExportManifest {
        target,
        cot_policy: cot,
        dataset_version: None,
        hub_commit_sha: None,
        n_records,
        n_admitted,
        decontam_index_id: None,
        build_inputs_hash,
        multi_turn_loss: MultiTurnLoss::default(),
        diversity: None,
    };
    Ok((bytes, manifest))
}

/// BLAKE3 of the sorted per-row `record_hash`es — a stable, order-independent content hash of the
/// shard for the manifest's `build_inputs_hash`. Reads the already-resolved (never empty) hashes
/// off the projected rows.
fn shard_content_hash(rows: &[Projected]) -> String {
    let mut hashes: Vec<&str> = rows.iter().map(|r| r.record_hash.as_str()).collect();
    hashes.sort_unstable();
    let mut h = Hasher::new();
    for rh in hashes {
        h.update(rh.as_bytes());
        h.update(b"\n");
    }
    h.finalize().to_hex().to_string()
}

/// Render a [`Message`] slice's clean text (helper exposed for callers projecting outside the
/// batch path). Mirrors the in-batch projection.
#[must_use]
pub fn clean_messages_json(messages: &[Message]) -> String {
    let clean: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": content_text(&m.content) }))
        .collect();
    serde_json::to_string(&clean).unwrap_or_default()
}
