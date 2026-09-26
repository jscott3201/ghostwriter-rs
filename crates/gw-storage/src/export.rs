//! Arrow/Parquet columnar export of admitted records → a [`gw_schema::ExportManifest`].
//!
//! This is a **columnar dump, not template rendering**: each record is projected into a flat
//! set of Arrow columns (ids, hashes, the conversation as one canonical JSON-string column,
//! scores) and written to Parquet. Full target-template rendering (Gemma-4 / ChatML / ShareGPT
//! byte shapes) belongs to `gw-format`; here we store the `messages` column and record the
//! [`CotPolicy`] / [`TrlFormat`] in the manifest, so a downstream renderer has everything it needs
//! without re-reading SQLite.
//!
//! ## One conversation column, losslessly
//!
//! `messages_json` holds the canonical [`Message`] slice serialized by serde — the SAME bytes a
//! `records.record_json` envelope carries for those turns. Nothing is flattened, dropped or
//! re-ordered on the way out, so `content: null` stays `null` (distinct from `""`), multimodal
//! parts stay parts, and `tool_calls` / `tool_call_id` / `name` / `reasoning` /
//! `reasoning_details` (including a retained `raw_arguments` wire text) all survive. A consumer
//! decodes the column straight back into `Vec<Message>`; there is no second column that has to
//! agree with the first by index.
//!
//! ## Reading a shard
//!
//! [`ExportSchemaVersion`] names the column contract a shard was written under and is recorded in
//! every manifest. [`export_parquet_bytes`] carries a worked read-back example. v1 shards (the
//! historical, lossy `{role, content}` + parallel `reasoning_json` pair) are still readable, but
//! only by a text-only consumer — see the enum's docs for the exact break.
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
    CotPolicy, ExportManifest, ExportSchemaVersion, Message, MultiTurnLoss, TrainingRecord,
    TrlFormat, Verdict,
};
use parquet::arrow::ArrowWriter;

use crate::error::{Result, StorageError};

/// The flat columnar projection of one record (one Parquet row). `messages_json` is a JSON-string
/// column to dodge deep-nesting schema churn (DATA-SCHEMA §4.1); it holds the whole canonical
/// conversation, so there is no parallel per-turn column that can drift out of alignment with it.
struct Projected {
    record_id: String,
    training_area: String,
    record_hash: String,
    prompt_hash: String,
    verdict: Option<String>,
    judge_aggregate: Option<f64>,
    reasoning_tokens: u32,
    messages_json: String,
}

/// Project a record into its export row. `messages_json` is the canonical conversation, serialized
/// straight from [`TrainingRecord::messages`] — every structural field kept (INVARIANT a: reasoning
/// stays a sibling of content, never inlined; INVARIANT i: the `tool_call_id` result link travels
/// with its turn).
///
/// `record_hash` falls back to a freshly computed content hash when the envelope's own hash is
/// empty, so an externally-constructed record can never export `record_hash = ""`.
fn project(rec: &TrainingRecord) -> Result<Projected> {
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
        messages_json: canonical_messages_json(&rec.messages)?,
    })
}

/// Serialize a conversation to the canonical `messages_json` payload: the serde form of the
/// [`Message`] slice, in order.
///
/// This is the SINGLE conversation policy on this path — the Parquet column and
/// [`clean_messages_json`] both call it, so a caller projecting outside the batch path cannot end
/// up with a second, quietly different (or lossy) shape. Deserializing the result yields a
/// byte-equal `Vec<Message>`.
///
/// # Errors
/// Returns [`StorageError::Serde`] if a message fails to serialize.
fn canonical_messages_json(messages: &[Message]) -> Result<String> {
    Ok(serde_json::to_string(messages)?)
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

/// The fixed Arrow schema for the export projection. Every column is non-nullable except the two
/// that genuinely can be absent (verdict / judge aggregate).
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
/// or move reasoning into a loss region (that is `gw-format`'s job) — it stores the conversation
/// column and the policy alongside it. The written shard's reader contract is
/// [`ExportSchemaVersion::CURRENT`]; read the file back as shown on
/// [`export_parquet_bytes`].
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
/// # Reading a shard back
///
/// A consumer needs two things: the [`ExportSchemaVersion`] from the manifest, and the
/// `messages_json` column. There is no transform to reverse — the column decodes straight into
/// `Vec<Message>`, with the `tool_call_id` result link, the `content: null` / `""` distinction and
/// the retained `raw_arguments` wire text all intact:
///
/// ```
/// # use gw_schema::{
/// #     Content, CotPolicy, ExportSchemaVersion, FunctionCall, Message, Provenance, Role,
/// #     ToolCall, TrainingRecord, TrlFormat, Verdict,
/// # };
/// # use gw_storage::export_parquet_bytes;
/// # use arrow::array::{Array as _, StringArray};
/// # use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
/// # fn plain(role: Role, content: Content) -> Message {
/// #     Message { role, content, reasoning: None, reasoning_details: None, tool_calls: None, tool_call_id: None, name: None }
/// # }
/// # fn minimal_record() -> TrainingRecord {
/// #     let call = ToolCall { id: Some("read-a".into()), function: FunctionCall {
/// #         name: "read_file".into(), arguments: serde_json::json!({"start_line": 1}), raw_arguments: None } };
/// #     let calling_turn = Message { role: Role::Assistant, content: Content::Null, reasoning: Some("need the file".into()),
/// #         reasoning_details: None, tool_calls: Some(vec![call]), tool_call_id: None, name: None };
/// #     let result_turn = Message { role: Role::Tool, content: Content::Text("ok".into()), reasoning: None,
/// #         reasoning_details: None, tool_calls: None, tool_call_id: Some("read-a".into()), name: Some("read_file".into()) };
/// #     TrainingRecord {
/// #         record_id: "r1".into(),
/// #         schema_version: semver::Version::new(1, 0, 0),
/// #         dataset_version: None,
/// #         training_area: "toy".into(),
/// #         tags: vec![],
/// #         messages: vec![plain(Role::User, Content::Text("read it".into())), calling_turn, result_turn],
/// #         tools: None,
/// #         provenance: Provenance { run_id: "run-1".into(), parent_ids: vec![],
/// #             teacher: gw_schema::TeacherRef { provider: "openrouter".into(),
/// #                 slug: "toy/teacher".into(), served_by: None, model_card_revision: None },
/// #             user_synth_model: None, user_turn_kind: None, in_scope_safe: None,
/// #             judge_models: vec![], harness_version: "0.1.0".into(), git_commit: None },
/// #         generation: Default::default(),
/// #         verification_contract: None,
/// #         execution_evidence: None,
/// #         verification: Default::default(),
/// #         judging: gw_schema::Judging { verdict: Some(Verdict::Admit), aggregate: Some(0.9),
/// #             ..Default::default() },
/// #         reasoning_quality: None,
/// #         lifecycle: Default::default(),
/// #         hashes: Default::default(),
/// #         cost: Default::default(),
/// #     }
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let (bytes, manifest) = tokio::runtime::Runtime::new()?.block_on(export_parquet_bytes(
///     &[minimal_record()],
///     TrlFormat::Gemma4,
///     CotPolicy::Masked,
/// ))?;
/// assert_eq!(manifest.column_schema_version, ExportSchemaVersion::CURRENT);
///
/// // Read the one conversation column back and decode it into the canonical type.
/// let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes))?
///     .build()?;
/// let mut turns: Vec<Message> = Vec::new();
/// for batch in reader {
///     let batch = batch?;
///     let col = batch.column_by_name("messages_json").unwrap()
///         .as_any().downcast_ref::<StringArray>().unwrap();
///     for i in 0..col.len() {
///         turns.extend(serde_json::from_str::<Vec<Message>>(col.value(i))?);
///     }
/// }
/// // Identity, the null body and the CoT all survived the round trip.
/// assert_eq!(turns[1].content, Content::Null);
/// assert_eq!(turns[1].tool_calls.as_ref().unwrap()[0].id.as_deref(), Some("read-a"));
/// assert_eq!(turns[2].tool_call_id.as_deref(), Some("read-a"));
/// # Ok(())
/// # }
/// ```
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
        column_schema_version: ExportSchemaVersion::CURRENT,
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

/// Serialize a [`Message`] slice to the canonical export JSON — the exact bytes of the
/// `messages_json` column (helper exposed for callers projecting outside the batch path).
///
/// It returns the SAME policy the column carries, so a caller cannot drift into a second, quietly
/// different shape. The name is historical: "clean" here means *not template-rendered* (the
/// loss-region / target policy is applied by `gw-format` at render time), NOT "content-only" —
/// `reasoning`, tool calls and result links are all present.
#[must_use]
pub fn clean_messages_json(messages: &[Message]) -> String {
    canonical_messages_json(messages).unwrap_or_default()
}
