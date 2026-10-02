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
//! v3 adds nullable canonical `task_json` with reviewed task provenance and its numeric contract.
//! v4 adds strict `origin_json`; v5 adds nullable canonical `tools_json` for the complete tool
//! definitions. Absent tools stay SQL null, distinct from an explicit empty JSON list.
//! The schema, projection, row identity, and writer follow the artifact's stored column version;
//! v2 receipt recovery continues to use its exact original eight-column projection.
//!
//! ## Reading a shard
//!
//! [`gw_schema::ExportSchemaVersion`] names the column contract a shard was written under and is recorded in
//! every manifest. [`export_parquet_bytes`] carries a worked read-back example. v1 shards (the
//! historical, lossy `{role, content}` + parallel `reasoning_json` pair) are still readable, but
//! only by a text-only consumer — see the enum's docs for the exact break.
//!
//! The Arrow + Parquet writers are **synchronous** and do file I/O, so [`crate::Store::publish_export`] runs
//! them inside [`tokio::task::spawn_blocking`] to keep the async runtime unblocked. Tests write
//! to an in-memory `Vec<u8>` buffer (the Parquet `ArrowWriter` accepts any [`std::io::Write`]).

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use blake3::Hasher;
use gw_schema::{
    CotPolicy, ExportArtifact, ExportManifest, ExportOptions, ExportSchemaVersion, ExportScope,
    ExportTaskProjection, LifecycleState, Message, TrainingRecord, TrlFormat, Verdict,
};
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use crate::artifact::{ARTIFACT_METADATA_KEY, ExportPlan};
use crate::error::Result;

/// The flat columnar projection of one record (one Parquet row). `messages_json` is a JSON-string
/// column to dodge deep-nesting schema churn (DATA-SCHEMA §4.1); it holds the whole canonical
/// conversation, so there is no parallel per-turn column that can drift out of alignment with it.
#[derive(Debug, Clone)]
pub(crate) struct Projected {
    pub record_id: String,
    pub training_area: String,
    pub record_hash: String,
    pub prompt_hash: String,
    pub verdict: Option<String>,
    pub judge_aggregate: Option<f64>,
    pub reasoning_tokens: u32,
    pub messages_json: String,
    pub task_json: Option<String>,
    pub origin_json: Option<String>,
    pub tools_json: Option<String>,
}

/// Project a record into its export row. `messages_json` is the canonical conversation, serialized
/// straight from [`TrainingRecord::messages`] — every structural field kept (INVARIANT a: reasoning
/// stays a sibling of content, never inlined; INVARIANT i: the `tool_call_id` result link travels
/// with its turn).
///
/// `record_hash` falls back to a freshly computed content hash when the envelope's own hash is
/// empty, so an externally-constructed record can never export `record_hash = ""`.
pub(crate) fn project(rec: &TrainingRecord, version: ExportSchemaVersion) -> Result<Projected> {
    if rec.origin.generated().is_none() && version < ExportSchemaVersion::RecordOrigins {
        return Err(crate::artifact::integrity(
            "reference origin cannot be down-projected",
        ));
    }
    crate::reference_records::validate_record(rec)?;
    let task_json = match version {
        ExportSchemaVersion::ReviewedTasks
        | ExportSchemaVersion::RecordOrigins
        | ExportSchemaVersion::ToolDefinitions => rec
            .task_provenance
            .as_ref()
            .map(|task| {
                let projection = ExportTaskProjection {
                    provenance: task.clone(),
                    verification_contract: rec.verification_contract.clone().ok_or_else(|| {
                        crate::artifact::integrity(
                            "reviewed task record lacks verification contract",
                        )
                    })?,
                };
                projection
                    .validate(&rec.messages)
                    .map_err(crate::artifact::integrity)?;
                canonical_task_json(&projection)
            })
            .transpose()?,
        ExportSchemaVersion::CanonicalMessages if rec.task_provenance.is_none() => None,
        ExportSchemaVersion::CanonicalMessages => {
            return Err(crate::artifact::integrity(
                "v2 publication cannot attest task provenance added after preparation",
            ));
        }
        ExportSchemaVersion::RoleContentText => {
            return Err(crate::artifact::integrity(
                "unsupported export column schema version",
            ));
        }
    };
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
        task_json,
        tools_json: if version == ExportSchemaVersion::ToolDefinitions {
            rec.tools.as_deref().map(canonical_tools_json).transpose()?
        } else {
            None
        },
        origin_json: (version >= ExportSchemaVersion::RecordOrigins)
            .then(|| canonical_origin_json(&rec.origin.projection()))
            .transpose()?,
    })
}

/// Encode definitions losslessly with recursively sorted object keys.
pub(crate) fn canonical_tools_json(tools: &[serde_json::Value]) -> Result<String> {
    let mut value = serde_json::to_value(tools)?;
    value.sort_all_objects();
    Ok(serde_json::to_string(&value)?)
}

pub(crate) fn canonical_origin_json(origin: &gw_schema::ExportRecordOrigin) -> Result<String> {
    origin.validate().map_err(crate::artifact::integrity)?;
    Ok(serde_json::to_string(&serde_json::to_value(origin)?)?)
}

pub(crate) fn canonical_task_json(task: &ExportTaskProjection) -> Result<String> {
    let mut value = serde_json::to_value(task)?;
    value.sort_all_objects();
    Ok(serde_json::to_string(&value)?)
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
/// Returns [`crate::StorageError::Serde`] if a message fails to serialize.
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

/// The Arrow schema for the stored version. Verdict, judge aggregate, and v3 task provenance can
/// genuinely be absent; other columns are required.
pub(crate) fn export_schema(version: ExportSchemaVersion) -> Result<Arc<Schema>> {
    let mut fields = vec![
        Field::new("record_id", DataType::Utf8, false),
        Field::new("training_area", DataType::Utf8, false),
        Field::new("record_hash", DataType::Utf8, false),
        Field::new("prompt_hash", DataType::Utf8, false),
        Field::new("verdict", DataType::Utf8, true),
        Field::new("judge_aggregate", DataType::Float64, true),
        Field::new("reasoning_tokens", DataType::UInt32, false),
        Field::new("messages_json", DataType::Utf8, false),
    ];
    match version {
        ExportSchemaVersion::ReviewedTasks
        | ExportSchemaVersion::RecordOrigins
        | ExportSchemaVersion::ToolDefinitions => {
            fields.push(Field::new("task_json", DataType::Utf8, true));
            if version >= ExportSchemaVersion::RecordOrigins {
                fields.push(Field::new("origin_json", DataType::Utf8, false));
            }
        }
        ExportSchemaVersion::CanonicalMessages => (),
        ExportSchemaVersion::RoleContentText => {
            return Err(crate::artifact::integrity(
                "unsupported export column schema version",
            ));
        }
    }
    if version == ExportSchemaVersion::ToolDefinitions {
        fields.push(Field::new("tools_json", DataType::Utf8, true));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// Assemble the projected rows into a single Arrow [`RecordBatch`].
pub(crate) fn build_batch(rows: &[Projected], version: ExportSchemaVersion) -> Result<RecordBatch> {
    let mut columns: Vec<ArrayRef> = vec![
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
    if matches!(
        version,
        ExportSchemaVersion::ReviewedTasks
            | ExportSchemaVersion::RecordOrigins
            | ExportSchemaVersion::ToolDefinitions
    ) {
        columns.push(Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.task_json.as_deref()),
        )));
    } else if rows.iter().any(|r| r.task_json.is_some()) {
        return Err(crate::artifact::integrity(
            "older export schema cannot represent task provenance",
        ));
    }
    if version >= ExportSchemaVersion::RecordOrigins {
        if rows.iter().any(|r| r.origin_json.is_none()) {
            return Err(crate::artifact::integrity("v4 requires origin_json"));
        }
        columns.push(Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.origin_json.as_deref()),
        )));
    } else if rows.iter().any(|r| r.origin_json.is_some()) {
        return Err(crate::artifact::integrity(
            "older export schema cannot contain origin_json",
        ));
    }
    if version == ExportSchemaVersion::ToolDefinitions {
        columns.push(Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.tools_json.as_deref()),
        )));
    } else if rows.iter().any(|r| r.tools_json.is_some()) {
        return Err(crate::artifact::integrity(
            "older export schema cannot contain tools_json",
        ));
    }
    Ok(RecordBatch::try_new(export_schema(version)?, columns)?)
}

/// Encode the rows to an in-memory Parquet byte buffer (sync; called from `spawn_blocking`).
pub(crate) fn write_parquet<W: std::io::Write + Send>(
    rows: &[Projected],
    artifact: &ExportArtifact,
    output: W,
) -> Result<()> {
    let batch = build_batch(rows, artifact.manifest.column_schema_version)?;
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            ARTIFACT_METADATA_KEY.to_string(),
            serde_json::to_string(artifact)?,
        )]))
        .build();
    let mut writer = ArrowWriter::try_new(output, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Encode records eligible for SFT to a Parquet byte buffer + manifest, entirely off the async
/// runtime. Uses the same verdict-and-lifecycle eligibility gate as [`crate::Store::publish_export`].
///
/// # Reading a shard back
///
/// A consumer needs two things: the [`gw_schema::ExportSchemaVersion`] from the manifest, and the
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
/// #         origin: gw_schema::RecordOrigin::Generated(Box::new(gw_schema::GeneratedOrigin {
/// #             provenance: Provenance { run_id: "run-1".into(), parent_ids: vec![],
/// #             teacher: gw_schema::TeacherRef { provider: "openrouter".into(),
/// #                 slug: "toy/teacher".into(), served_by: None, model_card_revision: None },
/// #             user_synth_model: None, user_turn_kind: None, in_scope_safe: None,
/// #             judge_models: vec![], harness_version: "0.1.0".into(), git_commit: None },
/// #         generation: Default::default(),
/// #         })),
/// #         task_provenance: None,
/// #         verification_contract: None,
/// #         execution_evidence: None,
/// #         verification: Default::default(),
/// #         judging: gw_schema::Judging { verdict: Some(Verdict::Admit), aggregate: Some(0.9),
/// #             ..Default::default() },
/// #         reasoning_quality: None,
/// #         lifecycle: gw_schema::Lifecycle { state: gw_schema::LifecycleState::Admitted, ..Default::default() },
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
/// Returns [`crate::StorageError`] on a serialization or Arrow/Parquet failure.
pub async fn export_parquet_bytes(
    records: &[TrainingRecord],
    target: TrlFormat,
    cot: CotPolicy,
) -> Result<(Vec<u8>, ExportManifest)> {
    let records = records.to_vec();
    tokio::task::spawn_blocking(move || {
        let plan = ExportPlan::prepare(
            &records,
            ExportOptions {
                target,
                cot_policy: cot,
                dataset_version: None,
                scope: ExportScope::Records,
            },
        )?;
        let mut bytes = Vec::new();
        write_parquet(&plan.rows, plan.artifact(), &mut bytes)?;
        Ok((bytes, plan.artifact().manifest.clone()))
    })
    .await?
}

/// Judging records the individual grade; lifecycle records whether selection admitted the trace.
/// Both are required so retained siblings and interrupted selection cannot enter an SFT dataset.
/// Preference preparation reuses this predicate for its chosen side without admitting negatives
/// into SFT export.
pub fn is_selected_admitted(record: &TrainingRecord) -> bool {
    (match &record.origin {
        gw_schema::RecordOrigin::Generated(_) => record.judging.verdict == Some(Verdict::Admit),
        gw_schema::RecordOrigin::ReviewedReference(_) => {
            crate::reference_records::validate_record(record).is_ok()
        }
    }) && matches!(
        record.lifecycle.state,
        LifecycleState::Admitted | LifecycleState::Formatted | LifecycleState::Exported
    )
}

/// BLAKE3 of the sorted per-row `record_hash`es — a stable, order-independent content hash of the
/// shard for the manifest's `build_inputs_hash`. Reads the already-resolved (never empty) hashes
/// off the projected rows.
pub(crate) fn shard_content_hash(rows: &[Projected]) -> String {
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
