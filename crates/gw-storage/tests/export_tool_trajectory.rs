//! Export evidence: an admitted conversation survives the Parquet projection with every structural
//! field intact (INVARIANT a reasoning-as-sibling, INVARIANT i tool-result identity).
//!
//! The export path used to project each turn down to `{role, content}` with the content flattened to
//! a string, and kept reasoning in a second, index-aligned column. That is enough for a text-only
//! consumer and lossy for everything else: `content: null` collapsed into `""`, multimodal parts were
//! re-encoded as a string inside a string, and `tool_calls` / `name` / the `tool_call_id` result link
//! were dropped outright — so two results of the SAME function became indistinguishable. These tests
//! pin the replacement: ONE canonical column, decoded independently by a real Parquet reader, and
//! compared field-for-field against the record the store holds.
//!
//! The conversation mirrors the synthetic `gw-format` tool-trajectory fixture (two same-name calls
//! with distinct arguments, one string-encoded on the wire, results arriving in reversed order). It
//! lives here as its own integration crate so it cannot collide with the shared `store.rs` suite.

use arrow::array::{Array, StringArray};
use gw_schema::{
    Content, ContentPart, CotPolicy, ExportManifest, ExportSchemaVersion, FunctionCall, Lifecycle,
    LifecycleState, Message, Provenance, ReasoningDetail, Role, ToolCall, TrainingRecord,
    TrlFormat, Verdict,
};
use gw_storage::{RecordFilter, Store, clean_messages_json, export_parquet_bytes};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// The whole `Content` variant vocabulary in one conversation: text, multimodal parts, and the
/// explicitly-absent `null` of a tool-calling turn.
fn content_variants() -> Vec<Message> {
    vec![
        Message {
            role: Role::System,
            content: Content::Parts(vec![
                ContentPart::Text {
                    text: "Inspect a toy module.".into(),
                },
                ContentPart::ImageUrl {
                    image_url: "file:///toy.png".into(),
                },
            ]),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        Message {
            role: Role::User,
            content: Content::Text("Read two regions.\nQuoted: \"λ\".".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        Message {
            // A tool-calling turn: the ONLY output is the call array, so content is null, not "".
            role: Role::Assistant,
            content: Content::Null,
            reasoning: Some("Two regions; only the ids distinguish them.".into()),
            reasoning_details: Some(vec![ReasoningDetail::Text {
                text: "Two regions; only the ids distinguish them.".into(),
                signature: Some("sig-1".into()),
                id: Some("rd-1".into()),
                format: Some("openai".into()),
                index: 0,
            }]),
            tool_calls: Some(vec![
                ToolCall {
                    id: Some("read-a".into()),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: serde_json::json!({"filepath": "toy.py", "start_line": 1}),
                        raw_arguments: None,
                    },
                },
                ToolCall {
                    id: Some("read-b".into()),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: serde_json::json!({
                            "filepath": "toy.py", "start_line": 20, "end_line": 22
                        }),
                        // The provider sent this string-encoded; the raw wire text is retained beside
                        // the normalized object, so the normalization stays recoverable.
                        raw_arguments: Some(
                            "{\"filepath\": \"toy.py\", \"start_line\": 20, \"end_line\": 22}"
                                .into(),
                        ),
                    },
                },
            ]),
            tool_call_id: None,
            name: None,
        },
        Message {
            // The SECOND call's result arrives first: order in the array is not call order.
            role: Role::Tool,
            content: Content::Text("{\"status\": \"error\"}".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: Some("read-b".into()),
            name: Some("read_file".into()),
        },
        Message {
            role: Role::Tool,
            content: Content::Text(
                "{\"status\": \"ok\", \"content\": \"def toy():\\n    return \\\"λ\\\"\\n\"}"
                    .into(),
            ),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: Some("read-a".into()),
            name: Some("read_file".into()),
        },
        Message {
            role: Role::Assistant,
            content: Content::Text("The second read failed; no repair is claimed.".into()),
            reasoning: Some("Region 20 was unreadable; stop.".into()),
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ]
}

fn tool_record(record_id: &str, run_id: &str, verdict: Option<Verdict>) -> TrainingRecord {
    TrainingRecord {
        record_id: record_id.into(),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "swe-agents".into(),
        tags: vec!["tool-use".into()],
        messages: content_variants(),
        tools: None,
        provenance: Provenance {
            run_id: run_id.into(),
            parent_ids: vec![],
            teacher: gw_schema::TeacherRef {
                provider: "openrouter".into(),
                slug: "z-ai/glm-5.2".into(),
                served_by: Some("Parasail".into()),
                model_card_revision: None,
            },
            user_synth_model: None,
            user_turn_kind: None,
            in_scope_safe: Some(true),
            judge_models: vec![],
            harness_version: "0.1.0".into(),
            git_commit: None,
        },
        generation: Default::default(),
        verification_contract: None,
        verification: Default::default(),
        judging: gw_schema::Judging {
            verdict,
            aggregate: Some(0.9),
            ..Default::default()
        },
        reasoning_quality: None,
        lifecycle: Lifecycle::default(),
        hashes: Default::default(),
        cost: Default::default(),
    }
}

async fn seeded_store() -> Store {
    let store = Store::open_in_memory().await.unwrap();
    store
        .create_run("run-e", "{\"budget\":25}", Some(25.0))
        .await
        .unwrap();
    store
}

/// Export the scanned run and decode the shard's conversation column back into canonical turns —
/// an INDEPENDENT read path (a real Parquet reader + a fresh `serde_json` decode), not a re-use of
/// the writer's own types.
async fn export_and_decode(
    records: &[TrainingRecord],
) -> (Vec<String>, Vec<String>, ExportSchemaVersion) {
    let (bytes, manifest) = export_parquet_bytes(records, TrlFormat::Gemma4, CotPolicy::Masked)
        .await
        .unwrap();
    assert_eq!(manifest.column_schema_version, ExportSchemaVersion::CURRENT);

    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes))
        .unwrap()
        .build()
        .unwrap();
    let mut record_ids = Vec::new();
    let mut conversations = Vec::new();
    for batch in reader {
        let batch = batch.expect("a readable Parquet batch");
        let ids = batch
            .column_by_name("record_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let messages = batch
            .column_by_name("messages_json")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..ids.len() {
            record_ids.push(ids.value(i).to_string());
            let turns: Vec<Message> = serde_json::from_str(messages.value(i))
                .expect("the canonical column decodes into Message[]");
            conversations.push(serde_json::to_string(&turns).unwrap());
        }
    }
    (record_ids, conversations, manifest.column_schema_version)
}

/// put → scan → export → decode: the conversation a consumer receives is the conversation that was
/// stored, field for field and in order.
#[tokio::test]
async fn exported_trajectory_decodes_back_to_the_identical_conversation() {
    let store = seeded_store().await;
    store
        .put(&tool_record("rec-e", "run-e", Some(Verdict::Admit)))
        .await
        .unwrap();

    let scanned = store
        .scan(&RecordFilter::new().run_id("run-e"))
        .await
        .unwrap();
    let (ids, conversations, _) = export_and_decode(&scanned).await;

    assert_eq!(ids, vec!["rec-e".to_string()], "one admitted row");
    let decoded: Vec<Message> = serde_json::from_str(&conversations[0]).unwrap();
    assert_eq!(
        decoded,
        content_variants(),
        "every structural field and the turn order must survive the projection"
    );
}

/// The same round trip, with each load-bearing field named — so a future regression is reported as
/// the specific field that broke, not as one opaque inequality.
#[tokio::test]
async fn canonical_column_keeps_every_structural_field() {
    let store = seeded_store().await;
    store
        .put(&tool_record("rec-e", "run-e", Some(Verdict::Admit)))
        .await
        .unwrap();
    let scanned = store
        .scan(&RecordFilter::new().run_id("run-e"))
        .await
        .unwrap();
    let (_, conversations, _) = export_and_decode(&scanned).await;
    let turns: Vec<Message> = serde_json::from_str(&conversations[0]).unwrap();
    let original = &content_variants();

    // Content variant: parts stay parts (not a JSON string inside a string), and the explicitly
    // absent body of a tool-calling turn stays absent.
    assert!(matches!(&turns[0].content, Content::Parts(parts) if parts.len() == 2));
    assert_eq!(turns[0].content, original[0].content);
    assert_eq!(
        turns[2].content,
        Content::Null,
        "null content must not be flattened into an empty string"
    );
    assert_ne!(turns[2].content, Content::Text(String::new()));

    // Reasoning is a sibling of content (INVARIANT a), with its structured details intact.
    assert_eq!(
        turns[2].reasoning.as_deref(),
        Some("Two regions; only the ids distinguish them.")
    );
    let detail = &turns[2].reasoning_details.as_ref().unwrap()[0];
    assert_eq!(detail, &original[2].reasoning_details.as_ref().unwrap()[0]);
    assert_eq!(
        turns[5].reasoning.as_deref(),
        Some("Region 20 was unreadable; stop.")
    );

    // Tool calls: both same-name calls, distinct arguments, and the retained raw wire text.
    let calls = turns[2].tool_calls.as_ref().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id.as_deref(), Some("read-a"));
    assert_eq!(calls[1].id.as_deref(), Some("read-b"));
    assert_eq!(calls[0].function.arguments["start_line"], 1);
    assert_eq!(calls[1].function.arguments["end_line"], 22);
    assert_eq!(
        calls[1].function.raw_arguments.as_deref(),
        Some("{\"filepath\": \"toy.py\", \"start_line\": 20, \"end_line\": 22}"),
        "the pre-normalization wire text must survive export"
    );

    // The result links, in the order the results actually arrived (reversed vs. the call order).
    assert_eq!(turns[3].role, Role::Tool);
    assert_eq!(turns[3].tool_call_id.as_deref(), Some("read-b"));
    assert_eq!(turns[3].name.as_deref(), Some("read_file"));
    assert_eq!(turns[4].tool_call_id.as_deref(), Some("read-a"));
    assert_ne!(turns[3].tool_call_id, turns[4].tool_call_id);

    // Escapes and unicode in a provider-returned result body.
    assert!(matches!(&turns[4].content, Content::Text(t) if t.contains('λ') && t.contains("\\n")));
}

/// Negative control for the shape this change replaced. The old `{role, content}` projection is
/// still DESERIALIZABLE — which is exactly the hazard: it decodes without error and silently merges
/// two results of the same function into one indistinguishable pair, so a consumer cannot tell the
/// trajectories apart. This is the named reason the column set is versioned rather than reshaped in
/// place.
#[test]
fn v1_lossy_projection_cannot_tell_two_tool_results_apart() {
    /// The v1 (v2-era predecessor) projection, reproduced here as the negative control: role plus a
    /// content string, with `null` flattened to `""`.
    fn v1_projection(messages: &[Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|m| {
                let content = match &m.content {
                    Content::Text(t) => t.clone(),
                    Content::Parts(parts) => serde_json::to_string(parts).unwrap_or_default(),
                    Content::Null => String::new(),
                };
                serde_json::json!({ "role": m.role, "content": content })
            })
            .collect()
    }

    let original = content_variants();
    let other = {
        // The same trajectory with the two result links swapped — a DIFFERENT trajectory.
        let mut m = original.clone();
        let (first, second) = (m[3].tool_call_id.take(), m[4].tool_call_id.take());
        m[3].tool_call_id = second;
        m[4].tool_call_id = first;
        m
    };

    // The lossy projection still parses as `Message[]` (no error surfaces to the consumer)...
    let decoded: Vec<Message> =
        serde_json::from_value(serde_json::Value::Array(v1_projection(&original))).unwrap();
    assert_eq!(decoded[2].content, Content::Text(String::new()));
    assert!(decoded[2].tool_calls.is_none(), "the call array is dropped");
    assert!(decoded[2].reasoning.is_none(), "the CoT is dropped");
    assert!(
        decoded[3].tool_call_id.is_none(),
        "the result link is dropped"
    );
    assert!(decoded[3].name.is_none(), "the tool name is dropped");

    // ...and the two trajectories collapse to the SAME bytes: the loss is not cosmetic.
    assert_eq!(
        serde_json::to_string(&v1_projection(&original)).unwrap(),
        serde_json::to_string(&v1_projection(&other)).unwrap(),
        "a lossy projection cannot distinguish which call each result answers"
    );

    // The canonical column distinguishes them, which is what the export now ships.
    assert_ne!(
        serde_json::to_string(&original).unwrap(),
        serde_json::to_string(&other).unwrap()
    );
}

/// The batch column and the public helper must not be two policies. A caller projecting outside the
/// batch path gets the same bytes the shard carries.
#[tokio::test]
async fn clean_messages_json_is_the_column_policy() {
    let record = tool_record("rec-e", "run-e", Some(Verdict::Admit));
    let (bytes, _) = export_parquet_bytes(
        std::slice::from_ref(&record),
        TrlFormat::Gemma4,
        CotPolicy::Masked,
    )
    .await
    .unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes))
        .unwrap()
        .build()
        .unwrap();
    let mut cells = Vec::new();
    for batch in reader {
        let batch = batch.expect("a readable Parquet batch");
        let column = batch
            .column_by_name("messages_json")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..column.len() {
            cells.push(column.value(i).to_string());
        }
    }
    assert_eq!(cells.len(), 1);
    assert_eq!(
        clean_messages_json(&record.messages),
        cells[0],
        "the helper and the column must be the same serializer"
    );
    // And it is lossless, not a lossy text-only shape under an old name.
    let turns: Vec<Message> = serde_json::from_str(&cells[0]).unwrap();
    assert_eq!(turns, content_variants());
    assert_eq!(turns[3].tool_call_id.as_deref(), Some("read-b"));
    assert_eq!(turns[2].content, Content::Null);
}

/// The crash/restart path before projection: `advance_lifecycle` rewrites the whole envelope and a
/// resume re-reads it, so the export must still be lossless for a record that took that path.
#[tokio::test]
async fn lifecycle_advance_and_resume_export_without_loss() {
    let store = seeded_store().await;
    store
        .put(&tool_record("rec-e", "run-e", Some(Verdict::Admit)))
        .await
        .unwrap();
    store
        .advance_lifecycle("rec-e", LifecycleState::Verified, None)
        .await
        .unwrap();
    store
        .advance_lifecycle("rec-e", LifecycleState::Admitted, Some("passed gate"))
        .await
        .unwrap();
    store
        .checkpoint(
            "run-e",
            0,
            "admitted",
            &serde_json::json!({"seed_offset": 64}),
        )
        .await
        .unwrap();
    let point = store.resume_cursor("run-e", 0).await.unwrap().unwrap();
    assert_eq!(point.state, "admitted");

    let scanned = store
        .scan(&RecordFilter::new().run_id("run-e"))
        .await
        .unwrap();
    let (_, conversations, _) = export_and_decode(&scanned).await;
    let turns: Vec<Message> = serde_json::from_str(&conversations[0]).unwrap();
    assert_eq!(
        turns,
        content_variants(),
        "identity must survive a lifecycle rewrite and a resume"
    );
    assert_eq!(turns[3].tool_call_id.as_deref(), Some("read-b"));
    assert_eq!(turns[4].tool_call_id.as_deref(), Some("read-a"));
}

/// Admission is still the only gate, and the manifest counts stay exact: the input set is counted,
/// the admitted subset is written, and no other verdict is admitted by accident.
#[tokio::test]
async fn only_admit_verdicts_are_written() {
    let records = vec![
        tool_record("admit", "run-e", Some(Verdict::Admit)),
        tool_record("reject", "run-e", Some(Verdict::Reject)),
        tool_record("review", "run-e", Some(Verdict::NeedsReview)),
        tool_record("unjudged", "run-e", None),
    ];
    let (bytes, manifest) = export_parquet_bytes(&records, TrlFormat::Gemma4, CotPolicy::Masked)
        .await
        .unwrap();
    assert_eq!(manifest.n_records, 4, "n_records counts the input set");
    assert_eq!(manifest.n_admitted, 1, "n_admitted counts what was written");
    assert!(!manifest.build_inputs_hash.is_empty());

    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes))
        .unwrap()
        .build()
        .unwrap();
    let mut ids = Vec::new();
    let mut verdicts = Vec::new();
    for batch in reader {
        let batch = batch.expect("a readable Parquet batch");
        let id_col = batch
            .column_by_name("record_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let verdict_col = batch
            .column_by_name("verdict")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..id_col.len() {
            ids.push(id_col.value(i).to_string());
            verdicts.push(verdict_col.value(i).to_string());
        }
    }
    assert_eq!(
        ids,
        vec!["admit".to_string()],
        "only the Admit row is written"
    );
    assert_eq!(
        verdicts,
        vec!["admit".to_string()],
        "the verdict column matches the gate"
    );
    for excluded in ["reject", "review", "unjudged"] {
        assert!(
            !ids.iter().any(|id| id == excluded),
            "{excluded} must not be admitted into the shard"
        );
    }
}

/// The two admission layers, stated once so the split stays deliberate: the storage exporter filters
/// on the JUDGE VERDICT only, and the engine narrows the input to lifecycle-admitted records before
/// handing it over (then reports the whole-run population as `n_records`). A record that carries an
/// Admit verdict but a non-admitted lifecycle state is written by the exporter — it is the engine's
/// pre-filter, not this one, that keeps it out of a real shard. Both layers are covered here and in
/// `gw-engine`'s shard-export suite.
#[tokio::test]
async fn storage_gate_is_the_verdict_and_lifecycle_narrowing_belongs_to_the_engine() {
    let mut rejected_lifecycle = tool_record("stale", "run-e", Some(Verdict::Admit));
    rejected_lifecycle.lifecycle.state = LifecycleState::Rejected;
    let (_, manifest) =
        export_parquet_bytes(&[rejected_lifecycle], TrlFormat::Gemma4, CotPolicy::Masked)
            .await
            .unwrap();
    assert_eq!(
        manifest.n_admitted, 1,
        "the storage gate is verdict-only; lifecycle narrowing happens upstream"
    );
}

/// What the export does NOT carry, stated rather than left for a consumer to discover.
///
/// Two distinct gaps, and neither is a silent one:
///
/// 1. Provider wire extensions outside the canonical type (`source_extension`, the OpenAI
///    `tool_calls[].type` discriminator) have no field in [`Message`], so they are absent from the
///    canonical record and therefore absent from every export. The canonical record is NOT a
///    superset of the wire form — `gw-format`'s fixture records the same gap for ingest, and it
///    applies unchanged here.
/// 2. The v1 shard shape (a parallel `reasoning_json` column) is gone rather than kept in step. It
///    is a documented, versioned break: a manifest written by this build names
///    [`ExportSchemaVersion::CanonicalMessages`], and a manifest without that key reads back as v1,
///    so a reader can tell which contract a file was written under.
#[tokio::test]
async fn export_records_what_it_cannot_preserve() {
    let store = seeded_store().await;
    store
        .put(&tool_record("rec-e", "run-e", Some(Verdict::Admit)))
        .await
        .unwrap();
    let scanned = store
        .scan(&RecordFilter::new().run_id("run-e"))
        .await
        .unwrap();
    let (bytes, manifest) = export_parquet_bytes(&scanned, TrlFormat::Gemma4, CotPolicy::Masked)
        .await
        .unwrap();

    // (2) The versioned transition: the shard names its contract, and the drifting parallel column
    // is not carried forward.
    assert_eq!(
        manifest.column_schema_version,
        ExportSchemaVersion::CanonicalMessages
    );
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes))
        .unwrap()
        .build()
        .unwrap();
    for batch in reader {
        let batch = batch.expect("a readable Parquet batch");
        assert!(batch.schema().field_with_name("messages_json").is_ok());
        assert!(batch.schema().field_with_name("reasoning_json").is_err());
    }
    // A v1-era manifest (the key absent) still reads as v1 rather than claiming the current shape.
    let legacy: ExportManifest = serde_json::from_str(
        r#"{"target":"chatml","cot_policy":"masked","n_records":1,"n_admitted":1,
            "build_inputs_hash":"h"}"#,
    )
    .unwrap();
    assert_eq!(
        legacy.column_schema_version,
        ExportSchemaVersion::RoleContentText
    );

    // (1) Wire-only provider fields have nowhere to live in the canonical type, so the export does
    // not claim them: nothing in a shard should look like a captured raw payload.
    let turns: Vec<Message> =
        serde_json::from_str(&clean_messages_json(&scanned[0].messages)).unwrap();
    let json = serde_json::to_string(&turns).unwrap();
    for absent in ["source_extension", "\"type\":\"function\""] {
        assert!(
            !json.contains(absent),
            "an unmodelled wire field must not appear in an export: {absent}"
        );
    }
    assert_eq!(turns.len(), content_variants().len());
}
