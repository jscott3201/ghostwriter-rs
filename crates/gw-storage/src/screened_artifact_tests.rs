//! Self-consistent adversarial edits to genuine CLI-published artifacts. Recompute all affected
//! identities so failures cannot rely on stale outer checksums.
use super::*;
use crate::export::{canonical_task_json, write_parquet};
use crate::screening_binding::screening_hash;
use gw_schema::*;

fn fixture() -> (ExportArtifact, Vec<Projected>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../adapters/trl/tests/fixtures/screened-all.parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
    let metadata = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .unwrap()
        .iter()
        .find(|entry| entry.key == ARTIFACT_METADATA_KEY)
        .unwrap();
    let artifact: ExportArtifact = serde_json::from_str(metadata.value.as_ref().unwrap()).unwrap();
    let rows = builder
        .build()
        .unwrap()
        .flat_map(|batch| read_batch(&batch.unwrap(), ExportSchemaVersion::ReviewedTasks).unwrap())
        .collect();
    (artifact, rows)
}

fn encode(mut artifact: ExportArtifact, rows: &[Projected]) -> Vec<u8> {
    artifact.manifest.build_inputs_hash = shard_content_hash(rows);
    artifact.artifact_id = artifact_identity(&artifact, rows).unwrap();
    let mut bytes = vec![];
    write_parquet(rows, &artifact, &mut bytes).unwrap();
    bytes
}

// Reconstruct this known synthetic generator's raw inputs, never a general artifact-to-record API.
// Equality assertions prove the mutations below start from every genuine per-record binding.
fn synthetic_records(artifact: &ExportArtifact, rows: &[Projected]) -> Vec<TrainingRecord> {
    let plan = &artifact.screening.as_ref().unwrap().plan;
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let task: ExportTaskProjection =
                serde_json::from_str(row.task_json.as_ref().unwrap()).unwrap();
            let messages: Vec<Message> = serde_json::from_str(&row.messages_json).unwrap();
            let binding = plan
                .population
                .iter()
                .find(|b| b.record.record_id == row.record_id)
                .unwrap();
            let record: TrainingRecord = serde_json::from_value(serde_json::json!({
                "record_id": row.record_id,
                "schema_version": "1.0.0",
                "training_area": row.training_area,
                "messages": messages,
                "task_provenance": task.provenance,
                "verification_contract": task.verification_contract,
                "generation": {
                    "n_completions": 2,
                    "completion_index": index,
                    "sibling_group_id": crate::prompt_hash(&messages).unwrap()
                },
                "provenance": {
                    "run_id": binding.record.run_id,
                    "teacher": {"provider": "fixture", "slug": "synthetic"},
                    "harness_version": "fixture"
                },
                "judging": {"verdict": "admit"},
                "lifecycle": {"state": "admitted"}
            }))
            .unwrap();
            assert_eq!(
                crate::screening_binding::capture_screening_input_for(
                    &record,
                    &plan.declaration.policy,
                    artifact.manifest.column_schema_version
                )
                .unwrap(),
                *binding
            );
            record
        })
        .collect()
}

fn rebuild_report(artifact: &mut ExportArtifact, records: &[TrainingRecord]) {
    let witness = artifact.screening.as_mut().unwrap();
    let plan = &mut witness.plan;
    plan.population = records
        .iter()
        .map(|r| {
            crate::screening_binding::capture_screening_input_for(
                r,
                &plan.declaration.policy,
                artifact.manifest.column_schema_version,
            )
            .unwrap()
        })
        .collect();
    plan.required_fields = records
        .iter()
        .flat_map(|record| {
            classify_screening_source(&record.messages, record.tools.as_deref()).required_fields
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    plan.policy_id = screening_hash("screening-policy-v1", &plan.declaration.policy).unwrap();
    plan.screening_input_id = screening_hash(
        "screening-captured-population-v2",
        &(&plan.declaration, &plan.population),
    )
    .unwrap();
    plan.protected_input_id = screening_hash(
        "screening-protected-inputs-v1",
        &plan
            .protected_inputs
            .iter()
            .map(|input| (&input.canonical_id, &input.input_id))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    plan.plan_id.clear();
    plan.plan_id = screening_hash("frozen-screening-plan-v2", plan).unwrap();
    witness.population_id = screening_hash(
        "screened-publication-population-v2",
        &(&plan.declaration.runs, &plan.population),
    )
    .unwrap();
}

#[test]
fn genuine_current_and_historical_raw_artifacts_remain_valid_controls() {
    let (artifact, rows) = fixture();
    synthetic_records(&artifact, &rows);
    assert!(verify_artifact_snapshot(encode(artifact, &rows)).is_ok());
    for name in ["v2-empty", "v2-text", "v3-empty", "v3-text", "v3-task"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../adapters/trl/tests/fixtures/{name}.parquet"));
        assert!(
            verify_artifact_snapshot(std::fs::read(path).unwrap()).is_ok(),
            "{name}"
        );
    }
}

#[test]
fn genuine_plan_rejects_rehashed_contradictory_row_record_hash() {
    let (artifact, mut rows) = fixture();
    rows[0].record_hash = "0".repeat(64);
    assert!(
        verify_artifact_snapshot(encode(artifact, &rows)).is_err(),
        "row hash contradicts genuine captured population after both outer identities were rebuilt"
    );
}

#[test]
fn genuine_train_component_rejects_rehashed_test_task_with_unchanged_semantic_identity() {
    let (artifact, mut rows) = fixture();
    let mut task: ExportTaskProjection =
        serde_json::from_str(rows[0].task_json.as_ref().unwrap()).unwrap();
    let original = task.provenance.identity.clone();
    task.provenance.split.role = TaskSplitRole::Test;
    task.validate(&serde_json::from_str::<Vec<Message>>(&rows[0].messages_json).unwrap())
        .unwrap();
    assert_eq!(task.provenance.identity, original);
    rows[0].task_json = Some(canonical_task_json(&task).unwrap());
    assert!(
        verify_artifact_snapshot(encode(artifact, &rows)).is_err(),
        "valid Test declaration contradicts genuine Train source evidence despite unchanged record hash"
    );
}

#[test]
fn genuine_source_projection_rejects_rehashed_message_tool_task_and_reasoning_metadata_changes() {
    let (artifact, original) = fixture();
    let records = synthetic_records(&artifact, &original);
    let mut accepted = vec![];
    for mutation in [
        "message",
        "reasoning_id",
        "reasoning_index",
        "reasoning_signature",
        "reasoning_format",
        "task_group",
        "task_rights",
        "task_contract",
        "training_area",
        "verdict",
        "tool_call",
    ] {
        let mut rows = original.clone();
        let mut messages: Vec<Message> = serde_json::from_str(&rows[0].messages_json).unwrap();
        if mutation.starts_with("reasoning_") {
            let ReasoningDetail::Text {
                id,
                index,
                signature,
                format,
                ..
            } = &mut messages[1].reasoning_details.as_mut().unwrap()[0]
            else {
                panic!("text fixture")
            };
            match mutation {
                "reasoning_id" => *id = Some("changed".into()),
                "reasoning_index" => *index = 9,
                "reasoning_signature" => *signature = Some("changed".into()),
                _ => *format = Some("changed".into()),
            }
            let mut changed = records[0].clone();
            changed.messages = messages.clone();
            assert_eq!(
                crate::record_hash(&changed).unwrap(),
                rows[0].record_hash,
                "legacy content hash intentionally omits these detail fields"
            );
        } else if mutation.starts_with("task_") {
            let mut task: ExportTaskProjection =
                serde_json::from_str(rows[0].task_json.as_ref().unwrap()).unwrap();
            match mutation {
                "task_group" => task.provenance.group.id.push_str("changed"),
                "task_rights" => task.provenance.rights.reviewer.push_str("changed"),
                _ => {
                    task.verification_contract.oracle = Oracle::Literal {
                        expected: "99".into(),
                    }
                }
            }
            // The contract mutation also needs a matching task semantic identity; valid group/rights
            // mutations leave it unchanged. The existing task validator rejects the contract case.
            rows[0].task_json = Some(canonical_task_json(&task).unwrap());
        } else {
            match mutation {
                "message" => messages[3].content = Content::Text("different answer".into()),
                "training_area" => rows[0].training_area = "different".into(),
                "verdict" => rows[0].verdict = Some("reject".into()),
                "tool_call" => {
                    messages[3].tool_calls = Some(vec![ToolCall {
                        id: Some("call".into()),
                        function: FunctionCall {
                            name: "different".into(),
                            arguments: serde_json::json!({"changed":true}),
                            raw_arguments: None,
                        },
                    }])
                }
                _ => unreachable!(),
            }
        }
        rows[0].messages_json = serde_json::to_string(&messages).unwrap();
        if verify_artifact_snapshot(encode(artifact.clone(), &rows)).is_ok() {
            accepted.push(mutation);
        }
    }
    assert!(
        accepted.is_empty(),
        "self-consistent row substitutions accepted: {accepted:?}"
    );
}

#[test]
fn invalid_policy_domains_fail_after_every_policy_population_plan_and_artifact_identity_is_rebuilt()
{
    let (original, rows) = fixture();
    let records = synthetic_records(&original, &rows);
    let mut accepted = vec![];
    let ceilings = ScreeningLimits::default();
    for case in 0..18 {
        let mut artifact = original.clone();
        let policy = &mut artifact.screening.as_mut().unwrap().plan.declaration.policy;
        match case {
            0 => policy.ngram = [0, 13],
            1 => policy.ngram = [14, 13],
            2 => policy.min_overlap_tokens = 0,
            3 => policy.min_overlap_tokens = 65_537,
            _ => {
                let pair = case - 4;
                let high = pair % 2 == 1;
                match pair / 2 {
                    0 => {
                        policy.limits.total_text_bytes = if high {
                            ceilings.total_text_bytes + 1
                        } else {
                            0
                        }
                    }
                    1 => {
                        policy.limits.segment_bytes =
                            if high { ceilings.segment_bytes + 1 } else { 0 }
                    }
                    2 => {
                        policy.limits.segment_tokens =
                            if high { ceilings.segment_tokens + 1 } else { 0 }
                    }
                    3 => policy.limits.segments = if high { ceilings.segments + 1 } else { 0 },
                    4 => {
                        policy.limits.distinct_shingles = if high {
                            ceilings.distinct_shingles + 1
                        } else {
                            0
                        }
                    }
                    5 => {
                        policy.limits.shingle_token_work = if high {
                            ceilings.shingle_token_work + 1
                        } else {
                            0
                        }
                    }
                    _ => {
                        policy.limits.comparisons = if high { ceilings.comparisons + 1 } else { 0 }
                    }
                }
            }
        }
        rebuild_report(&mut artifact, &records);
        if verify_artifact_snapshot(encode(artifact, &rows)).is_ok() {
            accepted.push(case);
        }
    }
    assert!(
        accepted.is_empty(),
        "unsupported policy cases accepted after complete identity rebuild: {accepted:?}"
    );
}

#[test]
fn incomplete_protected_summaries_fail_after_every_affected_identity_is_rebuilt() {
    let (original, rows) = fixture();
    let records = synthetic_records(&original, &rows);
    let mut accepted = vec![];
    for mutation in [
        "empty_union",
        "missing_required",
        "missing_additional",
        "duplicate_id",
        "unsorted_ids",
        "empty_items",
        "duplicate_items",
        "empty_revision",
        "empty_content_digest",
        "coverage_languages",
        "coverage_media",
        "coverage_fields",
        "coverage_incomplete",
    ] {
        let mut artifact = original.clone();
        let plan = &mut artifact.screening.as_mut().unwrap().plan;
        match mutation {
            "empty_union" => plan.protected_inputs.clear(),
            "missing_required" => {
                plan.protected_inputs.pop();
            }
            "missing_additional" => plan
                .declaration
                .policy
                .additional_protected_sets
                .push("required-extra".into()),
            "duplicate_id" => {
                plan.protected_inputs[1].canonical_id =
                    plan.protected_inputs[0].canonical_id.clone()
            }
            "unsorted_ids" => plan.protected_inputs.reverse(),
            "empty_items" => plan.protected_inputs[0].item_ids.clear(),
            "duplicate_items" => {
                let id = plan.protected_inputs[0].item_ids[0].clone();
                plan.protected_inputs[0].item_ids.push(id);
            }
            "empty_revision" => plan.protected_inputs[0].source_revision.clear(),
            "empty_content_digest" => plan.protected_inputs[0].content_digest.clear(),
            "coverage_languages" => plan.protected_inputs[0].coverage.languages.clear(),
            "coverage_media" => plan.protected_inputs[0].coverage.media = vec!["image".into()],
            "coverage_fields" => plan.protected_inputs[0].coverage.fields.clear(),
            "coverage_incomplete" => plan.protected_inputs[0].coverage.complete = false,
            _ => unreachable!(),
        }
        plan.counts.supplied_protected_sets = plan.protected_inputs.len() as u64;
        // Protected input IDs are opaque without protected text. A consistent producer can change
        // them; structural rejection must still follow from the summary itself.
        for input in &mut plan.protected_inputs {
            input.input_id = screening_hash(
                "self-consistent-test-protected-input",
                &(
                    mutation,
                    &input.canonical_id,
                    &input.coverage,
                    &input.item_ids,
                ),
            )
            .unwrap();
        }
        rebuild_report(&mut artifact, &records);
        if verify_artifact_snapshot(encode(artifact, &rows)).is_ok() {
            accepted.push(mutation);
        }
    }
    assert!(
        accepted.is_empty(),
        "unsupported protected summaries accepted after complete identity rebuild: {accepted:?}"
    );
}

#[test]
fn earlier_screened_metadata_is_explicitly_unsupported() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../adapters/trl/tests/fixtures/screened-legacy-unbound.parquet");
    let error = verify_artifact_snapshot(std::fs::read(path).unwrap()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported export metadata version"),
        "{error}"
    );
}

#[test]
fn complete_protected_summary_cannot_omit_a_visible_reasoning_field_after_rehashing() {
    let (mut artifact, rows) = fixture();
    let records = synthetic_records(&artifact, &rows);
    assert!(records[0].messages.iter().any(|message| {
        message.reasoning_details.as_ref().is_some_and(|details| {
            details
                .iter()
                .any(|detail| matches!(detail, ReasoningDetail::Text { .. }))
        })
    }));
    let summary = &mut artifact.screening.as_mut().unwrap().plan.protected_inputs[0];
    summary.coverage.fields = vec![ScreeningField::Content];
    summary.input_id =
        screening_hash("rehashed-protected-field-coverage", &summary.coverage).unwrap();
    rebuild_report(&mut artifact, &records);
    assert!(
        verify_artifact_snapshot(encode(artifact, &rows)).is_err(),
        "Complete summary omits visible ReasoningDetail after every affected identity is rebuilt"
    );
}

#[test]
fn lowering_both_claimed_union_and_summary_cannot_hide_visible_fields() {
    let (mut artifact, rows) = fixture();
    let records = synthetic_records(&artifact, &rows);
    for summary in &mut artifact.screening.as_mut().unwrap().plan.protected_inputs {
        summary.coverage.fields = vec![ScreeningField::Content];
        summary.input_id = screening_hash("lowered-protected-fields", &summary.coverage).unwrap();
    }
    rebuild_report(&mut artifact, &records);
    let plan = &mut artifact.screening.as_mut().unwrap().plan;
    plan.required_fields = vec![ScreeningField::Content];
    plan.plan_id.clear();
    plan.plan_id = screening_hash("frozen-screening-plan-v2", plan).unwrap();
    assert!(verify_artifact_snapshot(encode(artifact, &rows)).is_err());
}

fn call(arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id: Some("call".into()),
        function: FunctionCall {
            name: "synthetic_tool".into(),
            arguments,
            raw_arguments: None,
        },
    }
}

#[test]
fn unsupported_source_shapes_cannot_be_rehashed_into_a_complete_screened_artifact() {
    let (original, original_rows) = fixture();
    let original_records = synthetic_records(&original, &original_rows);
    let mut accepted = vec![];
    for case in [
        "image",
        "audio",
        "encrypted",
        "nonobject_arguments",
        "duplicate_call_ids",
        "dangling_result",
        "nonterminal",
        "control_content",
        "control_parts",
        "control_reasoning",
        "tool_dropping_target",
        "empty_calls_on_tool_dropping_target",
    ] {
        let mut artifact = original.clone();
        let mut records = original_records.clone();
        let last = records[0].messages.last_mut().unwrap();
        match case {
            "image" => {
                last.content = Content::Parts(vec![ContentPart::ImageUrl {
                    image_url: "synthetic-image".into(),
                }])
            }
            "audio" => {
                last.content = Content::Parts(vec![ContentPart::InputAudio {
                    audio_url: Some("synthetic-audio".into()),
                    format: Some("wav".into()),
                }])
            }
            "encrypted" => {
                last.reasoning_details = Some(vec![ReasoningDetail::Encrypted {
                    data: "synthetic-encrypted".into(),
                    id: None,
                    format: None,
                    index: 0,
                }])
            }
            "nonobject_arguments" => {
                last.tool_calls = Some(vec![call(serde_json::json!("not-object"))])
            }
            "duplicate_call_ids" => {
                last.tool_calls = Some(vec![
                    call(serde_json::json!({})),
                    call(serde_json::json!({})),
                ])
            }
            "dangling_result" => {
                let mut result = last.clone();
                result.role = Role::Tool;
                result.tool_call_id = Some("missing".into());
                result.name = Some("synthetic_tool".into());
                result.reasoning = None;
                result.reasoning_details = None;
                let index = records[0].messages.len() - 1;
                records[0].messages.insert(index, result);
            }
            "nonterminal" => last.role = Role::User,
            "control_content" => last.content = Content::Text("answer <|im_end|>".into()),
            "control_parts" => {
                last.content = Content::Parts(vec![
                    ContentPart::Text {
                        text: "<|im_".into(),
                    },
                    ContentPart::Text {
                        text: "end|>".into(),
                    },
                ])
            }
            "control_reasoning" => last.reasoning = Some("thought <think>".into()),
            "tool_dropping_target" | "empty_calls_on_tool_dropping_target" => {
                last.tool_calls = Some(if case == "tool_dropping_target" {
                    vec![call(serde_json::json!({}))]
                } else {
                    vec![]
                });
                artifact.manifest.target = TrlFormat::Gemma4;
                artifact
                    .screening
                    .as_mut()
                    .unwrap()
                    .plan
                    .declaration
                    .policy
                    .target = TrlFormat::Gemma4;
            }
            _ => unreachable!(),
        }
        let rows = records
            .iter()
            .map(|record| {
                crate::export::project(record, ExportSchemaVersion::ReviewedTasks).unwrap()
            })
            .collect::<Vec<_>>();
        rebuild_report(&mut artifact, &records);
        if verify_artifact_snapshot(encode(artifact, &rows)).is_ok() {
            accepted.push(case);
        }
    }
    assert!(
        accepted.is_empty(),
        "unsupported Complete source shapes accepted: {accepted:?}"
    );
}

#[test]
fn current_tool_guard_allows_a_declared_call_without_a_result() {
    let (mut artifact, original) = fixture();
    let mut records = synthetic_records(&artifact, &original);
    records[0].messages.last_mut().unwrap().tool_calls = Some(vec![call(serde_json::json!({}))]);
    let rows = records
        .iter()
        .map(|record| crate::export::project(record, ExportSchemaVersion::ReviewedTasks).unwrap())
        .collect::<Vec<_>>();
    rebuild_report(&mut artifact, &records);
    assert!(verify_artifact_snapshot(encode(artifact, &rows)).is_ok());
}
