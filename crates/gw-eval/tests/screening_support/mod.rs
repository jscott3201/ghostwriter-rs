//! Synthetic declarations and protected contents. They assert no real benchmark coverage.
#![allow(dead_code)]
use gw_schema::*;

pub fn message(role: Role, text: &str) -> Message {
    Message {
        role,
        content: Content::Text(text.into()),
        reasoning: None,
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

pub fn record(id: &str, prompt: &str) -> TrainingRecord {
    let document = NumericTaskDocument::from_json(include_str!(
        "../../../../examples/reviewed-numeric-tasks.json"
    ))
    .unwrap();
    let mut task = document.tasks[0].clone();
    task.task_id = id.into();
    task.source.item = id.into();
    task.group.id = id.into();
    task.prompt = TaskPrompt::User {
        content: prompt.into(),
    };
    task.split.role = TaskSplitRole::Train;
    let messages = vec![message(Role::User, prompt), message(Role::Assistant, "42")];
    TrainingRecord {
        record_id: id.into(),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "synthetic-screening".into(),
        tags: vec![],
        messages: messages.clone(),
        tools: None,
        origin: gw_schema::RecordOrigin::Generated(Box::new(gw_schema::GeneratedOrigin {
            provenance: Provenance {
                run_id: format!("run-{id}"),
                parent_ids: vec![],
                teacher: TeacherRef {
                    provider: "fixture".into(),
                    slug: "synthetic".into(),
                    served_by: None,
                    model_card_revision: None,
                },
                user_synth_model: None,
                user_turn_kind: None,
                in_scope_safe: Some(true),
                judge_models: vec![],
                harness_version: "fixture".into(),
                git_commit: None,
            },
            generation: Generation {
                n_completions: Some(1),
                completion_index: Some(0),
                sibling_group_id: Some(gw_storage::prompt_hash(&messages).unwrap()),
                ..Default::default()
            },
        })),
        task_provenance: Some(TaskProvenance::from_task(&task).unwrap()),
        verification_contract: Some(task.verification.contract()),
        execution_evidence: None,
        verification: Verification::default(),
        judging: Judging {
            verdict: Some(Verdict::Admit),
            ..Default::default()
        },
        reasoning_quality: None,
        lifecycle: Lifecycle {
            state: LifecycleState::Admitted,
            ..Default::default()
        },
        hashes: Hashes::default(),
        cost: Default::default(),
    }
}

pub fn key(record: &TrainingRecord) -> ScreeningRecordId {
    ScreeningRecordId {
        run_id: record.run_id().to_owned(),
        record_id: record.record_id.clone(),
    }
}

pub fn declaration(records: &[TrainingRecord]) -> ScreeningDeclaration {
    let mut tasks = std::collections::BTreeMap::<(String, String, String), Vec<_>>::new();
    let mut siblings = std::collections::BTreeMap::<(String, String), Vec<_>>::new();
    for record in records {
        if let Some(task) = &record.task_provenance {
            tasks
                .entry((
                    task.source.namespace.clone(),
                    task.source.item.clone(),
                    task.source.revision.clone(),
                ))
                .or_default()
                .push(key(record));
        }
        if let Some(group) = record
            .origin
            .generated()
            .and_then(|g| g.generation.sibling_group_id.as_ref())
        {
            siblings
                .entry((record.run_id().to_owned(), group.clone()))
                .or_default()
                .push(key(record));
        }
    }
    ScreeningDeclaration {
        version: SCREENING_VERSION,
        runs: DeclaredRunSet {
            run_ids: records
                .iter()
                .map(|r| r.run_id().to_owned())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
        },
        output: ScreeningOutputScope {
            run_id: records[0].run_id().to_owned(),
            record_ids: records
                .iter()
                .filter(|r| r.run_id() == records[0].run_id())
                .map(|r| r.record_id.clone())
                .collect(),
        },
        policy: ScreeningPolicy::default(),
        expected_tasks: tasks
            .into_iter()
            .map(
                |((namespace, item, revision), records)| ExpectedScreeningTask {
                    namespace,
                    item,
                    revision,
                    records,
                },
            )
            .collect(),
        siblings: siblings
            .into_iter()
            .map(
                |((run_id, sibling_group_id), records)| DeclaredScreeningSiblings {
                    run_id,
                    sibling_group_id,
                    records,
                },
            )
            .collect(),
    }
}

pub fn protected() -> Vec<ProtectedScreeningSet> {
    CANONICAL_PROTECTED_BENCHMARKS
        .iter()
        .map(|id| {
            let items = vec![ProtectedScreeningItem {
                item_id: "synthetic-item".into(),
                language: "en".into(),
                prompt: vec![message(Role::User, &format!("fixture-private-{id}"))],
                responses: vec![message(Role::Assistant, &format!("private-answer-{id}"))],
            }];
            ProtectedScreeningSet {
                version: SCREENING_VERSION,
                canonical_id: (*id).into(),
                source_revision: "synthetic-v1".into(),
                content_digest: gw_eval::screening::protected_screening_content_digest(&items)
                    .unwrap(),
                rights: Some(ProtectedScreeningRights {
                    reviewer: "synthetic-fixture".into(),
                    evidence: vec!["fixture authored for this test".into()],
                    screening_permitted: true,
                }),
                coverage: ProtectedScreeningCoverage {
                    complete: true,
                    languages: vec!["en".into()],
                    media: vec!["text".into()],
                    fields: vec![
                        ScreeningField::Content,
                        ScreeningField::Reasoning,
                        ScreeningField::ReasoningDetail,
                        ScreeningField::ToolArguments,
                        ScreeningField::ToolName,
                        ScreeningField::ToolResult,
                        ScreeningField::ToolDefinition,
                    ],
                },
                normalization: LEXICAL_SCREEN_RECIPE.into(),
                items,
            }
        })
        .collect()
}

pub fn rebind_task(record: &mut TrainingRecord) {
    let old = record.task_provenance.clone().unwrap();
    let mut task = NumericTaskDocument::from_json(include_str!(
        "../../../../examples/reviewed-numeric-tasks.json"
    ))
    .unwrap()
    .tasks
    .remove(0);
    task.task_id = old.task_id;
    task.source = old.source;
    task.group = old.group;
    task.split = old.split;
    let Content::Text(prompt) = &record.messages[0].content else {
        panic!("fixture requires first user text")
    };
    task.prompt = TaskPrompt::User {
        content: prompt.clone(),
    };
    record.task_provenance = Some(TaskProvenance::from_task(&task).unwrap());
    record.verification_contract = Some(task.verification.contract());
    record
        .origin
        .generated_mut()
        .expect("generated record")
        .generation
        .sibling_group_id = Some(gw_storage::prompt_hash(&record.messages).unwrap());
}
