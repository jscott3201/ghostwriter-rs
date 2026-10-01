mod screening_support;
use gw_eval::screening::{prepare_screening, validate_screening_plan};
use gw_schema::*;
use gw_storage::{ExportPurpose, StorageError};
mod screened_publication_support;
use screened_publication_support::*;
use screening_support::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[tokio::test]
async fn complete_source_plan_publishes_exact_filtered_scope_with_population_proof() {
    let a = record("a", "ordinary training question");
    let mut b = record("b", "alpha beta gamma delta epsilon zeta eta theta");
    b.provenance.run_id = a.provenance.run_id.clone();
    let mut c = record("c", "unrelated held out question");
    c.task_provenance.as_mut().unwrap().split.role = TaskSplitRole::Test;
    let rows = vec![a, b, c];
    let mut sets = protected();
    sets[0].items[0].responses[0].content =
        Content::Text("alpha beta gamma delta epsilon zeta eta theta".into());
    sets[0].content_digest =
        gw_eval::screening::protected_screening_content_digest(&sets[0].items).unwrap();
    let plan = prepare_screening(&rows, &declaration(&rows), &sets, None).unwrap();
    assert_eq!(
        plan.lexical_status,
        LexicalScreeningStatus::MatchQuarantined
    );
    assert_eq!(plan.eligible_output, vec![key(&rows[0])]);
    let out = Temp::new();
    let store = setup(&rows, &out).await;
    let published = store
        .publish_screened_export(
            options(&plan),
            plan.clone(),
            out.artifact(),
            ExportPurpose::Standalone,
            validate(sets),
        )
        .await
        .unwrap();
    assert_eq!(published.artifact.metadata_version, 3);
    assert_eq!(published.artifact.manifest.n_records, 2);
    assert_eq!(published.artifact.manifest.n_admitted, 1);
    let metadata = serde_json::to_value(&published.artifact).unwrap();
    assert_eq!(metadata["screening"]["plan"]["plan_id"], plan.plan_id);
    assert_eq!(
        metadata["screening"]["population_check"],
        "transaction_checked"
    );
    assert_eq!(
        metadata["screening"]["plan"]["counts"]["population_records"],
        3
    );
    assert_eq!(
        metadata["screening"]["members"].as_array().unwrap().len(),
        1
    );
    let verified =
        gw_storage::verify_artifact_snapshot(std::fs::read(out.artifact()).unwrap()).unwrap();
    assert_eq!(verified.artifact, published.artifact);
    assert_eq!(
        store.get("a").await.unwrap().lifecycle.state,
        LifecycleState::Admitted
    );
    assert!(store.lifecycle_history("a").await.unwrap().is_empty());
    assert!(
        !serde_json::to_string(&metadata)
            .unwrap()
            .contains("alpha beta gamma")
    );
}

#[tokio::test]
async fn self_consistent_wrong_groups_must_reach_the_trusted_rerun_and_fail_without_side_effects() {
    let rows = vec![
        record("a", "identical full prompt"),
        record("b", "identical full prompt"),
    ];
    let sets = protected();
    let mut plan = prepare_screening(&rows, &declaration(&rows), &sets, None).unwrap();
    assert_eq!(plan.groups.len(), 1);
    let split = plan.groups[0].split.clone();
    plan.groups = rows
        .iter()
        .map(|row| FrozenScreeningGroup {
            group_id: hash("screening-group-anchor-v1", &key(row)),
            members: vec![key(row)],
            split: split.clone(),
            quarantined: false,
            reasons: vec![],
        })
        .collect();
    plan.groups.sort_by(|a, b| a.group_id.cmp(&b.group_id));
    plan.edges.clear();
    plan.counts.groups = 2;
    let identity_groups: Vec<_> = plan
        .groups
        .iter()
        .map(|group| {
            (
                &group.group_id,
                &group.members,
                &group.split,
                &group.reasons,
            )
        })
        .collect();
    plan.grouping_id = hash(
        "screening-grouping-splits-v1",
        &(identity_groups, &plan.edges),
    );
    plan.plan_id.clear();
    plan.plan_id = hash("frozen-screening-plan-v2", &plan);
    let out = Temp::new();
    let store = setup(&rows, &out).await;
    let invoked = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&invoked);
    let error = store
        .publish_screened_export(
            options(&plan),
            plan,
            out.artifact(),
            ExportPurpose::Engine,
            move |captured, candidate| {
                observed.store(true, Ordering::SeqCst);
                validate_screening_plan(captured, &sets, candidate)
                    .map_err(|error| StorageError::Export(error.to_string()))
            },
        )
        .await
        .unwrap_err();
    assert!(
        invoked.load(Ordering::SeqCst),
        "mandatory rerun was bypassed: {error}"
    );
    assert!(error.to_string().contains("stale or altered"));
    assert!(!out.artifact().exists());
    let pool = sqlx::SqlitePool::connect(&format!(
        "sqlite://{}",
        out.0.join("store.sqlite").display()
    ))
    .await
    .unwrap();
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM export_receipts")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count.0, 0);
    pool.close().await;
    for row in &rows {
        assert!(
            store
                .lifecycle_history(&row.record_id)
                .await
                .unwrap()
                .is_empty()
        );
    }
}

#[tokio::test]
async fn screened_engine_ack_and_explicit_recovery_do_not_stale_their_own_inputs() {
    let rows = vec![record("a", "ordinary training question")];
    let sets = protected();
    let plan = prepare_screening(&rows, &declaration(&rows), &sets, None).unwrap();
    let out = Temp::new();
    let store = setup(&rows, &out).await;
    let published = store
        .publish_screened_export(
            options(&plan),
            plan,
            out.artifact(),
            ExportPurpose::Engine,
            validate(sets),
        )
        .await
        .unwrap();
    assert_eq!(published.advanced_record_ids, vec!["a"]);
    let original = std::fs::read(out.artifact()).unwrap();
    for remove in [false, true, false] {
        if remove {
            std::fs::remove_file(out.artifact()).unwrap();
        }
        let recovered = store
            .resume_export(&published.publication_id)
            .await
            .unwrap();
        assert_eq!(recovered.artifact, published.artifact);
        assert!(recovered.advanced_record_ids.is_empty());
        assert_eq!(std::fs::read(out.artifact()).unwrap(), original);
    }
    assert_eq!(
        store.get("a").await.unwrap().lifecycle.state,
        LifecycleState::Exported
    );
    assert_eq!(store.lifecycle_history("a").await.unwrap().len(), 1);
}

#[tokio::test]
async fn required_fields_cover_excluded_source_records_and_tool_definitions() {
    let a = record("a", "ordinary training question");
    let mut excluded = record("b", "different excluded question");
    excluded.lifecycle.state = LifecycleState::Rejected;
    excluded.judging.verdict = Some(Verdict::Reject);
    excluded.messages[1].reasoning = Some(String::new());
    excluded.tools = Some(vec![serde_json::json!({"": 0})]);
    let rows = vec![a, excluded];
    let declared = declaration(&rows);
    let sets = protected();
    let plan = prepare_screening(&rows, &declared, &sets, None).unwrap();
    assert!(plan.incomplete.is_empty());
    assert_eq!(
        plan.required_fields,
        vec![
            ScreeningField::Content,
            ScreeningField::Reasoning,
            ScreeningField::ToolDefinition
        ]
    );
    assert_eq!(plan.eligible_output, vec![key(&rows[0])]);
    let mut missing = sets.clone();
    missing[0]
        .coverage
        .fields
        .retain(|field| *field != ScreeningField::ToolDefinition);
    let incomplete = prepare_screening(&rows, &declared, &missing, None).unwrap();
    assert!(
        incomplete
            .incomplete
            .iter()
            .any(|issue| issue.code == "incomplete_protected_coverage")
    );
    let out = Temp::new();
    let store = setup(&rows, &out).await;
    let published = store
        .publish_screened_export(
            options(&plan),
            plan.clone(),
            out.artifact(),
            ExportPurpose::Standalone,
            validate(sets),
        )
        .await
        .unwrap();
    let verified =
        gw_storage::verify_artifact_snapshot(std::fs::read(out.artifact()).unwrap()).unwrap();
    assert_eq!(verified.artifact, published.artifact);
    assert_eq!(
        verified
            .artifact
            .screening
            .as_ref()
            .unwrap()
            .plan
            .required_fields,
        plan.required_fields
    );
    assert_eq!(verified.artifact.manifest.n_admitted, 1);
}
