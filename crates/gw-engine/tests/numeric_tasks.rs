//! Prepared task identities and full-plan invalid-input preflight for library callers.
mod common;
use common::*;
use gw_engine::{
    CapturedSeedPlan, Engine, EventSink, InMemorySeedSource, NumericTaskSource, SeedSource,
};
use gw_schema::{NumericTaskDocument, TaskSplitRole};
use gw_storage::{RecordFilter, Store};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
const INPUT: &str = include_str!("../../../examples/reviewed-numeric-tasks.json");

fn digest(source: &impl SeedSource) -> String {
    CapturedSeedPlan::capture(source)
        .unwrap()
        .identity()
        .content_hash
        .clone()
}
fn first(source: &impl SeedSource) -> gw_generate::UserTurnCandidate {
    source.items_for_shard(0).remove(0).candidate
}

#[test]
fn materialization_preserves_seed_shard_offset_and_separates_semantic_and_plan_identity() {
    let doc = NumericTaskDocument::from_json(INPUT).unwrap();
    let source = NumericTaskSource::from_document(doc.clone(), 3).unwrap();
    assert_eq!(source.len(), 2);
    assert_eq!(source.shard_count(), 3);
    for ordinal in 0..2 {
        let item = source.items_for_shard(ordinal).remove(0);
        assert_eq!((item.seed, item.offset), (ordinal, 0));
        assert_eq!(
            gw_engine::record_id("run", ordinal, item.seed, 0, 0),
            format!("run-s{ordinal}-seed{ordinal}-a0-c0")
        );
        assert_eq!(
            item.candidate.task_provenance.as_ref().unwrap().task_id,
            doc.tasks[ordinal as usize].task_id
        );
    }
    assert!(source.items_for_shard(2).is_empty());
    let formatted = NumericTaskSource::from_json(&serde_json::to_string(&doc).unwrap(), 3).unwrap();
    assert_eq!(digest(&source), digest(&formatted));
    let semantic = first(&source).task_provenance.unwrap().identity;
    for variant in 0..10 {
        let mut changed = doc.clone();
        let task = &mut changed.tasks[0];
        match variant {
            0 => task.source.revision.push_str("-changed"),
            1 => {
                task.prompt = gw_schema::TaskPrompt::User {
                    content: "New arithmetic question. FINAL: format required.".into(),
                }
            }
            2 => {
                task.verification.oracle = gw_schema::NumericTaskOracle::Literal {
                    expected: "6".into(),
                }
            }
            3 => task.verification.numeric.extraction = gw_schema::NumericExtraction::WholeContent,
            4 => task.verification.numeric.tolerance.absolute = 0.25,
            5 => task.rights.reviewer.push_str("-changed"),
            6 => task.split.role = TaskSplitRole::Test,
            7 => task.verification.answer_policy = gw_schema::VerificationPolicy::Advisory,
            8 => task
                .observations
                .qc
                .evidence
                .push("Another reviewed observation".into()),
            _ => task.group.id.push_str("-changed"),
        }
        let changed = NumericTaskSource::from_document(changed, 3).unwrap();
        assert_ne!(
            digest(&source),
            digest(&changed),
            "plan omitted variant {variant}"
        );
        let changed_semantic = first(&changed).task_provenance.unwrap().identity;
        assert_eq!(
            semantic == changed_semantic,
            variant >= 5,
            "semantic variant {variant}"
        );
    }
    let mut reordered = doc;
    reordered.tasks.reverse();
    let reordered = NumericTaskSource::from_document(reordered, 3).unwrap();
    assert_ne!(digest(&source), digest(&reordered));
    assert_eq!(
        semantic,
        reordered.items_for_shard(1)[0]
            .candidate
            .task_provenance
            .as_ref()
            .unwrap()
            .identity
    );
}

#[tokio::test]
async fn malformed_library_tasks_and_forged_materialization_fail_before_launch_or_clients() {
    let doc = NumericTaskDocument::from_json(INPUT).unwrap();
    let mut invalid = doc.clone();
    invalid.tasks[1].source.revision.clear();
    assert!(NumericTaskSource::from_document(invalid, 1).is_err());
    assert!(NumericTaskSource::from_json("{bad-json}", 1).is_err());
    for variant in 0..6 {
        let store = Store::open_in_memory().await.unwrap();
        let teacher = Arc::new(ScriptedTeacher::new(vec![], 0));
        let judge = Arc::new(ScriptedJudge::new(vec![]));
        let engine = Engine::new(
            clients(
                store.clone(),
                teacher.clone(),
                judge.clone(),
                EventSink::disconnected(),
            ),
            area_k1(one_judge(), lenient_thresholds()),
            2,
        );
        let source = NumericTaskSource::from_document(doc.clone(), 1).unwrap();
        let mut candidates: Vec<_> = source
            .items_for_shard(0)
            .into_iter()
            .map(|item| item.candidate)
            .collect();
        match variant {
            0 => {
                candidates[1]
                    .task_provenance
                    .as_mut()
                    .unwrap()
                    .identity
                    .digest = "caller-claimed-hash".into()
            }
            1 => {
                candidates[1].message.content =
                    gw_schema::Content::Text("changed prompt without updating identity".into())
            }
            2 => {
                candidates[1]
                    .contract
                    .numeric
                    .as_mut()
                    .unwrap()
                    .tolerance
                    .relative = f64::INFINITY
            }
            3 => candidates[1].answerable = false,
            4 => {
                let group = candidates[0]
                    .task_provenance
                    .as_ref()
                    .unwrap()
                    .group
                    .clone();
                let task = candidates[1].task_provenance.as_mut().unwrap();
                task.group = group;
                task.split.role = TaskSplitRole::Test;
            }
            _ => candidates[1] = candidates[0].clone(),
        }
        let source = InMemorySeedSource::new(candidates, 2);
        assert!(
            engine
                .run("invalid-tasks", &source, CancellationToken::new())
                .await
                .is_err(),
            "variant {variant}"
        );
        assert_eq!(teacher.call_count(), 0);
        assert_eq!(judge.call_count(), 0);
        assert!(store.scan(&RecordFilter::new()).await.unwrap().is_empty());
        assert!(
            store
                .model_launches("invalid-tasks")
                .await
                .unwrap()
                .is_empty()
        );
        let runs: i64 = sqlx::query_scalar("SELECT count(*) FROM runs")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
        assert_eq!(runs, 0, "preflight registered invalid input");
    }
}

#[test]
fn plain_prompt_candidates_have_explicit_absent_axes_and_no_reviewed_task_claim() {
    let candidate = good_candidate("Plain prompt");
    assert!(candidate.task_provenance.is_none());
    assert_eq!(
        candidate.contract.answer_policy,
        Some(gw_schema::VerificationPolicy::Absent)
    );
    assert_eq!(
        candidate.contract.execution_policy,
        Some(gw_schema::VerificationPolicy::Absent)
    );
}
