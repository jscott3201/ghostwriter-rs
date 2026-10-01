//! Hermetic reviewed arithmetic tasks through verification, election, replay, and self-contained export.
mod common;
use arrow::array::{Array, StringArray};
use common::*;
use gw_engine::{Engine, EventSink, ExportSpec, NumericTaskSource, SeedSource};
use gw_schema::{
    CotPolicy, ExportTaskProjection, LifecycleState, NumericTaskDocument, TaskProvenance,
    TrlFormat, VerificationOutcome,
};
use gw_storage::{ArtifactVerification, RecordFilter, Store, verify_artifact};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::{
    fs::File,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio_util::sync::CancellationToken;
const INPUT: &str = include_str!("../../../examples/reviewed-numeric-tasks.json");

struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "gw-numeric-pipeline-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn export(&self) -> ExportSpec {
        ExportSpec {
            dst: self.0.join("tasks.parquet"),
            target: TrlFormat::OpenAiMessages,
            cot: CotPolicy::Supervised,
            dataset_version: None,
        }
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn document() -> NumericTaskDocument {
    NumericTaskDocument::from_json(INPUT).unwrap()
}
fn one_task() -> NumericTaskSource {
    let mut doc = document();
    doc.tasks.truncate(1);
    NumericTaskSource::from_document(doc, 1).unwrap()
}
fn fact(record: &gw_schema::TrainingRecord) -> VerificationOutcome {
    record
        .verification
        .interpretation
        .as_ref()
        .unwrap()
        .answer
        .observation
        .as_ref()
        .unwrap()
        .outcome
}

#[tokio::test]
async fn arithmetic_corpus_exports_only_elected_verified_rows_and_reopens_exact_provenance() {
    let temp = Temp::new();
    let db = temp.0.join("corpus.sqlite");
    let store = Store::open(&db).await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![
            answer_cot("FINAL: 5", 0.01),
            answer_cot("FINAL: 6", 0.01),
            answer_cot("Unclear answer", 0.01),
            answer_cot("3", 0.01),
            answer_cot("4", 0.01),
            answer_cot("Perhaps 3 or 4", 0.01),
        ],
        6,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.99, "accept"),
        &judge_body(0.99, "accept"),
    ]));
    let source = NumericTaskSource::from_json(INPUT, 1).unwrap();
    let expected: Vec<_> = source
        .items_for_shard(0)
        .into_iter()
        .map(|item| item.candidate.task_provenance.unwrap())
        .collect();
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            EventSink::disconnected(),
        ),
        area_k(one_judge(), lenient_thresholds(), 3),
        1,
    )
    .with_export(temp.export());
    let report = engine
        .run("corpus", &source, CancellationToken::new())
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!(report.exported, 2);
    assert_eq!(teacher.call_count(), 6);
    assert_eq!(judge.call_count(), 2);
    let records = store
        .scan(&RecordFilter::new().run_id("corpus"))
        .await
        .unwrap();
    assert_eq!(records.len(), 6);
    for task in &expected {
        let siblings: Vec<_> = records
            .iter()
            .filter(|rec| rec.task_provenance.as_ref().unwrap().task_id == task.task_id)
            .collect();
        assert_eq!(siblings.len(), 3);
        for record in &siblings {
            assert_eq!(record.task_provenance.as_ref(), Some(task));
            assert_eq!(
                record
                    .origin
                    .generated()
                    .expect("generated record")
                    .generation
                    .sibling_group_id
                    .as_deref(),
                Some(record.hashes.prompt_hash.as_str())
            );
            assert_ne!(
                record
                    .origin
                    .generated()
                    .expect("generated record")
                    .generation
                    .sibling_group_id
                    .as_deref(),
                Some(task.group.id.as_str())
            );
        }
        for (outcome, state) in [
            (VerificationOutcome::Pass, LifecycleState::Exported),
            (VerificationOutcome::Fail, LifecycleState::Rejected),
            (VerificationOutcome::Unknown, LifecycleState::NeedsReview),
        ] {
            let matches: Vec<_> = siblings
                .iter()
                .filter(|record| fact(record) == outcome)
                .collect();
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].lifecycle.state, state);
            if outcome != VerificationOutcome::Pass {
                assert!(matches[0].judging.panel.is_empty());
            }
        }
    }
    // Teacher-visible messages remain exactly the declared prompts; no reference answer/metadata injection.
    for request in teacher.seen_requests() {
        assert_eq!(request.messages.len(), 1);
        assert!(
            source
                .items_for_shard(0)
                .iter()
                .any(|item| request.messages[0] == item.candidate.message)
        );
    }
    let artifact = match verify_artifact(temp.export().dst).unwrap() {
        ArtifactVerification::Verified(value) => value,
        other => panic!("{other:?}"),
    };
    assert_eq!(artifact.manifest.n_records, 6);
    assert_eq!(artifact.manifest.n_admitted, 2);
    assert_eq!(
        artifact.manifest.column_schema_version,
        gw_schema::ExportSchemaVersion::CURRENT
    );
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(temp.export().dst).unwrap())
        .unwrap()
        .build()
        .unwrap();
    let mut exported = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        let ids = batch
            .column_by_name("record_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let tasks = batch
            .column_by_name("task_json")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            assert!(!tasks.is_null(i));
            let task: ExportTaskProjection = serde_json::from_str(tasks.value(i)).unwrap();
            let record = records
                .iter()
                .find(|record| record.record_id == ids.value(i))
                .unwrap();
            assert_eq!(fact(record), VerificationOutcome::Pass);
            assert_eq!(task.provenance, record.task_provenance.clone().unwrap());
            assert_eq!(
                task.verification_contract,
                record.verification_contract.clone().unwrap()
            );
            exported.push(task.provenance);
        }
    }
    exported.sort_by(|a, b| a.task_id.cmp(&b.task_id));
    assert_eq!(exported, expected);
    store.raw_pool().close().await;
    let reopened = Store::open(&db).await.unwrap();
    let loaded = reopened
        .scan(&RecordFilter::new().run_id("corpus"))
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(loaded).unwrap(),
        serde_json::to_value(records).unwrap()
    );
    reopened.raw_pool().close().await;
}

#[tokio::test]
async fn correct_numeric_answer_still_requires_quality_and_cannot_force_export() {
    let temp = Temp::new();
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![answer_cot("FINAL: 5", 0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.1, "reject")]));
    let report = Engine::new(
        clients(
            store.clone(),
            teacher,
            judge.clone(),
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    )
    .with_export(temp.export())
    .run("quality", &one_task(), CancellationToken::new())
    .await
    .unwrap();
    assert_eq!(report.exported, 0);
    assert_eq!(judge.call_count(), 1);
    let records = store.scan(&RecordFilter::new()).await.unwrap();
    assert_eq!(fact(&records[0]), VerificationOutcome::Pass);
    assert_eq!(records[0].lifecycle.state, LifecycleState::Rejected);
    let ArtifactVerification::Verified(artifact) = verify_artifact(temp.export().dst).unwrap()
    else {
        panic!("missing artifact")
    };
    assert_eq!(artifact.manifest.n_admitted, 0);
}

#[tokio::test]
async fn retry_and_pre_generation_error_stubs_keep_exact_task_provenance() {
    let source = one_task();
    let expected = source.items_for_shard(0)[0]
        .candidate
        .task_provenance
        .clone()
        .unwrap();
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![
            answer_cot("Initial explanation.\nFINAL: 5", 0.01),
            answer_cot("Improved explanation.\nFINAL: 5", 0.01),
        ],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.65, "revise"),
        &judge_body(0.99, "accept"),
    ]));
    Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    )
    .run("retry", &source, CancellationToken::new())
    .await
    .unwrap();
    let records = store.scan(&RecordFilter::new()).await.unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(teacher.call_count(), 2);
    let retry = records
        .iter()
        .find(|record| record.record_id.contains("-a1-"))
        .unwrap();
    assert_eq!(retry.lifecycle.state, LifecycleState::Formatted);
    for record in records {
        assert_eq!(record.task_provenance.as_ref(), Some(&expected));
        assert_eq!(
            record.verification_contract.as_ref(),
            Some(&source.items_for_shard(0)[0].candidate.contract)
        );
    }
    for k in [1, 2] {
        let store = Store::open_in_memory().await.unwrap();
        let teacher = Arc::new(FailingTeacher::new(1, 0.01));
        let judge = Arc::new(ScriptedJudge::new(vec![]));
        let report = Engine::new(
            clients(
                store.clone(),
                teacher,
                judge.clone(),
                EventSink::disconnected(),
            ),
            area_k(one_judge(), lenient_thresholds(), k),
            1,
        )
        .run("stub", &source, CancellationToken::new())
        .await
        .unwrap();
        assert_eq!(report.errored, 1);
        assert_eq!(judge.call_count(), 0);
        let records = store.scan(&RecordFilter::new()).await.unwrap();
        assert_eq!(records.len(), k as usize);
        let error = records
            .iter()
            .find(|record| record.lifecycle.state == LifecycleState::Error)
            .unwrap();
        assert_eq!(error.messages.len(), 1);
        for record in records {
            assert_eq!(record.task_provenance.as_ref(), Some(&expected));
        }
    }
}

#[tokio::test]
async fn compatible_interrupted_task_replay_reuses_generation_and_changed_inputs_leave_state_untouched()
 {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![answer_cot("FINAL: 5", 0.01), answer_cot("3", 0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.99, "accept"),
        &judge_body(0.99, "accept"),
    ]));
    let clients = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let source = NumericTaskSource::from_json(INPUT, 1).unwrap();
    sqlx::query("CREATE TRIGGER stop_before_checkpoint BEFORE INSERT ON checkpoints BEGIN SELECT RAISE(FAIL,'interrupted-numeric-checkpoint'); END").execute(store.raw_pool()).await.unwrap();
    let error = Engine::new(clients.clone(), area.clone(), 1)
        .run("replay", &source, CancellationToken::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("interrupted-numeric-checkpoint"));
    assert_eq!(teacher.call_count(), 1);
    assert_eq!(judge.call_count(), 1);
    sqlx::query("DROP TRIGGER stop_before_checkpoint")
        .execute(store.raw_pool())
        .await
        .unwrap();
    let records = store.scan(&RecordFilter::new()).await.unwrap();
    let bytes = serde_json::to_vec(&records).unwrap();
    let accounting = store.accounting_snapshot("replay").await.unwrap();
    let launches = store.model_launches("replay").await.unwrap();
    let metadata: (String, String, String) =
        sqlx::query_as("SELECT config_json,created_at,status FROM runs WHERE run_id='replay'")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
    for variant in 0..4 {
        let mut doc = document();
        match variant {
            0 => {
                doc.tasks[0].verification.oracle = gw_schema::NumericTaskOracle::Literal {
                    expected: "7".into(),
                }
            }
            1 => doc.tasks[0].source.revision.push_str("-changed"),
            2 => doc.tasks[0].split.role = gw_schema::TaskSplitRole::Test,
            _ => doc.tasks[0]
                .rights
                .evidence
                .push("changed reviewed rights".into()),
        }
        let changed = NumericTaskSource::from_document(doc, 1).unwrap();
        assert!(
            Engine::new(clients.clone(), area.clone(), 1)
                .run("replay", &changed, CancellationToken::new())
                .await
                .is_err()
        );
        assert_eq!(teacher.call_count(), 1);
        assert_eq!(judge.call_count(), 1);
        assert_eq!(
            store.accounting_snapshot("replay").await.unwrap(),
            accounting
        );
        assert_eq!(store.model_launches("replay").await.unwrap(), launches);
        assert_eq!(
            serde_json::to_vec(&store.scan(&RecordFilter::new()).await.unwrap()).unwrap(),
            bytes
        );
        let after: (String, String, String) =
            sqlx::query_as("SELECT config_json,created_at,status FROM runs WHERE run_id='replay'")
                .fetch_one(store.raw_pool())
                .await
                .unwrap();
        assert_eq!(after, metadata);
    }
    let reformatted =
        NumericTaskSource::from_json(&serde_json::to_string(&document()).unwrap(), 1).unwrap();
    let report = Engine::new(clients, area, 1)
        .run("replay", &reformatted, CancellationToken::new())
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!(teacher.call_count(), 2);
    assert_eq!(judge.call_count(), 2);
    for record in store.scan(&RecordFilter::new()).await.unwrap() {
        let task = document()
            .tasks
            .into_iter()
            .find(|task| task.task_id == record.task_provenance.as_ref().unwrap().task_id)
            .unwrap();
        assert_eq!(
            record.task_provenance,
            Some(TaskProvenance::from_task(&task).unwrap())
        );
    }
}
