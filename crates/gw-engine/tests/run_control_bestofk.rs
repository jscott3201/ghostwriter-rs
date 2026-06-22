//! Best-of-k run-control regressions: Abort/cancellation must not double-admit a sibling group.

mod common;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::{Array, StringArray};
use common::*;
use gw_engine::{
    Engine, EngineEvent, EventSink, ExportSpec, InMemorySeedSource, SeedSource, drive,
    drive_to_judged, load_cursor, record_id, step,
};
use gw_generate::{
    RecordContext, SamplingPreset, SiblingPlan, TeacherCall, assemble, generate_assistant,
    synthesize_user_turn,
};
use gw_schema::{BudgetBreach, CotPolicy, LifecycleState, TeacherRef, TrainingRecord, TrlFormat};
use gw_storage::{RecordFilter, Store, now_rfc3339, prompt_hash};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tokio::sync::mpsc::Receiver;
use tokio_util::sync::CancellationToken;

fn source(prompts: &[&str], shards: usize) -> InMemorySeedSource {
    InMemorySeedSource::new(
        prompts
            .iter()
            .map(|prompt| good_candidate(prompt))
            .collect(),
        shards,
    )
}

fn temp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "gw-engine-run-control-{}-{n}-{suffix}",
        std::process::id()
    ));
    path
}

fn sidecar_path(dst: &Path) -> PathBuf {
    let mut path: OsString = dst.as_os_str().to_owned();
    path.push(".manifest.json");
    PathBuf::from(path)
}

fn cleanup_export(dst: &Path) {
    let _ = std::fs::remove_file(dst);
    let _ = std::fs::remove_file(sidecar_path(dst));
}

fn export_spec(dst: PathBuf) -> ExportSpec {
    ExportSpec {
        dst,
        target: TrlFormat::ChatML,
        cot: CotPolicy::Masked,
        dataset_version: None,
    }
}

fn exported_record_ids(dst: &Path) -> Vec<String> {
    let file = std::fs::File::open(dst).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let mut ids = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        let column = batch
            .column_by_name("record_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        ids.extend((0..column.len()).map(|i| column.value(i).to_string()));
    }
    ids
}

fn cancel_on_state(
    mut rx: Receiver<EngineEvent>,
    cancel: CancellationToken,
    state: LifecycleState,
) -> tokio::task::JoinHandle<bool> {
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if matches!(event, EngineEvent::StateAdvanced { to, .. } if to == state) {
                cancel.cancel();
                return true;
            }
        }
        false
    })
}

async fn records(store: &Store, run_id: &str) -> Vec<TrainingRecord> {
    store
        .scan(&RecordFilter::new().run_id(run_id))
        .await
        .unwrap()
}

fn admitted_record_ids(records: &[TrainingRecord]) -> Vec<String> {
    let mut ids: Vec<_> = records
        .iter()
        .filter(|record| {
            matches!(
                record.lifecycle.state,
                LifecycleState::Admitted | LifecycleState::Formatted | LifecycleState::Exported
            )
        })
        .map(|record| record.record_id.clone())
        .collect();
    ids.sort();
    ids
}

async fn seed_sibling_assistant_generated(
    store: &Store,
    teacher: &Arc<ScriptedTeacher>,
    run_id: &str,
    record_id: &str,
    completion_index: u32,
    n_completions: u32,
) -> TrainingRecord {
    let gated =
        synthesize_user_turn(good_candidate("12*8?"), &gw_generate::NullEmbedder, &[]).unwrap();
    let call = TeacherCall::new("z-ai/glm-5.2", vec![gated.candidate.message.clone()], 16384)
        .with_sampling(SamplingPreset::official().with_seed(0));
    let turn = generate_assistant(teacher.as_ref(), &gated, &call)
        .await
        .unwrap();
    let ctx = RecordContext {
        record_id: record_id.into(),
        run_id: run_id.into(),
        training_area: "math".into(),
        harness_version: "0.1.0-test".into(),
        git_commit: None,
        now_rfc3339: now_rfc3339(),
        user_synth_model: None,
    };
    let teacher_ref = TeacherRef {
        provider: "openrouter".into(),
        slug: "z-ai/glm-5.2".into(),
        served_by: None,
        model_card_revision: None,
    };
    let plan = SiblingPlan {
        completion_index,
        n_completions,
        sampling: call.sampling,
    };
    let mut rec = assemble(
        &ctx,
        &gated,
        turn,
        teacher_ref,
        call.generation(),
        Some(plan),
    );
    rec.generation.sibling_group_id = Some(prompt_hash(&rec.messages).unwrap());
    store.put(&rec).await.unwrap();
    store.get(record_id).await.unwrap()
}

async fn seed_assistant_generated(
    store: &Store,
    teacher: &Arc<ScriptedTeacher>,
    run_id: &str,
    record_id: &str,
) -> TrainingRecord {
    store.create_run(run_id, "{}", Some(25.0)).await.unwrap();
    seed_sibling_assistant_generated(store, teacher, run_id, record_id, 0, 1).await
}

#[tokio::test]
async fn partial_revising_winner_resume_does_not_elect_judged_runner_up() {
    let dst = temp_path("partial-revising-winner.parquet");
    cleanup_export(&dst);
    let run_id = "run-partial-revising-window";
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![
            answer_cot("winner-attempt", 0.01),
            answer_cot("runner-attempt", 0.01),
            answer_cot("winner-retry", 0.01),
            answer_cot("runner-retry-if-buggy", 0.01),
        ],
        4,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.70, "revise"),
        &judge_body(0.60, "revise"),
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let seed_source = source(&["q0"], 1);
    store
        .validate_or_record_run_partition(run_id, 1, &seed_source.prompts_hash().unwrap())
        .await
        .unwrap();
    store.create_run(run_id, "{}", Some(25.0)).await.unwrap();

    let item = seed_source.items_for_shard(0).remove(0);
    let area = area_k(one_judge(), lenient_thresholds(), 2);
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        25.0,
        EventSink::disconnected(),
    );
    let winner_id = record_id(run_id, 0, item.seed, 0, 0);
    let runner_id = record_id(run_id, 0, item.seed, 0, 1);

    let winner = seed_sibling_assistant_generated(&store, &teacher, run_id, &winner_id, 0, 2).await;
    let winner = drive_to_judged(winner, &cl, &area, &CancellationToken::new())
        .await
        .unwrap();
    let winner = step(winner, &cl, &area).await.unwrap();
    assert_eq!(winner.lifecycle.state, LifecycleState::Revising);

    let runner = seed_sibling_assistant_generated(&store, &teacher, run_id, &runner_id, 1, 2).await;
    let runner = drive_to_judged(runner, &cl, &area, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(runner.lifecycle.state, LifecycleState::Judged);

    let engine = Engine::new(cl, area, 1).with_export(export_spec(dst.clone()));
    let report = engine
        .run(run_id, &seed_source, CancellationToken::new())
        .await
        .unwrap();

    assert!(report.completed);
    assert_eq!(report.admitted, 1);
    assert_eq!(
        teacher.call_count(),
        3,
        "resume spends only the established Revising winner's retry"
    );
    assert_eq!(judge.call_count(), 3);
    let final_records = records(&store, run_id).await;
    assert_eq!(
        final_records
            .iter()
            .filter(|record| record.lifecycle.state == LifecycleState::Rejected)
            .count(),
        1,
        "the stranded runner-up is retained, not elected"
    );
    let admitted_ids = admitted_record_ids(&final_records);
    assert_eq!(admitted_ids.len(), 1);
    assert_eq!(exported_record_ids(&dst), admitted_ids);

    cleanup_export(&dst);
}

#[tokio::test]
async fn revising_winner_cancel_window_resumes_without_second_admit_and_exports_one_row() {
    let dst = temp_path("revising-winner.parquet");
    cleanup_export(&dst);
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![
            answer_cot("winner-attempt", 0.01),
            answer_cot("runner-attempt", 0.01),
            answer_cot("winner-retry", 0.01),
            answer_cot("runner-retry-if-buggy", 0.01),
        ],
        4,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.70, "revise"),
        &judge_body(0.60, "revise"),
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let seed_source = source(&["q0"], 1);
    let cancel = CancellationToken::new();
    let (sink, rx) = EventSink::subscribe();
    let cancel_task = cancel_on_state(rx, cancel.clone(), LifecycleState::Revising);
    let first_engine = Engine::new(
        clients(store.clone(), teacher.clone(), judge.clone(), 25.0, sink),
        area_k(one_judge(), lenient_thresholds(), 2),
        1,
    )
    .with_on_breach(BudgetBreach::Abort);

    let first = first_engine
        .run("run-revising-window", &seed_source, cancel)
        .await
        .unwrap();
    drop(first_engine);
    assert!(
        cancel_task.await.unwrap(),
        "test must cancel after the elected winner reaches Revising"
    );

    assert!(!first.completed);
    let first_records = records(&store, "run-revising-window").await;
    assert_eq!(
        first_records
            .iter()
            .filter(|record| record.lifecycle.state == LifecycleState::Revising)
            .count(),
        1
    );
    assert_eq!(
        first_records
            .iter()
            .filter(|record| record.lifecycle.state == LifecycleState::Rejected)
            .count(),
        1,
        "finalization is atomic once entered: the runner-up is retained before the halt"
    );
    assert_eq!(
        load_cursor(&store, "run-revising-window", 0)
            .await
            .unwrap()
            .next_offset,
        0
    );

    let resumed_engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            25.0,
            EventSink::disconnected(),
        ),
        area_k(one_judge(), lenient_thresholds(), 2),
        1,
    )
    .with_on_breach(BudgetBreach::Abort)
    .with_export(export_spec(dst.clone()));
    let resumed = resumed_engine
        .run(
            "run-revising-window",
            &seed_source,
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert!(resumed.completed);
    assert_eq!(resumed.admitted, 1);
    let final_records = records(&store, "run-revising-window").await;
    let admitted_ids = admitted_record_ids(&final_records);
    assert_eq!(
        admitted_ids.len(),
        1,
        "a Revising established winner must not allow a second elected admit"
    );
    assert_eq!(exported_record_ids(&dst), admitted_ids);

    cleanup_export(&dst);
}

#[tokio::test]
async fn formatted_winner_cancel_window_retains_runner_and_resumes_one_admit() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![answer_cot("winner", 0.01), answer_cot("runner", 0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.90, "accept"),
    ]));
    let seed_source = source(&["q0"], 1);
    let cancel = CancellationToken::new();
    let (sink, rx) = EventSink::subscribe();
    let cancel_task = cancel_on_state(rx, cancel.clone(), LifecycleState::Formatted);
    let first_engine = Engine::new(
        clients(store.clone(), teacher.clone(), judge.clone(), 25.0, sink),
        area_k(one_judge(), lenient_thresholds(), 2),
        1,
    );

    let first = first_engine
        .run("run-formatted-window", &seed_source, cancel)
        .await
        .unwrap();
    drop(first_engine);
    assert!(
        cancel_task.await.unwrap(),
        "test must cancel after the winner reaches Formatted"
    );

    assert!(!first.completed);
    let first_records = records(&store, "run-formatted-window").await;
    assert_eq!(admitted_record_ids(&first_records).len(), 1);
    assert_eq!(
        first_records
            .iter()
            .filter(|record| record.lifecycle.state == LifecycleState::Rejected)
            .count(),
        1,
        "runner-up is retained even if cancellation fires during winner finalization"
    );
    assert_eq!(
        load_cursor(&store, "run-formatted-window", 0)
            .await
            .unwrap()
            .next_offset,
        0
    );

    let resumed_engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            25.0,
            EventSink::disconnected(),
        ),
        area_k(one_judge(), lenient_thresholds(), 2),
        1,
    );
    let resumed = resumed_engine
        .run(
            "run-formatted-window",
            &seed_source,
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert!(resumed.completed);
    assert_eq!(resumed.admitted, 1);
    assert_eq!(teacher.call_count(), 2, "resume does not re-spend siblings");
    assert_eq!(judge.call_count(), 2, "resume reuses judged siblings");
    assert_eq!(
        admitted_record_ids(&records(&store, "run-formatted-window").await).len(),
        1
    );
}

#[tokio::test]
async fn in_group_abort_budget_gate_commits_settled_group_and_resume_no_respend() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![answer_cot("first", 0.01), answer_cot("second", 0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.90, "accept"),
    ]));
    let seed_source = source(&["q0"], 1);
    let first_engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            0.005,
            EventSink::disconnected(),
        ),
        area_k(one_judge(), lenient_thresholds(), 2),
        1,
    )
    .with_on_breach(BudgetBreach::Abort);

    let first = first_engine
        .run(
            "run-k2-ingroup-abort",
            &seed_source,
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert!(!first.completed);
    assert_eq!(
        teacher.call_count(),
        2,
        "fan-out permits both siblings to pass the budget gate before either charge lands"
    );
    assert_eq!(judge.call_count(), 2);
    let states = records(&store, "run-k2-ingroup-abort")
        .await
        .into_iter()
        .map(|record| record.lifecycle.state)
        .collect::<Vec<_>>();
    assert_eq!(states.len(), 2);
    assert!(
        states.iter().any(|state| matches!(
            state,
            LifecycleState::Exported | LifecycleState::Admitted | LifecycleState::Formatted
        )),
        "one sibling is still admitted before the abort boundary: {states:?}"
    );
    assert!(
        states.contains(&LifecycleState::Rejected),
        "the non-winning sibling is retained: {states:?}"
    );
    assert_eq!(
        load_cursor(&store, "run-k2-ingroup-abort", 0)
            .await
            .unwrap()
            .next_offset,
        1
    );

    let resumed_engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            25.0,
            EventSink::disconnected(),
        ),
        area_k(one_judge(), lenient_thresholds(), 2),
        1,
    )
    .with_on_breach(BudgetBreach::Abort);
    let resumed = resumed_engine
        .run(
            "run-k2-ingroup-abort",
            &seed_source,
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert!(resumed.completed);
    assert_eq!(resumed.admitted, 1);
    assert_eq!(resumed.rejected, 1);
    assert_eq!(
        teacher.call_count(),
        2,
        "resume does not re-spend already generated siblings"
    );
    assert_eq!(judge.call_count(), 2);
}

#[tokio::test]
async fn pre_cancelled_drive_helpers_return_unchanged_without_provider_calls() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![good_cot(0.01), good_cot(0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());

    let assistant =
        seed_assistant_generated(&store, &teacher, "run-pre-cancel-a", "rec-assistant").await;
    let cancel = CancellationToken::new();
    cancel.cancel();
    let unchanged = drive_to_judged(assistant.clone(), &cl, &area, &cancel)
        .await
        .unwrap();
    assert_eq!(unchanged.record_id, assistant.record_id);
    assert_eq!(
        unchanged.lifecycle.state,
        LifecycleState::AssistantGenerated
    );
    assert_eq!(teacher.call_count(), 1);
    assert_eq!(judge.call_count(), 0);

    let verified_seed =
        seed_assistant_generated(&store, &teacher, "run-pre-cancel-v", "rec-verified").await;
    let verified = step(verified_seed, &cl, &area).await.unwrap();
    assert_eq!(verified.lifecycle.state, LifecycleState::Verified);
    assert_eq!(teacher.call_count(), 2);
    assert_eq!(judge.call_count(), 0);

    let unchanged = drive(verified.clone(), &cl, &area, &cancel).await.unwrap();
    assert_eq!(unchanged.record_id, verified.record_id);
    assert_eq!(unchanged.lifecycle.state, LifecycleState::Verified);
    assert_eq!(teacher.call_count(), 2);
    assert_eq!(judge.call_count(), 0);
}
