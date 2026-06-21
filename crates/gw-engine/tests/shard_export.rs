//! End-of-run shard export tests: opt-in Parquet write, skip cases, and non-fatal export failure.

mod common;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use common::*;
use gw_engine::{Engine, EngineEvent, EventSink, ExportSpec};
use gw_schema::{CotPolicy, ExportManifest, TrlFormat};
use gw_storage::Store;
use tokio::sync::mpsc::Receiver;
use tokio_util::sync::CancellationToken;

fn temp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "gw-engine-export-{}-{n}-{suffix}",
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
        dataset_version: Some(semver::Version::new(1, 2, 3)),
    }
}

fn drain_events(rx: &mut Receiver<EngineEvent>) -> Vec<EngineEvent> {
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    events
}

fn event_position<F>(events: &[EngineEvent], pred: F) -> Option<usize>
where
    F: Fn(&EngineEvent) -> bool,
{
    events.iter().position(pred)
}

fn accepting_engine(
    store: Store,
    cap_usd: f64,
    sink: EventSink,
    export: Option<ExportSpec>,
) -> Engine {
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let cl = clients(store, teacher, judge, cap_usd, sink);
    let engine = Engine::new(cl, area_k1(one_judge(), lenient_thresholds()), 1);
    match export {
        Some(spec) => engine.with_export(spec),
        None => engine,
    }
}

#[tokio::test]
async fn completed_run_writes_parquet_manifest_and_export_event_before_finish() {
    let dst = temp_path("completed.parquet");
    cleanup_export(&dst);
    let store = Store::open_in_memory().await.unwrap();
    let (sink, mut rx) = EventSink::subscribe();
    let engine = accepting_engine(store, 25.0, sink, Some(export_spec(dst.clone())));

    let report = engine
        .run("run-export", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    assert!(report.completed);
    assert_eq!(report.admitted, 1);
    assert!(dst.exists(), "Parquet shard is written");
    let sidecar = sidecar_path(&dst);
    assert!(sidecar.exists(), "manifest sidecar is written");

    let sidecar_manifest: ExportManifest =
        serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    assert_eq!(sidecar_manifest.n_admitted, 1);
    assert_eq!(
        sidecar_manifest.dataset_version,
        Some(semver::Version::new(1, 2, 3))
    );
    assert_eq!(sidecar_manifest.target, TrlFormat::ChatML);
    assert_eq!(sidecar_manifest.cot_policy, CotPolicy::Masked);

    let events = drain_events(&mut rx);
    let export_pos = event_position(&events, |event| {
        matches!(
            event,
            EngineEvent::ShardExported {
                manifest: ExportManifest { n_admitted: 1, .. },
                ..
            }
        )
    })
    .expect("ShardExported emitted");
    let finish_pos = event_position(&events, |event| {
        matches!(
            event,
            EngineEvent::RunFinished {
                completed: true,
                ..
            }
        )
    })
    .expect("RunFinished emitted");
    assert!(
        export_pos < finish_pos,
        "ShardExported must be emitted before RunFinished"
    );

    cleanup_export(&dst);
}

#[tokio::test]
async fn halted_run_skips_shard_export_even_with_admitted_records() {
    let dst = temp_path("halted.parquet");
    cleanup_export(&dst);
    let store = Store::open_in_memory().await.unwrap();
    let (sink, mut rx) = EventSink::subscribe();
    let engine = accepting_engine(store, 0.01, sink, Some(export_spec(dst.clone())));

    let report = engine
        .run("run-halted", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    assert!(!report.completed, "spent cap means the run is halted");
    assert_eq!(
        report.admitted, 1,
        "the admitted record itself still stands"
    );
    assert!(!dst.exists());
    assert!(!sidecar_path(&dst).exists());
    let events = drain_events(&mut rx);
    assert!(
        !events.iter().any(|event| matches!(
            event,
            EngineEvent::ShardExported { .. } | EngineEvent::ShardExportFailed { .. }
        )),
        "halted runs skip auto-export"
    );

    cleanup_export(&dst);
}

#[tokio::test]
async fn zero_admitted_run_skips_shard_export() {
    let dst = temp_path("zero-admitted.parquet");
    cleanup_export(&dst);
    let store = Store::open_in_memory().await.unwrap();
    let (sink, mut rx) = EventSink::subscribe();
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.2, "reject")]));
    let cl = clients(store, teacher, judge, 25.0, sink);
    let engine = Engine::new(cl, area_k1(one_judge(), lenient_thresholds()), 1)
        .with_export(export_spec(dst.clone()));

    let report = engine
        .run("run-zero", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    assert!(report.completed);
    assert_eq!(report.admitted, 0);
    assert_eq!(report.rejected, 1);
    assert!(!dst.exists());
    assert!(!sidecar_path(&dst).exists());
    let events = drain_events(&mut rx);
    assert!(
        !events.iter().any(|event| matches!(
            event,
            EngineEvent::ShardExported { .. } | EngineEvent::ShardExportFailed { .. }
        )),
        "runs with no admitted records skip auto-export"
    );

    cleanup_export(&dst);
}

#[tokio::test]
async fn export_failure_emits_event_and_keeps_run_successful() {
    let missing_parent = temp_path("missing-parent");
    let dst = missing_parent.join("out.parquet");
    let store = Store::open_in_memory().await.unwrap();
    let (sink, mut rx) = EventSink::subscribe();
    let engine = accepting_engine(store, 25.0, sink, Some(export_spec(dst.clone())));

    let report = engine
        .run(
            "run-export-fail",
            &one_item_source(),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert!(report.completed);
    assert_eq!(report.admitted, 1);
    assert!(!dst.exists());
    let events = drain_events(&mut rx);
    let failure_pos = event_position(&events, |event| {
        matches!(event, EngineEvent::ShardExportFailed { error, .. } if error.contains("io error"))
    })
    .expect("ShardExportFailed emitted");
    let finish_pos = event_position(&events, |event| {
        matches!(
            event,
            EngineEvent::RunFinished {
                completed: true,
                ..
            }
        )
    })
    .expect("RunFinished emitted");
    assert!(
        failure_pos < finish_pos,
        "ShardExportFailed must be emitted before RunFinished"
    );
}
