//! The real standalone command must export the same selected trace as automatic end-of-run export.

mod common;

use std::collections::VecDeque;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};

use gw_engine::{
    AccountingPolicy, AreaConfig, Clients, Engine, EventSink, ExportSpec, InMemorySeedSource,
};
use gw_generate::{NullEmbedder, UserSeed, UserTurnCandidate, user_message};
use gw_judge::{AreaThresholds, NullSandboxOracle, PanelJudge};
use gw_providers::{
    ChatRequest, CompletionTokensDetails, DeltaStream, Provider, StreamChatFuture, StreamDelta,
    Usage,
};
use gw_schema::{
    CotPolicy, ExportManifest, LifecycleState, Oracle, ReasoningDetail, TrlFormat, Verdict,
    VerificationContract, VerificationKind,
};
use gw_storage::{
    ArtifactVerification, RecordFilter, Store, export_parquet_bytes, verify_artifact,
};
use tokio_util::sync::CancellationToken;

use common::{cleanup_db, unique_temp_path};

struct ScriptedProvider(Mutex<VecDeque<StreamDelta>>);

impl Provider for ScriptedProvider {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        let delta = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .expect("no provider re-spend");
        Box::pin(async move {
            let stream: DeltaStream = Box::pin(futures::stream::iter([Ok(delta)]));
            Ok(stream)
        })
    }
}

fn teacher_answer(answer: &str) -> StreamDelta {
    StreamDelta {
        content: Some(answer.into()),
        reasoning: Some(format!("Reasoning for {answer}")),
        reasoning_details: Some(vec![ReasoningDetail::Text {
            text: format!("Reasoning for {answer}"),
            signature: None,
            id: None,
            format: None,
            index: 0,
        }]),
        finish_reason: Some("stop".into()),
        usage: Some(Usage {
            prompt_tokens: Some(20),
            completion_tokens: Some(200),
            total_tokens: Some(220),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(150),
            }),
            cost: Some(0.01),
        }),
        ..Default::default()
    }
}

fn standalone_export(db: &Path, out: &Path, run_id: Option<&str>) -> ExportManifest {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gw"));
    command
        .args(["gen", "export", "--db"])
        .arg(db)
        .arg("--out")
        .arg(out)
        .args(["--format", "chat-ml", "--cot", "masked"]);
    if let Some(run_id) = run_id {
        command.args(["--run-id", run_id]);
    }
    let output = command.output().expect("run gw gen export");
    assert!(
        output.status.success(),
        "gw gen export failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("CLI prints an export manifest")
}

#[tokio::test]
async fn standalone_export_matches_automatic_best_of_k_and_replay() {
    let db = unique_temp_path("best-of-k.sqlite");
    let automatic = unique_temp_path("automatic.parquet");
    let standalone = unique_temp_path("standalone.parquet");
    let store = Store::open(&db).await.unwrap();
    let teacher = Arc::new(ScriptedProvider(Mutex::new(VecDeque::from([
        teacher_answer("96"),
        teacher_answer("97"),
    ]))));
    let judge = Arc::new(ScriptedProvider(Mutex::new(VecDeque::from([
        StreamDelta {
            content: Some(r#"{"score":0.95,"verdict":"accept"}"#.into()),
            finish_reason: Some("stop".into()),
            ..Default::default()
        },
        StreamDelta {
            content: Some(r#"{"score":0.90,"verdict":"accept"}"#.into()),
            finish_reason: Some("stop".into()),
            ..Default::default()
        },
    ]))));
    let clients = Clients::new(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        Arc::new(NullEmbedder),
        Arc::new(NullSandboxOracle),
        AccountingPolicy::ObservationOnly,
        EventSink::disconnected(),
        "test",
    );
    let area = AreaConfig::new(
        "math",
        "test-teacher",
        vec![PanelJudge::new("test-judge", "test-family")],
        "Grade the trace.",
    )
    .with_k(2)
    .with_thresholds(AreaThresholds {
        accept_threshold: 0.80,
        reject_below: 0.50,
        min_n_eff: 1.0,
        min_n_eff_ratio: 0.3,
    });
    let engine = Engine::new(clients, area, 1).with_export(ExportSpec {
        dst: automatic.clone(),
        target: TrlFormat::ChatML,
        cot: CotPolicy::Masked,
        dataset_version: None,
    });
    let source = InMemorySeedSource::new(
        vec![UserTurnCandidate {
            message: user_message("What is 12*8?"),
            seed: UserSeed::default(),
            contract: VerificationContract {
                kind: VerificationKind::None,
                oracle: Oracle::None,
                answer_marker: None,
            },
            answerable: true,
            difficulty_targeted: true,
            in_scope: true,
        }],
        1,
    );

    let report = engine
        .run("k2", &source, CancellationToken::new())
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!((report.admitted, report.rejected), (1, 1));
    let records = store.scan(&RecordFilter::new().run_id("k2")).await.unwrap();
    assert_eq!(records.len(), 2);
    assert!(
        records
            .iter()
            .all(|r| r.judging.verdict == Some(Verdict::Admit))
    );
    let winner = records
        .iter()
        .find(|r| {
            matches!(
                r.lifecycle.state,
                LifecycleState::Admitted | LifecycleState::Formatted | LifecycleState::Exported
            )
        })
        .unwrap();
    let loser = records
        .iter()
        .find(|r| r.lifecycle.state == LifecycleState::Rejected)
        .unwrap();
    assert_ne!(winner.hashes.record_hash, loser.hashes.record_hash);
    let (_, winner_manifest) = export_parquet_bytes(
        std::slice::from_ref(winner),
        TrlFormat::ChatML,
        CotPolicy::Masked,
    )
    .await
    .unwrap();
    let ArtifactVerification::Verified(auto_artifact) = verify_artifact(&automatic).unwrap() else {
        panic!("missing automatic metadata");
    };
    let auto_manifest = auto_artifact.manifest.clone();
    let automatic_bytes = std::fs::read(&automatic).unwrap();
    assert_eq!((auto_manifest.n_records, auto_manifest.n_admitted), (2, 1));
    assert_eq!(
        auto_manifest.build_inputs_hash,
        winner_manifest.build_inputs_hash
    );

    for run_filter in [Some("k2"), None] {
        let cli_manifest = standalone_export(&db, &standalone, run_filter);
        assert_eq!(
            cli_manifest, auto_manifest,
            "standalone and automatic selection must agree"
        );
        let ArtifactVerification::Verified(actual) = verify_artifact(&standalone).unwrap() else {
            panic!("missing standalone metadata");
        };
        assert_eq!(actual.manifest, auto_manifest);
        if run_filter.is_some() {
            assert_eq!(actual, auto_artifact);
        } else {
            assert_ne!(
                actual.artifact_id, auto_artifact.artifact_id,
                "scope participates in identity"
            );
        }
    }

    // Replaying the completed run cannot re-spend or change either shard or the retained audit row.
    assert!(teacher.0.lock().unwrap().is_empty());
    assert!(judge.0.lock().unwrap().is_empty());
    let replay = engine
        .run("k2", &source, CancellationToken::new())
        .await
        .unwrap();
    assert!(replay.completed);
    assert_eq!((replay.admitted, replay.rejected), (1, 1));
    assert_eq!(std::fs::read(&automatic).unwrap(), automatic_bytes);
    assert_eq!(
        standalone_export(&db, &standalone, Some("k2")),
        auto_manifest
    );
    assert_eq!(std::fs::read(&standalone).unwrap(), automatic_bytes);
    assert_eq!(
        store.scan(&RecordFilter::new().run_id("k2")).await.unwrap(),
        records
    );

    drop(engine);
    drop(store);
    for path in [automatic, standalone] {
        std::fs::remove_file(path).unwrap();
    }
    cleanup_db(&db);
}
