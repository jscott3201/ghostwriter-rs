//! Shared test fixtures: a unique temp path minter and a minimal `TrainingRecord` builder, mirroring
//! the gw-storage test record so the eval/export handler tests have a real persisted corpus.
//!
//! This module is compiled into EACH integration-test crate independently, so a helper used by only
//! some of them reads as dead code in the others — allow it crate-wide for this shared module.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use gw_schema::{
    Content, JudgeVote, Judging, Lifecycle, LifecycleState, Message, Provenance, Role, TeacherRef,
    TrainingRecord, Verdict,
};
use gw_storage::Store;

/// A process-unique temp file path with the given suffix (no external temp-file crate; the OS temp
/// dir is used and the caller removes the file).
pub fn unique_temp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("gw-cli-it-{}-{n}-{suffix}", std::process::id()));
    path
}

/// Build a minimal-but-valid record for `run_id`, with a chosen verdict + aggregate + verifier-pass
/// flag + a sibling-group key.
///
/// `group` becomes the record's USER-turn content, so siblings sharing a `group` get the SAME
/// `prompt_hash` once `Store::put` recomputes it (the separation diagnostic groups by `prompt_hash`),
/// and distinct groups get distinct hashes. (Setting `hashes.prompt_hash` directly would be
/// overwritten by `put`, which recomputes it from the message content — so the group key must live in
/// the content, not a pre-set hash.)
#[must_use]
pub fn record(
    record_id: &str,
    run_id: &str,
    verdict: Option<Verdict>,
    aggregate: Option<f64>,
    all_passed: bool,
    group: &str,
) -> TrainingRecord {
    let mut judging = Judging {
        verdict,
        aggregate,
        ..Default::default()
    };
    if let Some(a) = aggregate {
        judging.panel.push(JudgeVote {
            judge_model: "judge-x".into(),
            rubric_id: Some("rubric-1".into()),
            temperature: Some(0.0),
            top_p: None,
            seed: None,
            score: a,
            dimensions: None,
            rationale: None,
            raw_response: None,
        });
    }
    let verification = gw_schema::Verification {
        all_passed,
        ..Default::default()
    };

    TrainingRecord {
        record_id: record_id.into(),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "rust-async".into(),
        tags: vec![],
        messages: vec![
            Message {
                // The group key IS the user-turn content, so siblings in a group share a prompt_hash.
                role: Role::User,
                content: Content::Text(format!("question for group {group}")),
                reasoning: None,
                reasoning_details: None,
                tool_calls: None,
                name: None,
            },
            Message {
                role: Role::Assistant,
                content: Content::Text("96".into()),
                reasoning: Some("12*8 = 96".into()),
                reasoning_details: None,
                tool_calls: None,
                name: None,
            },
        ],
        tools: None,
        provenance: Provenance {
            run_id: run_id.into(),
            parent_ids: vec![],
            teacher: TeacherRef {
                provider: "openrouter".into(),
                slug: "z-ai/glm-5.2".into(),
                served_by: None,
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
        verification,
        judging,
        reasoning_quality: None,
        lifecycle: Lifecycle::default(),
        // Left at default; `Store::put` recomputes the content hashes (incl. prompt_hash) on insert.
        hashes: Default::default(),
        cost: Default::default(),
    }
}

/// Open a file-backed store at `path`, create `run_id`, and `put` every record. The returned store is
/// dropped by the caller; `path` (and `-wal`/`-shm` siblings) should be cleaned up after.
pub async fn seed_store(path: &std::path::Path, run_id: &str, records: &[TrainingRecord]) -> Store {
    let store = Store::open(path).await.expect("open store");
    store
        .create_run(run_id, "{}", Some(25.0))
        .await
        .expect("create run");
    for rec in records {
        store.put(rec).await.expect("put record");
    }
    store
}

/// Advance a record to `Admitted` so the export path (verdict==Admit) writes it.
pub async fn admit(store: &Store, record_id: &str) {
    store
        .advance_lifecycle(record_id, LifecycleState::Admitted, Some("test admit"))
        .await
        .expect("advance to Admitted");
}

/// Remove the SQLite file and its WAL/SHM siblings.
pub fn cleanup_db(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
    for ext in ["-wal", "-shm"] {
        let mut p = path.as_os_str().to_owned();
        p.push(ext);
        let _ = std::fs::remove_file(PathBuf::from(p));
    }
}
