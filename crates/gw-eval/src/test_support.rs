//! Test-only synthetic [`TrainingRecord`] builders shared by the unit tests.
//!
//! These construct minimal-but-valid envelopes pinned to the three fields the separation
//! diagnostic reads: `hashes.prompt_hash` (the sibling-group id), `verification.all_passed`, and
//! `judging.aggregate`. Everything else is defaulted. Compiled only under `cfg(test)`.

use gw_schema::{
    Content, Generation, Hashes, Judging, Lifecycle, Message, Provenance, Role, TeacherRef,
    TrainingRecord, Verification,
};

/// One synthetic sibling in group `prompt_hash` with the given verifier outcome and optional
/// reasoning-quality aggregate.
///
/// `record_id` is derived from the hash + a per-call counter so siblings stay distinct without a
/// clock or RNG.
pub fn scored_sibling(
    prompt_hash: &str,
    all_passed: bool,
    aggregate: Option<f64>,
) -> TrainingRecord {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);

    TrainingRecord {
        record_id: format!("{prompt_hash}-{n}"),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "test-area".into(),
        tags: vec![],
        messages: vec![Message {
            role: Role::Assistant,
            content: Content::Text("answer".into()),
            reasoning: Some("because".into()),
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }],
        tools: None,
        provenance: Provenance {
            run_id: "run-1".into(),
            parent_ids: vec![],
            teacher: TeacherRef {
                provider: "openrouter".into(),
                slug: "test/model".into(),
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
        generation: Generation::default(),
        verification_contract: None,
        verification: Verification {
            checks: vec![],
            all_passed,
        },
        judging: Judging {
            aggregate,
            ..Default::default()
        },
        reasoning_quality: None,
        lifecycle: Lifecycle::default(),
        // The diagnostic reads prompt_hash directly; set it explicitly (no Store to recompute it).
        hashes: Hashes {
            prompt_hash: prompt_hash.into(),
            ..Default::default()
        },
        cost: Default::default(),
    }
}

/// A whole all-pass (or all-fail) sibling group: one [`scored_sibling`] per aggregate in `aggs`.
pub fn group(prompt_hash: &str, all_passed: bool, aggs: &[f64]) -> Vec<TrainingRecord> {
    aggs.iter()
        .map(|&a| scored_sibling(prompt_hash, all_passed, Some(a)))
        .collect()
}
