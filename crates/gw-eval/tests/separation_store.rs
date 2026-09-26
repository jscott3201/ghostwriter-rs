//! End-to-end integration test for the async Store-reading wrapper
//! [`gw_eval::analyze_store`].
//!
//! In-process: in-memory SQLite, no network, no temp files. The point is to prove the wrapper
//! grouping survives a real `Store::put` round-trip — in particular that siblings sharing a
//! prompt land in ONE group keyed by the store-recomputed `prompt_hash`, while a different prompt
//! forms a separate group.

use gw_eval::{SeparationConfig, analyze_store};
use gw_schema::{
    Content, Generation, Hashes, Judging, Lifecycle, Message, Provenance, Role, TeacherRef,
    TrainingRecord, Verification,
};
use gw_storage::{RecordFilter, Store};

/// A record whose PROMPT (the user turn) is `prompt`, so the store-recomputed `prompt_hash`
/// groups all siblings sharing that prompt. `idx` keeps sibling `record_id`s distinct.
fn sibling(
    prompt: &str,
    idx: u32,
    all_passed: bool,
    aggregate: Option<f64>,
    answer: &str,
) -> TrainingRecord {
    TrainingRecord {
        record_id: format!("{prompt}-{idx}"),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "rust-async".into(),
        tags: vec![],
        messages: vec![
            Message {
                role: Role::User,
                content: Content::Text(prompt.into()),
                reasoning: None,
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Assistant,
                content: Content::Text(answer.into()),
                reasoning: Some("reasoning".into()),
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ],
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
        execution_evidence: None,
        verification: Verification {
            all_passed,
            ..Default::default()
        },
        judging: Judging {
            aggregate,
            ..Default::default()
        },
        reasoning_quality: None,
        lifecycle: Lifecycle::default(),
        // Store::put recomputes hashes from content; leave empty so we exercise that path.
        hashes: Hashes::default(),
        cost: Default::default(),
    }
}

#[tokio::test]
async fn analyze_store_groups_siblings_by_recomputed_prompt_hash() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .create_run("run-1", "{\"budget\":25}", Some(25.0))
        .await
        .unwrap();

    // Prompt A: an all-pass group of 3 scored siblings, argmax 0.9 > group mean 0.5.
    for (i, (agg, ans)) in [(0.9, "a1"), (0.5, "a2"), (0.1, "a3")].iter().enumerate() {
        let rec = sibling("What is 12*8?", i as u32, true, Some(*agg), ans);
        store.put(&rec).await.unwrap();
    }
    // Prompt B: a mixed group (one pass, one fail) — decidable.
    store
        .put(&sibling("Capital of France?", 0, true, None, "Paris"))
        .await
        .unwrap();
    store
        .put(&sibling("Capital of France?", 1, false, None, "London"))
        .await
        .unwrap();

    let report = analyze_store(&store, &RecordFilter::new(), &SeparationConfig::default())
        .await
        .unwrap();

    // Two distinct prompts ⇒ two groups, no singletons.
    assert_eq!(report.n_groups, 2);
    assert_eq!(report.n_singletons, 0);
    // Prompt A is all-pass; Prompt B is mixed.
    assert_eq!(report.n_allpass, 1);
    assert_eq!(report.n_mixed, 1);
    assert_eq!(report.n_allfail, 0);
    // 1 mixed / 2 multi-sibling groups.
    assert!((report.decidable_fraction - 0.5).abs() < 1e-12);
    // The all-pass group is selector-eligible (3 scored siblings); argmax 0.9 vs mean 0.5.
    assert_eq!(report.n_selector_eligible, 1);
    assert!((report.selector_mean - 0.9).abs() < 1e-12);
    assert!((report.control_per_prompt_mean - 0.5).abs() < 1e-12);
    assert!((report.selector_mean_gap - 0.4).abs() < 1e-12);
    assert!((report.selector_winrate - 1.0).abs() < 1e-12);
}

#[tokio::test]
async fn analyze_store_respects_record_filter() {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("run-1", "{}", Some(25.0)).await.unwrap();
    // A single mixed group in run-1.
    store
        .put(&sibling("Q?", 0, true, None, "yes"))
        .await
        .unwrap();
    store
        .put(&sibling("Q?", 1, false, None, "no"))
        .await
        .unwrap();

    // Filtering to a non-existent run yields an empty corpus ⇒ all-zero report.
    let empty = analyze_store(
        &store,
        &RecordFilter::new().run_id("does-not-exist"),
        &SeparationConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(empty.n_groups, 0);
    assert!((empty.decidable_fraction - 0.0).abs() < 1e-12);
    // Empty corpus has 0 decidable groups ⇒ low_data WARN.
    assert!(empty.low_data);
}
