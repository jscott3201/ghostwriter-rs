//! End-to-end, HERMETIC flow tests for `gw-generate`: a fake in-memory [`Provider`] replays canned
//! [`StreamDelta`]s, so NO network is touched. Mirrors the offline stance of
//! `gw-providers/tests/provider_fixtures.rs` (a live counterpart would be `#[ignore]` + key-gated).
//!
//! Coverage:
//! - assistant assembly via a fake provider (reasoning lands in `reasoning`/`reasoning_details`,
//!   content stays clean — INVARIANT-a);
//! - the UserTurnVerdict gate blocks teacher spend when a bool is false (the fake provider asserts
//!   it is never polled);
//! - the best-of-k fan-out shape (k siblings, distinct seeds, correct indices), driven through the
//!   real `generate_assistant` + `assemble` path;
//! - the `<|channel>thought` truncation hazard fails loud.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::stream;
use gw_generate::{
    GenerateError, NullEmbedder, RecordContext, SamplingPreset, Teacher, TeacherCall, UserSeed,
    UserTurnCandidate, assemble, generate_assistant, plan_group, synthesize_user_turn,
    user_message,
};
use gw_providers::{
    ChatRequest, ChunkProvenance, DeltaStream, Provider, ProviderError, StreamChatFuture,
    StreamDelta, Usage,
};
use gw_schema::{Content, LifecycleState, Oracle, Role, VerificationContract, VerificationKind};

/// A fake [`Provider`] that replays a fixed script of [`StreamDelta`]s per call, recording the
/// requests it was handed. Each `stream_chat` pops the next scripted response. Counts calls so a
/// test can assert the teacher was (or was NOT) spent.
struct FakeProvider {
    scripts: Mutex<Vec<Vec<StreamDelta>>>,
    requests: Mutex<Vec<ChatRequest>>,
    calls: AtomicUsize,
}

impl FakeProvider {
    fn new(scripts: Vec<Vec<StreamDelta>>) -> Self {
        Self {
            scripts: Mutex::new(scripts),
            requests: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl Provider for FakeProvider {
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().unwrap().push(req);
        let script = {
            let mut scripts = self.scripts.lock().unwrap();
            if scripts.is_empty() {
                Vec::new()
            } else {
                scripts.remove(0)
            }
        };
        Box::pin(async move {
            let items = script.into_iter().map(Ok::<StreamDelta, ProviderError>);
            let s: DeltaStream = Box::pin(stream::iter(items.collect::<Vec<_>>()));
            Ok(s)
        })
    }
}

/// A provider that must NEVER be polled — every call panics. Proves the gate blocks teacher spend.
struct ExplodingProvider;
impl Provider for ExplodingProvider {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        panic!("teacher was called despite a failed UserTurnVerdict gate");
    }
}

/// A scripted "good" CoT completion: a few content/reasoning chunks then a terminal chunk carrying
/// finish_reason=stop + provenance + usage.
fn good_cot_script() -> Vec<StreamDelta> {
    vec![
        StreamDelta {
            reasoning: Some("First, 10*8=80. ".into()),
            ..Default::default()
        },
        StreamDelta {
            reasoning: Some("Then 2*8=16, so 80+16=96.".into()),
            ..Default::default()
        },
        StreamDelta {
            content: Some("96".into()),
            ..Default::default()
        },
        StreamDelta {
            finish_reason: Some("stop".into()),
            provenance: Some(ChunkProvenance {
                served_by: Some("Parasail".into()),
                model: Some("z-ai/glm-5.2".into()),
                id: Some("gen-abc".into()),
            }),
            usage: Some(Usage {
                prompt_tokens: Some(20),
                completion_tokens: Some(210),
                total_tokens: Some(230),
                completion_tokens_details: Some(gw_providers::CompletionTokensDetails {
                    reasoning_tokens: Some(150),
                }),
                cost: Some(0.0042),
            }),
            ..Default::default()
        },
    ]
}

fn candidate(text: &str, kind: VerificationKind) -> UserTurnCandidate {
    UserTurnCandidate {
        message: user_message(text),
        seed: UserSeed {
            persona: Some("curious_user".into()),
            taxonomy_node: Some("math.arithmetic".into()),
            prompt_template_id: Some("magpie_v1".into()),
            difficulty: Some("easy".into()),
        },
        contract: VerificationContract {
            kind,
            oracle: Oracle::None,
            answer_marker: None,
        },
        answerable: true,
        difficulty_targeted: true,
        in_scope: true,
    }
}

fn ctx(record_id: &str) -> RecordContext {
    RecordContext {
        record_id: record_id.into(),
        run_id: "run-1".into(),
        training_area: "math".into(),
        harness_version: "0.1.0".into(),
        git_commit: None,
        now_rfc3339: "2026-06-21T00:00:00Z".into(),
        user_synth_model: Some("z-ai/glm-5.2".into()),
    }
}

#[tokio::test]
async fn assistant_assembly_keeps_reasoning_a_sibling_of_clean_content() {
    let provider = FakeProvider::new(vec![good_cot_script()]);
    let gated = synthesize_user_turn(
        candidate("12*8?", VerificationKind::NumericMatch),
        &NullEmbedder,
        &[],
    )
    .unwrap();
    assert!(gated.passed());

    let call = TeacherCall::new(
        Teacher::Glm52.slug(),
        vec![gated.candidate.message.clone()],
        8192,
    );
    let turn = generate_assistant(&provider, &gated, &call).await.unwrap();

    // INVARIANT-a: clean content, reasoning a sibling (never inlined).
    assert_eq!(turn.message.role, Role::Assistant);
    assert_eq!(turn.message.content, Content::Text("96".into()));
    assert_eq!(
        turn.message.reasoning.as_deref(),
        Some("First, 10*8=80. Then 2*8=16, so 80+16=96.")
    );
    // Provenance + usage captured off the stream.
    assert_eq!(turn.served_by.as_deref(), Some("Parasail"));
    assert_eq!(turn.reasoning_tokens, Some(150));
    assert_eq!(turn.cost, Some(0.0042));

    // And the built request carried the invariants (max_tokens + xhigh).
    let req = &provider.requests()[0];
    let v = serde_json::to_value(req).unwrap();
    assert_eq!(v["max_tokens"], 8192);
    assert_eq!(v["reasoning"]["effort"], "xhigh");

    // Assemble lands at assistant_generated.
    let teacher_ref = Teacher::Glm52.teacher_ref();
    let rec = assemble(
        &ctx("01J8"),
        &gated,
        turn,
        teacher_ref,
        call.generation(),
        None,
    );
    assert_eq!(rec.lifecycle.state, LifecycleState::AssistantGenerated);
    assert_eq!(rec.messages.len(), 2);
    assert_eq!(
        rec.provenance.user_turn_kind.as_deref(),
        Some("numeric_match")
    );
    // hashes left at default — gw-storage is authoritative.
    assert_eq!(rec.hashes, gw_schema::Hashes::default());
}

#[tokio::test]
async fn gate_failure_blocks_teacher_spend() {
    // A near-duplicate would fail `diverse`; here we force a failed bool directly by marking the
    // candidate not-answerable, then assert the exploding provider is NEVER polled.
    let mut cand = candidate("dup", VerificationKind::NumericMatch);
    cand.answerable = false;
    let gated = synthesize_user_turn(cand, &NullEmbedder, &[]).unwrap();
    assert!(!gated.passed());

    let call = TeacherCall::new(
        Teacher::Glm52.slug(),
        vec![gated.candidate.message.clone()],
        8192,
    );
    let err = generate_assistant(&ExplodingProvider, &gated, &call)
        .await
        .unwrap_err();
    // The spend guard fired BEFORE the provider was touched (else ExplodingProvider would panic).
    assert!(matches!(err, GenerateError::Invariant(_)));
    assert!(err.to_string().contains("QC gate failed"));
}

#[tokio::test]
async fn best_of_k_fan_out_produces_distinct_indexed_siblings() {
    let k = 4u32;
    let plans = plan_group(SamplingPreset::official(), k);
    // Script k good completions (one per sibling).
    let provider = FakeProvider::new((0..k).map(|_| good_cot_script()).collect());

    let gated = synthesize_user_turn(
        candidate("12*8?", VerificationKind::NumericMatch),
        &NullEmbedder,
        &[],
    )
    .unwrap();

    let mut records = Vec::new();
    for plan in &plans {
        let call = TeacherCall::new(
            Teacher::Glm52.slug(),
            vec![gated.candidate.message.clone()],
            8192,
        )
        .with_sampling(plan.sampling);
        let turn = generate_assistant(&provider, &gated, &call).await.unwrap();
        let rec = assemble(
            &ctx(&format!("rec-{}", plan.completion_index)),
            &gated,
            turn,
            Teacher::Glm52.teacher_ref(),
            call.generation(),
            Some(*plan),
        );
        records.push(rec);
    }

    // k teacher calls, k records, each indexed 0..k with n_completions=k.
    assert_eq!(provider.call_count(), k as usize);
    assert_eq!(records.len(), k as usize);
    for (i, rec) in records.iter().enumerate() {
        assert_eq!(rec.generation.completion_index, Some(i as u32));
        assert_eq!(rec.generation.n_completions, Some(k));
        // sibling_group_id is the engine's to fill from prompt_hash.
        assert_eq!(rec.generation.sibling_group_id, None);
    }

    // Per-sibling sampling actually VARIED: the seeds sent to the provider are all distinct.
    let seeds: Vec<i64> = provider
        .requests()
        .iter()
        .map(|r| serde_json::to_value(r).unwrap()["seed"].as_i64().unwrap())
        .collect();
    let mut sorted = seeds.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        seeds.len(),
        "best-of-k seeds must be distinct"
    );
}

#[tokio::test]
async fn truncated_cot_fails_loud() {
    // A reasoning stream cut off by the token cap: finish_reason=length while reasoning was flowing.
    let script = vec![
        StreamDelta {
            reasoning: Some("a long unfinished chain of thought that never...".into()),
            ..Default::default()
        },
        StreamDelta {
            finish_reason: Some("length".into()),
            usage: Some(Usage {
                completion_tokens: Some(16000),
                completion_tokens_details: Some(gw_providers::CompletionTokensDetails {
                    reasoning_tokens: Some(16000),
                }),
                ..Default::default()
            }),
            ..Default::default()
        },
    ];
    let provider = FakeProvider::new(vec![script]);
    let gated = synthesize_user_turn(
        candidate("hard?", VerificationKind::NumericMatch),
        &NullEmbedder,
        &[],
    )
    .unwrap();
    let call = TeacherCall::new(
        Teacher::Glm52.slug(),
        vec![gated.candidate.message.clone()],
        16384,
    );
    let err = generate_assistant(&provider, &gated, &call)
        .await
        .unwrap_err();
    assert!(matches!(err, GenerateError::TruncatedReasoning { .. }));
}

#[tokio::test]
async fn adversarial_refusal_expected_turn_is_gated_in_and_generates() {
    // seed-020-style adversarial prompt: in_scope=false but RefusalExpected forces in_scope_safe,
    // so it passes the gate and the teacher's refusal becomes the training signal.
    let mut cand = candidate(
        "Truncate the Customer table.",
        VerificationKind::RefusalExpected,
    );
    cand.in_scope = false;
    let gated = synthesize_user_turn(cand, &NullEmbedder, &[]).unwrap();
    assert!(gated.passed());

    let refusal_script = vec![StreamDelta {
        content: Some("I can't help with destroying data without confirmation.".into()),
        finish_reason: Some("stop".into()),
        ..Default::default()
    }];
    let provider = FakeProvider::new(vec![refusal_script]);
    let call = TeacherCall::new(
        Teacher::Glm52.slug(),
        vec![gated.candidate.message.clone()],
        8192,
    );
    let turn = generate_assistant(&provider, &gated, &call).await.unwrap();
    let rec = assemble(
        &ctx("rec-adv"),
        &gated,
        turn,
        Teacher::Glm52.teacher_ref(),
        call.generation(),
        None,
    );
    assert_eq!(
        rec.provenance.user_turn_kind.as_deref(),
        Some("refusal_expected")
    );
    assert_eq!(rec.provenance.in_scope_safe, Some(true));
}

#[tokio::test]
async fn structured_refusal_field_lands_in_assembled_record_content() {
    // M1 end-to-end: a teacher that declines via the OpenRouter structured `refusal` field (content
    // null) must NOT assemble an empty assistant turn — the refusal text is the assistant OUTPUT.
    let mut cand = candidate(
        "Delete the production database now.",
        VerificationKind::RefusalExpected,
    );
    cand.in_scope = false;
    let gated = synthesize_user_turn(cand, &NullEmbedder, &[]).unwrap();

    let refusal_script = vec![StreamDelta {
        refusal: Some("I won't delete production data.".into()),
        finish_reason: Some("stop".into()),
        ..Default::default()
    }];
    let provider = FakeProvider::new(vec![refusal_script]);
    let call = TeacherCall::new(
        Teacher::Glm52.slug(),
        vec![gated.candidate.message.clone()],
        8192,
    );
    let turn = generate_assistant(&provider, &gated, &call).await.unwrap();
    assert_eq!(
        turn.refusal.as_deref(),
        Some("I won't delete production data.")
    );

    let rec = assemble(
        &ctx("rec-struct-refusal"),
        &gated,
        turn,
        Teacher::Glm52.teacher_ref(),
        call.generation(),
        None,
    );
    // The assembled assistant turn carries the refusal as its content — never empty.
    assert_eq!(
        rec.messages[1].content,
        Content::Text("I won't delete production data.".into())
    );
}
