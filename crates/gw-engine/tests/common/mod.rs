//! Shared HERMETIC test harness for the `gw-engine` integration tests.
//!
//! Every side-effecting client is a fake: a scripted teacher [`Provider`], a scripted judge
//! [`Provider`], the [`NullEmbedder`], the [`NullSandboxOracle`], and `Store::open_in_memory`. NO
//! network is touched and every run is deterministic. The unhappy-path tests (crash-resume, single
//! revise, best-of-k retain, R-prior, never-re-spend, budget cutoff, verdict mapping, replay) build on
//! these fakes.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use gw_engine::{AreaConfig, BudgetMeter, Clients, EventSink, InMemorySeedSource};
use gw_generate::{NullEmbedder, UserSeed, UserTurnCandidate, user_message};
use gw_judge::{AreaThresholds, NullSandboxOracle, PanelJudge};
use gw_providers::{
    ChatRequest, CompletionTokensDetails, DeltaStream, Provider, ProviderError, StreamChatFuture,
    StreamDelta, Usage,
};
use gw_schema::{Oracle, ReasoningDetail, VerificationContract, VerificationKind};
use gw_storage::Store;

/// One streamed `reasoning.text` detail block (how a real OpenRouter teacher emits structured CoT).
/// The Verify reasoning-present hard gate requires a non-empty `reasoning.text` detail, so every
/// "good" CoT script carries one. Public so per-test custom scripts can satisfy the gate.
pub fn reasoning_text_detail(text: &str) -> ReasoningDetail {
    ReasoningDetail::Text {
        text: text.to_string(),
        signature: None,
        id: None,
        format: None,
        index: 0,
    }
}

/// A scripted teacher provider: replays canned CoT completions in call order, counting calls so a
/// test can assert the teacher was (or was NOT) re-spent. Panics if called more than `max_calls`.
pub struct ScriptedTeacher {
    scripts: Mutex<Vec<Vec<StreamDelta>>>,
    calls: AtomicUsize,
    max_calls: usize,
}

impl ScriptedTeacher {
    /// A teacher that replays `scripts` in order, allowing up to `max_calls` total calls.
    pub fn new(scripts: Vec<Vec<StreamDelta>>, max_calls: usize) -> Self {
        Self {
            scripts: Mutex::new(scripts),
            calls: AtomicUsize::new(0),
            max_calls,
        }
    }

    /// Total teacher calls so far.
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Provider for ScriptedTeacher {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        assert!(
            n <= self.max_calls,
            "teacher re-spent: called {n} times (max {})",
            self.max_calls
        );
        let script = {
            let mut s = self.scripts.lock().unwrap();
            if s.is_empty() {
                good_cot(0.01)
            } else {
                s.remove(0)
            }
        };
        Box::pin(async move {
            let items = script.into_iter().map(Ok::<StreamDelta, ProviderError>);
            let stream: DeltaStream = Box::pin(futures::stream::iter(items.collect::<Vec<_>>()));
            Ok(stream)
        })
    }
}

/// A scripted judge provider: replays canned judge-JSON bodies in call order. A judge call streams one
/// content chunk; the cache elides repeat calls for the same key.
pub struct ScriptedJudge {
    bodies: Mutex<Vec<String>>,
    calls: AtomicUsize,
}

impl ScriptedJudge {
    /// A judge that replays `bodies` (judge-response JSON strings) in order.
    pub fn new(bodies: Vec<&str>) -> Self {
        Self {
            bodies: Mutex::new(bodies.into_iter().map(str::to_string).collect()),
            calls: AtomicUsize::new(0),
        }
    }

    /// Total judge calls so far.
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Provider for ScriptedJudge {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let body = {
            let mut b = self.bodies.lock().unwrap();
            if b.is_empty() {
                "{\"score\":0.5,\"verdict\":\"uncertain\"}".to_string()
            } else {
                b.remove(0)
            }
        };
        Box::pin(async move {
            let delta = StreamDelta {
                content: Some(body),
                finish_reason: Some("stop".into()),
                ..Default::default()
            };
            let stream: DeltaStream = Box::pin(futures::stream::iter(vec![Ok::<
                StreamDelta,
                ProviderError,
            >(delta)]));
            Ok(stream)
        })
    }
}

/// A judge that returns `Err(ProviderError::Decode)` (a terminal fault → `JudgeError::Provider` →
/// record-level) on the `fail_on`-th call (1-based; `0` = EVERY call), and a passing judge body
/// otherwise. Injects a JUDGE-RAIL fault on a record already at `Verified` — the mid-drive fault the
/// no-clobber guard must STILL terminalize to `Error` (F1) — or a SYSTEMIC judge outage when it fails
/// every call (the circuit-breaker must then trip, F2). Counts calls.
pub struct FailingJudge {
    calls: AtomicUsize,
    fail_on: usize,
    body: String,
}

impl FailingJudge {
    /// Fails on call `fail_on` (1-based; `0` = every call); otherwise replays `body`.
    pub fn new(fail_on: usize, body: &str) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            fail_on,
            body: body.to_string(),
        }
    }

    /// Total calls so far.
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Provider for FailingJudge {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let fail = self.fail_on == 0 || n == self.fail_on;
        let body = self.body.clone();
        Box::pin(async move {
            if fail {
                Err(ProviderError::Decode("injected judge-rail fault".into()))
            } else {
                let delta = StreamDelta {
                    content: Some(body),
                    finish_reason: Some("stop".into()),
                    ..Default::default()
                };
                let stream: DeltaStream =
                    Box::pin(futures::stream::iter(vec![
                        Ok::<StreamDelta, ProviderError>(delta),
                    ]));
                Ok(stream)
            }
        })
    }
}

/// A provider that PANICS if ever called — proves a path never touches the teacher.
pub struct ExplodingTeacher;
impl Provider for ExplodingTeacher {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        panic!("teacher was called when it must not have been (never-re-spend / budget guard)");
    }
}

/// A teacher that returns `Err(ProviderError)` (a TERMINAL, non-retryable fault) on the `fail_on`-th
/// call (1-based) and a good CoT on every other call. Lets a test inject a RECORD-LEVEL fault on one
/// record and confirm the run isolates it and continues (E5/E10).
pub struct FailingTeacher {
    calls: AtomicUsize,
    fail_on: usize,
    cost_usd: f64,
}

impl FailingTeacher {
    /// A teacher that fails on call `fail_on` (1-based), otherwise streams a good CoT at `cost_usd`.
    pub fn new(fail_on: usize, cost_usd: f64) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            fail_on,
            cost_usd,
        }
    }

    /// Total calls so far.
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Provider for FailingTeacher {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let fail = n == self.fail_on;
        let cost = self.cost_usd;
        Box::pin(async move {
            if fail {
                // A RECORD-LEVEL provider fault (a malformed/garbled response for THIS record) — surfaces
                // as GenerateError::Provider(Decode) → record-level, isolated to one record. (NOT a
                // systemic MissingApiKey/Config/auth fault, which the engine treats as infra-fatal.)
                Err(ProviderError::Decode(
                    "injected terminal teacher fault".into(),
                ))
            } else {
                let items = good_cot(cost)
                    .into_iter()
                    .map(Ok::<StreamDelta, ProviderError>);
                let stream: DeltaStream =
                    Box::pin(futures::stream::iter(items.collect::<Vec<_>>()));
                Ok(stream)
            }
        })
    }
}

/// A teacher that returns the SAME terminal `ProviderError` (built from an HTTP `status`) on EVERY
/// call — a SYSTEMIC fault that fails every record identically: an invalid/revoked key (401, → fail-fast
/// abort), or a persistent non-auth error (404, record-level → only the circuit-breaker stops the
/// churn). Counts calls so a test can assert fail-fast vs. breaker-trip cardinality (F2).
pub struct AlwaysFailingTeacher {
    status: u16,
    calls: AtomicUsize,
}

impl AlwaysFailingTeacher {
    /// A teacher that fails every call with `ProviderError::from_status(status, None)`.
    pub fn new(status: u16) -> Self {
        Self {
            status,
            calls: AtomicUsize::new(0),
        }
    }

    /// Total calls so far.
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Provider for AlwaysFailingTeacher {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let status = self.status;
        Box::pin(async move { Err(ProviderError::from_status(status, None)) })
    }
}

/// A teacher whose `stream_chat` BLOCKS on a shared barrier until `n` concurrent callers arrive, then
/// all release together — so a test can force EXACTLY `n` teacher calls to be in flight at once (all
/// past the budget gate) before any of them charges. Counts calls. Used to pin the concurrent
/// budget-overshoot bound (E8).
pub struct BarrierTeacher {
    barrier: Arc<tokio::sync::Barrier>,
    calls: AtomicUsize,
    cost_usd: f64,
}

impl BarrierTeacher {
    /// A teacher that releases once `n` calls are simultaneously in flight, each streaming a good CoT
    /// at `cost_usd`.
    pub fn new(n: usize, cost_usd: f64) -> Self {
        Self {
            barrier: Arc::new(tokio::sync::Barrier::new(n)),
            calls: AtomicUsize::new(0),
            cost_usd,
        }
    }

    /// Total calls so far.
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Provider for BarrierTeacher {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let barrier = Arc::clone(&self.barrier);
        let cost = self.cost_usd;
        Box::pin(async move {
            // Block until `n` callers are here — forces `n` concurrent in-flight teacher calls, all of
            // which already passed `may_dispatch()` before any charged the meter.
            barrier.wait().await;
            let items = good_cot(cost)
                .into_iter()
                .map(Ok::<StreamDelta, ProviderError>);
            let stream: DeltaStream = Box::pin(futures::stream::iter(items.collect::<Vec<_>>()));
            Ok(stream)
        })
    }
}

/// A "good" CoT completion script: reasoning + content + a terminal chunk carrying usage with the
/// given `cost_usd` and positive reasoning tokens (so the Verify reasoning-present gate passes).
pub fn good_cot(cost_usd: f64) -> Vec<StreamDelta> {
    vec![
        StreamDelta {
            reasoning: Some("Let me reason: 12*8=96.".into()),
            reasoning_details: Some(vec![reasoning_text_detail("Let me reason: 12*8=96.")]),
            ..Default::default()
        },
        StreamDelta {
            content: Some("96".into()),
            ..Default::default()
        },
        StreamDelta {
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                prompt_tokens: Some(20),
                completion_tokens: Some(200),
                total_tokens: Some(220),
                completion_tokens_details: Some(CompletionTokensDetails {
                    reasoning_tokens: Some(150),
                }),
                cost: Some(cost_usd),
            }),
            ..Default::default()
        },
    ]
}

/// Like [`good_cot`] but with a custom final answer, so two siblings drawing the SAME prompt produce
/// DISTINCT content (distinct `record_hash` → distinct judge cache key) while sharing one
/// `prompt_hash` (the sibling group id). Used to give best-of-k siblings independent grades.
pub fn answer_cot(answer: &str, cost_usd: f64) -> Vec<StreamDelta> {
    vec![
        StreamDelta {
            reasoning: Some(format!("Reasoning toward {answer}.")),
            reasoning_details: Some(vec![reasoning_text_detail(&format!(
                "Reasoning toward {answer}."
            ))]),
            ..Default::default()
        },
        StreamDelta {
            content: Some(answer.to_string()),
            ..Default::default()
        },
        StreamDelta {
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                prompt_tokens: Some(20),
                completion_tokens: Some(200),
                total_tokens: Some(220),
                completion_tokens_details: Some(CompletionTokensDetails {
                    reasoning_tokens: Some(150),
                }),
                cost: Some(cost_usd),
            }),
            ..Default::default()
        },
    ]
}

/// A judge-response body at `score` voting `accept` (or `reject`/`revise` if the score implies).
pub fn judge_body(score: f64, verdict: &str) -> String {
    format!("{{\"score\":{score},\"verdict\":\"{verdict}\"}}")
}

/// A single-judge panel (k=1: the correlation guard is inert by construction).
pub fn one_judge() -> Vec<PanelJudge> {
    vec![PanelJudge::new("judge-a", "fam-a").with_rubric("r")]
}

/// A three-judge panel (k=3: triggers the NON-IDENTITY correlation prior).
pub fn three_judges() -> Vec<PanelJudge> {
    vec![
        PanelJudge::new("judge-a", "fam-a").with_rubric("r"),
        PanelJudge::new("judge-b", "fam-b").with_rubric("r"),
        PanelJudge::new("judge-c", "fam-c").with_rubric("r"),
    ]
}

/// Thresholds that ADMIT a clean panel even at the cold-start rho=0.7 (relaxed n_eff floor) — for
/// tests that want to reach `Admitted` with a multi-judge panel.
pub fn lenient_thresholds() -> AreaThresholds {
    AreaThresholds {
        accept_threshold: 0.80,
        reject_below: 0.50,
        min_n_eff: 1.0,
        min_n_eff_ratio: 0.3,
    }
}

/// A candidate USER turn that passes the four-bool QC gate.
pub fn good_candidate(text: &str) -> UserTurnCandidate {
    UserTurnCandidate {
        message: user_message(text),
        seed: UserSeed {
            persona: Some("curious_user".into()),
            taxonomy_node: Some("math.arithmetic".into()),
            prompt_template_id: Some("magpie_v1".into()),
            difficulty: Some("easy".into()),
        },
        contract: VerificationContract {
            kind: VerificationKind::None,
            oracle: Oracle::None,
            answer_marker: None,
        },
        answerable: true,
        difficulty_targeted: true,
        in_scope: true,
    }
}

/// A candidate with a NumericMatch verification contract pinned to a literal `expected` answer, so the
/// engine's Verify rail runs the answer-correctness check (E4). A teacher answer that does not match
/// `expected` is caught on the deterministic rail.
pub fn numeric_candidate(text: &str, expected: &str) -> UserTurnCandidate {
    let mut c = good_candidate(text);
    c.contract = VerificationContract {
        kind: VerificationKind::NumericMatch,
        oracle: Oracle::Literal {
            expected: expected.into(),
        },
        answer_marker: None,
    };
    c
}

/// A candidate with a RefusalExpected contract (adversarial-by-construction): the correct behavior is a
/// refusal; a clear compliance is a hard verifier reject (E4).
pub fn refusal_candidate(text: &str) -> UserTurnCandidate {
    let mut c = good_candidate(text);
    c.contract = VerificationContract {
        kind: VerificationKind::RefusalExpected,
        oracle: Oracle::RefusalPolicy {
            policy_id: "p1".into(),
        },
        answer_marker: None,
    };
    c
}

/// Build a `Clients` bundle over the given teacher/judge providers + a fresh in-memory store, with the
/// given budget cap.
pub fn clients(
    store: Store,
    teacher: Arc<dyn Provider>,
    judge: Arc<dyn Provider>,
    cap_usd: f64,
    events: EventSink,
) -> Clients {
    Clients::new(
        store,
        teacher,
        judge,
        Arc::new(NullEmbedder),
        Arc::new(NullSandboxOracle),
        BudgetMeter::new(cap_usd),
        events,
        "0.1.0-test",
    )
}

/// A single-trace area config (k=1) over a judge panel + rubric, with the given thresholds.
pub fn area_k1(judges: Vec<PanelJudge>, thresholds: AreaThresholds) -> AreaConfig {
    AreaConfig::new("math", "z-ai/glm-5.2", judges, "Grade the trace.")
        .with_thresholds(thresholds)
        .with_cot_required(true)
        .with_k(1)
}

/// A best-of-k area config (k siblings) over a judge panel + rubric, with the given thresholds.
pub fn area_k(judges: Vec<PanelJudge>, thresholds: AreaThresholds, k: u32) -> AreaConfig {
    AreaConfig::new("math", "z-ai/glm-5.2", judges, "Grade the trace.")
        .with_thresholds(thresholds)
        .with_cot_required(true)
        .with_k(k)
}

/// A k=1 area config whose rule comparator is RULE-ONLY AUTHORITATIVE: a verifier answer non-match is a
/// HARD reject (the opt-in path), so a wrong answer can never be rescued/admitted by the panel.
pub fn area_rule_authoritative(judges: Vec<PanelJudge>, thresholds: AreaThresholds) -> AreaConfig {
    let mut cfg = area_k1(judges, thresholds);
    cfg.rule_only_authoritative = true;
    cfg
}

/// A one-item, one-shard seed source for the common single-record test.
pub fn one_item_source() -> InMemorySeedSource {
    InMemorySeedSource::new(vec![good_candidate("What is 12*8?")], 1)
}
