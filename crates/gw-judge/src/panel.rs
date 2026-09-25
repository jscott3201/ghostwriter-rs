//! Rail 2 — the LLM JudgePanel (LLM-as-jury), everywhere a verifier can't decide (JUDGE-DESIGN
//! §1.2, §4, §5.7).
//!
//! A **panel** of LLM judges grades the trace in a **blind, independent, sealed** first pass: each
//! judge is a separate, isolated [`Provider`] call (no judge prompt ever contains another judge's
//! output — the structural defense against herding, §5.7). The per-judge
//! [`Grade`]s are then handed to the consensus math (`consensus.rs`), NEVER aggregated by
//! majority/mean here.
//!
//! This module owns:
//! - [`Grade`] — one judge's sealed outcome (score + verdict + meta-prediction + rationale + raw).
//! - [`PanelJudge`] — the per-judge config (slug, rubric, scoring, sampling) consumed to build the
//!   request and the cache key.
//! - [`JudgeScoring`] — `GEvalLogprob | IntegerLikert` (the §4.2 routing pin; the actual scoring
//!   path used is recorded on the grade for audit).
//! - the judge-prompt builder, the response parser, and [`grade_one`] / [`grade_panel`] — the
//!   sealed parallel pass over the injected provider, each call wrapped by the never-re-spend cache
//!   (`cache.rs`).
//!
//! All judge model calls go through the `gw-providers` [`Provider`] trait, so tests inject a fake
//! provider returning canned judge JSON — no network.

use std::collections::BTreeMap;

use futures::StreamExt;
use gw_providers::{ChatRequest, Provider, ReasoningParam, StreamDelta};
use gw_schema::{Content, JudgeVote, Message, ReasoningEffort, Role};
use serde::Deserialize;

use crate::decision::Verdict;
use crate::error::{JudgeError, Result};

/// How a judge's pointwise score is produced (JUDGE-DESIGN §4.2). The pin is per-judge; the actual
/// path used is recorded on the grade (`scoring_used` in [`Grade::raw`]) for audit, since a
/// `GEvalLogprob` judge whose route lacks logprobs/structured-outputs auto-falls-back to
/// `IntegerLikert`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JudgeScoring {
    /// G-Eval logprob-weighted continuous scoring — requires a route advertising BOTH `logprobs`
    /// AND `structured_outputs`.
    GEvalLogprob,
    /// Discrete integer Likert (1–5 / 1–10), no logprob de-quantization. The fallback path.
    IntegerLikert,
}

/// Default explicit judge reasoning budget. Judges need enough reasoning to inspect a trace, but
/// unlike teachers they must preserve content headroom for the verdict JSON.
pub const DEFAULT_JUDGE_REASONING_MAX_TOKENS: u32 = 2_000;

/// Default combined judge completion cap, including hidden reasoning plus visible verdict content.
pub const DEFAULT_JUDGE_MAX_TOKENS: u32 = 3_500;

/// Minimum visible-content headroom reserved for the judge verdict JSON and short rationale.
pub const MIN_JUDGE_VERDICT_TOKENS: u32 = 1_500;

impl JudgeScoring {
    /// A stable token for audit (`scoring_used`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            JudgeScoring::GEvalLogprob => "g_eval_logprob",
            JudgeScoring::IntegerLikert => "integer_likert",
        }
    }
}

/// The reasoning knob applied to a judge request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JudgeReasoning {
    /// OpenRouter effort mode. Prefer [`JudgeReasoning::MaxTokens`] for judges when possible,
    /// because effort mode does not reserve a visible verdict budget.
    Effort(ReasoningEffort),
    /// Explicit reasoning-token cap.
    MaxTokens(u32),
}

impl JudgeReasoning {
    fn to_param(self) -> ReasoningParam {
        match self {
            Self::Effort(effort) => ReasoningParam::effort(effort),
            Self::MaxTokens(max_tokens) => ReasoningParam::max_tokens(max_tokens),
        }
    }

    fn min_overall_tokens(self) -> Option<u32> {
        match self {
            Self::MaxTokens(reasoning) => Some(reasoning.saturating_add(MIN_JUDGE_VERDICT_TOKENS)),
            Self::Effort(_) => None,
        }
    }
}

/// The judge-call sampling params actually applied (mirrors `gw_schema::JudgeSampling`, but local so
/// the panel does not depend on the config-side default policy). `temperature` is folded into the
/// cache key as `temperature_bits` (A2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JudgeSampling {
    /// Sampling temperature actually used (default policy 0.0; the CACHE, not temp=0, guarantees
    /// replay).
    pub temperature: f64,
    /// Nucleus top-p, or `None`.
    pub top_p: Option<f64>,
    /// Deterministic seed, or `None`.
    pub seed: Option<i64>,
}

impl Default for JudgeSampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
            seed: None,
        }
    }
}

/// One judge's configuration in the panel: its model slug, optional rubric id (part of the cache
/// key), scoring pin, and sampling. The `family` token drives same-family exclusion upstream
/// (`grader.rs`); it is the coarse model family, e.g. `"gemma"`, not the full slug.
#[derive(Debug, Clone, PartialEq)]
pub struct PanelJudge {
    /// OpenRouter model slug, e.g. `"deepseek/deepseek-v4-pro"`.
    pub slug: String,
    /// Coarse model family for same-family exclusion (§5.8), e.g. `"gemma"`.
    pub family: String,
    /// Rubric id (`rubric_id` in the cache key + `JudgeVote.rubric_id`), or `None`.
    pub rubric_id: Option<String>,
    /// Scoring pin (`GEvalLogprob | IntegerLikert`).
    pub scoring: JudgeScoring,
    /// Judge-call sampling.
    pub sampling: JudgeSampling,
    /// Combined hidden-reasoning plus visible-content cap for the judge completion.
    pub max_tokens: u32,
    /// Judge reasoning budget/mode. Defaults to explicit bounded reasoning, not effort mode.
    pub reasoning: Option<JudgeReasoning>,
}

impl PanelJudge {
    /// A minimal judge: a slug + family, integer-Likert scoring, default (temp-0) sampling, no
    /// rubric. The builder-style setters layer the rest.
    #[must_use]
    pub fn new(slug: impl Into<String>, family: impl Into<String>) -> Self {
        Self {
            slug: slug.into(),
            family: family.into(),
            rubric_id: None,
            scoring: JudgeScoring::IntegerLikert,
            sampling: JudgeSampling::default(),
            max_tokens: DEFAULT_JUDGE_MAX_TOKENS,
            reasoning: Some(JudgeReasoning::MaxTokens(
                DEFAULT_JUDGE_REASONING_MAX_TOKENS,
            )),
        }
    }

    /// Set the rubric id. Chainable.
    #[must_use]
    pub fn with_rubric(mut self, rubric_id: impl Into<String>) -> Self {
        self.rubric_id = Some(rubric_id.into());
        self
    }

    /// Set the scoring pin. Chainable.
    #[must_use]
    pub fn with_scoring(mut self, scoring: JudgeScoring) -> Self {
        self.scoring = scoring;
        self
    }

    /// Set the sampling params. Chainable.
    #[must_use]
    pub fn with_sampling(mut self, sampling: JudgeSampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set the combined hidden-reasoning plus visible-content cap. If explicit reasoning tokens are
    /// also configured, [`effective_max_tokens`](Self::effective_max_tokens) preserves the verdict
    /// headroom floor even when this raw value is too low.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the judge reasoning mode. Chainable.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: Option<JudgeReasoning>) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Set an explicit judge reasoning-token cap. Chainable.
    #[must_use]
    pub fn with_reasoning_max_tokens(self, max_tokens: u32) -> Self {
        self.with_reasoning(Some(JudgeReasoning::MaxTokens(max_tokens)))
    }

    /// Set effort-mode reasoning for a judge. Prefer explicit max-tokens for production judges.
    #[must_use]
    pub fn with_reasoning_effort(self, effort: ReasoningEffort) -> Self {
        self.with_reasoning(Some(JudgeReasoning::Effort(effort)))
    }

    /// Combined max-token cap that preserves the minimum visible verdict floor when the judge uses
    /// an explicit reasoning budget.
    #[must_use]
    pub fn effective_max_tokens(&self) -> u32 {
        self.reasoning
            .and_then(JudgeReasoning::min_overall_tokens)
            .map_or(self.max_tokens, |floor| self.max_tokens.max(floor))
    }

    fn reasoning_param(&self) -> Option<ReasoningParam> {
        self.reasoning.map(JudgeReasoning::to_param)
    }
}

/// One judge's sealed grade — collected independently, persisted before reveal (§5.7).
///
/// `score` is normalized to `[0, 1]`. `verdict` is the per-grade 4-variant [`Verdict`].
/// `meta_prediction` is the judge's prediction of the panel-mean verdict score (the SP/BTS "how
/// will the others vote" elicitation, §5.1) — recorded for the SP tie-break. `confidence` is the
/// judge's STATED confidence — RECORDED, never used as an aggregation weight (§4.4). `raw` carries
/// the provider response + `scoring_used` for audit.
#[derive(Debug, Clone, PartialEq)]
pub struct Grade {
    /// The judge model slug.
    pub judge_model: String,
    /// Normalized score in `[0, 1]`.
    pub score: f64,
    /// The per-grade verdict.
    pub verdict: Verdict,
    /// Stated confidence in `[0, 1]` — audit-only, NEVER a weight.
    pub confidence: f64,
    /// Predicted panel-mean score in `[0, 1]` (SP/BTS meta-prediction), or `None` if not elicited.
    pub meta_prediction: Option<f64>,
    /// Per-criterion sub-scores (also the B7 reject-axis keys, e.g. `"over_refusal"`,
    /// `"groundedness_sycophancy"`).
    ///
    /// PHASE-0 STATUS (V6): these are **RECORDED and round-tripped through the cache for audit, but
    /// NOT YET ENFORCED**. The spec'd enforcement — the §5.8 SOFT minority-veto that lets a B7 axis
    /// route a trace to review — is a tracked follow-up; `grader::HybridGrader::decide` does not
    /// read `dimensions`. Do not mistake their inertness for a live gate.
    pub dimensions: Option<BTreeMap<String, f64>>,
    /// The judge's free-text rationale (required on a reject; feeds the revise loop + Dung nodes).
    pub rationale: Option<String>,
    /// Raw provider response + `scoring_used`, for the immutable audit record.
    pub raw: serde_json::Value,
    /// The sampling temperature actually used (folded into the cache key).
    pub temperature: f64,
    /// Top-p actually used.
    pub top_p: Option<f64>,
    /// Seed actually used.
    pub seed: Option<i64>,
    /// The rubric id used.
    pub rubric_id: Option<String>,
}

impl Grade {
    /// Project this sealed grade to the persisted [`JudgeVote`] (DATA-SCHEMA §1.7), preserving the
    /// recorded sampling params (which feed the cache key) and the raw response.
    #[must_use]
    pub fn to_vote(&self) -> JudgeVote {
        JudgeVote {
            judge_model: self.judge_model.clone(),
            rubric_id: self.rubric_id.clone(),
            temperature: Some(self.temperature),
            top_p: self.top_p,
            seed: self.seed,
            score: self.score,
            dimensions: self.dimensions.clone(),
            rationale: self.rationale.clone(),
            raw_response: Some(self.raw.to_string()),
        }
    }
}

/// The minimal JSON contract a judge is asked to emit (so the panel can parse it deterministically).
/// A judge that cannot emit this (or emits garbage) yields a [`Verdict::Uncertain`] grade rather
/// than crashing the panel — one bad judge must not sink the consensus.
#[derive(Debug, Deserialize)]
struct JudgeResponse {
    /// `0..1` (or `1..10` — normalized by [`normalize_score`]).
    score: f64,
    /// `"accept" | "revise" | "reject" | "uncertain"`.
    verdict: String,
    #[serde(default)]
    confidence: Option<f64>,
    /// Predicted panel-mean score (the SP/BTS meta-prediction).
    #[serde(default)]
    meta_prediction: Option<f64>,
    #[serde(default)]
    dimensions: Option<BTreeMap<String, f64>>,
    #[serde(default)]
    rationale: Option<String>,
}

/// Normalize a raw judge score onto `[0, 1]`: a value already in `[0, 1]` passes through; a `1..10`
/// Likert is divided by 10. Anything outside `[0, 10]` is clamped. This keeps the threshold
/// comparison and the weighted aggregate on one scale regardless of a judge's rubric granularity.
fn normalize_score(raw: f64) -> f64 {
    if !raw.is_finite() {
        return 0.0;
    }
    let s = if raw > 1.0 { raw / 10.0 } else { raw };
    s.clamp(0.0, 1.0)
}

/// Parse a judge verdict token to the per-grade [`Verdict`]; an unknown token is `Uncertain`.
fn parse_verdict(token: &str) -> Verdict {
    match token.trim().to_ascii_lowercase().as_str() {
        "accept" | "admit" => Verdict::Accept,
        "revise" => Verdict::Revise,
        "reject" => Verdict::Reject,
        _ => Verdict::Uncertain,
    }
}

/// Extract the JSON object body from a judge completion, tolerating the two most common LLM output
/// shapes: a fenced ```json … ``` block and a prose preamble/suffix around the object. Strips a
/// leading/trailing markdown code fence, then returns the substring from the FIRST `{` to the LAST
/// `}` (the outermost object). Returns `None` if no brace pair is present at all.
fn extract_json(text: &str) -> Option<&str> {
    // Drop a leading ```json / ``` fence and a trailing ``` fence if present.
    let mut t = text.trim();
    if let Some(rest) = t.strip_prefix("```") {
        // Skip an optional language tag on the opening fence line (e.g. ```json\n…).
        let after_tag = rest.find('\n').map_or(rest, |nl| &rest[nl + 1..]);
        t = after_tag
            .trim_end()
            .strip_suffix("```")
            .unwrap_or(after_tag);
        t = t.trim();
    }
    // Extract the outermost {...} so a prose preamble/suffix around the object still parses.
    let start = t.find('{')?;
    let end = t.rfind('}')?;
    if end > start {
        Some(&t[start..=end])
    } else {
        None
    }
}

/// Parse the accumulated judge response text into a [`Grade`]. A SUCCESSFUL deserialize (including a
/// judge that legitimately voted `"uncertain"`) returns `Ok`; a STRUCTURAL parse failure on a
/// non-empty body returns [`JudgeError::JudgeParse`] (V2) so the caller does NOT freeze a degenerate
/// `0.0`/`Uncertain` grade in the cache and the engine can retry — mirroring the empty-completion
/// path. Markdown-fenced JSON and a prose-wrapped object are tolerated via [`extract_json`].
///
/// # Errors
/// Returns [`JudgeError::JudgeParse`] if neither the raw text nor the extracted `{...}` body
/// deserializes into a `JudgeResponse`.
fn parse_grade(
    judge: &PanelJudge,
    response_text: &str,
    scoring_used: JudgeScoring,
) -> Result<Grade> {
    let raw = serde_json::json!({
        "response": response_text,
        "scoring_used": scoring_used.as_str(),
    });
    // Try the trimmed body first, then the fence/prose-stripped {...} extraction.
    let parsed = serde_json::from_str::<JudgeResponse>(response_text.trim()).or_else(|first_err| {
        match extract_json(response_text) {
            Some(body) => serde_json::from_str::<JudgeResponse>(body),
            None => Err(first_err),
        }
    });
    match parsed {
        Ok(r) => Ok(Grade {
            judge_model: judge.slug.clone(),
            score: normalize_score(r.score),
            verdict: parse_verdict(&r.verdict),
            confidence: r.confidence.unwrap_or(0.0).clamp(0.0, 1.0),
            meta_prediction: r.meta_prediction.map(normalize_score),
            dimensions: r.dimensions,
            rationale: r.rationale,
            raw,
            temperature: judge.sampling.temperature,
            top_p: judge.sampling.top_p,
            seed: judge.sampling.seed,
            rubric_id: judge.rubric_id.clone(),
        }),
        Err(e) => Err(JudgeError::JudgeParse(format!(
            "judge {} returned an unparseable response: {e}",
            judge.slug
        ))),
    }
}

/// Build the (non-CoT) judge [`ChatRequest`]: the rubric/system framing + the candidate trace, with
/// the judge's sampling applied and `max_tokens` ALWAYS set (spine invariant g). The judge is asked
/// to emit the `JudgeResponse` JSON. Judges default to bounded `reasoning.max_tokens`, not `xhigh`
/// effort, so the combined completion cap still leaves visible verdict headroom.
#[must_use]
pub fn build_judge_request(
    judge: &PanelJudge,
    rubric: &str,
    candidate_render: &str,
) -> ChatRequest {
    let system = Message {
        role: Role::System,
        content: Content::Text(format!(
            "You are a strict grading judge. Apply the rubric and return ONLY a JSON object with \
             keys: score (0..1), verdict (\"accept\"|\"revise\"|\"reject\"|\"uncertain\"), \
             confidence (0..1), meta_prediction (0..1, your prediction of the panel-mean score), \
             dimensions (object of criterion->0..1), rationale (string, required on reject). \
             Rubric:\n{rubric}"
        )),
        reasoning: None,
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    let user = Message {
        role: Role::User,
        content: Content::Text(format!("Grade this candidate trace:\n{candidate_render}")),
        reasoning: None,
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    let mut req = ChatRequest::new(judge.slug.clone(), vec![system, user])
        .with_temperature(judge.sampling.temperature)
        .with_max_tokens(judge.effective_max_tokens());
    if let Some(reasoning) = judge.reasoning_param() {
        req = req.with_reasoning(reasoning);
    }
    if let Some(p) = judge.sampling.top_p {
        req = req.with_top_p(p);
    }
    if let Some(s) = judge.sampling.seed {
        req = req.with_seed(s);
    }
    req
}

struct DrainedContent {
    content: String,
    finish_reason: Option<String>,
    native_finish_reason: Option<String>,
}

impl DrainedContent {
    fn ingest(&mut self, delta: StreamDelta) {
        if let Some(c) = delta.content {
            self.content.push_str(&c);
        }
        if delta.finish_reason.is_some() {
            self.finish_reason = delta.finish_reason;
        }
        if delta.native_finish_reason.is_some() {
            self.native_finish_reason = delta.native_finish_reason;
        }
    }

    fn hit_length_cap(&self) -> bool {
        self.finish_reason.as_deref() == Some("length")
            || self.native_finish_reason.as_deref() == Some("length")
    }
}

/// Drain a judge's streamed completion into its concatenated content text. Judge calls are non-CoT
/// (a verdict, not a reasoning trace); we only need the content. A streamed provider error is
/// surfaced, never swallowed.
async fn drain_content(mut stream: gw_providers::DeltaStream) -> Result<DrainedContent> {
    let mut drained = DrainedContent {
        content: String::new(),
        finish_reason: None,
        native_finish_reason: None,
    };
    while let Some(item) = stream.next().await {
        let delta = item?;
        drained.ingest(delta);
    }
    Ok(drained)
}

fn empty_completion_error(
    judge: &PanelJudge,
    drained: &DrainedContent,
    max_tokens: Option<u32>,
    reasoning: Option<ReasoningParam>,
) -> JudgeError {
    if drained.hit_length_cap() {
        JudgeError::JudgeParse(format!(
            "judge {} was truncated at max_tokens before emitting a verdict \
             (finish_reason={:?}, native_finish_reason={:?}, max_tokens={:?}, reasoning={:?}); \
             raise judge max_tokens or lower the judge reasoning budget",
            judge.slug, drained.finish_reason, drained.native_finish_reason, max_tokens, reasoning
        ))
    } else {
        JudgeError::JudgeParse(format!(
            "judge {} returned an empty completion (finish_reason={:?}, native_finish_reason={:?})",
            judge.slug, drained.finish_reason, drained.native_finish_reason
        ))
    }
}

/// Call ONE judge over the injected provider and parse its sealed [`Grade`]. This spends a judge
/// token; the caller (`cache.rs`) wraps it so a cache hit skips this entirely. The scoring path
/// used is recorded for audit (here always the judge's pin; the route-capability fallback is the
/// engine's to resolve and pass in).
///
/// # Errors
/// Returns [`JudgeError::Provider`] if the judge call cannot be established or the stream resets, or
/// [`JudgeError::JudgeParse`] if the (non-empty) completion does not parse into a grade (V2: an
/// unparseable body errors rather than freezing a degenerate `Uncertain`/`0.0` in the cache, so the
/// engine can retry). A judge that legitimately voted `"uncertain"` parses fine and is NOT an error.
pub async fn grade_one<P: Provider + ?Sized>(
    provider: &P,
    judge: &PanelJudge,
    rubric: &str,
    candidate_render: &str,
) -> Result<Grade> {
    let req = build_judge_request(judge, rubric, candidate_render);
    let max_tokens = req.max_tokens;
    let reasoning = req.reasoning;
    let stream = provider.stream_chat(req).await?;
    let drained = drain_content(stream).await?;
    if drained.content.trim().is_empty() {
        return Err(empty_completion_error(
            judge, &drained, max_tokens, reasoning,
        ));
    }
    match parse_grade(judge, &drained.content, judge.scoring) {
        Ok(grade) => Ok(grade),
        Err(_) if drained.hit_length_cap() => Err(empty_completion_error(
            judge, &drained, max_tokens, reasoning,
        )),
        Err(err) => Err(err),
    }
}

/// Grade the whole panel in a BLIND, INDEPENDENT, SEALED first pass (§5.7): every judge is a
/// separate provider call and no judge sees another's verdict. Calls run concurrently; the grades
/// are returned in panel order.
///
/// This is the raw (uncached) path. Production callers go through `cache.rs`'s
/// [`grade_panel_cached`](crate::grade_panel_cached) so a re-run never re-spends. Returns
/// [`JudgeError::EmptyPanel`] if `judges` is empty (the consensus math must never run on no votes).
///
/// # Errors
/// Returns the FIRST judge call that fails to establish/stream (a transport failure is real — a
/// silently-dropped judge would corrupt the design effect). Unparseable-but-streamed responses
/// become `Uncertain` grades and do not error.
pub async fn grade_panel<P: Provider + ?Sized>(
    provider: &P,
    judges: &[PanelJudge],
    rubric: &str,
    candidate_render: &str,
) -> Result<Vec<Grade>> {
    if judges.is_empty() {
        return Err(JudgeError::EmptyPanel(
            "grade_panel requires at least one judge".into(),
        ));
    }
    // Sealed parallel pass: no judge prompt contains another judge's output.
    let futures = judges
        .iter()
        .map(|j| grade_one(provider, j, rubric, candidate_render));
    let grades = futures::future::try_join_all(futures).await?;
    Ok(grades)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_providers::{DeltaStream, ProviderError, StreamChatFuture, StreamDelta};
    use std::sync::Mutex;

    /// A fake provider that replays canned judge JSON per request, keyed by call order. Records the
    /// requests it saw so tests can assert the sealed prompts never leak another judge's output.
    struct FakeJudgeProvider {
        responses: Mutex<Vec<String>>,
        seen: Mutex<Vec<ChatRequest>>,
    }

    impl FakeJudgeProvider {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().map(str::to_string).collect()),
                seen: Mutex::new(Vec::new()),
            }
        }
    }

    impl Provider for FakeJudgeProvider {
        fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
            self.seen.lock().unwrap().push(req);
            let body = {
                let mut r = self.responses.lock().unwrap();
                if r.is_empty() {
                    "{\"score\":0.5,\"verdict\":\"uncertain\"}".to_string()
                } else {
                    r.remove(0)
                }
            };
            Box::pin(async move {
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
            })
        }
    }

    #[test]
    fn normalize_handles_likert_and_unit() {
        assert!((normalize_score(0.8) - 0.8).abs() < 1e-12);
        assert!((normalize_score(8.0) - 0.8).abs() < 1e-12);
        assert!((normalize_score(11.0) - 1.0).abs() < 1e-12);
        assert!((normalize_score(-1.0) - 0.0).abs() < 1e-12);
        assert_eq!(normalize_score(f64::NAN), 0.0);
    }

    #[test]
    fn parse_verdict_tokens() {
        assert_eq!(parse_verdict("accept"), Verdict::Accept);
        assert_eq!(parse_verdict("ADMIT"), Verdict::Accept);
        assert_eq!(parse_verdict("reject"), Verdict::Reject);
        assert_eq!(parse_verdict("revise"), Verdict::Revise);
        assert_eq!(parse_verdict("nonsense"), Verdict::Uncertain);
    }

    #[test]
    fn parse_grade_reads_full_response() {
        let judge = PanelJudge::new("deepseek/deepseek-v4-pro", "deepseek");
        let g = parse_grade(
            &judge,
            "{\"score\":0.9,\"verdict\":\"accept\",\"confidence\":0.8,\"meta_prediction\":0.7,\"rationale\":\"clean\"}",
            JudgeScoring::IntegerLikert,
        )
        .unwrap();
        assert_eq!(g.verdict, Verdict::Accept);
        assert!((g.score - 0.9).abs() < 1e-12);
        assert!((g.confidence - 0.8).abs() < 1e-12);
        assert_eq!(g.meta_prediction, Some(0.7));
        assert_eq!(g.raw["scoring_used"], "integer_likert");
    }

    #[test]
    fn unparseable_response_errors_not_frozen() {
        // V2: an unparseable body returns Err (mirroring the empty-completion path) so it is NOT
        // cached as a degenerate Uncertain/0.0 — the engine can retry.
        let judge = PanelJudge::new("m", "fam");
        let err = parse_grade(&judge, "not json at all", JudgeScoring::GEvalLogprob).unwrap_err();
        assert!(matches!(err, JudgeError::JudgeParse(_)));
    }

    #[test]
    fn fenced_json_parses() {
        // V2: the most common LLM output shape — a ```json fenced block — must parse.
        let judge = PanelJudge::new("m", "fam");
        let fenced = "```json\n{\"score\":0.85,\"verdict\":\"accept\"}\n```";
        let g = parse_grade(&judge, fenced, JudgeScoring::IntegerLikert).unwrap();
        assert_eq!(g.verdict, Verdict::Accept);
        assert!((g.score - 0.85).abs() < 1e-12);
    }

    #[test]
    fn prose_wrapped_json_parses() {
        // V2: a prose preamble/suffix around the object still parses (outermost {...} extraction).
        let judge = PanelJudge::new("m", "fam");
        let prose = "Here is my grade:\n{\"score\":0.4,\"verdict\":\"reject\"}\nHope that helps!";
        let g = parse_grade(&judge, prose, JudgeScoring::IntegerLikert).unwrap();
        assert_eq!(g.verdict, Verdict::Reject);
    }

    #[test]
    fn legitimate_uncertain_vote_parses_and_is_not_an_error() {
        // A judge that genuinely voted "uncertain" parses fine — only a STRUCTURAL failure errors.
        let judge = PanelJudge::new("m", "fam");
        let g = parse_grade(
            &judge,
            "{\"score\":0.5,\"verdict\":\"uncertain\"}",
            JudgeScoring::IntegerLikert,
        )
        .unwrap();
        assert_eq!(g.verdict, Verdict::Uncertain);
    }

    #[tokio::test]
    async fn grade_one_streams_and_parses() {
        let provider = FakeJudgeProvider::new(vec!["{\"score\":0.85,\"verdict\":\"accept\"}"]);
        let judge = PanelJudge::new("z-ai/glm-5.2", "glm");
        let g = grade_one(&provider, &judge, "rubric", "trace")
            .await
            .unwrap();
        assert_eq!(g.verdict, Verdict::Accept);
        assert!((g.score - 0.85).abs() < 1e-12);
    }

    #[tokio::test]
    async fn grade_panel_is_sealed_no_cross_leak() {
        let provider = FakeJudgeProvider::new(vec![
            "{\"score\":0.9,\"verdict\":\"accept\"}",
            "{\"score\":0.2,\"verdict\":\"reject\"}",
        ]);
        let judges = vec![PanelJudge::new("a", "fa"), PanelJudge::new("b", "fb")];
        let grades = grade_panel(&provider, &judges, "rubric", "trace")
            .await
            .unwrap();
        assert_eq!(grades.len(), 2);
        // No judge prompt may contain another judge's verdict/score — sealed first pass.
        let seen = provider.seen.lock().unwrap();
        for req in seen.iter() {
            let body = serde_json::to_string(&req).unwrap();
            assert!(!body.contains("\\\"verdict\\\":\\\"accept\\\""));
            assert!(!body.contains("\\\"verdict\\\":\\\"reject\\\""));
        }
    }

    #[tokio::test]
    async fn empty_panel_is_an_error() {
        let provider = FakeJudgeProvider::new(vec![]);
        let err = grade_panel(&provider, &[], "r", "t").await.unwrap_err();
        assert!(matches!(err, JudgeError::EmptyPanel(_)));
    }

    #[tokio::test]
    async fn empty_completion_is_a_parse_error() {
        struct EmptyProvider;
        impl Provider for EmptyProvider {
            fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
                Box::pin(async move {
                    let stream: DeltaStream =
                        Box::pin(futures::stream::iter(vec![
                            Ok::<StreamDelta, ProviderError>(StreamDelta {
                                finish_reason: Some("stop".into()),
                                ..Default::default()
                            }),
                        ]));
                    Ok(stream)
                })
            }
        }
        let judge = PanelJudge::new("m", "fam");
        let err = grade_one(&EmptyProvider, &judge, "r", "t")
            .await
            .unwrap_err();
        assert!(matches!(err, JudgeError::JudgeParse(_)));
    }

    #[test]
    fn judge_request_defaults_to_bounded_reasoning_with_verdict_headroom() {
        let judge = PanelJudge::new("m", "fam");
        let req = build_judge_request(&judge, "rubric", "trace");
        assert_eq!(req.max_tokens, Some(DEFAULT_JUDGE_MAX_TOKENS));
        assert_eq!(
            req.reasoning,
            Some(ReasoningParam::max_tokens(
                DEFAULT_JUDGE_REASONING_MAX_TOKENS
            ))
        );
        assert_eq!(req.temperature, Some(0.0));
    }

    #[test]
    fn judge_request_clamps_explicit_reasoning_to_verdict_headroom() {
        let judge = PanelJudge::new("m", "fam")
            .with_max_tokens(1_000)
            .with_reasoning_max_tokens(800);
        let req = build_judge_request(&judge, "rubric", "trace");
        assert_eq!(req.max_tokens, Some(800 + MIN_JUDGE_VERDICT_TOKENS));
        assert_eq!(req.reasoning, Some(ReasoningParam::max_tokens(800)));
    }

    #[test]
    fn judge_request_keeps_effort_mode_max_tokens_unfloored() {
        let judge = PanelJudge::new("m", "fam")
            .with_max_tokens(1_000)
            .with_reasoning_effort(ReasoningEffort::Xhigh);
        let req = build_judge_request(&judge, "rubric", "trace");
        assert_eq!(req.max_tokens, Some(1_000));
        assert_eq!(
            req.reasoning,
            Some(ReasoningParam::effort(ReasoningEffort::Xhigh))
        );
    }
}
