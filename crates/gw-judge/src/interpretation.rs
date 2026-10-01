//! Pure production JSON-score interpretation, shared by live grading and offline intake.
use crate::Verdict;
use serde::Deserialize;
use std::collections::BTreeMap;

/// Interpreted response fields, before judge metadata or provider provenance is attached.
#[derive(Debug, Clone, PartialEq)]
pub struct InterpretedJudgeResponse {
    /// Current normalized quality score.
    pub score: f64,
    /// Current four-way verdict.
    pub verdict: Verdict,
    /// Audit-only confidence, with the existing default and clamping.
    pub confidence: f64,
    /// Optional normalized prediction of the panel mean.
    pub meta_prediction: Option<f64>,
    /// Optional dimensions as supplied by the response.
    pub dimensions: Option<BTreeMap<String, f64>>,
    /// Optional rationale text.
    pub rationale: Option<String>,
}

/// Parse exact concatenated response text using production extraction and normalization.
/// Parsing establishes neither stream termination nor truthful historical collection.
///
/// # Errors
/// Returns the same structural JSON error used by production grading for malformed responses.
pub fn interpret_judge_response(
    response_text: &str,
) -> Result<InterpretedJudgeResponse, serde_json::Error> {
    let r = serde_json::from_str::<JudgeResponse>(response_text.trim()).or_else(|first_err| {
        match extract_json(response_text) {
            Some(body) => serde_json::from_str::<JudgeResponse>(body),
            None => Err(first_err),
        }
    })?;
    Ok(InterpretedJudgeResponse {
        score: normalize_score(r.score),
        verdict: parse_verdict(&r.verdict),
        confidence: r.confidence.unwrap_or(0.0).clamp(0.0, 1.0),
        meta_prediction: r.meta_prediction.map(normalize_score),
        dimensions: r.dimensions,
        rationale: r.rationale,
    })
}

/// The minimal JSON contract a judge is asked to emit (so the panel can parse it deterministically).
/// A structurally invalid response returns [`crate::JudgeError::JudgeParse`] and is not cached.
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

/// Normalize a raw JSON score onto `[0, 1]`: values above 1 are divided by 10, then the result is
/// clamped to `[0, 1]`. Non-finite values become zero. This is the existing normalization policy,
/// shared by scores and meta-predictions; it does not derive scores from token probabilities.
pub(crate) fn normalize_score(raw: f64) -> f64 {
    if !raw.is_finite() {
        return 0.0;
    }
    let s = if raw > 1.0 { raw / 10.0 } else { raw };
    s.clamp(0.0, 1.0)
}

/// Parse a judge verdict token to the per-grade [`Verdict`]; an unknown token is `Uncertain`.
pub(crate) fn parse_verdict(token: &str) -> Verdict {
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
