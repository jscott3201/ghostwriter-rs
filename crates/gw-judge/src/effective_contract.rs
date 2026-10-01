//! Live evidence identity, independent of cache keys and audit labels.

use std::collections::HashMap;

use gw_providers::{ChatRequest, ReasoningParam};

use crate::panel::JUDGE_INTERPRETATION_VERSION;
use crate::{Grade, JudgeError, JudgeScoring, PanelJudge, Result, build_judge_request};

/// An immutable description of the actual judge request and supported JSON interpretation.
///
/// Construct this from the request used to obtain a live observation. Family, rubric audit IDs,
/// record IDs, and provider receipts are not inputs. This establishes contract equality, not
/// empirical independence or proof that a remote provider executed the request.
///
/// The value is deliberately not deserializable. Current cache hits reconstruct it from the exact
/// request whose versioned key matched; historical stored decisions do not acquire new evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveJudgeContract {
    fingerprint: [u8; 32],
    model: String,
    temperature: f64,
    top_p: Option<f64>,
    seed: Option<i64>,
    cached_temperature: f64,
    cached_top_p: Option<f64>,
}

impl EffectiveJudgeContract {
    /// Bind an actual request to the current JSON-score response interpretation.
    ///
    /// This supports direct library callers supplying their own observations as well as the
    /// production request builder. All serialized request fields participate, including prompts,
    /// sampling, reasoning, and routing. Signed zero in sampling is one effective setting.
    ///
    /// # Errors
    /// Requires explicit finite temperature in `[0,2]`, optional finite top-p in `[0,1]`, a positive
    /// completion cap, and a positive reasoning cap when one is supplied.
    pub fn json_score(request: &ChatRequest) -> Result<Self> {
        Self::with_interpretation(request, JUDGE_INTERPRETATION_VERSION)
    }

    fn with_interpretation(request: &ChatRequest, interpretation_version: u32) -> Result<Self> {
        let Some(temperature) = request.temperature else {
            return Err(JudgeError::Invariant(
                "effective judge request requires an explicit temperature".into(),
            ));
        };
        if !temperature.is_finite()
            || !(0.0..=2.0).contains(&temperature)
            || request
                .top_p
                .is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p))
        {
            return Err(JudgeError::Invariant(
                "judge sampling requires finite temperature in [0,2] and top_p in [0,1]".into(),
            ));
        }
        if request.max_tokens.is_none_or(|cap| cap == 0)
            || matches!(
                request.reasoning,
                Some(ReasoningParam::MaxTokens { max_tokens: 0 })
            )
        {
            return Err(JudgeError::Invariant(
                "judge effective completion and reasoning token caps must be positive".into(),
            ));
        }
        let mut normalized = request.clone();
        normalized.temperature = Some(normalize_zero(temperature));
        normalized.top_p = request.top_p.map(normalize_zero);
        let bytes = serde_json::to_vec(&serde_json::json!({
            "request": normalized,
            "scoring_method": JudgeScoring::JsonScore.as_str(),
            "interpretation_version": interpretation_version,
        }))
        .map_err(|error| {
            JudgeError::Invariant(format!(
                "could not serialize effective judge request: {error}"
            ))
        })?;
        Ok(Self {
            fingerprint: *blake3::hash(&bytes).as_bytes(),
            model: request.model.clone(),
            temperature,
            top_p: request.top_p,
            seed: request.seed,
            cached_temperature: cache_float_projection(temperature)?,
            cached_top_p: request.top_p.map(cache_float_projection).transpose()?,
        })
    }

    fn matches_grade(&self, grade: &Grade) -> bool {
        self.model == grade.judge_model
            && (self.temperature == grade.temperature
                || self.cached_temperature == grade.temperature)
            && (self.top_p == grade.top_p || self.cached_top_p == grade.top_p)
            && self.seed == grade.seed
            && grade
                .raw
                .get("scoring_used")
                .is_none_or(|value| value.as_str() == Some(JudgeScoring::JsonScore.as_str()))
            && grade
                .raw
                .get("interpretation_version")
                .is_none_or(|value| value.as_u64() == Some(u64::from(JUDGE_INTERPRETATION_VERSION)))
    }
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

// The existing cache JSON parser can round a serialized f64 by one representable value. Accept
// exactly that persisted audit projection as well as the original live value, never an epsilon.
// The fingerprint above always binds the actual request; this projection cannot merge contracts
// or change cache keys, payloads, or historical data.
fn cache_float_projection(value: f64) -> Result<f64> {
    let restored: serde_json::Value = serde_json::to_vec(&serde_json::json!(value))
        .and_then(|bytes| serde_json::from_slice(&bytes))
        .map_err(|error| {
            JudgeError::Invariant(format!("could not project cached judge sampling: {error}"))
        })?;
    restored.as_f64().ok_or_else(|| {
        JudgeError::Invariant("cached judge sampling projection was not a number".into())
    })
}

pub(crate) fn validate_unique<'a>(
    contracts: impl IntoIterator<Item = &'a EffectiveJudgeContract>,
) -> Result<()> {
    let mut positions = HashMap::new();
    for (position, contract) in contracts.into_iter().enumerate() {
        if let Some(first) = positions.insert(contract.fingerprint, position) {
            return Err(JudgeError::DuplicateJudgeEvidence {
                first,
                duplicate: position,
            });
        }
    }
    Ok(())
}

/// Validate the effective requests and interpretation of a configured panel before spending.
///
/// Audit aliases and raw token caps that resolve to one request do not create another observation.
/// This check applies to both automatic admission and review-only collection; count feasibility is
/// assessed separately by [`crate::assess_panel`].
///
/// # Errors
/// Rejects invalid effective request settings and duplicate contracts.
pub fn validate_judge_panel(judges: &[PanelJudge], rubric: &str) -> Result<()> {
    let contracts = judges
        .iter()
        .map(|judge| EffectiveJudgeContract::json_score(&build_judge_request(judge, rubric, "")))
        .collect::<Result<Vec<_>>>()?;
    validate_unique(&contracts)
}

/// The live consensus boundary requires evidence even when grades were supplied by a caller.
pub(crate) fn validate_grade_evidence(grades: &[Grade]) -> Result<()> {
    let contracts = grades
        .iter()
        .enumerate()
        .map(|(position, grade)| {
            let contract = grade.effective_contract.as_ref().ok_or_else(|| {
                JudgeError::Invariant(format!(
                    "panel position {position} has missing effective judge evidence"
                ))
            })?;
            if !contract.matches_grade(grade) {
                return Err(JudgeError::Invariant(format!(
                    "panel position {position} has inconsistent effective judge evidence"
                )));
            }
            Ok(contract)
        })
        .collect::<Result<Vec<_>>>()?;
    validate_unique(contracts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ChatRequest {
        build_judge_request(&PanelJudge::new("judge", "family"), "rubric", "candidate")
    }

    #[test]
    fn response_interpretation_and_actual_request_fields_participate() {
        let original = request();
        let before = EffectiveJudgeContract::json_score(&original).unwrap();
        let next = EffectiveJudgeContract::with_interpretation(
            &original,
            JUDGE_INTERPRETATION_VERSION + 1,
        )
        .unwrap();
        assert_ne!(before.fingerprint, next.fingerprint);
        validate_unique([&before, &next]).unwrap();
        let changes: [fn(&mut ChatRequest); 8] = [
            |r| r.model.push('x'),
            |r| {
                r.messages[0].content =
                    gw_schema::Content::Text("different response instructions".into())
            },
            |r| r.temperature = Some(0.2),
            |r| r.top_p = Some(0.8),
            |r| r.seed = Some(7),
            |r| r.max_tokens = Some(8_000),
            |r| r.reasoning = None,
            |r| r.provider = Some(gw_providers::ProviderRouting::pin("different-provider")),
        ];
        for change in changes {
            let mut changed = original.clone();
            change(&mut changed);
            let after = EffectiveJudgeContract::json_score(&changed).unwrap();
            assert_ne!(before.fingerprint, after.fingerprint);
            validate_unique([&before, &after]).unwrap();
        }
        let mut precise = original;
        precise.temperature = Some(20.0 / 13.0);
        let mut adjacent = precise.clone();
        adjacent.temperature = precise.temperature.map(f64::next_down);
        assert_ne!(
            serde_json::to_vec(&precise).unwrap(),
            serde_json::to_vec(&adjacent).unwrap()
        );
        let precise = EffectiveJudgeContract::json_score(&precise).unwrap();
        let adjacent = EffectiveJudgeContract::json_score(&adjacent).unwrap();
        assert_ne!(precise.fingerprint, adjacent.fingerprint);
        validate_unique([&precise, &adjacent]).unwrap();
    }

    #[test]
    fn missing_nonfinite_and_invalid_controls_cannot_establish_live_evidence() {
        let changes: [fn(&mut ChatRequest); 8] = [
            |r| r.temperature = None,
            |r| r.temperature = Some(f64::NAN),
            |r| r.temperature = Some(f64::INFINITY),
            |r| r.top_p = Some(f64::NAN),
            |r| r.top_p = Some(-0.1),
            |r| r.max_tokens = None,
            |r| r.max_tokens = Some(0),
            |r| r.reasoning = Some(ReasoningParam::max_tokens(0)),
        ];
        for change in changes {
            let mut changed = request();
            change(&mut changed);
            assert!(matches!(
                EffectiveJudgeContract::json_score(&changed),
                Err(JudgeError::Invariant(_))
            ));
        }
    }
}
