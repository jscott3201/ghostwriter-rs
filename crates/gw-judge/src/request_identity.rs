//! Identity for the request and JSON interpretation actually used by the judge cache.
//!
//! The namespace replaces legacy folded rubric keys. A contract revision is independent of request
//! bytes: changing decoding/normalization semantics must invalidate grades even when the prompt is
//! unchanged. Routing/endpoint defaults applied inside a provider and resolved model revisions are
//! not represented by this layer.

use gw_providers::ChatRequest;
use serde::Serialize;

use crate::error::{JudgeError, Result};
use crate::panel::{JUDGE_INTERPRETATION_VERSION, JudgeScoring};

/// Pin an ordered judge's effective request/scoring contract using the production request builder
/// and the same versioned fingerprint as the never-re-spend cache. The candidate slot is fixed;
/// the separately captured input plan and generation contract bind future candidate contents.
///
/// # Errors
/// Rejects non-finite/out-of-domain sampling and invalid effective token caps before canonicalization.
pub fn judge_request_contract(
    judge: &crate::PanelJudge,
    rubric: &str,
) -> Result<serde_json::Value> {
    let request = crate::build_judge_request(judge, rubric, "");
    crate::EffectiveJudgeContract::json_score(&request)?;
    Ok(
        serde_json::json!({"request": request, "request_identity": request_fingerprint(&request, judge.rubric_id.as_deref())?, "family": judge.family, "rubric_id": judge.rubric_id, "scoring": JudgeScoring::JsonScore.as_str(), "interpretation_version": JUDGE_INTERPRETATION_VERSION}),
    )
}

#[derive(Serialize)]
struct RequestIdentity<'a> {
    request: &'a ChatRequest,
    rubric_id: Option<&'a str>,
    scoring_method: &'static str,
    interpretation_version: u32,
    // Preserve bit identity even where JSON float serialization is not injective (e.g. NaN bits).
    temperature_bits: Option<u64>,
    top_p_bits: Option<u64>,
}

/// Fingerprint the built request and optional audit rubric identity, without widening storage keys.
pub(crate) fn request_fingerprint(
    request: &ChatRequest,
    rubric_id: Option<&str>,
) -> Result<String> {
    fingerprint_version(request, rubric_id, JUDGE_INTERPRETATION_VERSION)
}

fn fingerprint_version(
    request: &ChatRequest,
    rubric_id: Option<&str>,
    version: u32,
) -> Result<String> {
    let identity = RequestIdentity {
        request,
        rubric_id,
        scoring_method: JudgeScoring::JsonScore.as_str(),
        interpretation_version: version,
        temperature_bits: request.temperature.map(f64::to_bits),
        top_p_bits: request.top_p.map(f64::to_bits),
    };
    let bytes = serde_json::to_vec(&identity).map_err(|error| {
        JudgeError::Invariant(format!(
            "could not serialize judge request identity: {error}"
        ))
    })?;
    Ok(format!(
        "judge-request-v2:{}",
        blake3::hash(&bytes).to_hex()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PanelJudge, build_judge_request};
    use gw_schema::Content;

    #[test]
    fn changed_interpretation_version_or_prompt_template_has_a_new_identity() {
        let judge = PanelJudge::new("judge", "family");
        let mut request = build_judge_request(&judge, "rubric", "trace");
        let current = request_fingerprint(&request, None).unwrap();
        assert!(current.starts_with("judge-request-v2:"));
        assert_eq!(current, request_fingerprint(&request, None).unwrap());
        let next = fingerprint_version(&request, None, JUDGE_INTERPRETATION_VERSION + 1).unwrap();
        assert_ne!(
            current, next,
            "identical requests under different decoders cannot share grades"
        );

        let Content::Text(system) = &mut request.messages[0].content else {
            panic!("system text")
        };
        system.push_str("\nA changed prompt template.");
        assert_ne!(current, request_fingerprint(&request, None).unwrap());
    }

    #[test]
    fn sampling_float_bits_survive_json_equivalence() {
        let judge = PanelJudge::new("judge", "family");
        for is_temperature in [true, false] {
            let mut first = build_judge_request(&judge, "rubric", "trace");
            let mut second = first.clone();
            let a = Some(f64::from_bits(0x7ff8_0000_0000_0001));
            let b = Some(f64::from_bits(0x7ff8_0000_0000_0002));
            if is_temperature {
                first.temperature = a;
                second.temperature = b;
            } else {
                first.top_p = a;
                second.top_p = b;
            }
            assert_eq!(
                serde_json::to_vec(&first).unwrap(),
                serde_json::to_vec(&second).unwrap()
            );
            assert_ne!(
                request_fingerprint(&first, None).unwrap(),
                request_fingerprint(&second, None).unwrap()
            );
        }
    }
}
