//! Shared effective request builders and versioned pure execution behavior.
use crate::{AreaConfig, Result};
use gw_generate::{ReasoningPolicy, SamplingPreset, TeacherCall};
use gw_schema::Message;

pub(crate) const REVISION_SEED_OFFSET: i64 = 1_000_000;
pub(crate) const TRUNCATION_RETRY_MAX_TOKENS: u32 = 32_000;

pub(crate) fn teacher_call(
    area: &AreaConfig,
    messages: Vec<Message>,
    sampling: SamplingPreset,
) -> TeacherCall {
    let mut call =
        TeacherCall::new(&area.teacher_slug, messages, area.max_tokens).with_sampling(sampling);
    if let Some(tokens) = area.teacher_reasoning_max_tokens {
        call = call.with_reasoning(ReasoningPolicy::MaxTokens(tokens));
    }
    call
}
pub(crate) fn revision_seed(seed: i64, completion_index: u32) -> i64 {
    seed.wrapping_add(i64::from(completion_index))
        .wrapping_add(REVISION_SEED_OFFSET)
}
pub(crate) fn retry_max_tokens(tokens: u32) -> Option<u32> {
    if tokens >= TRUNCATION_RETRY_MAX_TOKENS {
        return None;
    }
    Some(((u64::from(tokens) * 3).div_ceil(2)).min(u64::from(TRUNCATION_RETRY_MAX_TOKENS)) as u32)
}
pub(crate) fn contract(area: &AreaConfig) -> Result<serde_json::Value> {
    area.assess_admission()?;
    let base = teacher_call(area, vec![], SamplingPreset::official().with_seed(0));
    let initial = base.build()?;
    let retry = retry_max_tokens(base.max_tokens)
        .map(|tokens| {
            let mut widened = base.clone();
            widened.max_tokens = tokens;
            widened.build()
        })
        .transpose()?;
    let revision = teacher_call(
        area,
        vec![],
        SamplingPreset::official().with_seed(revision_seed(0, 0)),
    )
    .build()?;
    let judges = area
        .judges
        .iter()
        .map(|judge| gw_judge::judge_request_contract(judge, &area.rubric))
        .collect::<gw_judge::Result<Vec<_>>>()?;
    Ok(serde_json::json!({
        "training_area": area.training_area,
        "generation": {"teacher_template": initial, "k": area.k.max(1), "seed_binding": "captured-seed-wrapping-add-completion-index-v1", "truncation_retry": retry, "truncation_attempts": 1, "revision_template": revision, "revision_seed_offset": REVISION_SEED_OFFSET, "revision_attempts": 1, "revision_truncation_retry": false, "second_revise": "reject"},
        "verification": {"interpretation_version": gw_schema::VERIFICATION_INTERPRETATION_VERSION,"contract": "verification-policy-typed-facts-v2", "numeric": "strict-decimal-binary64-final-line-marker-max-abs-rel-v1", "cot_required": area.cot_required, "execution_evidence": "required-tests-keyed-evidence-v2"},
        "judging": {"request_contracts": judges, "candidate_render": "openai-messages-supervised-v1", "rubric": area.rubric, "correlation_rho": area.correlation_rho, "weights": "equal", "consensus": "weighted-design-effect-sp-bts-v1", "evidence": "distinct-effective-request-and-interpretation-v1", "admission_intent": area.admission_intent,
            "thresholds": {"accept_threshold": area.thresholds.accept_threshold, "reject_below": area.thresholds.reject_below, "min_n_eff": area.thresholds.min_n_eff, "min_n_eff_ratio": area.thresholds.min_n_eff_ratio}},
        "selection": "established-winner-else-max-admissible-aggregate-tie-lowest-completion-index-v1",
        "qc": gw_generate::user_qc_contract(),
        "priors": {"implementation": "run-scoped-in-memory-exact-cosine-v1", "selection": "admitted-formatted-exported-in-same-run", "ownership": "one-vector-per-record", "exclude": "all-records-of-current-seed-item", "initialization": "lazy-before-fresh-generation"}
    }))
}
