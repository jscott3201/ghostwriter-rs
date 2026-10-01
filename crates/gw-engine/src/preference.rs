//! Pure preference preparation; no provider, store, Engine instance or runtime is needed.
use gw_schema::{
    AdmissionIntent, PREFERENCE_VERSION, PreferenceAssessment, PreferenceBindingStatus,
    PreferenceEvidence, PreferenceRecord, PreferenceTermination, TrainingRecord, Verdict,
};
use gw_storage::{capture_preference_source, is_selected_admitted, preference_hash};

use crate::{EngineError, Result, grade::verifier_grade_for_reasoning};

fn invalid(reason: &str) -> EngineError {
    EngineError::Invariant(format!("preference preparation: {reason}"))
}

fn decisive(record: &TrainingRecord, cot_required: bool) -> Result<gw_judge::VerifierGrade> {
    let verifier = verifier_grade_for_reasoning(record, cot_required)?;
    let facts = verifier
        .verification
        .interpretation
        .as_ref()
        .ok_or_else(|| invalid("missing verification facts"))?;
    if [&facts.reasoning, &facts.answer, &facts.execution]
        .into_iter()
        .any(|axis| {
            axis.observation.as_ref().is_some_and(|observation| {
                observation.outcome == gw_schema::VerificationOutcome::Unknown
            })
        })
        || record.verification.needs_review.is_some()
    {
        return Err(invalid(
            "uncertain verification evidence cannot establish preference",
        ));
    }
    let judging = &record.judging;
    if judging.admission_intent != AdmissionIntent::Automatic
        || !matches!(judging.verdict, Some(Verdict::Admit | Verdict::Reject))
        || judging.panel.is_empty()
        || judging.decisive_count != Some(judging.panel.len())
    {
        return Err(invalid(
            "preference requires decisive automatic judge evidence",
        ));
    }
    Ok(verifier)
}

/// Validate an exact judge-ranking assessment and return conversational preference arrays.
///
/// Supplied snapshots must equal fresh captures of both records; scores come from those records,
/// not arbitrary supplied ranking values. Chosen selection and task-authority gates are reused.
/// Protocol revision is a declaration; execution lineage and receipt/output binding stay unbound,
/// and stop/length remain unknown. This result does not certify tokenization or DPO likelihoods.
///
/// # Errors
/// Rejects unsupported versions/policies, stale evidence/hashes, identities from different runs or
/// areas, unsupported structure, uncertain/tied scores, unmet strict margins, unselected chosen
/// records, or authoritative verification failures. Performs no I/O or provider calls.
pub fn prepare_preference_pair(
    chosen: &TrainingRecord,
    rejected: &TrainingRecord,
    assessment: &PreferenceAssessment,
) -> Result<PreferenceRecord> {
    let policy = &assessment.policy;
    if assessment.version != PREFERENCE_VERSION || policy.version != PREFERENCE_VERSION {
        return Err(invalid("unsupported preference version"));
    }
    if policy.protocol_revision.trim().is_empty()
        || !policy.minimum_margin.is_finite()
        || !(0.0..=1.0).contains(&policy.minimum_margin)
    {
        return Err(invalid(
            "protocol revision and finite margin in [0,1] are required",
        ));
    }
    let projection = gw_format::project_preference_messages(chosen, rejected, policy.cot_policy)?;
    let fresh_chosen = capture_preference_source(chosen)?;
    let fresh_rejected = capture_preference_source(rejected)?;
    if fresh_chosen != assessment.chosen
        || fresh_rejected != assessment.rejected
        || preference_hash("ghostwriter.preference.source.v1", &fresh_chosen)?
            != preference_hash("ghostwriter.preference.source.v1", &assessment.chosen)?
        || preference_hash("ghostwriter.preference.source.v1", &fresh_rejected)?
            != preference_hash("ghostwriter.preference.source.v1", &assessment.rejected)?
    {
        return Err(invalid(
            "supplied source scores or material snapshots do not match the records",
        ));
    }
    if chosen.record_id == rejected.record_id
        || chosen.provenance.run_id != rejected.provenance.run_id
        || chosen.training_area != rejected.training_area
        || fresh_chosen.prompt_hash != fresh_rejected.prompt_hash
        || chosen.task_provenance != rejected.task_provenance
        || chosen.verification_contract != rejected.verification_contract
    {
        return Err(invalid(
            "distinct source IDs and matching run, area, prompt and task contracts are required",
        ));
    }
    let chosen_verifier = decisive(chosen, policy.cot_required)?;
    decisive(rejected, policy.cot_required)?;
    if !is_selected_admitted(chosen) {
        return Err(invalid("chosen record is not selected and admitted"));
    }
    if chosen_verifier.is_hard_reject() || chosen_verifier.blocks_admission() {
        return Err(invalid("chosen record fails authoritative verification"));
    }
    let chosen_score = chosen
        .judging
        .aggregate
        .ok_or_else(|| invalid("missing chosen aggregate"))?;
    let rejected_score = rejected
        .judging
        .aggregate
        .ok_or_else(|| invalid("missing rejected aggregate"))?;
    if chosen_score - rejected_score <= policy.minimum_margin {
        return Err(invalid(
            "chosen score must strictly exceed rejected score plus the margin",
        ));
    }
    let original_prefix_hash = preference_hash(
        "ghostwriter.preference.prefix.v1",
        &projection.original_prefix,
    )?;
    let decision_evidence_hash = preference_hash(
        "ghostwriter.preference.decision.v1",
        &(assessment, &projection.original_prefix),
    )?;
    let evidence = PreferenceEvidence {
        assessment: assessment.clone(),
        original_prefix: projection.original_prefix,
        original_prefix_hash,
        decision_evidence_hash,
        chosen_selected: true,
        chosen_termination: PreferenceTermination::Unknown,
        rejected_termination: PreferenceTermination::Unknown,
        receipt_output_binding: PreferenceBindingStatus::Unbound,
        decision_execution_binding: PreferenceBindingStatus::Unbound,
    };
    let pair_id = preference_hash(
        "ghostwriter.preference.pair.v1",
        &(
            &projection.prompt,
            &projection.chosen,
            &projection.rejected,
            &evidence,
        ),
    )?;
    Ok(PreferenceRecord {
        version: PREFERENCE_VERSION,
        pair_id,
        prompt: projection.prompt,
        chosen: projection.chosen,
        rejected: projection.rejected,
        evidence,
    })
}

#[cfg(test)]
mod tests;
