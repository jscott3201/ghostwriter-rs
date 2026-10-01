//! Shared raw inputs for the pure planner and transactional publication checks.
use crate::{Result, artifact::integrity};
use gw_schema::{
    Message, ScreeningInputBinding, ScreeningPolicy, ScreeningRecordId, TaskProvenance,
    TrainingRecord, Verdict, VerificationContract,
};
use serde::Serialize;

pub(crate) fn screening_hash(domain: &str, value: &impl Serialize) -> Result<String> {
    crate::canonical_json_hash(&serde_json::to_value((domain, value))?)
}

/// Bind all source-planner inputs while excluding publication history and equivalent eligible states.
///
/// # Errors
/// Rejects duplicate/empty parent identities or noncanonical record/policy serialization.
pub fn capture_screening_input(
    record: &TrainingRecord,
    policy: &ScreeningPolicy,
) -> Result<ScreeningInputBinding> {
    let mut parents = record.provenance.parent_ids.clone();
    parents.sort();
    if parents.iter().any(|parent| parent.trim().is_empty())
        || parents.windows(2).any(|p| p[0] == p[1])
    {
        return Err(integrity("duplicate or empty declared parent identity"));
    }
    let key = ScreeningRecordId {
        run_id: record.provenance.run_id.clone(),
        record_id: record.record_id.clone(),
    };
    let record_hash = crate::record_hash(record)?;
    let export_projection_id = screening_projection_id(
        &record.training_area,
        &record.messages,
        record.task_provenance.as_ref(),
        record.verification_contract.as_ref(),
        record.judging.verdict,
    )?;
    let screening_input_id = screening_hash(
        "screening-record-input-v2",
        &serde_json::json!({
            "record":key,"record_hash":record_hash,"export_projection_id":export_projection_id,"task":record.task_provenance,
            "messages":record.messages,"tools":record.tools,
            "verification_contract":record.verification_contract,"parents":parents,
            "siblings":{"group":record.generation.sibling_group_id,"index":record.generation.completion_index,"count":record.generation.n_completions},
            "eligible":crate::is_selected_admitted(record),"verdict":record.judging.verdict,
            "teacher":record.provenance.teacher,"policy":policy
        }),
    )?;
    Ok(ScreeningInputBinding {
        record: key,
        record_hash,
        export_projection_id,
        screening_input_id,
    })
}

/// The same pure typed projection is derived from raw records and independently decoded rows.
/// Top-level tool definitions are not exported by v3; full raw bindings retain them separately.
/// Scores, token costs and publication history are deliberately outside this screening digest.
pub(crate) fn screening_projection_id(
    training_area: &str,
    messages: &[Message],
    task: Option<&TaskProvenance>,
    verification_contract: Option<&VerificationContract>,
    verdict: Option<Verdict>,
) -> Result<String> {
    screening_hash(
        "screening-export-projection-v1",
        &serde_json::json!({
            "training_area":training_area,"messages":messages,"task":task,
            "verification_contract":verification_contract,"verdict":verdict
        }),
    )
}
