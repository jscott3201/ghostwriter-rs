//! Capture a candidate identity using the shared record-content hashing contract.
use gw_schema::{CandidateBinding, TrainingRecord};

/// Recompute candidate/run/area/content identity without trusting stored hash fields.
///
/// # Errors
/// Returns a serialization error if the canonical content projection cannot be encoded.
pub fn capture_candidate_binding(record: &TrainingRecord) -> crate::Result<CandidateBinding> {
    Ok(CandidateBinding {
        record_id: record.record_id.clone(),
        run_id: record.provenance.run_id.clone(),
        training_area: record.training_area.clone(),
        prompt_hash: crate::prompt_hash(&record.messages)?,
        record_hash: crate::record_hash(record)?,
    })
}
