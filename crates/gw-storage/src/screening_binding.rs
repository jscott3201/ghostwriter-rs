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
    capture_screening_input_for(record, policy, gw_schema::ExportSchemaVersion::CURRENT)
}

pub(crate) fn capture_screening_input_for(
    record: &TrainingRecord,
    policy: &ScreeningPolicy,
    version: gw_schema::ExportSchemaVersion,
) -> Result<ScreeningInputBinding> {
    let origin = (version >= gw_schema::ExportSchemaVersion::RecordOrigins)
        .then(|| record.origin.projection());
    let generated = record.origin.generated();
    let mut parents = generated.map_or_else(Vec::new, |g| g.provenance.parent_ids.clone());
    parents.sort();
    if parents.iter().any(|parent| parent.trim().is_empty())
        || parents.windows(2).any(|p| p[0] == p[1])
    {
        return Err(integrity("duplicate or empty declared parent identity"));
    }
    let key = ScreeningRecordId {
        run_id: record.run_id().to_owned(),
        record_id: record.record_id.clone(),
    };
    let record_hash = crate::record_hash(record)?;
    let export_projection_id = screening_projection_id(
        &record.training_area,
        &record.messages,
        record.task_provenance.as_ref(),
        record.verification_contract.as_ref(),
        record.judging.verdict,
        origin.as_ref(),
        (version == gw_schema::ExportSchemaVersion::ToolDefinitions)
            .then_some(record.tools.as_deref()),
    )?;
    let mut input = serde_json::json!({
        "record":key,"record_hash":record_hash,"export_projection_id":export_projection_id,"task":record.task_provenance,
        "messages":record.messages,"tools":record.tools,
        "verification_contract":record.verification_contract,"parents":parents,
        "siblings":generated.map(|g| serde_json::json!({"group":g.generation.sibling_group_id,"index":g.generation.completion_index,"count":g.generation.n_completions})),
        "eligible":crate::is_selected_admitted(record),"verdict":record.judging.verdict,
        "teacher":generated.map(|g| &g.provenance.teacher),"policy":policy
    });
    if let Some(origin) = &origin {
        input["origin"] = serde_json::to_value(origin)?;
    }
    let screening_input_id = screening_hash(
        if version == gw_schema::ExportSchemaVersion::ToolDefinitions {
            "screening-record-input-v4-tools"
        } else if origin.is_some() {
            "screening-record-input-v3-origins"
        } else {
            "screening-record-input-v2"
        },
        &input,
    )?;
    Ok(ScreeningInputBinding {
        record: key,
        record_hash,
        export_projection_id,
        screening_input_id,
    })
}

/// The same pure typed projection is derived from raw records and independently decoded rows.
/// Historical v3/v4 projections omit definitions; v5 binds their explicit presence and payload.
/// Scores, token costs and publication history are deliberately outside this screening digest.
pub(crate) fn screening_projection_id(
    training_area: &str,
    messages: &[Message],
    task: Option<&TaskProvenance>,
    verification_contract: Option<&VerificationContract>,
    verdict: Option<Verdict>,
    origin: Option<&gw_schema::ExportRecordOrigin>,
    tools: Option<Option<&[serde_json::Value]>>,
) -> Result<String> {
    let mut value = serde_json::json!({
        "training_area":training_area,"messages":messages,"task":task,
        "verification_contract":verification_contract,"verdict":verdict
    });
    if let Some(origin) = origin {
        value["origin"] = serde_json::to_value(origin)?;
    }
    if let Some(tools) = tools {
        value["tools"] = serde_json::to_value(tools)?;
    }
    screening_hash(
        if tools.is_some() {
            "screening-export-projection-v3-tools"
        } else if origin.is_some() {
            "screening-export-projection-v2-origins"
        } else {
            "screening-export-projection-v1"
        },
        &value,
    )
}
