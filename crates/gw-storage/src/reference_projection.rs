//! Independent public row checks. Stable hashes bind declarations; storage supplies local authority.
use crate::{Result, artifact::integrity, export::Projected, reference_registration::id};
use gw_schema::*;

pub(crate) fn validate(row: &Projected, version: ExportSchemaVersion) -> Result<()> {
    if version < ExportSchemaVersion::RecordOrigins {
        if row.origin_json.is_some() {
            return Err(integrity("historical row contains v4 origin"));
        }
        return Ok(());
    }
    let json = row
        .origin_json
        .as_ref()
        .ok_or_else(|| integrity("v4 row lacks required origin"))?;
    let origin: ExportRecordOrigin = serde_json::from_str(json)?;
    origin.validate().map_err(integrity)?;
    if crate::export::canonical_origin_json(&origin)? != *json {
        return Err(integrity("origin must use strict canonical typed JSON"));
    }
    let ExportRecordOrigin::ReviewedReference(reference) = origin else {
        if row.verdict.as_deref() != Some("admit") {
            return Err(integrity("generated export lacks its admission verdict"));
        }
        return Ok(());
    };
    let task: ExportTaskProjection = serde_json::from_str(
        row.task_json
            .as_ref()
            .ok_or_else(|| integrity("reference row lacks task"))?,
    )?;
    let messages: Vec<Message> = serde_json::from_str(&row.messages_json)?;
    task.validate(&messages).map_err(integrity)?;
    let [_, answer] = messages.as_slice() else {
        return Err(integrity("reference row requires exact prompt and module"));
    };
    let Content::Text(code) = &answer.content else {
        return Err(integrity("reference module must be text"));
    };
    let Oracle::CodingSuite { suite } = &task.verification_contract.oracle else {
        return Err(integrity("reference row requires coding suite"));
    };
    if row.record_id
        != id(
            "ghostwriter.reference-record.v1",
            &(&reference.batch_id, &reference.member_id),
        )?
        || reference.permitted_use != TaskPermittedUse::Training
        || task.provenance.split.role != TaskSplitRole::Train
        || !task
            .provenance
            .rights
            .permitted_uses
            .contains(&TaskPermittedUse::Training)
        || suite.suite_id != reference.suite_id
        || coding_digest("ghostwriter.coding-module.v1", code.as_bytes())
            != reference.reference_code_id
        || row.verdict.is_some()
        || row.judge_aggregate.is_some()
        || row.reasoning_tokens != 0
        || row.tools_json.is_some()
        || answer.role != Role::Assistant
        || answer.reasoning.is_some()
        || answer.reasoning_details.is_some()
        || answer.tool_calls.is_some()
        || answer.tool_call_id.is_some()
        || answer.name.is_some()
    {
        return Err(integrity(
            "reference row contradicts its origin, task, module or absent judge facts",
        ));
    }
    Ok(())
}
