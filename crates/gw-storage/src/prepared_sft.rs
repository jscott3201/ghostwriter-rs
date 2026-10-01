//! Import a whole captured prepared input only after all structure and source bindings pass.
use std::collections::{BTreeMap, BTreeSet};

use gw_schema::{
    ExportArtifact, ExportTaskProjection, Message, MultiTurnLoss, PreparedSftDeclaredTask,
    PreparedSftExampleSource, PreparedSftPayload, PreparedSftScreeningIdentity, Role,
    ScreenedExampleLayout, TaskSplitRole, TrlFormat, decode_prepared_sft_frame,
};
use serde::{Deserialize, Serialize};

use crate::artifact::{integrity, verify_snapshot_with_rows};
use crate::export::Projected;
use crate::{ArtifactSnapshotReport, Result};

/// Separate verification evidence for a complete input build; no model execution is implied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftVerificationReport {
    /// Version of this captured-byte receipt.
    pub report_version: u32,
    /// Complete payload/source identity from the original framing.
    pub build_id: String,
    /// Complete captured build size.
    pub byte_length: u64,
    /// Ordinary BLAKE3 of the complete build bytes.
    pub snapshot_blake3: String,
    /// Actual verification result from the embedded Parquet, never a caller-asserted report.
    pub source_verification: ArtifactSnapshotReport,
    /// Number of wholly validated ordered examples.
    pub example_count: u64,
    /// Rust structure, accounting and source bindings passed.
    pub structural_source_validation: String,
    /// Official tokenizer/template replay must run in the pinned Python consumer.
    pub tokenizer_replay: String,
}

/// Opaque whole-build import. Accessors expose only validated owned state; no partial import exists.
#[derive(Debug)]
pub struct VerifiedPreparedSft {
    payload: PreparedSftPayload,
    report: PreparedSftVerificationReport,
}
impl VerifiedPreparedSft {
    /// Fully validated payload; official tokenizer semantics still require Python replay.
    #[must_use]
    pub fn payload(&self) -> &PreparedSftPayload {
        &self.payload
    }
    /// Receipt computed from this exact captured build and its independently reverified source.
    #[must_use]
    pub fn report(&self) -> &PreparedSftVerificationReport {
        &self.report
    }
}

/// Validate one immutable build, verify its original source, and reconcile every source binding.
///
/// # Errors
/// Rejects unsupported framing/JSON, incomplete or contradictory examples, source/policy mismatch,
/// corrupt or unverifiable Parquet, or unknown source references. No partial result is returned.
pub fn verify_prepared_sft_snapshot(snapshot: Vec<u8>) -> Result<VerifiedPreparedSft> {
    let frame = decode_prepared_sft_frame(&snapshot).map_err(integrity)?;
    let payload = PreparedSftPayload::from_json(frame.payload).map_err(integrity)?;
    let (source, rows) = verify_snapshot_with_rows(frame.source.to_vec())?;
    if payload.source.artifact_id != source.artifact.artifact_id
        || payload.source.byte_length != source.byte_length
        || payload.source.snapshot_blake3 != source.snapshot_blake3
        || payload.manifest.source_record_count != rows.len() as u64
    {
        return Err(integrity("prepared SFT source snapshot/count mismatch"));
    }
    validate_policy(&payload, &source.artifact)?;
    validate_bindings(&payload, &source.artifact, &rows)?;
    let report = PreparedSftVerificationReport {
        report_version: 1,
        build_id: frame.build_id,
        byte_length: snapshot.len() as u64,
        snapshot_blake3: blake3::hash(&snapshot).to_hex().to_string(),
        source_verification: source,
        example_count: payload.examples.len() as u64,
        structural_source_validation: "passed".into(),
        tokenizer_replay: "not_run".into(),
    };
    Ok(VerifiedPreparedSft { payload, report })
}

fn screening_identity(artifact: &ExportArtifact) -> Option<PreparedSftScreeningIdentity> {
    artifact
        .screening
        .as_ref()
        .map(|w| PreparedSftScreeningIdentity {
            plan_id: w.plan.plan_id.clone(),
            policy_id: w.plan.policy_id.clone(),
            screening_input_id: w.plan.screening_input_id.clone(),
            protected_input_id: w.plan.protected_input_id.clone(),
            grouping_id: w.plan.grouping_id.clone(),
            population_id: w.population_id.clone(),
        })
}

fn validate_policy(payload: &PreparedSftPayload, artifact: &ExportArtifact) -> Result<()> {
    let recipe = &payload.manifest.recipe;
    if recipe.screening != screening_identity(artifact) {
        return Err(integrity("prepared SFT screening identity mismatch"));
    }
    if let Some(w) = &artifact.screening {
        let policy = &w.plan.declaration.policy;
        let layout = match recipe.multi_turn_loss {
            MultiTurnLoss::AllAssistant => ScreenedExampleLayout::AssistantPrefixV1,
            MultiTurnLoss::FinalTurnOnly => ScreenedExampleLayout::FullConversationFinalV1,
        };
        if artifact.manifest.target != TrlFormat::OpenAiMessages
            || policy.target != TrlFormat::OpenAiMessages
            || artifact.manifest.cot_policy != recipe.cot_policy
            || policy.cot_policy != recipe.cot_policy
            || artifact.manifest.multi_turn_loss != recipe.multi_turn_loss
            || policy.multi_turn_loss != recipe.multi_turn_loss
            || w.layout != layout
            || payload.manifest.qualification_limits["contamination_screening"]
                != serde_json::to_value(w.plan.lexical_status)?
        {
            return Err(integrity(
                "prepared SFT source screening policy or claim mismatch",
            ));
        }
    }
    Ok(())
}

fn validate_bindings(
    payload: &PreparedSftPayload,
    artifact: &ExportArtifact,
    rows: &[Projected],
) -> Result<()> {
    let known: BTreeMap<_, _> = rows
        .iter()
        .map(|row| (row.record_id.as_str(), row))
        .collect();
    let mut targets: BTreeMap<&str, BTreeSet<u64>> = BTreeMap::new();
    let mut rejected_records = BTreeSet::new();
    for example in &payload.examples {
        let row = known
            .get(example.source.record_id.as_str())
            .ok_or_else(|| integrity("unknown prepared SFT source record"))?;
        validate_source(&example.source, artifact, row)?;
        targets
            .entry(&example.source.record_id)
            .or_default()
            .insert(example.target_index);
    }
    for rejection in &payload.manifest.rejections {
        if !known.contains_key(rejection.record_id.as_str()) {
            return Err(integrity("unknown prepared SFT rejected record"));
        }
        if let Some(target) = rejection.target_index {
            targets
                .entry(&rejection.record_id)
                .or_default()
                .insert(target);
        } else {
            rejected_records.insert(rejection.record_id.as_str());
        }
    }
    for row in rows {
        let observed = targets.remove(row.record_id.as_str()).unwrap_or_default();
        if rejected_records.contains(row.record_id.as_str()) {
            if !observed.is_empty() {
                return Err(integrity(
                    "record-level rejection conflicts with enumerated targets",
                ));
            }
            continue;
        }
        let messages: Vec<Message> = serde_json::from_str(&row.messages_json)?;
        let mut expected: BTreeSet<_> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == Role::Assistant)
            .map(|(i, _)| i as u64)
            .collect();
        if payload.manifest.recipe.multi_turn_loss == MultiTurnLoss::FinalTurnOnly {
            expected = expected.last().copied().into_iter().collect();
        }
        if expected.is_empty() || observed != expected {
            return Err(integrity(
                "prepared SFT target partition is missing, extra, or inconsistent with source",
            ));
        }
    }
    Ok(())
}

fn validate_source(
    source: &PreparedSftExampleSource,
    artifact: &ExportArtifact,
    row: &Projected,
) -> Result<()> {
    if source.record_hash != row.record_hash
        || source.prompt_hash != row.prompt_hash
        || source.messages_json != row.messages_json
        || source.task_json != row.task_json
        || source.origin_json != row.origin_json
    {
        return Err(integrity(
            "prepared SFT source row differs from actual captured Parquet",
        ));
    }
    let declared = row
        .task_json
        .as_ref()
        .map(|json| -> Result<_> {
            let task: ExportTaskProjection = serde_json::from_str(json)?;
            let p = task.provenance;
            Ok(PreparedSftDeclaredTask {
                identity: p.identity,
                group: p.group,
                split: p.split,
                rights: p.rights,
            })
        })
        .transpose()?;
    if source.declared_task != declared
        || declared
            .as_ref()
            .is_some_and(|task| task.split.role != TaskSplitRole::Train)
    {
        return Err(integrity(
            "prepared SFT task declarations differ or use a held-out split",
        ));
    }
    match (&artifact.screening, &source.screening) {
        (Some(w), Some(binding)) => {
            let member = w
                .members
                .iter()
                .find(|member| member.record.record_id == row.record_id)
                .ok_or_else(|| integrity("prepared SFT source is not a screened member"))?;
            let population = w
                .plan
                .population
                .iter()
                .find(|item| item.record == member.record)
                .ok_or_else(|| integrity("missing prepared SFT source population binding"))?;
            let ids = screening_identity(artifact)
                .ok_or_else(|| integrity("missing prepared SFT screening identity"))?;
            if source.group_kind != "screened_connected_component"
                || source.group_id != member.component_id
                || binding.record != member.record
                || binding.component_id != member.component_id
                || binding.export_projection_id != population.export_projection_id
                || binding.plan_id != ids.plan_id
                || binding.policy_id != ids.policy_id
                || binding.screening_input_id != ids.screening_input_id
                || binding.protected_input_id != ids.protected_input_id
                || binding.grouping_id != ids.grouping_id
                || binding.population_id != ids.population_id
            {
                return Err(integrity(
                    "prepared SFT source component or screening binding mismatch",
                ));
            }
        }
        (None, None) => {
            let origin = row
                .origin_json
                .as_ref()
                .map(|json| serde_json::from_str::<gw_schema::ExportRecordOrigin>(json))
                .transpose()?;
            let reference = match &origin {
                Some(gw_schema::ExportRecordOrigin::ReviewedReference(value)) => Some(value),
                _ => None,
            };
            if let Some(reference) = reference {
                use sha2::Digest;
                let bytes = serde_json::to_vec(&serde_json::json!([
                    "ghostwriter.reference-component.v1",
                    reference.catalogue_id,
                    reference.component
                ]))?;
                let expected = format!("{:x}", sha2::Sha256::digest(bytes));
                if source.group_id != expected {
                    return Err(integrity("prepared reference component identity differs"));
                }
            }
            let kind = if reference.is_some() {
                "reviewed_reference_component"
            } else if declared.is_some() {
                "declared_task_group"
            } else {
                "source_record"
            };
            if source.group_kind != kind {
                return Err(integrity("prepared SFT source grouping basis mismatch"));
            }
        }
        _ => return Err(integrity("prepared SFT source screening presence mismatch")),
    }
    Ok(())
}

#[cfg(test)]
#[path = "prepared_sft_tests.rs"]
mod tests;
