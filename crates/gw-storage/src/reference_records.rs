//! Redacted reference envelopes and local publication eligibility.
use crate::{Result, artifact::integrity, record_data as data, reference_registration::id};
use gw_schema::*;
use sqlx::{Sqlite, Transaction};

/// Material supplied by the trusted runtime adapter after consuming an opaque fresh observation.
/// This type intentionally has no deserializer; saved reports must never be used by an import UI.
#[derive(Debug)]
pub struct ReferenceMemberObservation {
    pub(crate) member_id: String,
    pub(crate) native_result_id: String,
    pub(crate) verification: VerificationInterpretation,
    pub(crate) evidence: serde_json::Value,
}
impl ReferenceMemberObservation {
    /// Bind native material for a trusted in-process execution adapter. This constructor does not
    /// authenticate serialized reports; the application must retain its opaque runtime capability.
    ///
    /// # Errors
    /// Rejects any missing/non-Pass authoritative execution fact or inconsistent material.
    pub fn from_native_observation(
        member_id: String,
        native_result_id: String,
        verification: VerificationInterpretation,
        private_evidence: serde_json::Value,
    ) -> Result<Self> {
        native_pass(&verification)?;
        if private_evidence["artifact_id"].as_str() != Some(&native_result_id)
            || private_evidence["report"]["native_verification"]
                != serde_json::to_value(&verification)?
        {
            return Err(integrity(
                "reference native result does not bind its private evidence",
            ));
        }
        Ok(Self {
            member_id,
            native_result_id,
            verification,
            evidence: private_evidence,
        })
    }
}
pub(crate) fn native_pass(value: &VerificationInterpretation) -> Result<()> {
    if value.gate().map_err(integrity)? != (true, None)
        || value.answer.policy != VerificationPolicy::Absent
        || value.reasoning.policy != VerificationPolicy::Absent
        || value.execution.policy != VerificationPolicy::Authoritative
        || value.execution.observation.as_ref().map(|v| v.outcome)
            != Some(VerificationOutcome::Pass)
    {
        return Err(integrity(
            "reference requires a fresh authoritative native execution Pass",
        ));
    }
    Ok(())
}
pub(crate) fn origin(
    registered: &crate::RegisteredReferenceCatalogue,
    member: &ValidatedReferenceMember,
    observed: &ReferenceMemberObservation,
) -> Result<ReviewedReferenceOrigin> {
    if observed.member_id != member.member_id
        || observed.evidence["input"]["task"] != serde_json::to_value(&member.task)?
        || observed.evidence["input"]["code"].as_str() != Some(&member.code)
        || observed.evidence["input"]["code_id"].as_str() != Some(&member.reference_code_id)
        || observed.evidence["input"]["suite"] != serde_json::to_value(member.task.suite_binding())?
    {
        return Err(integrity(
            "reference observed input differs from registered member",
        ));
    }
    native_pass(&observed.verification)?;
    let value = ReviewedReferenceOrigin {
        version: 1,
        catalogue_id: registered.population().catalogue_id().into(),
        registration_id: registered.registration_id().into(),
        batch_id: registered.batch_id().into(),
        member_id: member.member_id.clone(),
        reference_code_id: member.reference_code_id.clone(),
        suite_id: member.task.suite_binding().suite_id,
        native_result_id: observed.native_result_id.clone(),
        authorship: member.authorship.clone(),
        component: member.component.clone(),
        permitted_use: if member.task.split.role == TaskSplitRole::Train {
            TaskPermittedUse::Training
        } else {
            TaskPermittedUse::Evaluation
        },
    };
    value.validate().map_err(integrity)?;
    Ok(value)
}
pub(crate) fn make_record(
    registered: &crate::RegisteredReferenceCatalogue,
    member: &ValidatedReferenceMember,
    observed: &ReferenceMemberObservation,
    at: &str,
) -> Result<TrainingRecord> {
    let origin = origin(registered, member, observed)?;
    if member.task.split.role != TaskSplitRole::Train {
        return Err(integrity(
            "held-out reference cannot become a TrainingRecord",
        ));
    }
    let mut record = TrainingRecord {
        record_id: id(
            "ghostwriter.reference-record.v1",
            &(&origin.batch_id, &origin.member_id),
        )?,
        schema_version: "1.0.0".parse().expect("static schema version"),
        dataset_version: None,
        training_area: registered.population().catalogue().training_area.clone(),
        tags: vec![],
        messages: vec![
            member.task.prompt(),
            Message {
                role: Role::Assistant,
                content: Content::Text(member.code.clone()),
                reasoning: None,
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ],
        tools: None,
        origin: RecordOrigin::ReviewedReference(Box::new(origin)),
        task_provenance: Some(TaskProvenance::from_coding_task(&member.task).map_err(integrity)?),
        verification_contract: Some(member.task.contract()),
        execution_evidence: None,
        verification: Verification {
            checks: vec![],
            all_passed: true,
            interpretation: Some(observed.verification.clone()),
            needs_review: None,
        },
        judging: Judging::default(),
        reasoning_quality: None,
        lifecycle: Lifecycle::default(),
        hashes: Hashes::default(),
        cost: Cost::default(),
    };
    crate::record_mutations::append(&mut record, LifecycleState::Verified, None, at)?;
    crate::record_mutations::append(&mut record, LifecycleState::Admitted, None, at)?;
    data::normalize(&record)
}
/// Pure reference-envelope consistency. It cannot establish application-owned registration.
pub(crate) fn validate_record(record: &TrainingRecord) -> Result<()> {
    let RecordOrigin::ReviewedReference(origin) = &record.origin else {
        return Ok(());
    };
    origin.validate().map_err(integrity)?;
    let task = record
        .task_provenance
        .as_ref()
        .ok_or_else(|| integrity("reference lacks task provenance"))?;
    let contract = record
        .verification_contract
        .as_ref()
        .ok_or_else(|| integrity("reference lacks verification contract"))?;
    let Oracle::CodingSuite { suite } = &contract.oracle else {
        return Err(integrity("reference lacks coding suite"));
    };
    let [prompt, answer] = record.messages.as_slice() else {
        return Err(integrity("reference requires exact prompt and module"));
    };
    let Content::Text(code) = &answer.content else {
        return Err(integrity("reference module must be text"));
    };
    task.validate_for(prompt, contract).map_err(integrity)?;
    let verification = record
        .verification
        .interpretation
        .as_ref()
        .ok_or_else(|| integrity("reference lacks native interpretation"))?;
    native_pass(verification)?;
    if record.record_id
        != id(
            "ghostwriter.reference-record.v1",
            &(&origin.batch_id, &origin.member_id),
        )?
        || task.split.role != TaskSplitRole::Train
        || origin.permitted_use != TaskPermittedUse::Training
        || !task
            .rights
            .permitted_uses
            .contains(&TaskPermittedUse::Training)
        || suite.suite_id != origin.suite_id
        || coding_digest("ghostwriter.coding-module.v1", code.as_bytes())
            != origin.reference_code_id
        || answer.role != Role::Assistant
        || answer.reasoning.is_some()
        || answer.reasoning_details.is_some()
        || answer.tool_calls.is_some()
        || answer.tool_call_id.is_some()
        || answer.name.is_some()
        || record.judging != Judging::default()
        || record.cost != Cost::default()
        || record.tools.is_some()
        || record.reasoning_quality.is_some()
        || record.execution_evidence.is_some()
        || !record.verification.all_passed
        || record.verification.needs_review.is_some()
    {
        return Err(integrity(
            "reference envelope contradicts its origin, native facts or training use",
        ));
    }
    Ok(())
}
pub(crate) async fn eligible(
    tx: &mut Transaction<'_, Sqlite>,
    record: &TrainingRecord,
) -> Result<()> {
    let RecordOrigin::ReviewedReference(origin) = &record.origin else {
        return Ok(());
    };
    validate_record(record)?;
    let stored: Option<String> = sqlx::query_scalar("SELECT m.origin_json FROM reference_members m JOIN reference_batches b ON b.batch_id=m.batch_id JOIN reference_registrations r ON r.registration_id=b.registration_id JOIN runs x ON x.run_id=b.batch_id WHERE m.record_id=? AND m.member_id=? AND m.batch_id=? AND m.split='train' AND b.member_count=112 AND r.registration_id=? AND r.catalogue_id=? AND x.run_kind='reviewed_reference'")
        .bind(&record.record_id).bind(&origin.member_id).bind(&origin.batch_id).bind(&origin.registration_id).bind(&origin.catalogue_id)
        .fetch_optional(&mut **tx).await?;
    if stored.as_deref() != Some(serde_json::to_string(origin)?.as_str()) {
        return Err(integrity(
            "reference is not an eligible committed registered Train member",
        ));
    }
    Ok(())
}
