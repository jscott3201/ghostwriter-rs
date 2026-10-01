//! Pure preference identity and material evidence capture; no store or runtime is required.
use gw_schema::{PreferenceSourceSnapshot, TrainingRecord};

use crate::{Result, StorageError, canonical_json_hash, completion_hash, prompt_hash, record_hash};

fn invalid(reason: &str) -> StorageError {
    StorageError::Export(reason.into())
}

fn finite(value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid(
            "preference material evidence contains a nonfinite number",
        ))
    }
}

/// Validate every floating-point value retained in a source snapshot before JSON canonicalization.
/// Raw rubric votes may use scales such as 1..10; only the aggregate has a `[0,1]` constraint.
fn validate_material_numbers(record: &TrainingRecord) -> Result<()> {
    let judging = &record.judging;
    for value in [
        judging.aggregate,
        judging.agreement,
        judging.n_eff,
        judging.threshold_at_decision,
    ]
    .into_iter()
    .flatten()
    {
        finite(value)?;
    }
    if judging
        .aggregate
        .is_some_and(|score| !(0.0..=1.0).contains(&score))
    {
        return Err(invalid("preference aggregate must be in [0,1]"));
    }
    for vote in &judging.panel {
        finite(vote.score)?;
        for value in [vote.temperature, vote.top_p].into_iter().flatten() {
            finite(value)?;
        }
        if let Some(dimensions) = &vote.dimensions {
            for value in dimensions.values() {
                finite(*value)?;
            }
        }
    }
    for check in &record.verification.checks {
        if let Some(score) = check.score {
            finite(score)?;
        }
    }
    if let Some(numeric) = record
        .verification_contract
        .as_ref()
        .and_then(|contract| contract.numeric.as_ref())
    {
        numeric.validate().map_err(invalid)?;
    }
    if let Some(quality) = &record.reasoning_quality {
        for value in [quality.reasoning_score, quality.fsf].into_iter().flatten() {
            finite(value)?;
        }
        for step in &quality.steps {
            finite(step.score)?;
        }
    }
    Ok(())
}

fn checked_hash(stored: &str, fresh: String) -> Result<String> {
    if !stored.is_empty() && stored != fresh {
        return Err(invalid(
            "preference source contains a populated stale content hash",
        ));
    }
    Ok(fresh)
}

/// Capture exact material evidence, recomputing missing hashes and rejecting populated stale ones.
/// This does not establish eligibility, prefix equality or a preference; preparation does that.
///
/// # Errors
/// Rejects blank source identities, nonfinite material numbers, out-of-range aggregates, stale
/// hashes, or serialization errors. Performs no I/O.
pub fn capture_preference_source(record: &TrainingRecord) -> Result<PreferenceSourceSnapshot> {
    for identity in [
        &record.record_id,
        &record.provenance.run_id,
        &record.training_area,
    ] {
        if identity.trim().is_empty() {
            return Err(invalid("preference source identities must be nonempty"));
        }
    }
    validate_material_numbers(record)?;
    Ok(PreferenceSourceSnapshot {
        record_id: record.record_id.clone(),
        training_area: record.training_area.clone(),
        provenance: record.provenance.clone(),
        record_hash: checked_hash(&record.hashes.record_hash, record_hash(record)?)?,
        prompt_hash: checked_hash(&record.hashes.prompt_hash, prompt_hash(&record.messages)?)?,
        completion_hash: checked_hash(
            &record.hashes.completion_hash,
            completion_hash(&record.messages)?,
        )?,
        original_messages: record.messages.clone(),
        judging: record.judging.clone(),
        verification: record.verification.clone(),
        verification_contract: record.verification_contract.clone(),
        task_provenance: record.task_provenance.clone(),
        execution_evidence: record.execution_evidence.clone(),
        reasoning_quality: record.reasoning_quality.clone(),
        lifecycle: record.lifecycle.clone(),
    })
}

/// Hash finite, validated preference data in a separate, explicitly versioned identity domain.
/// Callers must validate floating-point inputs before serialization (JSON turns NaN into null).
///
/// # Errors
/// Returns serialization errors without I/O.
pub fn preference_hash(domain: &str, value: &impl serde::Serialize) -> Result<String> {
    let value = serde_json::to_value(value)?;
    canonical_json_hash(&serde_json::json!({"domain": domain, "value": value}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_hash_reports_serialization_failure_without_panicking() {
        let unsupported = std::collections::BTreeMap::from([((1, 2), 3)]);
        assert!(preference_hash("test", &unsupported).is_err());
    }
}
