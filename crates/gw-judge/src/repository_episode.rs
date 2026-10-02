//! Consistency checking of supplied repository episodes; no execution or admission authority.
use crate::verifier::evidence::observe;
use gw_schema::{
    EvidenceBinding, REPOSITORY_EPISODE_MAX_BYTES, RepositoryEpisodeArtifact,
    RepositoryEpisodeAssessment, RepositoryEpisodeError, RepositoryEpisodeReceipt,
    RepositoryEpisodeRequest, RepositoryReportBinding, VerificationOutcome,
};

fn assessment(
    request: &RepositoryEpisodeRequest,
    task: Option<&str>,
    candidate: Option<&str>,
) -> RepositoryEpisodeAssessment {
    let mut result = RepositoryEpisodeAssessment {
        report_binding: RepositoryReportBinding::Absent,
        declared_outcome: VerificationOutcome::Unknown,
        observed_execution: VerificationOutcome::Unknown,
        training_eligible: false,
    };
    let Some(report) = request.report.as_ref().and_then(|r| r.execution.as_ref()) else {
        return result;
    };
    let (Some(task), Some(candidate)) = (task, candidate) else {
        result.report_binding = RepositoryReportBinding::Pending;
        return result;
    };
    if report.binding.task != task || report.binding.attempt != request.candidate.attempt {
        result.report_binding = RepositoryReportBinding::Foreign;
        return result;
    }
    if report.binding.patch_hash != candidate {
        result.report_binding = RepositoryReportBinding::Stale;
        return result;
    }
    result.report_binding = RepositoryReportBinding::Matched;
    let required = request
        .task
        .private_contract
        .as_ref()
        .map_or(&[][..], |c| c.required_test_ids.as_slice());
    result.declared_outcome = observe(
        Some(report),
        &EvidenceBinding {
            task: task.into(),
            attempt: request.candidate.attempt.clone(),
            patch_hash: candidate.into(),
        },
        required,
    )
    .outcome;
    result
}

/// Capture one complete portable repository artifact from strict caller-supplied JSON.
/// No providers, environment loaders, executors, database, or referenced resources are accessed.
/// The complete canonical artifact must fit the wire bound with one terminating newline reserved.
///
/// # Errors
/// Rejects invalid, oversized, duplicate or unknown protected fields and invalid internal bindings.
pub fn capture_repository_episode(
    bytes: &[u8],
) -> Result<RepositoryEpisodeArtifact, RepositoryEpisodeError> {
    let request = RepositoryEpisodeRequest::from_json(bytes)?;
    let identities = request.identities()?;
    let assessment = assessment(
        &request,
        identities.task.as_deref(),
        identities.candidate.as_deref(),
    );
    let artifact = RepositoryEpisodeArtifact {
        version: 1,
        request,
        identities,
        assessment,
    };
    let length = serde_json::to_vec(&artifact)
        .map_err(|_| RepositoryEpisodeError::new("repository serialization failed"))?
        .len();
    if length >= REPOSITORY_EPISODE_MAX_BYTES {
        return Err(RepositoryEpisodeError::new(
            "repository output exceeds 32 MiB bound",
        ));
    }
    Ok(artifact)
}

/// Independently recompute all saved content bindings and declaration diagnostics.
/// A passing declaration still yields observed Unknown and training eligibility false.
///
/// # Errors
/// Rejects malformed or tampered artifacts, noncanonical delta order, identities and diagnostics.
pub fn verify_repository_episode(
    bytes: &[u8],
) -> Result<RepositoryEpisodeReceipt, RepositoryEpisodeError> {
    let artifact = RepositoryEpisodeArtifact::from_json(bytes)?;
    let canonical = artifact.request.canonicalized()?;
    let encode = |value: &RepositoryEpisodeRequest| {
        serde_json::to_vec(value)
            .map_err(|_| RepositoryEpisodeError::new("repository serialization failed"))
    };
    if encode(&canonical)? != encode(&artifact.request)? {
        return Err(RepositoryEpisodeError::new(
            "repository artifact request is not canonical",
        ));
    }
    let identities = canonical.identities()?;
    if artifact.identities != identities {
        return Err(RepositoryEpisodeError::new(
            "repository artifact identity mismatch",
        ));
    }
    let assessment = assessment(
        &canonical,
        identities.task.as_deref(),
        identities.candidate.as_deref(),
    );
    if artifact.assessment != assessment {
        return Err(RepositoryEpisodeError::new(
            "repository declaration assessment mismatch",
        ));
    }
    Ok(RepositoryEpisodeReceipt {
        version: 1,
        identities,
        assessment,
    })
}
