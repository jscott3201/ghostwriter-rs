//! Fresh-observation-only batch builder. Saved CodingArtifact JSON is deliberately not an input.
use super::{CapturedCodingInput, ObservedCodingRun, observe_coding};
use anyhow::ensure;
use gw_schema::{CodingTaskDocument, ValidatedReferenceMember};
use gw_storage::{
    ReferenceImportOutcome, ReferenceMemberObservation, RegisteredReferenceCatalogue, Store,
};
use tokio_util::sync::CancellationToken;

struct FreshBatch {
    members: Vec<ReferenceMemberObservation>,
}
impl FreshBatch {
    fn push(
        &mut self,
        member: &ValidatedReferenceMember,
        observed: ObservedCodingRun,
    ) -> anyhow::Result<()> {
        let artifact = observed.consume();
        ensure!(
            artifact.input.task == member.task && artifact.input.code == member.code,
            "fresh observation differs from captured registered member"
        );
        self.members
            .push(ReferenceMemberObservation::from_native_observation(
                member.member_id.clone(),
                artifact.artifact_id.clone(),
                artifact.report.native_verification.clone(),
                serde_json::to_value(&artifact)?,
            )?);
        Ok(())
    }
}
/// Run every registered member freshly, then commit the entire batch or preserve its earlier facts.
/// This accepts only application-owned registration; saved reports never enter the private builder.
///
/// # Errors
/// Rejects cancellation, any non-Pass native member, changed bindings or persistence failure.
pub(crate) async fn import(
    store: &Store,
    registered: &RegisteredReferenceCatalogue,
    fresh_validation: bool,
    cancel: CancellationToken,
) -> anyhow::Result<(ReferenceImportOutcome, bool)> {
    if !fresh_validation
        && let Some(existing) = store.committed_reference_import(registered).await?
    {
        return Ok((existing, false));
    }
    let mut batch = FreshBatch {
        members: Vec::with_capacity(112),
    };
    for member in registered.population().members() {
        ensure!(
            !cancel.is_cancelled(),
            "reference import cancelled before complete batch"
        );
        let document = CodingTaskDocument {
            version: 1,
            tasks: vec![member.task.clone()],
        };
        let input = CapturedCodingInput::new(&document, &member.task.task_id, member.code.clone())?;
        let observed = observe_coding(input, cancel.clone()).await?;
        batch.push(member, observed)?;
    }
    ensure!(
        !cancel.is_cancelled(),
        "reference import cancelled before commit"
    );
    Ok((
        store
            .commit_reference_import(registered, &batch.members, cancel.cancelled())
            .await?,
        true,
    ))
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
