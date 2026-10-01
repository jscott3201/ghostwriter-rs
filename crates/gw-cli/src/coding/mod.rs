//! Offline local coding evaluation. Only fresh opaque runtime observations enter its native
//! consumer. Saved artifacts are declarations and explicit replay executes their captured code.
mod artifact;
mod consume;
mod container;
#[cfg(test)]
mod containment_tests;
#[cfg(all(test, unix))]
mod lifecycle_tests;
mod process;
mod runtime;
#[cfg(test)]
mod tests;
pub use artifact::{
    CapturedCodingInput, CodingArtifact, CodingCaseObservation, CodingCaseReason, CodingReport,
};
pub use runtime::CodingRuntimeIdentity;

use artifact::{digest, outcome};
use container::{Invocation, Stop, invoke};
use gw_schema::{CodingValue, TestStatus};
use runtime::{Docker, PROBE, WRAPPER, nonce};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Opaque non-deserializable runtime result. Private construction prevents caller-supplied reports
/// from reaching the new positive consumer. Dropping the evaluation future cancels its supervisor.
pub struct ObservedCodingRun {
    input: CapturedCodingInput,
    run_id: String,
    cases: Vec<CodingCaseObservation>,
}
impl ObservedCodingRun {
    /// Consume actual local observations through the native deterministic verifier.
    /// Quality-panel grading and model/student lineage are not asserted by this offline command.
    #[must_use]
    pub fn consume(self) -> CodingArtifact {
        let native_verification = consume::interpret(&self.input, &self.run_id, &self.cases);
        let report = CodingReport {
            run_id: self.run_id,
            outcome: outcome(&self.cases),
            cases: self.cases,
            native_verification,
        };
        let mut artifact = CodingArtifact {
            version: 1,
            artifact_id: String::new(),
            input: self.input,
            runtime: CodingRuntimeIdentity::expected(),
            report,
            replayed_declaration_id: None,
        };
        artifact.seal();
        artifact
    }
}

/// Evaluate captured code using the cached pinned local Docker recipe; never downloads or calls
/// providers. Cancellation waits for dispatched mutations and whole-container cleanup. If the
/// caller drops this future, the supervisor continues cleanup while its Tokio runtime remains alive.
///
/// # Errors
/// Rejects invalid input or unsupported runtime. Daemon/candidate execution uncertainty is recorded
/// as Unknown; a failed cleanup acknowledgment never becomes a positive native observation.
pub async fn observe_coding(
    input: CapturedCodingInput,
    cancel: CancellationToken,
) -> anyhow::Result<ObservedCodingRun> {
    observe(input, cancel, None).await
}
async fn observe(
    input: CapturedCodingInput,
    cancel: CancellationToken,
    docker: Option<Docker>,
) -> anyhow::Result<ObservedCodingRun> {
    input.validate()?;
    let local = cancel.child_token();
    let guard = local.clone().drop_guard();
    let task = tokio::spawn(supervise(input, local, docker));
    let result = task.await?;
    guard.disarm();
    result
}

/// Validate a saved declaration, re-execute its captured input, and compare complete stable
/// outcomes. A matching result establishes the fresh run only; it does not authenticate history.
///
/// # Errors
/// Rejects stale/inconsistent declarations before Docker, unsupported runtime, or fresh mismatch.
pub async fn replay_coding(
    saved: CodingArtifact,
    cancel: CancellationToken,
) -> anyhow::Result<CodingArtifact> {
    saved.validate()?;
    let mut fresh = observe_coding(saved.input.clone(), cancel).await?.consume();
    anyhow::ensure!(
        saved.stable_matches(&fresh),
        "fresh coding observation differs from saved declaration; saved success was not consumed"
    );
    fresh.replayed_declaration_id = Some(saved.artifact_id);
    fresh.seal();
    Ok(fresh)
}

async fn supervise(
    input: CapturedCodingInput,
    cancel: CancellationToken,
    test_docker: Option<Docker>,
) -> anyhow::Result<ObservedCodingRun> {
    let budget = cancel.clone();
    let timer = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(180)).await;
        budget.cancel();
    });
    let result = run(input, &cancel, test_docker).await;
    timer.abort();
    let _ = timer.await;
    result
}
async fn run(
    input: CapturedCodingInput,
    cancel: &CancellationToken,
    test_docker: Option<Docker>,
) -> anyhow::Result<ObservedCodingRun> {
    let docker = match test_docker {
        Some(docker) => docker,
        None => Docker::connect().await?,
    };
    let probe = invoke(&docker, PROBE, Vec::new(), cancel, Duration::from_secs(3)).await?;
    let mut runnable = probe.settled && probe.stop.is_none() && probe.output.as_ref().is_some_and(|out|
        out.status.success() && !out.exceeded && out.stderr.is_empty()
        && serde_json::from_slice::<serde_json::Value>(&out.stdout).ok() == Some(serde_json::json!({
            "implementation":"cpython","version":[3,12,14],"machine":"aarch64","uid":65534,"gid":65534})));
    let mut cases = Vec::new();
    for (case, case_id) in input.task.cases().into_iter().zip(&input.suite.case_ids) {
        if !runnable || cancel.is_cancelled() {
            let mut observation = CodingCaseObservation::not_run(case_id.clone());
            if cases.is_empty() && !probe.settled {
                observation.settled = false;
                observation.reason = CodingCaseReason::Infrastructure;
            }
            cases.push(observation);
            continue;
        }
        // Only this invocation's arguments enter the container. Expected values, the rest of the
        // private suite, provenance, labels, case identities, and saved verdicts remain outside.
        let payload = invocation_input(&input, case)?;
        anyhow::ensure!(
            payload.len() <= 1024 * 1024,
            "coding invocation exceeds its input bound"
        );
        let invocation = invoke(&docker, WRAPPER, payload, cancel, Duration::from_secs(3)).await?;
        let observation = classify(case_id.clone(), &case.expected, invocation);
        runnable = observation.settled && observation.status != TestStatus::Skipped;
        cases.push(observation);
    }
    Ok(ObservedCodingRun {
        input,
        run_id: nonce()?,
        cases,
    })
}
fn invocation_input(
    input: &CapturedCodingInput,
    case: &gw_schema::CodingCase,
) -> anyhow::Result<Vec<u8>> {
    Ok(serde_json::to_vec(
        &serde_json::json!({"code":input.code,"function":input.task.function,"arguments":case.arguments}),
    )?)
}
fn classify(
    case_id: String,
    expected: &CodingValue,
    invocation: Invocation,
) -> CodingCaseObservation {
    use CodingCaseReason::*;
    let mut observation = CodingCaseObservation {
        case_id,
        status: TestStatus::Skipped,
        reason: Infrastructure,
        result_id: None,
        exit_code: None,
        settled: invocation.settled,
        container_id: invocation.container_id,
        elapsed_ms: invocation.elapsed_ms,
    };
    let Some(output) = invocation.output else {
        return observation;
    };
    observation.reason = match invocation.stop {
        Some(Stop::Cancelled) => Cancelled,
        Some(Stop::Infrastructure) => Infrastructure,
        Some(Stop::WallLimit) => WallLimit,
        Some(Stop::OutputLimit) => OutputLimit,
        None => {
            observation.exit_code = output.status.code();
            if output.exceeded {
                OutputLimit
            } else if observation.exit_code.is_none() || observation.exit_code == Some(125) {
                Infrastructure
            } else if !output.status.success() {
                CandidateExit
            } else if !output.stderr.is_empty() {
                InvalidOutput
            } else if let Ok(value) = CodingValue::from_json(&output.stdout) {
                observation.result_id = Some(digest("ghostwriter.coding-result.v1", &value));
                if value == *expected {
                    Matched
                } else {
                    WrongResult
                }
            } else {
                InvalidOutput
            }
        }
    };
    observation.status = match observation.reason {
        Matched => TestStatus::Passed,
        Cancelled | Infrastructure | NotRun => TestStatus::Skipped,
        _ => TestStatus::Failed,
    };
    observation
}
