//! The per-record step machine — the heart of the engine (ARCHITECTURE §5).
//!
//! `step` drives ONE record through ONE lifecycle transition, given its inputs + the injected
//! [`Clients`]. It is PURE given those inputs (the only side effects are through the injected
//! clients), so it is replayable and HERMETICALLY testable. Crucially, `step` reads the record's
//! CURRENT `lifecycle.state` and advances from there — so a record persisted mid-flight (e.g. at
//! `AssistantGenerated`) re-enters at exactly that state on relaunch, NOT from `Seeded` (crash-resume,
//! INVARIANT: idempotent-crash-resume).
//!
//! ## Persist-after-every-transition (INVARIANT 4)
//!
//! Every transition is committed to the [`Store`](gw_storage::Store) BEFORE `step` returns — the
//! expensive teacher CoT is written at `AssistantGenerated`, never held only in memory. `step`
//! returns the advanced record (re-read from the store so the returned envelope matches what was
//! persisted), and the executor loops `step` until the record reaches a terminal state.
//!
//! ## Never-re-spend (INVARIANT 3)
//!
//! The teacher call at the `UserSynthesized → AssistantGenerated` edge is content-hash cached: if a
//! record already at `AssistantGenerated` (or beyond) is re-stepped, `step` never re-enters
//! generation. And within a single attempt, the judge calls go through `gw-judge`'s
//! `grade_panel_cached`, so a re-judge of the same content hits the cache. The teacher generation
//! itself is idempotent-by-state: once the assistant turn is persisted, no path re-generates it.
//!
//! ## What each edge does
//!
//! - `Seeded → UserSynthesized` — gate the candidate (already gated by the producer in v1; the
//!   record arrives at `AssistantGenerated` from `gw-generate::assemble`, so the engine seeds records
//!   directly into the pipeline at `AssistantGenerated` — see `crate::sibling`). This module
//!   handles the post-generation edges; the generation edge is owned by `crate::sibling` (best-of-k)
//!   so the fan-out and the single-trace path share one producer call site.
//! - `AssistantGenerated → Verified` — run the deterministic verifier rail (pure + the injected
//!   sandbox oracle).
//! - `Verified → Judged` — grade the panel (cached) and compute consensus with the NON-IDENTITY
//!   correlation prior, persisting the `Judging` block.
//! - `Judged → {Admitted | Rejected | Revising | NeedsReview}` — reconcile the `gw-judge::Decision`
//!   to the lifecycle (the verdict→lifecycle mapping).
//! - `Admitted → Formatted → Exported` — render + mark exported. `NeedsReview` / `Rejected` /
//!   `Revising` / `Error` are terminal for the automated pipeline (Revising re-enters generation via
//!   `crate::revise`; NeedsReview LEAVES the pipeline — not exported, not counted admitted).

use gw_judge::{
    CorrelationMatrix, GradeOutcome, HybridGrader, NullSandboxOracle, VerifierInput,
    grade_panel_cached, run_verifier,
};
use gw_schema::{CotPolicy, EvidenceBinding, LifecycleState, TrainingRecord, TrlFormat};
use gw_storage::record_hash;
use tokio_util::sync::CancellationToken;

use crate::clients::{AreaConfig, Clients};
use crate::error::{EngineError, Result};
use crate::event::EngineEvent;

/// Advance one record by exactly ONE lifecycle transition and persist the result.
///
/// Reads `rec.lifecycle.state`, performs the matching transition through the injected [`Clients`],
/// persists it (so the expensive result is never memory-only), emits a [`EngineEvent::StateAdvanced`],
/// and returns the re-read advanced record. A record already at a TERMINAL state
/// (`is_terminal`) is returned unchanged (idempotent — re-stepping a finished record is a no-op).
///
/// The `Revising` and `Judged → Revising` handoff is engine-owned and NOT done here: this function
/// reports a record at `Revising` as terminal-for-`step` so the executor (via `crate::revise`)
/// re-enters generation. Likewise the `Seeded → AssistantGenerated` generation edge is owned by
/// `crate::sibling`.
///
/// # Errors
/// - [`EngineError::Judge`] / [`EngineError::Generate`] / [`EngineError::Storage`] from the underlying
///   rail calls.
/// - [`EngineError::Invariant`] if a `k > 1` panel would be graded with an identity correlation matrix
///   (fail loud), or the record is at a state `step` cannot advance from.
pub async fn step(
    rec: TrainingRecord,
    clients: &Clients,
    area: &AreaConfig,
) -> Result<TrainingRecord> {
    match rec.lifecycle.state {
        LifecycleState::AssistantGenerated => verify(rec, clients, area).await,
        LifecycleState::Verified => judge(rec, clients, area).await,
        LifecycleState::Judged => reconcile(rec, clients, area).await,
        LifecycleState::Admitted => format_record(rec, clients).await,
        LifecycleState::Formatted => export_record(rec, clients).await,
        // Terminal-for-step states: the executor handles Revising (re-enter generation) and the
        // genuinely-terminal states are returned unchanged.
        LifecycleState::Seeded
        | LifecycleState::UserSynthesized
        | LifecycleState::Revising
        | LifecycleState::NeedsReview
        | LifecycleState::Rejected
        | LifecycleState::Exported
        | LifecycleState::Error => Ok(rec),
    }
}

/// `true` when a record needs no further automated `step` driving: a genuinely terminal state
/// (`Exported` / `Rejected` / `NeedsReview` / `Error`) OR a handoff state the executor owns
/// (`Revising`, re-entered via `crate::revise`). The driver loop stops calling `step` here.
#[must_use]
pub fn is_terminal(state: LifecycleState) -> bool {
    matches!(
        state,
        LifecycleState::Exported
            | LifecycleState::Rejected
            | LifecycleState::NeedsReview
            | LifecycleState::Error
            | LifecycleState::Revising
    )
}

/// Drive `rec` through `step` repeatedly until it reaches a state where `is_terminal` holds,
/// persisting after every transition. Returns the record at its terminal state, or at its last
/// persisted non-terminal boundary if `cancel` is set before the next transition starts. This is the
/// per-record driver the executor calls; the bounded revise re-entry is layered on top by
/// `crate::revise`.
///
/// # Errors
/// Propagates the first [`EngineError`] from any transition. On error the record's LAST persisted
/// state stands (so a relaunch resumes from there) — the error is surfaced, never swallowed.
pub async fn drive(
    mut rec: TrainingRecord,
    clients: &Clients,
    area: &AreaConfig,
    cancel: &CancellationToken,
) -> Result<TrainingRecord> {
    // A hard cap on transitions defends against a logic bug that fails to advance (it would otherwise
    // spin). The pipeline is at most ~6 forward edges from AssistantGenerated to Exported.
    for _ in 0..16 {
        if is_terminal(rec.lifecycle.state) {
            return Ok(rec);
        }
        if cancel.is_cancelled() {
            return Ok(rec);
        }
        let before = rec.lifecycle.state;
        rec = step(rec, clients, area).await?;
        // A non-advancing step on a non-terminal state is an engine bug — fail loud rather than spin.
        if rec.lifecycle.state == before && !is_terminal(before) {
            return Err(EngineError::Invariant(format!(
                "step did not advance record {} from {:?}",
                rec.record_id, before
            )));
        }
    }
    Err(EngineError::Invariant(format!(
        "record {} exceeded the transition cap without reaching a terminal state (state {:?})",
        rec.record_id, rec.lifecycle.state
    )))
}

/// Drive `rec` forward up to AND STOPPING AT `Judged` — it computes the verification + judging blocks
/// (and persists them) but does NOT reconcile the verdict to a lifecycle decision. Used by best-of-k:
/// every sibling is graded to `Judged` first, then the GROUP picks the single best to admit
/// (`crate::sibling`), so a non-best sibling is never independently admitted (INVARIANT 11).
///
/// A record already at/after `Judged` (or terminal) is returned unchanged.
///
/// # Errors
/// Propagates the first [`EngineError`] from the verify / judge transitions.
pub async fn drive_to_judged(
    mut rec: TrainingRecord,
    clients: &Clients,
    area: &AreaConfig,
    cancel: &CancellationToken,
) -> Result<TrainingRecord> {
    for _ in 0..8 {
        if cancel.is_cancelled() {
            return Ok(rec);
        }
        match rec.lifecycle.state {
            // Only the pre-judgment edges advance here; stop the moment Judged (or beyond) is reached.
            LifecycleState::AssistantGenerated | LifecycleState::Verified => {
                let before = rec.lifecycle.state;
                rec = step(rec, clients, area).await?;
                if rec.lifecycle.state == before {
                    return Err(EngineError::Invariant(format!(
                        "drive_to_judged did not advance record {} from {:?}",
                        rec.record_id, before
                    )));
                }
            }
            _ => return Ok(rec),
        }
    }
    Err(EngineError::Invariant(format!(
        "record {} did not reach Judged within the transition cap (state {:?})",
        rec.record_id, rec.lifecycle.state
    )))
}

/// The task/attempt/patch key an execution-evidence report must be bound to in order to be
/// applicable to `rec`.
///
/// `task` is the run, `attempt` is the record (a bounded-revise retry is a DISTINCT record id, so
/// a retry never inherits its parent's reports), and `patch_hash` is a content hash of the produced
/// candidate. Deriving all three from the envelope is what makes a stale or cross-attempt report
/// detectable without trusting the report's own claim about itself.
///
/// # Errors
/// Propagates a [`gw_storage`] hashing error (a content field that cannot be canonicalized).
pub fn evidence_key(rec: &TrainingRecord) -> Result<EvidenceBinding> {
    Ok(EvidenceBinding {
        task: rec.provenance.run_id.clone(),
        attempt: rec.record_id.clone(),
        patch_hash: gw_storage::completion_hash(&rec.messages)?,
    })
}

/// Resolve the precomputed execution report for `rec`, preferring the keyed source and falling back
/// to whatever the envelope already carries.
///
/// The source is a lookup keyed by the very patch hash it must match, so a report it returns cannot
/// be stale. The envelope fallback covers a crash-resume of the `verify` edge (the report was
/// already resolved and persisted) and a source that is no longer available; a carried report whose
/// binding no longer matches is classified `Unknown` by the verifier, never followed.
fn resolve_execution_evidence(
    rec: &TrainingRecord,
    clients: &Clients,
    key: &EvidenceBinding,
) -> Option<gw_schema::ExecutionEvidence> {
    clients
        .execution_evidence
        .evidence(key)
        .or_else(|| rec.execution_evidence.clone())
}

/// `AssistantGenerated → Verified`: run the deterministic verifier rail (pure + injected sandbox +
/// any injected execution-evidence source).
///
/// The per-record `VerificationContract` (kind + oracle) is threaded in from the record envelope
/// (`rec.verification_contract`, carried from the user-turn candidate by `gw-generate::assemble`), so
/// BOTH rails run: the always-authoritative reasoning-present hard gate AND the contract's
/// answer-correctness check (NumericMatch / SetMatch / SqlResultMatch / SchemaShape / RefusalExpected).
/// A wrong numeric answer or a complied-with adversarial prompt is therefore caught HERE on the
/// deterministic rail — on the happy path AND on a crash-resume of this edge (the contract is persisted
/// on the envelope, so the answer rail is not a function of in-memory state). A record with no contract
/// (`None`) or an `Oracle::None` contract is judge-only for the answer axis; the reasoning gate still
/// applies. `rule_only_authoritative` (off by default) governs whether a rule non-match hard-rejects
/// or routes to judge rescue (`rescue_negatives`).
///
/// The PRECOMPUTED execution axis runs independently of the contract: a candidate whose correctness
/// is only knowable by running it carries an external evaluator's report on the envelope, and the
/// report is resolved (and then persisted) here so the verdict survives the crash-resume of this
/// edge. The harness never executes anything itself.
async fn verify(
    rec: TrainingRecord,
    clients: &Clients,
    area: &AreaConfig,
) -> Result<TrainingRecord> {
    let key = evidence_key(&rec)?;
    let execution_evidence = resolve_execution_evidence(&rec, clients, &key);
    let input = VerifierInput {
        messages: &rec.messages,
        reasoning_tokens: rec.cost.reasoning_tokens,
        cot_required: area.cot_required,
        // Thread the persisted contract so the answer-correctness rail runs (E4): a present-CoT but
        // wrong-answer record is caught on the deterministic rail, not silently admitted on the panel.
        contract: rec.verification_contract.as_ref(),
        rule_only_authoritative: area.rule_only_authoritative,
        execution_evidence: execution_evidence.as_ref(),
        evidence_key: key,
    };
    let grade = run_verifier(&input, clients.sandbox.as_ref());

    // Persist the verification block AND the resolved execution report onto the envelope, then
    // advance the lifecycle. Persisting the report is what makes a crash-resume of THIS edge
    // re-verify to the same verdict from data alone, and it leaves the operator an audit trail of
    // which report decided the record.
    let mut updated = rec;
    updated.execution_evidence = execution_evidence;
    updated.verification = grade.verification;
    persist_envelope_and_advance(&updated, clients, LifecycleState::Verified, None).await?;
    reload(updated, clients).await
}

/// `Verified → Judged`: grade the panel (cached) + compute consensus with the NON-IDENTITY
/// correlation prior, persisting the `Judging` block.
async fn judge(
    rec: TrainingRecord,
    clients: &Clients,
    area: &AreaConfig,
) -> Result<TrainingRecord> {
    // The verifier rail already ran; re-derive its grade from the persisted verification block so the
    // hard gate is honored without re-running (pure). The block carries BOTH signals: a proven
    // failure (`all_passed == false`) and an undecidable deterministic axis (`needs_review`).
    let verifier_grade = crate::grade::verifier_grade_from_verification(&rec, area);

    // Short-circuit on a verifier hard reject — a proven failure sinks the record regardless of the
    // panel (no judge spend) — and on a record the deterministic rail could not decide: a panel
    // score is not a substitute for ground truth that could not be obtained, so such a record is
    // held for review without spending a judge token either.
    let outcome = if verifier_grade.is_hard_reject() || verifier_grade.blocks_admission() {
        let grader = HybridGrader::new(area.thresholds);
        // An empty correlation matrix is fine here (the panel is never consulted in either case).
        grader.grade(
            Some(&verifier_grade),
            &[],
            &[],
            Some(&area.training_area),
            &CorrelationMatrix::identity(0),
        )?
    } else {
        grade_and_consense(&rec, clients, area, verifier_grade).await?
    };

    let mut updated = rec;
    updated.judging = outcome.judging;
    persist_envelope_and_advance(&updated, clients, LifecycleState::Judged, None).await?;
    reload(updated, clients).await
}

/// Run the cached judge panel + the consensus with the NON-IDENTITY correlation prior. The
/// load-bearing R-prior wiring lives here: for a `k > 1` panel the matrix is
/// `CorrelationMatrix::uniform_offdiagonal(k, rho)`, asserted non-identity (fail loud otherwise).
async fn grade_and_consense(
    rec: &TrainingRecord,
    clients: &Clients,
    area: &AreaConfig,
    verifier_grade: gw_judge::VerifierGrade,
) -> Result<GradeOutcome> {
    let content_hash = record_hash(rec)?;
    let candidate_render = gw_format::render(
        &rec.messages,
        TrlFormat::OpenAiMessages,
        CotPolicy::Supervised,
    )?;

    let panel = grade_panel_cached(
        &clients.store,
        clients.judge.as_ref(),
        &area.judges,
        &area.rubric,
        &candidate_render,
        &content_hash,
    )
    .await?;

    let r = crate::grade::correlation_prior(panel.len(), area.correlation_rho)?;
    let grader = HybridGrader::new(area.thresholds);
    let outcome = grader.grade(
        Some(&verifier_grade),
        &panel,
        &[],
        Some(&area.training_area),
        &r,
    )?;
    Ok(outcome)
}

/// The tag stamped on a bounded-revise RETRY record (`attempt = 1`). When [`reconcile`] sees it, a
/// `Decision::Revise` is DOWNGRADED to a terminal `Rejected` rather than written as a second
/// `Revising` — enforcing the single-bound (INVARIANT 12: no second `revising`). Set by
/// `crate::revise::generate_retry`.
pub const REVISE_RETRY_TAG: &str = "revise_retry";

/// `Judged → {Admitted | Rejected | Revising | NeedsReview}`: reconcile the persisted `Judging` block
/// to the lifecycle via the `gw-judge::Decision` mapping (the verdict→lifecycle reconciliation).
///
/// The single-revise bound is enforced HERE: a record already tagged [`REVISE_RETRY_TAG`] (the
/// `attempt = 1` retry) that judges `Revise` AGAIN is mapped straight to `Rejected` — it never writes
/// a SECOND `Revising` transition. So there is exactly one `revising` per logical record.
async fn reconcile(
    rec: TrainingRecord,
    clients: &Clients,
    area: &AreaConfig,
) -> Result<TrainingRecord> {
    let decision = crate::grade::decision_from_judging(&rec, area)?;
    let already_revised = rec.tags.iter().any(|t| t == REVISE_RETRY_TAG);
    let (to, detail) = match (&decision, already_revised) {
        // A second revise on the retry is downgraded to a conservative terminal Reject (the bound).
        (gw_judge::Decision::Revise { .. }, true) => (
            LifecycleState::Rejected,
            "revise_bound_exhausted: second revise downgraded to reject".to_string(),
        ),
        _ => (
            decision.to_lifecycle(),
            decision.reason().as_str().to_string(),
        ),
    };
    persist_envelope_and_advance(&rec, clients, to, Some(&detail)).await?;
    reload(rec, clients).await
}

/// `Admitted → Formatted`: render the admitted record to the target template(s). v1 renders the
/// OpenAI-messages projection to validate the render path; the export shard owns the byte-exact
/// target render.
async fn format_record(rec: TrainingRecord, clients: &Clients) -> Result<TrainingRecord> {
    // Render to confirm the admitted record projects cleanly (fail loud on a control-token leak).
    let _ = gw_format::render(
        &rec.messages,
        TrlFormat::OpenAiMessages,
        CotPolicy::Supervised,
    )?;
    persist_envelope_and_advance(&rec, clients, LifecycleState::Formatted, None).await?;
    reload(rec, clients).await
}

/// `Formatted → Exported`: mark the record exported. The actual columnar shard write is the
/// executor's batch step (`Store::export_parquet` over all admitted records); here the per-record
/// lifecycle reaches its terminal-good state.
async fn export_record(rec: TrainingRecord, clients: &Clients) -> Result<TrainingRecord> {
    persist_envelope_and_advance(&rec, clients, LifecycleState::Exported, None).await?;
    reload(rec, clients).await
}

/// Persist the (already-mutated) envelope via `put` (idempotent upsert), then advance the lifecycle
/// in one transaction (`advance_lifecycle` records the transition + grows `lifecycle.history`). Emits
/// the [`EngineEvent::StateAdvanced`]. This is the single persist-after-every-transition site.
async fn persist_envelope_and_advance(
    rec: &TrainingRecord,
    clients: &Clients,
    to: LifecycleState,
    detail: Option<&str>,
) -> Result<()> {
    // 1. Upsert the envelope (carries the new verification/judging block; idempotent by record_id).
    clients.store.put(rec).await?;
    // 2. Advance the lifecycle (state + event-sourced history) in one transaction.
    clients
        .store
        .advance_lifecycle(&rec.record_id, to, detail)
        .await?;
    if to == LifecycleState::Admitted {
        crate::priors::append_record(&clients.priors, clients.embedder.as_ref(), rec);
    }
    clients.events.emit(EngineEvent::StateAdvanced {
        record_id: rec.record_id.clone(),
        to,
    });
    Ok(())
}

/// Re-read the record from the store after a transition so the returned envelope reflects exactly what
/// was persisted (including the `lifecycle.history` grown by `advance_lifecycle`). The store is the
/// source of truth; the in-memory `rec` we mutated does not carry the appended history row.
async fn reload(rec: TrainingRecord, clients: &Clients) -> Result<TrainingRecord> {
    Ok(clients.store.get(&rec.record_id).await?)
}

/// A `step` that runs with no sandbox oracle wired (the [`NullSandboxOracle`] default). Convenience
/// for callers that have not injected a real oracle; identical to `step` in every other respect.
#[doc(hidden)]
pub fn _null_sandbox() -> NullSandboxOracle {
    NullSandboxOracle
}
