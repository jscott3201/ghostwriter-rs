//! Best-of-k fan-out (D-BESTOFK; ARCHITECTURE §5, INVARIANT 11).
//!
//! A best-of-k group fans ONE gated USER turn into `k` sibling child records that share a
//! `sibling_group_id` (`== prompt_hash`) and carry distinct `completion_index` `0..k`. Each sibling
//! runs `assistant_generated → verified → judged` INDEPENDENTLY (driven by `crate::step`); the
//! engine then ADMITS the best by `judging.aggregate` (the verifier hard-gate must pass) and RETAINS
//! every rejected sibling. A rejected sibling is NEVER dropped while an admitted sibling exists — they
//! are kept (as `Rejected`) for DPO + audit + bad_patterns.
//!
//! ## The producer call site (generation edge)
//!
//! This module owns the `gw-generate` producer call: for each sibling it builds the seed-varied
//! [`SamplingPreset`], calls `synthesize_user_turn` (the QC gate) → `generate_assistant` (the single
//! teacher-spend, content-hash cached by state) → `assemble`, mints the deterministic record id, fills
//! the `sibling_group_id` from the canonical `prompt_hash` (which `gw-generate` deliberately leaves
//! `None` — it has no storage dep), and persists the record at `AssistantGenerated`. Both the
//! single-trace (`k = 1`) and fan-out (`k > 1`) paths share this one call site.
//!
//! ## Never-re-spend across the fan-out
//!
//! Each sibling's teacher call is generated ONCE: a sibling already persisted at `AssistantGenerated`
//! (or beyond) on a crash-restart is NOT re-generated — the producer call only runs for siblings the
//! store does not already carry. So a restart mid-fan-out re-enters each sibling at its last persisted
//! state, and the teacher is never re-spent for an already-generated sibling (INVARIANT 3).

use futures::StreamExt;
use gw_generate::{
    GatedUserTurn, GenerateError, ReasoningPolicy, RecordContext, SamplingPreset, TeacherCall,
    assemble, generate_assistant, plan_group, synthesize_user_turn,
};
use gw_schema::{BudgetBreach, LifecycleState, TeacherRef, TrainingRecord};
use gw_storage::{Store, now_rfc3339, prompt_hash};

use crate::clients::{AreaConfig, Clients};
use crate::control::RunControl;
use crate::error::{EngineError, Result};
use crate::event::EngineEvent;
use crate::seed::{SeedItem, record_id};
use crate::step::{drive, drive_to_judged};

const TRUNCATION_RETRY_MAX_TOKENS: u32 = 32_000;

/// Generate, persist, drive, and select the best of a best-of-k group for one seed item.
///
/// Fans `seed.candidate` into `area.k` siblings, drives each to a terminal state, then admits the best
/// admissible sibling and retains the rest. Returns the group's siblings (all driven to terminal),
/// the best record id (if any sibling was admitted), and persists the lifecycle of every member.
///
/// The budget gate is consulted BEFORE each sibling's teacher call: once the cap is reached, no new
/// teacher work is dispatched (the in-flight siblings already generated still finish). If the cap is
/// reached before ANY sibling generates, the group is skipped (returns an empty result).
///
/// # Errors
/// Propagates the first [`EngineError`] from generation / driving / persistence.
pub async fn run_group(
    run_id: &str,
    shard: i64,
    seed: &SeedItem,
    clients: &Clients,
    area: &AreaConfig,
    control: RunControl<'_>,
) -> Result<GroupOutcome> {
    let plans = plan_group(SamplingPreset::official().with_seed(seed.seed), area.k);
    let plan_count = plans.len();
    let mut interrupted = false;
    let ctx = GroupDrive {
        run_id,
        shard,
        seed,
        clients,
        area,
        control,
    };
    let mut settled: Vec<(u32, TrainingRecord)> = Vec::with_capacity(plan_count);
    let mut first_fatal = None;

    if control.is_cancelled() {
        interrupted = true;
    } else {
        // F1: drive EACH sibling independently and concurrently. Join-all semantics are load-bearing:
        // a single fatal must not cancel sibling futures mid-generation and strand teacher spend. We
        // let every in-flight sibling settle, then surface the first systemic fault.
        let mut outcomes = futures::stream::iter(plans)
            .map(|plan| {
                let rid = record_id(run_id, shard, seed.seed, 0, plan.completion_index);
                let completion_index = plan.completion_index;
                let ctx = &ctx;
                async move { (completion_index, drive_sibling(ctx, &rid, &plan).await) }
            })
            .buffer_unordered(plan_count);

        while let Some((completion_index, outcome)) = outcomes.next().await {
            match outcome {
                Ok(driven) => {
                    interrupted |= control.is_cancelled();
                    settled.push((completion_index, driven));
                }
                // The budget gate tripped before this sibling could generate (Drain). Under fan-out,
                // other siblings may already be in flight; those still settle before we finalize.
                Err(SiblingOutcome::BudgetGated) => {}
                // This sibling faulted at the record level: it is parked at `Error`; keep its (now
                // terminal) envelope in the group so the report counts it.
                Err(SiblingOutcome::Parked(parked)) => settled.push((completion_index, *parked)),
                // A systemic/infrastructure fault — record it and continue joining siblings.
                Err(SiblingOutcome::Fatal(e)) => {
                    if first_fatal.is_none() {
                        first_fatal = Some(e);
                    }
                }
            }
        }
    }

    if let Some(e) = first_fatal {
        return Err(e);
    }
    settled.sort_by_key(|(completion_index, _)| *completion_index);
    let mut siblings: Vec<TrainingRecord> = settled.into_iter().map(|(_, rec)| rec).collect();

    if control.is_cancelled() {
        interrupted = true;
    }
    let best = if interrupted {
        None
    } else {
        select_and_finalize(&mut siblings, clients, area, control).await?
    };
    interrupted |= control.is_cancelled();
    Ok(GroupOutcome {
        siblings,
        best,
        interrupted,
    })
}

/// The outcome of running one best-of-k group: every sibling (driven to terminal) and the admitted
/// member's record id (`None` if none was admitted — every sibling was rejected/escalated).
#[derive(Debug, Clone, PartialEq)]
pub struct GroupOutcome {
    /// All siblings reached so far, each at its latest persisted lifecycle state.
    pub siblings: Vec<TrainingRecord>,
    /// The record id of the admitted (best) sibling, if any.
    pub best: Option<String>,
    /// `true` when cancellation stopped this group at a persisted transition boundary.
    pub interrupted: bool,
}

/// The control-flow outcome of generating + driving ONE best-of-k sibling (the `Err` arms of
/// [`drive_sibling`]), so [`run_group`] can ISOLATE a per-sibling fault without unwinding the group
/// (F1). `BudgetGated` stops the fan-out (Drain); `Parked` carries the sibling already advanced to
/// `Error` (continue with the rest); `Fatal` is a systemic fault that aborts the run.
enum SiblingOutcome {
    /// The budget cap was reached before this sibling could generate — stop fanning out (Drain).
    BudgetGated,
    /// A record-level fault struck this sibling; it has been parked at `Error` (correctly attributed).
    /// Its terminal envelope is carried (boxed — it is far larger than the other variants) so the
    /// group/report still counts it.
    Parked(Box<TrainingRecord>),
    /// A systemic / infrastructure fault — propagate to abort the group and the run.
    Fatal(EngineError),
}

struct GroupDrive<'a> {
    run_id: &'a str,
    shard: i64,
    seed: &'a SeedItem,
    clients: &'a Clients,
    area: &'a AreaConfig,
    control: RunControl<'a>,
}

struct GeneratedTeacherAttempt {
    turn: gw_generate::AssistantTurn,
    call: TeacherCall,
}

/// Generate (if needed) and drive ONE best-of-k sibling to `Judged`, ISOLATING its faults (F1).
///
/// Returns the sibling driven to `Judged` (or its persisted state on crash-resume) on success. On
/// failure it classifies via [`EngineError::is_record_level`]: a RECORD-LEVEL fault parks THIS sibling
/// at `Error` (attributed to its own id) and returns [`SiblingOutcome::Parked`] so the caller keeps
/// finalizing the healthy survivors; a SYSTEMIC fault returns [`SiblingOutcome::Fatal`] (attributed,
/// for the audit trail) to abort the run. A budget-gated pre-generation stop returns
/// [`SiblingOutcome::BudgetGated`].
async fn drive_sibling(
    ctx: &GroupDrive<'_>,
    rid: &str,
    plan: &gw_generate::SiblingPlan,
) -> std::result::Result<TrainingRecord, SiblingOutcome> {
    // Crash-resume + never-re-spend: if this sibling is already persisted, drive it from its last state
    // rather than re-generating (the teacher is never re-spent for an already-generated id). A sibling
    // already parked at `Error` (a prior pass isolated it) is terminal — `drive_to_judged` returns it
    // unchanged, so a resume neither re-spends nor re-faults it.
    let rec = match ctx.clients.store.get(rid).await {
        Ok(existing) => existing,
        Err(gw_storage::StorageError::NotFound(_)) => {
            // Budget gate: stop dispatching NEW teacher work once the cap is reached (Drain).
            if !ctx.clients.budget.may_dispatch() {
                if ctx.control.on_breach() == BudgetBreach::Abort {
                    ctx.control.cancel();
                }
                return Err(SiblingOutcome::BudgetGated);
            }
            match generate_and_persist(ctx, rid, plan.sampling).await {
                Ok(rec) => rec,
                Err(e) => {
                    return Err(classify_sibling_fault(
                        e.attribute_to(rid),
                        rid,
                        ctx.seed,
                        ctx.run_id,
                        ctx.clients,
                        ctx.area,
                    )
                    .await);
                }
            }
        }
        // A store read fault is infrastructure — fatal (attributed for the audit log).
        Err(e) => {
            return Err(SiblingOutcome::Fatal(
                EngineError::from(e).attribute_to(rid),
            ));
        }
    };

    match drive_to_judged(rec, ctx.clients, ctx.area, ctx.control.token()).await {
        Ok(driven) => Ok(driven),
        Err(e) => Err(classify_sibling_fault(
            e.attribute_to(rid),
            rid,
            ctx.seed,
            ctx.run_id,
            ctx.clients,
            ctx.area,
        )
        .await),
    }
}

/// Classify a sibling fault into a [`SiblingOutcome`] (F1): a RECORD-LEVEL fault parks the sibling at
/// `Error` and returns `Parked` (so the group continues with the survivors); anything else is `Fatal`.
async fn classify_sibling_fault(
    err: EngineError,
    rid: &str,
    seed: &SeedItem,
    run_id: &str,
    clients: &Clients,
    area: &AreaConfig,
) -> SiblingOutcome {
    if !err.is_record_level() {
        return SiblingOutcome::Fatal(err);
    }
    match park_sibling_errored(rid, &err, seed, run_id, clients, area).await {
        Ok(parked) => SiblingOutcome::Parked(Box::new(parked)),
        // The park itself failed (a storage fault) — that IS infrastructure, so abort.
        Err(park_err) => SiblingOutcome::Fatal(park_err),
    }
}

/// Park ONE faulting sibling at [`LifecycleState::Error`] (F1) and emit [`EngineEvent::RecordErrored`],
/// returning the parked terminal envelope. If the sibling never persisted (the fault hit during
/// generation before the first `put`), a minimal stub (via [`crate::executor::error_stub`]) is
/// persisted first so the failure is queryable and counted. NO-CLOBBER: if the record somehow already
/// carries HEALTHY PROGRESS it is left intact (returned as-is) — a fault can never overwrite a good
/// record. The error message is recorded (never carries a secret — the wrapped errors are safe to log).
async fn park_sibling_errored(
    rid: &str,
    err: &EngineError,
    seed: &SeedItem,
    run_id: &str,
    clients: &Clients,
    area: &AreaConfig,
) -> Result<TrainingRecord> {
    let msg = err.to_string();
    match clients.store.get(rid).await {
        // NO-CLOBBER: a record already at healthy progress (or a non-Error terminal) is never parked.
        Ok(existing) if !is_sibling_parkable(existing.lifecycle.state) => {
            clients.events.emit(EngineEvent::RecordErrored {
                record_id: rid.to_string(),
                error: msg,
            });
            return Ok(existing);
        }
        Ok(_) => {}
        Err(gw_storage::StorageError::NotFound(_)) => {
            // The fault struck before generation persisted anything: persist a minimal stub at `Seeded`
            // (the candidate's prompt is known) so the failure is queryable + counted, then park it.
            let stub = crate::executor::error_stub(area, clients, run_id, rid, &seed.candidate);
            clients.store.put(&stub).await?;
        }
        Err(e) => return Err(e.into()),
    }
    clients
        .store
        .advance_lifecycle(rid, LifecycleState::Error, Some(&msg))
        .await?;
    clients.events.emit(EngineEvent::RecordErrored {
        record_id: rid.to_string(),
        error: msg,
    });
    Ok(clients.store.get(rid).await?)
}

/// `true` when a sibling at `state` may be PARKED at `Error` (F1). The faulting sibling is parked from
/// any IN-FLIGHT pre-decision state (`Seeded`/`UserSynthesized`/`AssistantGenerated`/`Verified`/`Judged`)
/// — a mid-drive fault (e.g. the judge rail faulting a sibling already at `Verified`) must STILL
/// terminalize it to `Error`, not silently strand it at a forward state. Only a DECIDED outcome (a
/// non-`Error` terminal or the `Revising` handoff) is never clobbered. Here `rid` is ALWAYS the faulting
/// sibling's own id (attributed in `drive_sibling`), so there is no cross-sibling mis-target risk — the
/// no-clobber refusal is purely defensive against re-parking an already-decided record.
fn is_sibling_parkable(state: LifecycleState) -> bool {
    !matches!(
        state,
        LifecycleState::Admitted
            | LifecycleState::Formatted
            | LifecycleState::Exported
            | LifecycleState::Rejected
            | LifecycleState::NeedsReview
            | LifecycleState::Revising
    )
}

/// Generate one sibling via the producer and persist it at `AssistantGenerated`. The single
/// teacher-spend site; charges the budget meter post-spend with the call's authoritative cost.
async fn generate_and_persist(
    group: &GroupDrive<'_>,
    rid: &str,
    sampling: SamplingPreset,
) -> Result<TrainingRecord> {
    // Gate the candidate (no teacher spend if the four-bool QC gate fails).
    // Best-effort under concurrency: simultaneous sibling groups may snapshot before either admits.
    // Dedup failures use the existing Error/circuit-breaker path by design.
    let priors = crate::priors::snapshot(&group.clients.priors);
    let gated: GatedUserTurn = synthesize_user_turn(
        group.seed.candidate.clone(),
        group.clients.embedder.as_ref(),
        &priors,
    )?;
    if !gated.passed() {
        return Err(EngineError::Generate(
            gw_generate::GenerateError::Invariant(format!(
                "user-turn QC gate failed for seed {} (answerable={}, difficulty_targeted={}, \
             diverse={}, in_scope_safe={})",
                group.seed.seed,
                gated.verdict.answerable,
                gated.verdict.difficulty_targeted,
                gated.verdict.diverse,
                gated.verdict.in_scope_safe,
            )),
        ));
    }

    let mut call = TeacherCall::new(
        group.area.teacher_slug.clone(),
        vec![gated.candidate.message.clone()],
        group.area.max_tokens,
    )
    .with_sampling(sampling);
    if let Some(reasoning_max_tokens) = group.area.teacher_reasoning_max_tokens {
        call = call.with_reasoning(ReasoningPolicy::MaxTokens(reasoning_max_tokens));
    }

    // The single teacher-spend, with one cost-accounted truncation retry if needed.
    let generated =
        generate_assistant_with_truncation_retry(group.clients, rid, &gated, &call, group.control)
            .await?;
    let cost_usd = generated.turn.cost.unwrap_or(0.0);

    let teacher_ref = TeacherRef {
        provider: "openrouter".to_string(),
        slug: group.area.teacher_slug.clone(),
        served_by: None,
        model_card_revision: None,
    };
    let record_ctx = RecordContext {
        record_id: rid.to_string(),
        run_id: group.run_id.to_string(),
        training_area: group.area.training_area.clone(),
        harness_version: group.clients.harness_version.clone(),
        git_commit: group.clients.git_commit.clone(),
        now_rfc3339: now_rfc3339(),
        user_synth_model: gated.candidate.seed.prompt_template_id.clone(),
    };

    let plan = plan_for(group.seed, sampling, group.area.k);
    let generation = generated.call.generation();
    let mut rec = assemble(
        &record_ctx,
        &gated,
        generated.turn,
        teacher_ref,
        generation,
        Some(plan),
    );

    // Fill the sibling_group_id from the canonical prompt_hash (gw-generate leaves it None; gw-storage
    // is authoritative for hashes, so compute it here from the same canonical projection).
    rec.generation.sibling_group_id = Some(sibling_group_id(&rec, &group.clients.store)?);
    let _ = group.shard; // shard rode into the record id; nothing else to stamp here.

    // Persist at AssistantGenerated (the expensive CoT is written before the next transition).
    group.clients.store.put(&rec).await?;
    group.clients.events.emit(EngineEvent::StateAdvanced {
        record_id: rec.record_id.clone(),
        to: LifecycleState::AssistantGenerated,
    });

    // Charge the budget AFTER the spend, with the call's authoritative cost.
    charge_teacher_cost(group.clients, &rec.record_id, cost_usd);

    // Re-read so the returned envelope matches what was persisted.
    Ok(group.clients.store.get(rid).await?)
}

async fn generate_assistant_with_truncation_retry(
    clients: &Clients,
    rid: &str,
    gated: &GatedUserTurn,
    call: &TeacherCall,
    control: RunControl<'_>,
) -> Result<GeneratedTeacherAttempt> {
    let err = match generate_assistant(clients.teacher.as_ref(), gated, call).await {
        Ok(turn) => {
            return Ok(GeneratedTeacherAttempt {
                turn,
                call: call.clone(),
            });
        }
        Err(err) => err,
    };
    let GenerateError::TruncatedReasoning { cost_usd, .. } = &err else {
        return Err(err.into());
    };
    charge_truncated_attempt(clients, rid, *cost_usd);
    if !clients.budget.may_dispatch() {
        if control.on_breach() == BudgetBreach::Abort {
            control.cancel();
        }
        return Err(err.into());
    }
    let Some(retry_tokens) = retry_max_tokens(call.max_tokens) else {
        return Err(err.into());
    };

    let mut retry = call.clone();
    retry.max_tokens = retry_tokens;
    match generate_assistant(clients.teacher.as_ref(), gated, &retry).await {
        Ok(turn) => Ok(GeneratedTeacherAttempt { turn, call: retry }),
        Err(err @ GenerateError::TruncatedReasoning { cost_usd, .. }) => {
            charge_truncated_attempt(clients, rid, cost_usd);
            Err(err.into())
        }
        Err(err) => Err(EngineError::from(err)),
    }
}

fn charge_truncated_attempt(clients: &Clients, rid: &str, cost_usd: Option<f64>) {
    if let Some(cost_usd) = cost_usd {
        charge_teacher_cost(clients, rid, cost_usd);
    }
}

fn charge_teacher_cost(clients: &Clients, rid: &str, cost_usd: f64) {
    let total = clients.budget.charge(cost_usd);
    clients.events.emit(EngineEvent::CostCharged {
        record_id: rid.to_string(),
        usd: cost_usd,
        run_total_usd: total,
    });
}

fn retry_max_tokens(max_tokens: u32) -> Option<u32> {
    if max_tokens >= TRUNCATION_RETRY_MAX_TOKENS {
        return None;
    }
    let widened =
        ((u64::from(max_tokens) * 3).div_ceil(2)).min(u64::from(TRUNCATION_RETRY_MAX_TOKENS));
    Some(widened as u32)
}

/// The canonical `sibling_group_id == prompt_hash` for a record, computed via the same hashing
/// `gw-storage::put` uses (so the group id agrees with the dedup key). The `store` argument is unused
/// at runtime but pins the dependency that owns the canonical hash; the computation is pure.
fn sibling_group_id(rec: &TrainingRecord, _store: &Store) -> Result<String> {
    Ok(prompt_hash(&rec.messages)?)
}

/// Build the `gw-generate::SiblingPlan` for one sibling (the assembler stamps `n_completions` +
/// `completion_index`; `sibling_group_id` is filled afterward from the prompt_hash).
fn plan_for(seed: &SeedItem, sampling: SamplingPreset, k: u32) -> gw_generate::SiblingPlan {
    // Re-derive the completion index from the sampling seed offset (seed.seed + index == sampling seed).
    let index = sampling
        .seed
        .map(|s| (s.wrapping_sub(seed.seed)) as u32)
        .unwrap_or(0);
    gw_generate::SiblingPlan {
        completion_index: index,
        n_completions: k.max(1),
        sampling,
    }
}

/// `true` when a sibling's state is in the admitted set (it passed admission): `Admitted`,
/// `Formatted`, or `Exported`.
fn is_admitted_state(state: LifecycleState) -> bool {
    matches!(
        state,
        LifecycleState::Admitted | LifecycleState::Formatted | LifecycleState::Exported
    )
}

/// `true` when a sibling is the established group winner from a prior pass.
///
/// An established winner is one already admitted or one driven to the `Revising` handoff. Retained
/// non-winners are forced to `Rejected`, so a sibling at `Revising` can only be the elected winner.
fn is_established_winner(state: LifecycleState) -> bool {
    is_admitted_state(state) || state == LifecycleState::Revising
}

/// Finalize a group of siblings: admit ONLY the single best, RETAIN the rest (INVARIANT 11), and stay
/// CRASH-IDEMPOTENT (E1) — never elect a SECOND winner when one was already admitted on a prior pass.
///
/// CRASH-RESUME GUARD (E1): a crash can land AFTER the best sibling reached `Admitted`/`Exported` or
/// the `Revising` handoff but BEFORE the shard cursor committed, so on relaunch this re-runs over the
/// SAME group. It first scans ALL siblings for an established winner ([`is_established_winner`]). If
/// one exists, THAT is the established winner: every other still-`Judged` sibling is driven to a
/// retained `Rejected` and NO second winner is elected (the established winner's id is returned). Only
/// when NO sibling is yet established does it elect a fresh winner: the highest-`judging.aggregate`
/// sibling among those whose verifier hard-gate passed AND whose re-derived decision is admissible. The
/// elected best is reconciled NATURALLY (`Judged → Admitted → … → Exported`, or `Judged → Revising`);
/// every OTHER `Judged` sibling whose own decision would ADMIT/revise is OVERRIDDEN to a retained
/// `Rejected`, and one whose natural outcome is Reject/Escalate reconciles to it (retained / parked).
///
/// Returns the admitted-or-revising best record id, if any.
async fn select_and_finalize(
    siblings: &mut [TrainingRecord],
    clients: &Clients,
    area: &AreaConfig,
    control: RunControl<'_>,
) -> Result<Option<String>> {
    // E1 crash-idempotency: if a sibling is ALREADY established as the winner from a prior pass, do NOT
    // re-elect. Retain every still-Judged sibling and return it.
    if let Some(winner_idx) = siblings
        .iter()
        .position(|s| is_established_winner(s.lifecycle.state))
    {
        // F3 (H-A): DRIVE the established winner before returning. A crash can land with the winner at
        // `Admitted`/`Formatted` (admitted but not yet exported); without this it would be stranded at
        // that state across resumes — an honest lifecycle requires it reach `Exported`. `drive` is
        // idempotent: a winner already at `Exported` returns immediately (no re-work, no re-spend).
        let finalized = drive(siblings[winner_idx].clone(), clients, area, control.token()).await?;
        let winner_id = finalized.record_id.clone();
        siblings[winner_idx] = finalized;
        retain_remaining_judged(siblings, Some(winner_idx), clients, area, control.token()).await?;
        return Ok(Some(winner_id));
    }

    // No established winner: elect the max-aggregate admissible sibling among the `Judged` candidates.
    let mut best_idx: Option<usize> = None;
    let mut best_agg = f64::NEG_INFINITY;
    for (i, sib) in siblings.iter().enumerate() {
        if sib.lifecycle.state != LifecycleState::Judged {
            continue; // already finalized (escalated / rejected) — not a fresh candidate.
        }
        if !sib.verification.all_passed {
            continue; // verifier hard-gate must pass to be admissible.
        }
        if is_admissible(sib, area)? {
            // Strict `>` so an aggregate TIE keeps the FIRST (lowest completion_index) candidate.
            let agg = sib.judging.aggregate.unwrap_or(f64::NEG_INFINITY);
            if agg > best_agg {
                best_agg = agg;
                best_idx = Some(i);
            }
        }
    }

    // Reconcile the elected best naturally; retain every other still-Judged sibling.
    let mut best_id = None;
    if let Some(i) = best_idx {
        let finalized = drive(siblings[i].clone(), clients, area, control.token()).await?;
        best_id = Some(finalized.record_id.clone());
        siblings[i] = finalized;
    }
    retain_remaining_judged(siblings, best_idx, clients, area, control.token()).await?;
    Ok(best_id)
}

/// Drive every still-`Judged` sibling (other than `keep_idx`, the elected/established winner) to a
/// RETAINED terminal state: a sibling whose own decision would ADMIT/revise is OVERRIDDEN to `Rejected`
/// (the group admits exactly one); one whose natural decision is Reject/Escalate reconciles to it
/// (retained / parked). Siblings already at a terminal state are left untouched.
async fn retain_remaining_judged(
    siblings: &mut [TrainingRecord],
    keep_idx: Option<usize>,
    clients: &Clients,
    area: &AreaConfig,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<()> {
    for (i, sib) in siblings.iter_mut().enumerate() {
        if Some(i) == keep_idx {
            continue;
        }
        if sib.lifecycle.state != LifecycleState::Judged {
            continue; // already finalized on its own (escalate / reject) — leave it (retained).
        }
        if is_admissible(sib, area)? {
            // OVERRIDE a non-winning admissible sibling to a retained Rejected (one admit per group).
            clients
                .store
                .advance_lifecycle(
                    &sib.record_id,
                    LifecycleState::Rejected,
                    Some("best_of_k_retained: not the admitted sibling"),
                )
                .await?;
            clients.events.emit(EngineEvent::StateAdvanced {
                record_id: sib.record_id.clone(),
                to: LifecycleState::Rejected,
            });
            *sib = clients.store.get(&sib.record_id).await?;
        } else {
            // Natural non-admitted outcome (Reject / Escalate→NeedsReview): reconcile + drive normally.
            *sib = drive(sib.clone(), clients, area, cancel).await?;
        }
    }
    Ok(())
}

/// `true` when a `Judged` sibling's re-derived decision is admissible (`Accept` or `Revise`).
fn is_admissible(sib: &TrainingRecord, area: &AreaConfig) -> Result<bool> {
    let decision = crate::grade::decision_from_judging(sib, area)?;
    Ok(matches!(
        decision,
        gw_judge::Decision::Accept { .. } | gw_judge::Decision::Revise { .. }
    ))
}
