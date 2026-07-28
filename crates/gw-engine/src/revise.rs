//! The bounded single revise loop (ARCHITECTURE §5, INVARIANT 12; D-LIFECYCLE).
//!
//! When the judge returns `Decision::Revise`, the record is persisted at `Revising` (by
//! `crate::step`'s reconcile edge). The ENGINE — not gw-judge — then re-enters generation for a
//! SINGLE bounded retry: it re-generates the assistant turn at a new attempt, re-drives the record
//! through verify → judge → reconcile, and BLOCKS a second `Revising`. A second `Decision::Revise` on
//! the retry is downgraded to a terminal `Rejected` (conservative), so there is no code path that
//! produces two `Revising` transitions for one logical record.
//!
//! ## How the single-bound is enforced
//!
//! The retry record is a DISTINCT record id (`attempt = 1` in the deterministic id), so the original
//! `Revising` record stays in the store as the audit trail of the first pass, and the retry is its own
//! row. The bound rests SOLELY on `step::reconcile`: the retry is stamped with
//! `step::REVISE_RETRY_TAG`, and reconcile maps a SECOND `Decision::Revise` on a tagged record straight
//! to `Rejected` — so the retry can NEVER write a second `Revising` transition. (`revise_once`
//! `debug_assert!`s this rather than re-implementing the downgrade, so there is no unreachable
//! second-guard a future change could mistake for live defense-in-depth.) This is idempotent: a
//! crash-restart that finds the `attempt = 1` retry already persisted resumes it from its last state
//! and never re-generates a second retry (the id is deterministic, so there is only ever one retry per
//! original).

use gw_generate::{
    GatedUserTurn, ReasoningPolicy, RecordContext, SamplingPreset, TeacherCall, assemble,
    generate_assistant, synthesize_user_turn,
};
use gw_schema::{BudgetBreach, LifecycleState, TeacherRef, TrainingRecord};
use gw_storage::{StorageError, now_rfc3339, prompt_hash};

use crate::clients::{AreaConfig, Clients};
use crate::control::RunControl;
use crate::error::{EngineError, Result};
use crate::event::EngineEvent;
use crate::seed::{SeedItem, record_id};
use crate::step::{drive, is_terminal};

/// Run the bounded single revise for a record that reached `Revising`.
///
/// Generates the `attempt = 1` retry (once — idempotent by deterministic id), drives it through the
/// pipeline, and BLOCKS a second `Revising`: if the retry is judged `Revise` again, it is forced to
/// `Rejected`. Returns the terminal retry record. If the original is NOT at `Revising`, this is a
/// no-op returning the original unchanged.
///
/// The budget gate is consulted before the retry's teacher call (a retry is new teacher work). If the
/// cap is reached, the retry is not generated and the original stays at `Revising` (it will be
/// re-attempted on a later relaunch under budget).
///
/// # Errors
/// Propagates the first [`EngineError`] from generation / driving / persistence.
pub async fn revise_once(
    run_id: &str,
    shard: i64,
    seed: &SeedItem,
    original: &TrainingRecord,
    clients: &Clients,
    area: &AreaConfig,
    control: RunControl<'_>,
) -> Result<TrainingRecord> {
    if original.lifecycle.state != LifecycleState::Revising {
        return Ok(original.clone());
    }
    if control.is_cancelled() {
        return Ok(original.clone());
    }

    let completion_index = original.generation.completion_index.unwrap_or(0);
    let retry_id = record_id(run_id, shard, seed.seed, 1, completion_index);

    // Crash-resume + never-re-spend: a retry already persisted is driven from its last state, never
    // re-generated (the deterministic id guarantees exactly one retry per original).
    let retry = match clients.store.get(&retry_id).await {
        Ok(existing) => existing,
        Err(StorageError::NotFound(_)) => {
            if !clients.budget.may_dispatch() {
                if control.on_breach() == BudgetBreach::Abort {
                    control.cancel();
                }
                // Budget exhausted: do not start the retry; the original stays at Revising.
                return Ok(original.clone());
            }
            generate_retry(
                run_id,
                &retry_id,
                completion_index,
                seed,
                original,
                clients,
                area,
            )
            .await
            // X1: attribute a record-level retry fault to the RETRY id so the shard parks the actual
            // faulting retry at `Error`, never a blindly-assumed `c0` (which may be the `Revising`
            // original or a healthy sibling).
            .map_err(|e| e.attribute_to(&retry_id))?
        }
        Err(e) => return Err(e.into()),
    };

    // Drive the retry to terminal. The single-bound rests SOLELY on `step::reconcile`: it is tagged
    // `REVISE_RETRY_TAG`, so a SECOND `Decision::Revise` on the retry is downgraded to `Rejected`
    // INSIDE reconcile and the retry can therefore NEVER land back at `Revising`. We assert that here
    // rather than re-implementing the downgrade (which would be unreachable defense-in-depth that a
    // future change could mistake for a live second guard).
    let driven = drive(retry, clients, area, control.token())
        .await
        .map_err(|e| e.attribute_to(&retry_id))?;
    if !control.is_cancelled() {
        debug_assert!(
            is_terminal(driven.lifecycle.state)
                && driven.lifecycle.state != LifecycleState::Revising,
            "the revise retry must terminate without a second Revising (the tag-in-reconcile bound); \
             got {:?}",
            driven.lifecycle.state
        );
    }
    Ok(driven)
}

/// Generate the `attempt = 1` retry record at `AssistantGenerated`. Mirrors the producer call in
/// `crate::sibling` but stamps `attempt = 1` into the id and re-uses the original's sibling group.
async fn generate_retry(
    run_id: &str,
    retry_id: &str,
    completion_index: u32,
    seed: &SeedItem,
    original: &TrainingRecord,
    clients: &Clients,
    area: &AreaConfig,
) -> Result<TrainingRecord> {
    let priors = crate::priors::snapshot(&clients.priors);
    let gated: GatedUserTurn =
        synthesize_user_turn(seed.candidate.clone(), clients.embedder.as_ref(), &priors)?;
    if !gated.passed() {
        return Err(EngineError::Generate(
            gw_generate::GenerateError::Invariant("revise retry: user-turn QC gate failed".into()),
        ));
    }

    // Vary the sampling seed for the retry so it is not a verbatim re-draw of the first attempt.
    let retry_seed = seed
        .seed
        .wrapping_add(i64::from(completion_index))
        .wrapping_add(1_000_000);
    let mut call = TeacherCall::new(
        area.teacher_slug.clone(),
        vec![gated.candidate.message.clone()],
        area.max_tokens,
    )
    .with_sampling(SamplingPreset::official().with_seed(retry_seed));
    if let Some(reasoning_max_tokens) = area.teacher_reasoning_max_tokens {
        call = call.with_reasoning(ReasoningPolicy::MaxTokens(reasoning_max_tokens));
    }

    let turn = generate_assistant(clients.teacher.as_ref(), &gated, &call).await?;
    let cost_usd = turn.cost.unwrap_or(0.0);

    let teacher_ref = TeacherRef {
        provider: "openrouter".to_string(),
        slug: area.teacher_slug.clone(),
        served_by: None,
        model_card_revision: None,
    };
    let ctx = RecordContext {
        record_id: retry_id.to_string(),
        run_id: run_id.to_string(),
        training_area: area.training_area.clone(),
        harness_version: clients.harness_version.clone(),
        git_commit: clients.git_commit.clone(),
        now_rfc3339: now_rfc3339(),
        user_synth_model: gated.candidate.seed.prompt_template_id.clone(),
    };

    let plan = gw_generate::SiblingPlan {
        completion_index,
        n_completions: original.generation.n_completions.unwrap_or(1),
        sampling: call.sampling,
    };
    let mut rec = assemble(
        &ctx,
        &gated,
        turn,
        teacher_ref,
        call.generation(),
        Some(plan),
    );
    // Keep the retry in the same sibling group as the original (same prompt → same prompt_hash).
    rec.generation.sibling_group_id = Some(prompt_hash(&rec.messages)?);
    // Record the lineage: the retry derives from the original Revising record.
    rec.provenance.parent_ids = vec![original.record_id.clone()];
    // Tag the retry so `step::reconcile` downgrades a SECOND revise straight to Rejected (the single
    // bound: the retry never writes a second `revising` transition).
    rec.tags.push(crate::step::REVISE_RETRY_TAG.to_string());

    clients.store.put(&rec).await?;
    clients.events.emit(EngineEvent::StateAdvanced {
        record_id: rec.record_id.clone(),
        to: LifecycleState::AssistantGenerated,
    });
    let total = clients.budget.charge(cost_usd);
    clients.events.emit(EngineEvent::CostCharged {
        record_id: rec.record_id.clone(),
        usd: cost_usd,
        run_total_usd: total,
    });

    Ok(clients.store.get(retry_id).await?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_id_is_attempt_one_and_distinct_from_original() {
        let original = record_id("run", 0, 5, 0, 0);
        let retry = record_id("run", 0, 5, 1, 0);
        assert_ne!(original, retry);
        assert!(retry.contains("-a1-"));
        assert!(original.contains("-a0-"));
    }
}
