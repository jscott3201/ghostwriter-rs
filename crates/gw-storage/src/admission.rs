//! Atomic operational launch registration and physical-request admission.
use crate::{Result, StorageError, Store, now_rfc3339};
use gw_schema::{
    AccountingCapability as Cap, AccountingHistory, AccountingPolicy, AccountingSnapshot,
    AdmissionDenial, AttemptIntent, AttemptReceipt, LaunchCoverage, PolicyState,
};

/// Inputs resolved before atomic semantic-manifest and operational launch registration.
pub struct LaunchRequest<'a> {
    /// Stable run identity.
    pub run_id: &'a str,
    /// Immutable, supported effective generation/admission contract.
    pub manifest: gw_schema::RunManifest,
    /// Whether an unknown run ID may be initialized.
    pub mode: crate::RunMode,
    /// Explicit launch policy.
    pub policy: &'a AccountingPolicy,
    /// Actual injected teacher capability.
    pub teacher: Cap,
    /// Actual injected judge capability.
    pub judge: Cap,
    /// Actual injected embedding capability.
    pub embedding: Cap,
}

/// Atomic pre-send outcome. Waiting never holds a database transaction open.
#[derive(Debug, PartialEq)]
pub enum AttemptAdmission {
    /// Intent was committed and exactly one physical send is authorized.
    Admitted(String),
    /// Every unresolved intent belongs to this live coordinator; wait and re-check.
    Wait,
    /// No physical send is authorized.
    Denied(AdmissionDenial),
}

impl Store {
    /// Atomically initialize or compare the immutable manifest before registering current policy
    /// and coverage. Compatible replay preserves original manifest bytes and creation time.
    ///
    /// # Errors
    /// Rejects incompatible, unpinned, unsupported or unknown replay runs without mutation;
    /// also returns policy/admission or storage failures.
    pub async fn register_accounting_launch(
        &self,
        request: LaunchRequest<'_>,
    ) -> Result<LaunchCoverage> {
        if !request.policy.is_valid() {
            return Err(StorageError::Attempt(
                "accounting limit must be finite and nonnegative".into(),
            ));
        }
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        let existing: Option<crate::run_manifest::StoredManifest> = sqlx::query_as(
            "SELECT config_json, shard_count, prompts_hash FROM runs WHERE run_id = ?",
        )
        .bind(request.run_id)
        .fetch_optional(&mut *tx)
        .await?;
        let fresh = existing.is_none();
        crate::run_manifest::check(
            request.run_id,
            existing.as_ref(),
            &request.manifest,
            request.mode,
        )?;
        let prior = crate::accounting::policy(&mut tx, request.run_id).await?;
        let complete = fresh || prior.as_ref().is_some_and(|(_, complete, _)| *complete);
        let unknown = [request.teacher, request.judge, request.embedding].contains(&Cap::Unknown);
        if let AccountingPolicy::FiniteUsd { .. } = request.policy {
            if unknown {
                return Err(StorageError::Admission(AdmissionDenial::UnknownCoverage));
            }
            if !complete {
                return Err(StorageError::Admission(AdmissionDenial::IncompleteHistory));
            }
            let summary = crate::accounting::snapshot(&mut tx, request.run_id).await?;
            // Same-policy launches can reuse the epoch, but never upgrade unknown coverage.
            if summary.unknown_coverage_lanes > 0 {
                return Err(StorageError::Admission(AdmissionDenial::UnknownCoverage));
            }
            let changing = prior
                .as_ref()
                .is_none_or(|(state, _, _)| &state.policy != request.policy);
            if changing && let Some(reason) = evidence_denial(&summary) {
                return Err(StorageError::Admission(reason));
            }
        }
        let epoch = match &prior {
            Some((state, _, _)) if &state.policy == request.policy => state.epoch,
            Some((state, _, _)) => state
                .epoch
                .checked_add(1)
                .ok_or_else(|| StorageError::Attempt("policy epoch exhausted".into()))?,
            None => 1,
        };
        let policy = PolicyState {
            version: 1,
            epoch,
            policy: request.policy.clone(),
        };
        if fresh {
            sqlx::query("INSERT INTO runs (run_id, config_json, budget_usd, status, created_at, shard_count, prompts_hash) VALUES (?, ?, NULL, 'running', ?, ?, ?)")
                .bind(request.run_id).bind(serde_json::to_string(&request.manifest)?).bind(now_rfc3339()).bind(request.manifest.input_plan.shard_items.len() as i64).bind(&request.manifest.input_plan.content_hash).execute(&mut *tx).await?;
        } else {
            sqlx::query("UPDATE runs SET status='running' WHERE run_id=?")
                .bind(request.run_id)
                .execute(&mut *tx)
                .await?;
        }
        sqlx::query("INSERT INTO run_accounting (run_id, policy_json, history_complete) VALUES (?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET policy_json=excluded.policy_json, revision=revision+1")
            .bind(request.run_id).bind(serde_json::to_string(&policy)?).bind(complete).execute(&mut *tx).await?;
        let launch_id: String = sqlx::query_scalar("SELECT lower(hex(randomblob(16)))")
            .fetch_one(&mut *tx)
            .await?;
        let coverage = LaunchCoverage {
            version: 1,
            run_id: request.run_id.into(),
            launch_id,
            history: if complete {
                AccountingHistory::RecordedFromCreation
            } else {
                AccountingHistory::Unknown
            },
            policy: Some(policy),
            teacher: request.teacher,
            judge: request.judge,
            embedding: request.embedding,
        };
        sqlx::query("INSERT INTO model_launches (launch_id, run_id, coverage_json, created_at) VALUES (?, ?, ?, ?)")
            .bind(&coverage.launch_id).bind(request.run_id).bind(serde_json::to_string(&coverage)?).bind(now_rfc3339()).execute(&mut *tx).await?;
        #[cfg(test)]
        self.test_boundary("launch", "precommit").await?;
        tx.commit().await?;
        #[cfg(test)]
        self.test_boundary("launch", "committed").await?;
        Ok(coverage)
    }

    /// Atomically re-check epoch, coverage, spend and live ownership before committing one intent.
    /// `live_attempts` must come from in-memory ownership, never from a stored launch identifier.
    pub async fn admit_model_attempt(
        &self,
        intent: &AttemptIntent,
        epoch: u64,
        live_attempts: &[String],
    ) -> Result<AttemptAdmission> {
        crate::attempts::validate_intent(intent)?;
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        let launch: String = sqlx::query_scalar(
            "SELECT coverage_json FROM model_launches WHERE launch_id = ? AND run_id = ?",
        )
        .bind(&intent.context.launch_id)
        .bind(&intent.context.run_id)
        .fetch_one(&mut *tx)
        .await?;
        let coverage = crate::attempts::decode_coverage(&launch)?;
        let summary = crate::accounting::snapshot(&mut tx, &intent.context.run_id).await?;
        let Some(policy) = summary.effective.as_ref() else {
            return Ok(AttemptAdmission::Denied(
                AdmissionDenial::UnregisteredContext,
            ));
        };
        if policy.epoch != epoch
            || coverage
                .policy
                .as_ref()
                .is_none_or(|captured| captured != policy)
        {
            return Ok(AttemptAdmission::Denied(AdmissionDenial::PolicySuperseded));
        }
        if let AccountingPolicy::FiniteUsd { limit_usd } = policy.policy {
            if summary.history != AccountingHistory::RecordedFromCreation {
                return Ok(AttemptAdmission::Denied(AdmissionDenial::IncompleteHistory));
            }
            if summary.unknown_coverage_lanes > 0 {
                return Ok(AttemptAdmission::Denied(AdmissionDenial::UnknownCoverage));
            }
            if summary.unresolved_attempts > 0 {
                let unresolved =
                    crate::accounting::receipts(&mut tx, &intent.context.run_id).await?;
                return Ok(
                    if unresolved
                        .iter()
                        .filter(|receipt| receipt.transport.is_none())
                        .all(|receipt| live_attempts.contains(&receipt.attempt_id))
                    {
                        AttemptAdmission::Wait
                    } else {
                        AttemptAdmission::Denied(AdmissionDenial::UnresolvedAttempts)
                    },
                );
            }
            if let Some(reason) = evidence_denial(&summary) {
                return Ok(AttemptAdmission::Denied(reason));
            }
            let known_usd = summary.known_usd.unwrap_or(0.0);
            if known_usd >= limit_usd {
                return Ok(AttemptAdmission::Denied(AdmissionDenial::LimitReached {
                    known_usd,
                    limit_usd,
                }));
            }
        }
        let id: String = sqlx::query_scalar("SELECT lower(hex(randomblob(16)))")
            .fetch_one(&mut *tx)
            .await?;
        let receipt = AttemptReceipt {
            attempt_id: id.clone(),
            policy_epoch: Some(epoch),
            intent: intent.clone(),
            metadata: Default::default(),
            observations: vec![],
            conflicts: vec![],
            transport: None,
            interpretation: None,
        };
        let now = now_rfc3339();
        sqlx::query("INSERT INTO model_attempts (attempt_id, run_id, launch_id, receipt_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)")
            .bind(&id).bind(&intent.context.run_id).bind(&intent.context.launch_id).bind(serde_json::to_string(&receipt)?).bind(&now).bind(&now).execute(&mut *tx).await?;
        crate::accounting::bump(&mut tx, &intent.context.run_id).await?;
        #[cfg(test)]
        self.test_boundary("intent", "precommit").await?;
        tx.commit().await?;
        #[cfg(test)]
        self.test_boundary("intent", "committed").await?;
        Ok(AttemptAdmission::Admitted(id))
    }
}
fn evidence_denial(summary: &AccountingSnapshot) -> Option<AdmissionDenial> {
    if summary.unresolved_attempts > 0 {
        Some(AdmissionDenial::UnresolvedAttempts)
    } else if summary.known_usd.is_none()
        || summary.invalid_cost_attempts > 0
        || summary.conflicting_attempts > 0
    {
        Some(AdmissionDenial::InvalidEvidence)
    } else if summary.unknown_cost_attempts > 0 {
        Some(AdmissionDenial::UnknownCost)
    } else {
        None
    }
}
