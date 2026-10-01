//! Explicit publication through an application-trusted full planner rerun.
use crate::artifact::{ExportPlan, artifact_identity, integrity};
use crate::publication::{destination_path, publication_error};
use crate::{ExportPublication, ExportPurpose, Result, Store};
use gw_schema::{ExportOptions, ExportScope, FrozenScreeningPlan, TrainingRecord};
use std::path::Path;

impl Store {
    /// Capture the complete declared population and publish a trusted, exactly rerun screening plan.
    ///
    /// The trusted application callback must run its authoritative planner against these captured
    /// rows, the candidate plan and protected contents captured by the application. The read
    /// transaction closes before this expensive validation; preparation rechecks membership and
    /// every raw input inside its write transaction. Deserializing a plan alone supplies no authority.
    /// An implicit retry must supply the same plan and validation. Explicit-ID recovery uses the
    /// saved receipt and requires no protected text. Await completion and serialize by destination,
    /// as with [`Store::publish_export`].
    ///
    /// # Errors
    /// Rejects failed validation, stale populations, incomplete plans, mismatched policies or
    /// publication failure. Incomplete plans fail even when no output rows would be emitted.
    pub async fn publish_screened_export<F>(
        &self,
        options: ExportOptions,
        screening: FrozenScreeningPlan,
        dst: impl AsRef<Path>,
        purpose: ExportPurpose,
        validate: F,
    ) -> Result<ExportPublication>
    where
        F: FnOnce(&[TrainingRecord], &FrozenScreeningPlan) -> Result<()> + Send + 'static,
    {
        if options.scope
            != (ExportScope::Run {
                run_id: screening.declaration.output.run_id.clone(),
            })
            || options.target != screening.declaration.policy.target
            || options.cot_policy != screening.declaration.policy.cot_policy
        {
            return Err(integrity(
                "screened publication options do not match its pinned plan",
            ));
        }
        let dst = dst.as_ref().to_path_buf();
        let dst = tokio::task::spawn_blocking(move || destination_path(&dst)).await??;
        let destination = dst
            .to_str()
            .ok_or_else(|| integrity("export destination is not valid UTF-8"))?
            .to_string();
        let pending = self
            .pending_export(&destination, purpose, &options, Some(&screening.plan_id))
            .await?;
        let mut tx = self.pool().begin().await?;
        let records =
            crate::record_data::screening_population(&mut tx, &screening.declaration.runs.run_ids)
                .await?;
        tx.commit().await?;
        // Only this private construction after callback success can attach publication authority.
        let plan = tokio::task::spawn_blocking(move || -> Result<ExportPlan> {
            validate(&records, &screening)?;
            check_bindings(
                &records,
                &screening,
                gw_schema::ExportSchemaVersion::CURRENT,
            )?;
            let output: Vec<_> = records
                .iter()
                .filter(|r| r.run_id() == screening.declaration.output.run_id)
                .cloned()
                .collect();
            let mut plan = ExportPlan::prepare_registered(&output, options)?;
            let selected: std::collections::BTreeSet<_> = screening
                .eligible_output
                .iter()
                .map(|id| id.record_id.as_str())
                .collect();
            plan.rows
                .retain(|row| selected.contains(row.record_id.as_str()));
            plan.artifact.metadata_version = gw_schema::ExportArtifact::SCREENED_VERSION;
            plan.artifact.manifest.n_admitted = plan.rows.len() as u64;
            plan.artifact.manifest.multi_turn_loss = screening.declaration.policy.multi_turn_loss;
            plan.artifact.manifest.build_inputs_hash =
                crate::export::shard_content_hash(&plan.rows);
            plan.artifact.screening = Some(Box::new(crate::screening_witness::qualification(
                screening,
            )?));
            plan.artifact.artifact_id = artifact_identity(&plan.artifact, &plan.rows)?;
            crate::artifact::validate_rows(&plan.artifact, &plan.rows)?;
            Ok(plan)
        })
        .await??;
        let recovering = pending.is_some();
        let receipt = if let Some(receipt) = pending {
            let restored = self
                .restore_export_plan(&receipt)
                .await
                .map_err(|e| publication_error(&receipt.publication_id, e))?;
            if restored.artifact != plan.artifact {
                return Err(publication_error(
                    &receipt.publication_id,
                    integrity("pending screened artifact differs from validated plan"),
                ));
            }
            receipt
        } else {
            self.prepare_export_receipt(&plan, &destination, purpose)
                .await?
        };
        self.finish_publication(plan, receipt, recovering).await
    }
}

fn check_bindings(
    records: &[TrainingRecord],
    plan: &FrozenScreeningPlan,
    version: gw_schema::ExportSchemaVersion,
) -> Result<()> {
    let current = records
        .iter()
        .map(|r| {
            crate::screening_binding::capture_screening_input_for(
                r,
                &plan.declaration.policy,
                version,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    if current != plan.population {
        return Err(integrity(
            "screened population membership or inputs changed",
        ));
    }
    Ok(())
}

pub(crate) async fn check_population(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    artifact: &gw_schema::ExportArtifact,
) -> Result<()> {
    if let Some(witness) = &artifact.screening {
        let records =
            crate::record_data::screening_population(tx, &witness.plan.declaration.runs.run_ids)
                .await?;
        check_bindings(
            &records,
            &witness.plan,
            artifact.manifest.column_schema_version,
        )?;
    }
    Ok(())
}
