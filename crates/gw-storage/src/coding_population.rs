//! One transaction captures current registered held-out membership and keeps its oracles native.
use crate::{Result, Store, artifact::integrity, reference_import, reference_registration};
use gw_schema::{
    CodingPopulation, CodingPopulationMember, CodingTrainingBinding, Content, ExportSchemaVersion,
    ReferenceCapture, ReviewedCodingTask, TaskProvenance, TaskSplitRole,
};

/// Opaque database capture. Only the store can bind complete committed membership to private tasks.
/// Its public view includes compact Train content bindings but no held-out answers or protected oracles.
pub struct CapturedCodingPopulation {
    public: CodingPopulation,
    tasks: Vec<ReviewedCodingTask>,
}
impl CapturedCodingPopulation {
    /// Complete ordered redacted population for generation and saved binding inspection.
    #[must_use]
    pub fn public(&self) -> &CodingPopulation {
        &self.public
    }
    /// Native controller-only reviewed tasks, in precisely the public membership order.
    /// Do not send these private oracle values to model generation or public artifacts.
    #[must_use]
    pub fn private_tasks(&self) -> &[ReviewedCodingTask] {
        &self.tasks
    }
}
impl Store {
    /// Capture one entire held-out split from a complete, current, committed reference import.
    /// Registration alone is insufficient. No saved report or caller population assertion is used.
    ///
    /// # Errors
    /// Rejects unknown/changed registration, uncommitted/incomplete import, Train or corrupt state.
    pub async fn capture_coding_population(
        &self,
        registration_id: &str,
        split: TaskSplitRole,
    ) -> Result<CapturedCodingPopulation> {
        if split == TaskSplitRole::Train {
            return Err(integrity("coding comparison requires a held-out split"));
        }
        let mut tx = self.pool().begin().await?;
        let raw: Option<String> = sqlx::query_scalar(
            "SELECT capture_json FROM reference_registrations WHERE registration_id=?",
        )
        .bind(registration_id)
        .fetch_optional(&mut *tx)
        .await?;
        let raw = raw.ok_or_else(|| integrity("unknown coding population registration"))?;
        if raw.len() > 64 * 1024 * 1024 {
            return Err(integrity("registered coding capture exceeds bound"));
        }
        let capture: ReferenceCapture = serde_json::from_str(&raw)?;
        let registered = reference_registration::registration(&capture)?;
        let current = reference_import::current(&mut tx, &registered).await?;
        if registered.registration_id() != registration_id || current.is_none() {
            return Err(integrity(
                "coding population lacks an exact complete committed reference import",
            ));
        }
        let current = current.expect("checked complete reference import");
        let mut training_members = vec![];
        for member in registered
            .population()
            .members()
            .iter()
            .filter(|m| m.task.split.role == TaskSplitRole::Train)
        {
            let record = current.records.iter().find(|r| matches!(&r.origin, gw_schema::RecordOrigin::ReviewedReference(o) if o.member_id == member.member_id)).ok_or_else(|| integrity("accepted Train record absent"))?;
            let projected = crate::export::project(record, ExportSchemaVersion::RecordOrigins)?;
            let task_json = projected
                .task_json
                .ok_or_else(|| integrity("accepted Train task absent"))?;
            let origin_json = projected
                .origin_json
                .ok_or_else(|| integrity("accepted Train origin absent"))?;
            let content_id = CodingTrainingBinding::compute_content_id(
                &projected.record_id,
                &projected.messages_json,
                &task_json,
                &origin_json,
            );
            let task = record
                .task_provenance
                .as_ref()
                .ok_or_else(|| integrity("accepted Train provenance absent"))?;
            training_members.push(CodingTrainingBinding {
                member_id: member.member_id.clone(),
                record_id: projected.record_id,
                content_id,
                task_identity: task.identity.clone(),
                group: task.group.clone(),
                component: member.component.clone(),
                suite_id: member.task.suite_binding().suite_id,
            });
        }
        let mut tasks = vec![];
        let mut members = vec![];
        for (ordinal, member) in registered.population().members().iter().enumerate() {
            if member.task.split.role != split {
                continue;
            }
            let Content::Text(prompt) = member.task.prompt().content else {
                unreachable!("validated coding prompt")
            };
            members.push(CodingPopulationMember {
                ordinal: ordinal as u32,
                member_id: member.member_id.clone(),
                provenance: TaskProvenance::from_coding_task(&member.task).map_err(integrity)?,
                component: member.component.clone(),
                suite: member.task.suite_binding(),
                prompt,
            });
            tasks.push(member.task.clone());
        }
        let mut public = CodingPopulation {
            version: 1,
            population_id: String::new(),
            catalogue_id: registered.population().catalogue_id().into(),
            registration_id: registration_id.into(),
            batch_id: registered.batch_id().into(),
            split,
            training_members,
            members,
        };
        public.population_id = public.computed_id();
        public.validate().map_err(integrity)?;
        tx.commit().await?;
        Ok(CapturedCodingPopulation { public, tasks })
    }
}
