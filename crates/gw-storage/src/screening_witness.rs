//! Versioned structural and identity checks. The lexical algorithm remains owned by gw-eval.
use std::collections::{BTreeMap, BTreeSet};

use gw_schema::*;

use crate::{Result, artifact::integrity, screening_binding::screening_hash};

pub(crate) fn layout(policy: MultiTurnLoss) -> ScreenedExampleLayout {
    match policy {
        MultiTurnLoss::AllAssistant => ScreenedExampleLayout::AssistantPrefixV1,
        MultiTurnLoss::FinalTurnOnly => ScreenedExampleLayout::FullConversationFinalV1,
    }
}

pub(crate) fn qualification(plan: FrozenScreeningPlan) -> Result<ScreenedExportQualification> {
    let by_member: BTreeMap<_, _> = plan
        .groups
        .iter()
        .flat_map(|g| g.members.iter().map(move |id| (id, &g.group_id)))
        .collect();
    let members = plan
        .eligible_output
        .iter()
        .map(|id| {
            Ok(ScreenedExportMember {
                record: id.clone(),
                component_id: by_member
                    .get(id)
                    .ok_or_else(|| integrity("selected component missing"))?
                    .to_string(),
            })
        })
        .collect::<Result<_>>()?;
    Ok(ScreenedExportQualification {
        version: 2,
        validation: ScreeningValidation::PlannerRerunV2,
        population_check: ScreenedPopulationCheck::TransactionChecked,
        population_id: screening_hash(
            "screened-publication-population-v2",
            &(&plan.declaration.runs, &plan.population),
        )?,
        layout: layout(plan.declaration.policy.multi_turn_loss),
        plan,
        members,
    })
}

fn ordered<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn plan_identity(plan: &FrozenScreeningPlan, depth: u32) -> Result<()> {
    if depth >= 16
        || plan.version != SCREENING_PLAN_VERSION
        || plan.declaration.version != SCREENING_VERSION
        || plan.declaration.policy.recipe != LEXICAL_SCREEN_RECIPE
    {
        return Err(integrity("unsupported screening plan version/recipe/depth"));
    }
    if canonical_screening_declaration(&plan.declaration).map_err(integrity)? != plan.declaration {
        return Err(integrity("screening declaration is not canonical"));
    }
    if plan.lexical_status != LexicalScreeningStatus::Incomplete {
        validate_complete_screening_protected(
            &plan.declaration.policy,
            &plan.required_fields,
            &plan.protected_inputs,
        )
        .map_err(integrity)?;
    }
    if plan.population.iter().any(|binding| {
        binding.export_projection_id.len() != 64
            || !binding
                .export_projection_id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }) {
        return Err(integrity(
            "unsupported or missing screening export projection binding",
        ));
    }
    let mut unsigned = plan.clone();
    unsigned.plan_id.clear();
    if plan.plan_id != screening_hash("frozen-screening-plan-v2", &unsigned)?
        || plan.policy_id != screening_hash("screening-policy-v1", &plan.declaration.policy)?
        || plan.screening_input_id
            != screening_hash(
                "screening-captured-population-v2",
                &(&plan.declaration, &plan.population),
            )?
        || plan.protected_input_id
            != screening_hash(
                "screening-protected-inputs-v1",
                &plan
                    .protected_inputs
                    .iter()
                    .map(|i| (&i.canonical_id, &i.input_id))
                    .collect::<Vec<_>>(),
            )?
    {
        return Err(integrity("screening report identity mismatch"));
    }
    let groups: Vec<_> = plan
        .groups
        .iter()
        .map(|g| {
            (
                &g.group_id,
                &g.members,
                &g.split,
                g.reasons
                    .iter()
                    .filter(|reason| reason.code != "protected_match")
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    if plan.grouping_id != screening_hash("screening-grouping-splits-v1", &(groups, &plan.edges))? {
        return Err(integrity("screening grouping identity mismatch"));
    }
    if let Some(previous) = &plan.previous {
        plan_identity(previous, depth + 1)?;
    }
    Ok(())
}

pub(crate) fn validate(artifact: &ExportArtifact, rows: &[crate::export::Projected]) -> Result<()> {
    match (artifact.metadata_version, &artifact.screening) {
        (1, None) => return Ok(()),
        (3, Some(_)) => {}
        _ => return Err(integrity("unsupported export metadata shape/version")),
    }
    let witness = artifact.screening.as_ref().expect("screened branch");
    let plan = &witness.plan;
    let declaration = &plan.declaration;
    plan_identity(plan, 0)?;
    if witness.version != 2
        || !matches!(
            artifact.manifest.column_schema_version,
            ExportSchemaVersion::ReviewedTasks | ExportSchemaVersion::RecordOrigins
        )
        || artifact.scope
            != (ExportScope::Run {
                run_id: declaration.output.run_id.clone(),
            })
        || artifact.manifest.target != declaration.policy.target
        || artifact.manifest.cot_policy != declaration.policy.cot_policy
        || artifact.manifest.multi_turn_loss != declaration.policy.multi_turn_loss
        || witness.layout != layout(declaration.policy.multi_turn_loss)
        || plan.lexical_status == LexicalScreeningStatus::Incomplete
        || !plan.incomplete.is_empty()
        || plan
            .protected_inputs
            .iter()
            .any(|input| !input.complete || !input.rights_declared)
        || witness.population_id
            != screening_hash(
                "screened-publication-population-v2",
                &(&declaration.runs, &plan.population),
            )?
    {
        return Err(integrity(
            "unsupported or incomplete screened qualification/policy",
        ));
    }
    if declaration.runs.run_ids.is_empty()
        || !ordered(&declaration.runs.run_ids)
        || declaration
            .runs
            .run_ids
            .iter()
            .any(|id| id.trim().is_empty())
        || !declaration
            .runs
            .run_ids
            .contains(&declaration.output.run_id)
        || !ordered(&declaration.output.record_ids)
        || !ordered(&plan.eligible_output)
    {
        return Err(integrity("screened declaration ordering or scope mismatch"));
    }
    let population: Vec<_> = plan.population.iter().map(|b| &b.record).collect();
    if population.is_empty()
        || !ordered(&population)
        || population.iter().any(|id| {
            id.record_id.trim().is_empty() || !declaration.runs.run_ids.contains(&id.run_id)
        })
    {
        return Err(integrity(
            "screened population is empty, duplicated, or outside scope",
        ));
    }
    let mut grouped = BTreeMap::new();
    let mut group_ids = BTreeSet::new();
    for group in &plan.groups {
        if group.members.is_empty()
            || !ordered(&group.members)
            || !group_ids.insert(&group.group_id)
            || group.quarantined != !group.reasons.is_empty()
        {
            return Err(integrity("invalid screened component shape"));
        }
        for id in &group.members {
            if grouped.insert(id, group).is_some() {
                return Err(integrity("duplicated screened component member"));
            }
        }
    }
    if grouped.keys().copied().collect::<Vec<_>>() != population {
        return Err(integrity("screened component population mismatch"));
    }
    let eligible: Vec<_> = witness
        .members
        .iter()
        .map(|member| &member.record)
        .collect();
    if eligible != plan.eligible_output.iter().collect::<Vec<_>>()
        || witness.members.len() != rows.len()
    {
        return Err(integrity("screened output membership mismatch"));
    }
    for (member, row) in witness.members.iter().zip(rows) {
        let group = grouped
            .get(&member.record)
            .ok_or_else(|| integrity("screened selected member missing"))?;
        if member.record.run_id != declaration.output.run_id
            || member.record.record_id != row.record_id
            || !declaration
                .output
                .record_ids
                .contains(&member.record.record_id)
            || member.component_id != group.group_id
            || group.quarantined
            || group.split.as_ref().map(|split| split.role) != Some(TaskSplitRole::Train)
        {
            return Err(integrity("screened row/component mismatch"));
        }
        let binding = plan
            .population
            .iter()
            .find(|binding| binding.record == member.record)
            .ok_or_else(|| integrity("screened row input binding missing"))?;
        let origin = row
            .origin_json
            .as_ref()
            .map(|json| serde_json::from_str::<ExportRecordOrigin>(json))
            .transpose()?;
        let expected_verdict = match origin {
            Some(ExportRecordOrigin::ReviewedReference(_)) => None,
            _ => Some("admit"),
        };
        if binding.record_hash != row.record_hash || row.verdict.as_deref() != expected_verdict {
            return Err(integrity(
                "screened row record hash or admitted verdict contradicts its source binding",
            ));
        }
        let messages: Vec<Message> = serde_json::from_str(&row.messages_json)?;
        let shape = classify_screening_training_source(
            &messages,
            None,
            declaration.policy.target,
            declaration.policy.multi_turn_loss,
        );
        if !shape.unsupported_reasons.is_empty()
            || !shape
                .required_fields
                .iter()
                .all(|field| plan.required_fields.contains(field))
        {
            return Err(integrity(
                "screened row has unsupported source shape or unclaimed field requirements",
            ));
        }
        let task: ExportTaskProjection = serde_json::from_str(
            row.task_json
                .as_deref()
                .ok_or_else(|| integrity("screened row lacks its captured task declaration"))?,
        )?;
        let projected = crate::screening_binding::screening_projection_id(
            &row.training_area,
            &messages,
            Some(&task.provenance),
            Some(&task.verification_contract),
            row.verdict
                .as_ref()
                .map(|v| serde_json::from_value(serde_json::Value::String(v.clone())))
                .transpose()?,
            row.origin_json
                .as_ref()
                .map(|json| serde_json::from_str::<ExportRecordOrigin>(json))
                .transpose()?
                .as_ref(),
        )?;
        if projected != binding.export_projection_id
            || group.split.as_ref() != Some(&task.provenance.split)
            || row.prompt_hash != crate::prompt_hash(&messages)?
        {
            return Err(integrity(
                "screened row projection contradicts its captured source semantics",
            ));
        }
    }
    let excluded: Vec<_> = plan.exclusions.iter().map(|e| &e.record).collect();
    let requested: BTreeSet<_> = declaration
        .output
        .record_ids
        .iter()
        .map(|id| ScreeningRecordId {
            run_id: declaration.output.run_id.clone(),
            record_id: id.clone(),
        })
        .collect();
    let actual: BTreeSet<_> = eligible
        .iter()
        .chain(&excluded)
        .map(|id| (*id).clone())
        .collect();
    let counts = &plan.counts;
    if !ordered(&excluded)
        || eligible.iter().any(|id| excluded.contains(id))
        || actual != requested
        || plan.exclusions.iter().any(|e| e.reasons.is_empty())
        || counts.population_records != population.len() as u64
        || counts.groups != plan.groups.len() as u64
        || counts.quarantined_groups != plan.groups.iter().filter(|g| g.quarantined).count() as u64
        || counts.requested_output_records != requested.len() as u64
        || counts.eligible_output_records != eligible.len() as u64
        || counts.excluded_output_records != excluded.len() as u64
        || counts.supplied_protected_sets != plan.protected_inputs.len() as u64
        || counts.protected_matches != plan.protected_matches.len() as u64
        || artifact.manifest.n_records
            != population
                .iter()
                .filter(|id| id.run_id == declaration.output.run_id)
                .count() as u64
        || artifact.manifest.n_admitted != eligible.len() as u64
    {
        return Err(integrity("screened counts or exclusion partition mismatch"));
    }
    Ok(())
}
