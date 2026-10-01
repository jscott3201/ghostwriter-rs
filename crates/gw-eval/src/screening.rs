//! Provider-free grouping and lexical screening of an operator-declared finite supplied corpus.
//! Parsing a plan never validates it; revalidation uses actual records and protected contents.
use crate::{
    screening_groups, screening_intake as intake, screening_lexical::TextIndex,
    screening_projection as projection,
};
use gw_schema::*;
use std::collections::{BTreeMap, BTreeSet};

/// Malformed declarations or stale frozen inputs, distinct from incomplete coverage in a report.
#[derive(Debug, thiserror::Error)]
#[error("screening input is invalid: {0}")]
pub struct ScreeningError(pub String);

fn same_bytes<T: serde::Serialize>(left: &T, right: &T) -> Result<bool, ScreeningError> {
    let encode =
        |value: &T| serde_json::to_vec(value).map_err(|error| ScreeningError(error.to_string()));
    Ok(encode(left)? == encode(right)?)
}

/// Prepare a deterministic report over every supplied record in the declared runs.
/// Records outside the run set are outside this bounded claim. No database or provider is opened.
///
/// # Errors
/// Rejects invalid declarations, duplicate identities, contradictory digests or stale predecessors.
pub fn prepare_screening(
    records: &[TrainingRecord],
    declaration: &ScreeningDeclaration,
    protected: &[ProtectedScreeningSet],
    previous: Option<&FrozenScreeningPlan>,
) -> Result<FrozenScreeningPlan, ScreeningError> {
    prepare(records, declaration, protected, previous, 0)
}

fn prepare(
    records: &[TrainingRecord],
    declaration: &ScreeningDeclaration,
    protected: &[ProtectedScreeningSet],
    previous: Option<&FrozenScreeningPlan>,
    depth: u32,
) -> Result<FrozenScreeningPlan, ScreeningError> {
    if depth >= 16 {
        return Err(ScreeningError(
            "screening predecessor chain exceeds 16 plans".into(),
        ));
    }
    let declaration = intake::canonical_declaration(declaration)?;
    let policy = &declaration.policy;
    let records = intake::population(records, &declaration)?;
    let protected = intake::canonical_protected(protected)?;
    let bindings = intake::bindings(&records, policy)?;
    if let Some(previous) = previous {
        if !same_bytes(&previous.declaration.policy, policy)? {
            return Err(ScreeningError(
                "stable extension requires identical policies".into(),
            ));
        }
        let old: BTreeSet<_> = previous
            .population
            .iter()
            .map(|binding| &binding.record)
            .collect();
        let retained: Vec<_> = records
            .iter()
            .filter(|record| old.contains(&intake::key(record)))
            .map(|record| (*record).clone())
            .collect();
        if retained.len() != old.len() {
            return Err(ScreeningError(
                "previous population member missing from extension".into(),
            ));
        }
        let actual = prepare(
            &retained,
            &previous.declaration,
            &protected,
            previous.previous.as_deref(),
            depth + 1,
        )?;
        if !same_bytes(&actual, previous)? {
            return Err(ScreeningError(
                "stale or altered previous screening plan".into(),
            ));
        }
    }
    let mut incomplete = vec![];
    if records.is_empty() {
        incomplete.push(intake::issue(
            "empty_screening_population",
            "declared corpus",
        ));
    }
    let mut edges = intake::declared_edges(&records, &declaration, &mut incomplete);
    let mut index = TextIndex::default();
    let mut projections = vec![];
    let mut protected_projections = vec![];
    let mut bounded = true;
    let classifications: Vec<_> = records
        .iter()
        .map(|record| {
            classify_screening_training_source(
                &record.messages,
                record.tools.as_deref(),
                policy.target,
                policy.multi_turn_loss,
            )
        })
        .collect();
    let fields: BTreeSet<_> = classifications
        .iter()
        .flat_map(|shape| shape.required_fields.iter().copied())
        .collect();
    for (record, classification) in records.iter().zip(classifications) {
        let owner = format!(
            "record/{}",
            intake::hash("screening-record-coordinate-v1", &intake::key(record))?
        );
        match projection::project_record(record, &owner, policy, classification, &mut index) {
            Ok(projected) => {
                for reason in &projected.unsupported {
                    incomplete.push(intake::issue(reason, intake::subject(record)));
                }
                projections.push(projected);
            }
            Err(reason) => {
                incomplete.push(intake::issue(reason, intake::subject(record)));
                bounded = false;
                break;
            }
        }
    }
    let mut protected_inputs =
        intake::protected_coverage(&protected, policy, &fields, &mut incomplete)?;
    if bounded {
        'sets: for set in &protected {
            for item in &set.items {
                let owner = format!(
                    "protected/{}",
                    intake::hash(
                        "screening-protected-coordinate-v1",
                        &(&set.canonical_id, &item.item_id)
                    )?
                );
                match projection::project_protected(
                    &set.canonical_id,
                    item,
                    &owner,
                    policy,
                    &mut index,
                ) {
                    Ok(projected) => {
                        for reason in &projected.unsupported {
                            incomplete.push(intake::issue(reason, set.canonical_id.clone()));
                        }
                        if !projected.unsupported.is_empty() {
                            protected_inputs
                                .iter_mut()
                                .find(|input| input.canonical_id == set.canonical_id)
                                .expect("captured set")
                                .complete = false;
                        }
                        protected_projections.push(projected);
                    }
                    Err(reason) => {
                        incomplete.push(intake::issue(reason, set.canonical_id.clone()));
                        bounded = false;
                        break 'sets;
                    }
                }
            }
        }
    }
    let mut matches = vec![];
    if bounded {
        let result = index
            .build(policy)
            .and_then(|()| {
                screening_groups::lexical_edges(
                    &records,
                    &projections,
                    &mut index,
                    policy,
                    &mut edges,
                )
            })
            .and_then(|()| {
                screening_groups::protected_matches(
                    &records,
                    &projections,
                    &protected_projections,
                    &mut index,
                    policy,
                    &mut matches,
                )
            });
        if let Err(reason) = result {
            incomplete.push(intake::issue(reason, "declared corpus"));
            bounded = false;
        }
    }
    if !bounded {
        for input in &mut protected_inputs {
            input.complete = false;
        }
    }
    edges.sort();
    edges.dedup();
    matches.sort();
    matches.dedup();
    if incomplete.is_empty() {
        validate_complete_screening_protected(
            policy,
            &fields.iter().copied().collect::<Vec<_>>(),
            &protected_inputs,
        )
        .map_err(|reason| ScreeningError(reason.into()))?;
    }
    incomplete.sort();
    incomplete.dedup();
    let (groups, grouping_id) = screening_groups::groups(&records, &edges, previous, &matches)?;
    let mut strata = vec![];
    for (i, record) in records.iter().enumerate() {
        let task = record.task_provenance.as_ref();
        strata.push(ScreeningRecordStratum {
            record: intake::key(record),
            task: task.map(|t| t.task_id.clone()),
            area: record.training_area.clone(),
            domain: task.map(|t| t.observations.domain.clone()),
            difficulty: task.map(|t| t.observations.difficulty.label.clone()),
            teacher: record
                .origin
                .generated()
                .map(|g| g.provenance.teacher.slug.clone()),
            tokens: bounded.then(|| {
                projections[i]
                    .segments
                    .iter()
                    .map(|&s| index.segments[s].tokens.len() as u64)
                    .sum()
            }),
        });
    }
    let by_id: BTreeMap<_, _> = records
        .iter()
        .map(|record| (intake::key(record), *record))
        .collect();
    let by_group: BTreeMap<_, _> = groups
        .iter()
        .flat_map(|group| group.members.iter().map(move |id| (id, group)))
        .collect();
    let (mut eligible_output, mut exclusions) = (vec![], vec![]);
    for id in &declaration.output.record_ids {
        let record = ScreeningRecordId {
            run_id: declaration.output.run_id.clone(),
            record_id: id.clone(),
        };
        let mut reasons = vec![];
        if !incomplete.is_empty() {
            reasons.push("incomplete_plan".into());
        }
        match by_id.get(&record) {
            Some(source) => {
                if !gw_storage::is_selected_admitted(source) {
                    reasons.push("not_selected_admitted".into());
                }
                let group = by_group[&record];
                if group.quarantined {
                    reasons.push("quarantined_component".into());
                }
                if group.split.as_ref().map(|split| split.role) != Some(TaskSplitRole::Train) {
                    reasons.push("not_train_component".into());
                }
            }
            None => reasons.push("missing_output_member".into()),
        }
        if reasons.is_empty() {
            eligible_output.push(record);
        } else {
            reasons.sort();
            exclusions.push(ScreeningExclusion { record, reasons });
        }
    }
    let lexical_status = if !incomplete.is_empty() {
        LexicalScreeningStatus::Incomplete
    } else if matches.is_empty() {
        LexicalScreeningStatus::CompleteNoMatch
    } else {
        LexicalScreeningStatus::MatchQuarantined
    };
    let policy_id = intake::hash("screening-policy-v1", policy)?;
    let screening_input_id = intake::hash(
        "screening-captured-population-v2",
        &(&declaration, &bindings),
    )?;
    let protected_input_id = intake::hash(
        "screening-protected-inputs-v1",
        &protected_inputs
            .iter()
            .map(|input| (&input.canonical_id, &input.input_id))
            .collect::<Vec<_>>(),
    )?;
    let mut plan = FrozenScreeningPlan {
        version: SCREENING_PLAN_VERSION,
        counts: ScreeningCounts {
            population_records: bindings.len() as u64,
            groups: groups.len() as u64,
            quarantined_groups: groups.iter().filter(|group| group.quarantined).count() as u64,
            requested_output_records: declaration.output.record_ids.len() as u64,
            eligible_output_records: eligible_output.len() as u64,
            excluded_output_records: exclusions.len() as u64,
            supplied_protected_sets: protected_inputs.len() as u64,
            protected_matches: matches.len() as u64,
        },
        declaration,
        population: bindings,
        required_fields: fields.into_iter().collect(),
        screening_input_id,
        policy_id,
        protected_inputs,
        protected_input_id,
        groups,
        grouping_id,
        edges,
        protected_matches: matches,
        lexical_status,
        semantic_status: SemanticScreeningStatus::NotRun,
        population_check: ScreeningPopulationCheck::SuppliedFilesOnly,
        lexical_scope: ScreeningLexicalScope::CanonicalSourceAndPinnedExportPolicy,
        effective_prompt_separation: EffectivePromptSeparation::Unknown,
        incomplete,
        eligible_output,
        exclusions,
        strata,
        previous: previous.cloned().map(Box::new),
        plan_id: String::new(),
    };
    plan.plan_id = intake::hash("frozen-screening-plan-v2", &plan)?;
    Ok(plan)
}

/// Recompute the entire plan from actual inputs; submitted report fields cannot certify themselves.
///
/// # Errors
/// Rejects any changed population/input or altered report field.
pub fn validate_screening_plan(
    records: &[TrainingRecord],
    protected: &[ProtectedScreeningSet],
    plan: &FrozenScreeningPlan,
) -> Result<(), ScreeningError> {
    let actual = prepare_screening(
        records,
        &plan.declaration,
        protected,
        plan.previous.as_deref(),
    )?;
    if !same_bytes(&actual, plan)? {
        return Err(ScreeningError("stale or altered screening plan".into()));
    }
    Ok(())
}

/// Compute a protected set's typed content digest in canonical item order.
///
/// # Errors
/// Rejects duplicate/empty item IDs or a serialization failure.
pub fn protected_screening_content_digest(
    items: &[gw_schema::ProtectedScreeningItem],
) -> Result<String, ScreeningError> {
    let mut items = items.to_vec();
    items.sort_by(|left, right| left.item_id.cmp(&right.item_id));
    if items.iter().any(|item| item.item_id.trim().is_empty())
        || items
            .windows(2)
            .any(|pair| pair[0].item_id == pair[1].item_id)
    {
        return Err(ScreeningError(
            "duplicate or empty protected item identity".into(),
        ));
    }
    let value = serde_json::json!(["protected-screening-content-v1", items]);
    gw_storage::canonical_json_hash(&value).map_err(|error| ScreeningError(error.to_string()))
}
