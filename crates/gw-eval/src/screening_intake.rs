//! Canonical supplied population/declaration intake, lineage closure and protected coverage.
use crate::screening::{ScreeningError, protected_screening_content_digest};
use gw_schema::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn hash(domain: &str, value: &impl Serialize) -> Result<String, ScreeningError> {
    let value =
        serde_json::to_value((domain, value)).map_err(|error| ScreeningError(error.to_string()))?;
    gw_storage::canonical_json_hash(&value).map_err(|error| ScreeningError(error.to_string()))
}
pub(crate) fn key(record: &TrainingRecord) -> ScreeningRecordId {
    ScreeningRecordId {
        run_id: record.provenance.run_id.clone(),
        record_id: record.record_id.clone(),
    }
}
pub(crate) fn issue(code: &str, subject: impl Into<String>) -> ScreeningIssue {
    ScreeningIssue {
        code: code.into(),
        subject: subject.into(),
    }
}
pub(crate) fn subject(record: &TrainingRecord) -> String {
    serde_json::to_string(&key(record)).expect("string record identity")
}
fn invalid(reason: &str) -> ScreeningError {
    ScreeningError(reason.into())
}
fn nonblank(text: &str) -> Result<(), ScreeningError> {
    if text.trim().is_empty() {
        return Err(invalid("empty declared identity"));
    }
    Ok(())
}
fn unique<T: Ord>(values: &mut [T]) -> Result<(), ScreeningError> {
    values.sort();
    if values.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid("duplicate declared identity"));
    }
    Ok(())
}
pub(crate) fn canonical_declaration(
    input: &ScreeningDeclaration,
) -> Result<ScreeningDeclaration, ScreeningError> {
    canonical_screening_declaration(input).map_err(invalid)
}

pub(crate) fn population<'a>(
    records: &'a [TrainingRecord],
    declaration: &ScreeningDeclaration,
) -> Result<Vec<&'a TrainingRecord>, ScreeningError> {
    let runs: BTreeSet<_> = declaration.runs.run_ids.iter().collect();
    let mut records: Vec<_> = records
        .iter()
        .filter(|r| runs.contains(&r.provenance.run_id))
        .collect();
    records.sort_by_key(|r| key(r));
    for record in &records {
        nonblank(&record.record_id)?;
    }
    if records.windows(2).any(|r| key(r[0]) == key(r[1])) {
        return Err(invalid("duplicate supplied population record"));
    }
    Ok(records)
}

pub(crate) fn bindings(
    records: &[&TrainingRecord],
    policy: &ScreeningPolicy,
) -> Result<Vec<ScreeningInputBinding>, ScreeningError> {
    records
        .iter()
        .map(|record| {
            gw_storage::capture_screening_input(record, policy)
                .map_err(|error| ScreeningError(error.to_string()))
        })
        .collect()
}

fn edge(
    edges: &mut Vec<ScreeningGroupEdge>,
    left: &TrainingRecord,
    right: &TrainingRecord,
    kind: &str,
) {
    let (mut left, mut right) = (key(left), key(right));
    if left == right {
        return;
    }
    if left > right {
        std::mem::swap(&mut left, &mut right);
    }
    edges.push(ScreeningGroupEdge {
        left,
        right,
        kind: kind.into(),
        evidence: vec![],
    });
}

pub(crate) fn declared_edges(
    records: &[&TrainingRecord],
    declaration: &ScreeningDeclaration,
    incomplete: &mut Vec<ScreeningIssue>,
) -> Vec<ScreeningGroupEdge> {
    let mut edges = vec![];
    let by_key: BTreeMap<_, _> = records.iter().map(|r| (key(r), *r)).collect();
    let mut by_id = BTreeMap::<&str, Vec<&TrainingRecord>>::new();
    let mut source_groups: BTreeMap<_, &TrainingRecord> = BTreeMap::new();
    let mut task_groups: BTreeMap<_, &TrainingRecord> = BTreeMap::new();
    let mut source_members = BTreeMap::<_, Vec<_>>::new();
    let mut sibling_members = BTreeMap::<_, Vec<_>>::new();
    for record in records {
        by_id.entry(&record.record_id).or_default().push(*record);
        let id = subject(record);
        match (
            &record.task_provenance,
            &record.verification_contract,
            record.messages.first(),
        ) {
            (Some(task), Some(contract), Some(prompt)) => {
                if task.validate_for(prompt, contract).is_err() {
                    incomplete.push(issue("invalid_task_declaration", id.clone()));
                }
                let source = (&task.source.namespace, &task.source.item);
                if let Some(first) = source_groups.get(&source) {
                    edge(&mut edges, first, record, "source_item_revision");
                } else {
                    source_groups.insert(source, *record);
                }
                let group = (&task.group.namespace, &task.group.id);
                if let Some(first) = task_groups.get(&group) {
                    edge(&mut edges, first, record, "declared_task_group");
                } else {
                    task_groups.insert(group, *record);
                }
                source_members
                    .entry((
                        &task.source.namespace,
                        &task.source.item,
                        &task.source.revision,
                    ))
                    .or_default()
                    .push(key(record));
            }
            _ => incomplete.push(issue("missing_task_evidence", id.clone())),
        }
        if let (Some(group), Some(count), Some(index)) = (
            &record.generation.sibling_group_id,
            record.generation.n_completions,
            record.generation.completion_index,
        ) {
            if count == 0
                || index >= count
                || gw_storage::prompt_hash(&record.messages).ok().as_ref() != Some(group)
            {
                incomplete.push(issue("invalid_sibling_identity", id));
            }
            sibling_members
                .entry((&record.provenance.run_id, group))
                .or_default()
                .push(*record);
        } else {
            incomplete.push(issue("missing_sibling_evidence", id));
        }
    }
    for task in &declaration.expected_tasks {
        let id = format!("{}/{}/{}", task.namespace, task.item, task.revision);
        let actual = source_members
            .remove(&(&task.namespace, &task.item, &task.revision))
            .unwrap_or_default();
        if actual != task.records || task.records.is_empty() {
            incomplete.push(issue("expected_task_membership", id));
        }
    }
    for ((namespace, item, revision), _) in source_members {
        incomplete.push(issue(
            "undeclared_source_item",
            format!("{namespace}/{item}/{revision}"),
        ));
    }
    for declared in &declaration.siblings {
        let members = sibling_members
            .remove(&(&declared.run_id, &declared.sibling_group_id))
            .unwrap_or_default();
        let actual: Vec<_> = members.iter().map(|r| key(r)).collect();
        let label = format!("{}/{}", declared.run_id, declared.sibling_group_id);
        if actual != declared.records || members.is_empty() {
            incomplete.push(issue("expected_sibling_membership", label.clone()));
        }
        if let Some(first) = members.first() {
            let count = first.generation.n_completions.unwrap_or(0);
            let indices: BTreeSet<_> = members
                .iter()
                .filter_map(|r| r.generation.completion_index)
                .collect();
            let prefix = first.messages.split_last().map(|(_, prefix)| prefix);
            if count as usize != members.len()
                || indices.len() != members.len()
                || members.iter().any(|r| {
                    r.generation.n_completions != Some(count)
                        || r.messages.split_last().map(|(_, prefix)| prefix) != prefix
                })
            {
                incomplete.push(issue("incomplete_or_inconsistent_siblings", label));
            }
            for member in members.iter().skip(1) {
                edge(&mut edges, first, member, "declared_sibling");
            }
        }
    }
    for ((run, group), _) in sibling_members {
        incomplete.push(issue("undeclared_sibling_set", format!("{run}/{group}")));
    }
    let mut parent_edges = BTreeMap::<ScreeningRecordId, Vec<ScreeningRecordId>>::new();
    for record in records {
        for parent in &record.provenance.parent_ids {
            match by_id.get(parent.as_str()).map(Vec::as_slice) {
                Some([found]) if key(found) != key(record) => {
                    edge(&mut edges, found, record, "declared_parent");
                    parent_edges
                        .entry(key(record))
                        .or_default()
                        .push(key(found));
                }
                _ => incomplete.push(issue("missing_or_ambiguous_parent", subject(record))),
            }
        }
    }
    // Iterative topological reduction avoids recursive traversal of supplied lineage chains.
    let mut pending: BTreeMap<_, _> = by_key
        .keys()
        .map(|id| (id.clone(), parent_edges.get(id).map_or(0, Vec::len)))
        .collect();
    let mut children = BTreeMap::<_, Vec<_>>::new();
    for (child, parents) in &parent_edges {
        for parent in parents {
            children.entry(parent).or_default().push(child);
        }
    }
    let mut ready: Vec<_> = pending
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut visited = 0;
    while let Some(id) = ready.pop() {
        visited += 1;
        for child in children.get(&id).into_iter().flatten() {
            let count = pending.get_mut(*child).expect("captured lineage member");
            *count -= 1;
            if *count == 0 {
                ready.push((*child).clone());
            }
        }
    }
    if visited != records.len() {
        incomplete.push(issue("cyclic_parent_lineage", "declared corpus"));
    }
    for id in &declaration.output.record_ids {
        if !by_key.contains_key(&ScreeningRecordId {
            run_id: declaration.output.run_id.clone(),
            record_id: id.clone(),
        }) {
            incomplete.push(issue("missing_output_member", id.clone()));
        }
    }
    edges.sort();
    edges.dedup();
    edges
}

pub(crate) fn canonical_protected(
    input: &[ProtectedScreeningSet],
) -> Result<Vec<ProtectedScreeningSet>, ScreeningError> {
    let mut sets = input.to_vec();
    sets.sort_by(|a, b| a.canonical_id.cmp(&b.canonical_id));
    if sets
        .windows(2)
        .any(|p| p[0].canonical_id == p[1].canonical_id)
    {
        return Err(invalid("duplicate protected-set identity"));
    }
    for set in &mut sets {
        nonblank(&set.canonical_id)?;
        nonblank(&set.source_revision)?;
        if set.version != SCREENING_VERSION || set.normalization != LEXICAL_SCREEN_RECIPE {
            return Err(invalid(
                "unsupported protected manifest/normalization version",
            ));
        }
        set.coverage = set.coverage.canonicalized().map_err(invalid)?;
        set.items.sort_by(|a, b| a.item_id.cmp(&b.item_id));
        if protected_screening_content_digest(&set.items)? != set.content_digest {
            return Err(invalid(
                "protected content digest disagrees with supplied items",
            ));
        }
        for item in &set.items {
            nonblank(&item.language)?;
        }
        if let Some(rights) = &mut set.rights {
            unique(&mut rights.evidence)?;
        }
    }
    Ok(sets)
}

pub(crate) fn protected_coverage(
    sets: &[ProtectedScreeningSet],
    policy: &ScreeningPolicy,
    fields: &BTreeSet<ScreeningField>,
    incomplete: &mut Vec<ScreeningIssue>,
) -> Result<Vec<ProtectedScreeningIdentity>, ScreeningError> {
    let required = policy.required_protected_sets();
    for id in &required {
        if !sets.iter().any(|set| &set.canonical_id == id) {
            incomplete.push(issue("missing_protected_set", id.clone()));
        }
    }
    let mut result = vec![];
    for set in sets {
        let start = incomplete.len();
        let rights_declared = set.rights.as_ref().is_some_and(|r| {
            r.screening_permitted
                && !r.reviewer.trim().is_empty()
                && !r.evidence.is_empty()
                && r.evidence.iter().all(|e| !e.trim().is_empty())
        });
        if !rights_declared {
            incomplete.push(issue(
                "unresolved_protected_rights",
                set.canonical_id.clone(),
            ));
        }
        if !set.coverage.complete || set.items.is_empty() {
            incomplete.push(issue(
                "incomplete_protected_contents",
                set.canonical_id.clone(),
            ));
        }
        if !policy
            .required_languages
            .iter()
            .all(|lang| set.coverage.languages.contains(lang))
            || set
                .items
                .iter()
                .any(|item| !set.coverage.languages.contains(&item.language))
            || set.coverage.media != ["text"]
            || !fields
                .iter()
                .all(|field| set.coverage.fields.contains(field))
        {
            incomplete.push(issue(
                "incomplete_protected_coverage",
                set.canonical_id.clone(),
            ));
        }
        result.push(ProtectedScreeningIdentity {
            canonical_id: set.canonical_id.clone(),
            source_revision: set.source_revision.clone(),
            content_digest: set.content_digest.clone(),
            input_id: hash("protected-screening-input-v1", set)?,
            complete: start == incomplete.len(),
            coverage: set.coverage.clone(),
            rights_declared,
            item_ids: set.items.iter().map(|item| item.item_id.clone()).collect(),
        });
    }
    Ok(result)
}
