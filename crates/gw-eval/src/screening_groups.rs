//! Selected-unit grouping and protected matches have separate graph effects.
use crate::{
    screening::ScreeningError,
    screening_intake::{hash, issue, key},
    screening_lexical::TextIndex,
    screening_projection::{PromptUnit, ProtectedProjection, RecordProjection},
};
use gw_schema::*;
use std::collections::{BTreeMap, BTreeSet};

fn exact_prompt(left: &PromptUnit, right: &PromptUnit, index: &TextIndex) -> bool {
    left.has_user_text
        && right.has_user_text
        && left.shape == right.shape
        && left.segments.len() == right.segments.len()
        && left
            .segments
            .iter()
            .zip(&right.segments)
            .all(|(a, b)| index.segments[*a].tokens == index.segments[*b].tokens)
}
fn whole_evidence(left: &PromptUnit, right: &PromptUnit) -> LexicalScreeningEvidence {
    LexicalScreeningEvidence {
        left: left.id.clone(),
        right: right.id.clone(),
        n: None,
        intersection: 0,
        union: 0,
    }
}
fn lexical_prompt(
    left: &PromptUnit,
    right: &PromptUnit,
    index: &mut TextIndex,
    policy: &ScreeningPolicy,
) -> Result<Option<Vec<LexicalScreeningEvidence>>, &'static str> {
    if !left.has_user_text
        || !right.has_user_text
        || left.shape != right.shape
        || left.segments.len() != right.segments.len()
    {
        return Ok(None);
    }
    let mut evidence = vec![];
    for (&a, &b) in left.segments.iter().zip(&right.segments) {
        if index.segments[a].tokens == index.segments[b].tokens {
            continue;
        }
        let matches = index.matches(a, b, policy)?;
        if matches.is_empty() {
            return Ok(None);
        }
        evidence.extend(matches);
    }
    Ok(Some(evidence))
}

pub(crate) fn lexical_edges(
    records: &[&TrainingRecord],
    projections: &[RecordProjection],
    index: &mut TextIndex,
    policy: &ScreeningPolicy,
    edges: &mut Vec<ScreeningGroupEdge>,
) -> Result<(), &'static str> {
    // Rows without emitted units remain captured and incomplete, but must not cause uncharged
    // quadratic scans of empty unit arrays.
    let active: Vec<_> = (0..records.len())
        .filter(|&i| !projections[i].units.is_empty())
        .collect();
    for (position, &left) in active.iter().enumerate() {
        for &right in &active[position + 1..] {
            for a in &projections[left].units {
                for b in &projections[right].units {
                    index.comparison(policy)?;
                    let mut add = |kind: &str, evidence: Vec<_>| {
                        edges.push(ScreeningGroupEdge {
                            left: key(records[left]),
                            right: key(records[right]),
                            kind: kind.into(),
                            evidence,
                        })
                    };
                    if a.prompt.has_user_text && b.prompt.has_user_text && a.rendered == b.rendered
                    {
                        add(
                            "exact_training_example",
                            vec![whole_evidence(&a.prompt, &b.prompt)],
                        );
                    }
                    if exact_prompt(&a.prompt, &b.prompt, index) {
                        add(
                            "exact_complete_prompt",
                            vec![whole_evidence(&a.prompt, &b.prompt)],
                        );
                    } else if let Some(evidence) =
                        lexical_prompt(&a.prompt, &b.prompt, index, policy)?
                    {
                        add("lexical_complete_prompt", evidence);
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn protected_matches(
    records: &[&TrainingRecord],
    projections: &[RecordProjection],
    protected: &[ProtectedProjection],
    index: &mut TextIndex,
    policy: &ScreeningPolicy,
    matches: &mut Vec<ProtectedScreeningMatch>,
) -> Result<(), &'static str> {
    for (record, projection) in records.iter().zip(projections) {
        if projection.units.is_empty() && projection.segments.is_empty() {
            continue;
        }
        for item in protected
            .iter()
            .filter(|item| item.prompt.has_user_text || !item.segments.is_empty())
        {
            let mut add = |evidence| {
                matches.push(ProtectedScreeningMatch {
                    record: key(record),
                    protected_set: item.set_id.clone(),
                    item_id: item.item_id.clone(),
                    evidence,
                })
            };
            for unit in &projection.units {
                index.comparison(policy)?;
                if exact_prompt(&unit.prompt, &item.prompt, index) {
                    add(whole_evidence(&unit.prompt, &item.prompt));
                }
            }
            for &left in &projection.segments {
                for &right in &item.segments {
                    for evidence in index.matches(left, right, policy)? {
                        add(evidence);
                    }
                }
            }
        }
    }
    Ok(())
}

fn root(parents: &mut [usize], mut node: usize) -> usize {
    while parents[node] != node {
        parents[node] = parents[parents[node]];
        node = parents[node];
    }
    node
}

pub(crate) fn groups(
    records: &[&TrainingRecord],
    edges: &[ScreeningGroupEdge],
    previous: Option<&FrozenScreeningPlan>,
    matches: &[ProtectedScreeningMatch],
) -> Result<(Vec<FrozenScreeningGroup>, String), ScreeningError> {
    let positions: BTreeMap<_, _> = records
        .iter()
        .enumerate()
        .map(|(i, r)| (key(r), i))
        .collect();
    let mut parents: Vec<_> = (0..records.len()).collect();
    for edge in edges {
        let a = root(&mut parents, positions[&edge.left]);
        let b = root(&mut parents, positions[&edge.right]);
        if a != b {
            parents[b] = a;
        }
    }
    let mut components = BTreeMap::<usize, Vec<usize>>::new();
    for i in 0..records.len() {
        let leader = root(&mut parents, i);
        components.entry(leader).or_default().push(i);
    }
    let old: BTreeMap<_, _> = previous
        .into_iter()
        .flat_map(|plan| &plan.groups)
        .flat_map(|group| group.members.iter().map(move |id| (id, &group.group_id)))
        .collect();
    let protected: BTreeSet<_> = matches.iter().map(|m| &m.record).collect();
    let mut groups = vec![];
    let mut identity_groups = vec![];
    for members in components.into_values() {
        let ids: Vec<_> = members.iter().map(|&i| key(records[i])).collect();
        let old_ids: BTreeSet<_> = ids.iter().filter_map(|id| old.get(id).copied()).collect();
        let group_id = match old_ids.len() {
            0 => hash("screening-group-anchor-v1", &ids[0])?,
            1 => (*old_ids.first().expect("one previous group")).clone(),
            _ => hash("screening-joined-prior-groups-v1", &old_ids)?,
        };
        let mut reasons = vec![];
        if old_ids.len() > 1
            || previous.is_some_and(|plan| {
                plan.groups.iter().any(|group| {
                    old_ids.contains(&group.group_id)
                        && group
                            .reasons
                            .iter()
                            .any(|reason| reason.code == "joined_prior_groups")
                })
            })
        {
            reasons.push(issue("joined_prior_groups", group_id.clone()));
        }
        let declared: Vec<_> = members
            .iter()
            .filter_map(|&i| records[i].task_provenance.as_ref().map(|task| &task.split))
            .collect();
        let conflict = declared
            .first()
            .is_some_and(|first| declared.iter().any(|split| split != first));
        let split = if conflict {
            reasons.push(issue("conflicting_split_assignments", group_id.clone()));
            None
        } else if declared.len() == members.len() {
            declared.first().map(|split| (*split).clone())
        } else {
            None
        };
        identity_groups.push((
            group_id.clone(),
            ids.clone(),
            split.clone(),
            reasons.clone(),
        ));
        if ids.iter().any(|id| protected.contains(id)) {
            reasons.push(issue("protected_match", group_id.clone()));
        }
        reasons.sort();
        groups.push(FrozenScreeningGroup {
            group_id,
            members: ids,
            split,
            quarantined: !reasons.is_empty(),
            reasons,
        });
    }
    groups.sort_by(|a, b| a.group_id.cmp(&b.group_id));
    identity_groups.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((
        groups,
        hash("screening-grouping-splits-v1", &(identity_groups, edges))?,
    ))
}
