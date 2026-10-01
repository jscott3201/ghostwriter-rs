//! Pure supported screening declaration and protected-summary contracts. No text matching or I/O.
use crate::*;
use std::collections::BTreeSet;

fn nonblank(text: &str) -> Result<(), &'static str> {
    if text.trim().is_empty() {
        Err("empty declared identity")
    } else {
        Ok(())
    }
}
fn unique<T: Ord>(values: &mut [T]) -> Result<(), &'static str> {
    values.sort();
    if values.windows(2).any(|pair| pair[0] == pair[1]) {
        Err("duplicate declared identity")
    } else {
        Ok(())
    }
}
fn strings(values: &mut [String]) -> Result<(), &'static str> {
    for value in values.iter() {
        nonblank(value)?;
    }
    unique(values)
}
fn record_ids(values: &mut [ScreeningRecordId]) -> Result<(), &'static str> {
    for value in values.iter() {
        nonblank(&value.run_id)?;
        nonblank(&value.record_id)?;
    }
    unique(values)
}

/// Validate supported v1 domains and return the canonical finite-corpus declaration.
///
/// # Errors
/// Rejects unsupported versions, policy domains/ceilings, empty or duplicate identities, and
/// contradictory run scopes. This checks shape only; the planner checks actual corpus contents.
pub fn canonical_screening_declaration(
    input: &ScreeningDeclaration,
) -> Result<ScreeningDeclaration, &'static str> {
    let mut declaration = input.clone();
    if declaration.version != SCREENING_VERSION {
        return Err("unsupported screening version");
    }
    strings(&mut declaration.runs.run_ids)?;
    if declaration.runs.run_ids.is_empty() {
        return Err("declared run set must be nonempty");
    }
    nonblank(&declaration.output.run_id)?;
    if !declaration
        .runs
        .run_ids
        .contains(&declaration.output.run_id)
    {
        return Err("output run is outside declared corpus");
    }
    strings(&mut declaration.output.record_ids)?;
    let policy = &mut declaration.policy;
    if policy.recipe != LEXICAL_SCREEN_RECIPE
        || policy.ngram[0] == 0
        || policy.ngram[0] > policy.ngram[1]
        || policy.ngram[1] > 65_536
        || policy.min_overlap_tokens == 0
        || policy.min_overlap_tokens > 65_536
        || !policy.jaccard_threshold.is_finite()
        || !(0.0..=1.0).contains(&policy.jaccard_threshold)
    {
        return Err("unsupported or invalid lexical policy");
    }
    let ceilings = ScreeningLimits::default();
    let bounds = [
        (policy.limits.total_text_bytes, ceilings.total_text_bytes),
        (policy.limits.segment_bytes, ceilings.segment_bytes),
        (policy.limits.segment_tokens, ceilings.segment_tokens),
        (policy.limits.segments, ceilings.segments),
        (policy.limits.distinct_shingles, ceilings.distinct_shingles),
        (
            policy.limits.shingle_token_work,
            ceilings.shingle_token_work,
        ),
        (policy.limits.comparisons, ceilings.comparisons),
    ];
    if bounds
        .into_iter()
        .any(|(limit, max)| limit == 0 || limit > max)
    {
        return Err("resource limits exceed supported v1 bounds or are zero");
    }
    strings(&mut policy.additional_protected_sets)?;
    strings(&mut policy.required_languages)?;
    if policy.required_languages.is_empty() {
        return Err("required languages must be explicitly declared");
    }
    for task in &mut declaration.expected_tasks {
        nonblank(&task.namespace)?;
        nonblank(&task.item)?;
        nonblank(&task.revision)?;
        record_ids(&mut task.records)?;
    }
    declaration.expected_tasks.sort_by(|a, b| {
        (&a.namespace, &a.item, &a.revision).cmp(&(&b.namespace, &b.item, &b.revision))
    });
    if declaration.expected_tasks.windows(2).any(|p| {
        (&p[0].namespace, &p[0].item, &p[0].revision)
            == (&p[1].namespace, &p[1].item, &p[1].revision)
    }) {
        return Err("duplicate expected source item revision");
    }
    for group in &mut declaration.siblings {
        nonblank(&group.run_id)?;
        nonblank(&group.sibling_group_id)?;
        record_ids(&mut group.records)?;
        if group.records.iter().any(|r| r.run_id != group.run_id) {
            return Err("sibling declarations cannot cross run identities");
        }
    }
    declaration
        .siblings
        .sort_by(|a, b| (&a.run_id, &a.sibling_group_id).cmp(&(&b.run_id, &b.sibling_group_id)));
    if declaration
        .siblings
        .windows(2)
        .any(|p| (&p[0].run_id, &p[0].sibling_group_id) == (&p[1].run_id, &p[1].sibling_group_id))
    {
        return Err("duplicate sibling declaration");
    }
    Ok(declaration)
}

impl ScreeningPolicy {
    /// Canonically ordered protected union: all required built-ins plus operator additions.
    #[must_use]
    pub fn required_protected_sets(&self) -> Vec<String> {
        CANONICAL_PROTECTED_BENCHMARKS
            .iter()
            .map(|id| (*id).to_owned())
            .chain(self.additional_protected_sets.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

impl ProtectedScreeningCoverage {
    /// Canonicalize the declared language, media and field sets without asserting completeness.
    ///
    /// # Errors
    /// Rejects blank or duplicate declaration members.
    pub fn canonicalized(&self) -> Result<Self, &'static str> {
        let mut coverage = self.clone();
        strings(&mut coverage.languages)?;
        strings(&mut coverage.media)?;
        unique(&mut coverage.fields)?;
        Ok(coverage)
    }
}

fn ordered<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}
fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Validate the text-free summaries required by a complete source-screening result.
///
/// Both the planner and artifact verifier use this supported-shape check. It checks canonical IDs,
/// the required protected union, nonempty item/content identities, and declared complete text/field/
/// language coverage. It does not recreate protected text or independently authenticate its claims.
///
/// # Errors
/// Rejects a missing required set, unsupported/contradictory coverage, or malformed summaries.
pub fn validate_complete_screening_protected(
    policy: &ScreeningPolicy,
    required_fields: &[ScreeningField],
    inputs: &[ProtectedScreeningIdentity],
) -> Result<(), &'static str> {
    if !ordered(required_fields) || !required_fields.contains(&ScreeningField::Content) {
        return Err(
            "complete source field requirements must be canonical and include task content",
        );
    }
    let ids: Vec<_> = inputs.iter().map(|input| &input.canonical_id).collect();
    if !ordered(&ids)
        || policy
            .required_protected_sets()
            .iter()
            .any(|id| !ids.contains(&id))
    {
        return Err("protected summaries must canonically contain the complete required union");
    }
    for input in inputs {
        nonblank(&input.canonical_id)?;
        nonblank(&input.source_revision)?;
        if !input.complete
            || !input.rights_declared
            || !digest(&input.content_digest)
            || !digest(&input.input_id)
            || input.item_ids.is_empty()
            || !ordered(&input.item_ids)
            || input.item_ids.iter().any(|id| id.trim().is_empty())
        {
            return Err(
                "complete protected summaries require nonempty canonical content and item identities",
            );
        }
        let coverage = &input.coverage;
        if coverage.canonicalized()? != *coverage
            || !coverage.complete
            || coverage.media != ["text"]
            || coverage.languages.is_empty()
            || !policy
                .required_languages
                .iter()
                .all(|language| coverage.languages.contains(language))
            || !required_fields
                .iter()
                .all(|field| coverage.fields.contains(field))
        {
            return Err("unsupported or incomplete protected coverage summary");
        }
    }
    Ok(())
}
