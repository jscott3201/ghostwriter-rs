//! Exact evidence joins and the independent prompt-level comparison. No I/O or provider calls.

use std::collections::{BTreeMap, BTreeSet};

use gw_schema::TrainingRecord;

use crate::outcomes::*;

pub(crate) fn analyze_outcomes(
    records: &[TrainingRecord],
    evidence: Option<&OutcomeEvidence>,
    cfg: &OutcomeConfig,
) -> OutcomeReport {
    let mut report = empty_report(records.len(), evidence, cfg);
    if report.confidence_level.is_none() {
        invalid(
            &mut report,
            OutcomeReason::InvalidConfiguration {
                field: "confidence_level".into(),
            },
        );
    }
    if cfg.min_evaluated_prompts == 0 {
        invalid(
            &mut report,
            OutcomeReason::InvalidConfiguration {
                field: "min_evaluated_prompts".into(),
            },
        );
    }
    let Some(evidence) = evidence else {
        report.reasons.push(OutcomeReason::MissingEvidence);
        return report;
    };
    validate_contract(evidence, &mut report);
    let declared = declared_members(evidence, &mut report);
    let matched = match_records(records, evidence, &declared, &mut report);
    let values = match_outcomes(evidence, &declared, &mut report);
    let groups = eligible_groups(&matched, &mut report);
    report.coverage.eligible_prompts = groups.len();

    // Any missing/unknown member blocks the whole declared corpus. In particular, no random
    // comparison is computed over a convenient subset of the candidates with known labels.
    if report.status == OutcomeStatus::InvalidEvidence
        || report.coverage.missing_outcomes > 0
        || report.coverage.unknown_outcomes > 0
    {
        return report;
    }
    if groups.is_empty() {
        report.reasons.push(OutcomeReason::NoEligiblePrompts);
        return report;
    }

    let mut selected_values = Vec::with_capacity(groups.len());
    let mut random_values = Vec::with_capacity(groups.len());
    let mut gaps = Vec::with_capacity(groups.len());
    for group in groups {
        // Indices are present and unique after validation. Sorting first reproduces the engine's
        // strict-greater update: a score tie preserves the lowest completion index.
        let mut ordered = group;
        ordered.sort_by_key(|record| record.generation.completion_index);
        let mut selected = ordered[0];
        for candidate in &ordered[1..] {
            if candidate.judging.aggregate > selected.judging.aggregate {
                selected = candidate;
            }
        }
        let selected_value = values[&selected.record_id];
        let random = ordered
            .iter()
            .map(|record| values[&record.record_id])
            .sum::<f64>()
            / ordered.len() as f64;
        selected_values.push(selected_value);
        random_values.push(random);
        gaps.push(selected_value - random);
    }
    let n = gaps.len();
    let mean_gap = mean(&gaps);
    // Using -ln(alpha) avoids overflow in the mathematically equivalent ln(1 / alpha).
    let radius = (-2.0 * (1.0 - cfg.confidence_level).ln() / n as f64).sqrt();
    let statistics = OutcomeStatistics {
        selector_mean: mean(&selected_values),
        random_mean: mean(&random_values),
        mean_gap,
        lower_bound: mean_gap - radius,
    };
    if !statistics.selector_mean.is_finite()
        || !statistics.random_mean.is_finite()
        || !statistics.mean_gap.is_finite()
        || !statistics.lower_bound.is_finite()
        || !(0.0..=1.0).contains(&statistics.selector_mean)
        || !(0.0..=1.0).contains(&statistics.random_mean)
        || !(-1.0..=1.0).contains(&statistics.mean_gap)
    {
        invalid(&mut report, OutcomeReason::InvalidArithmetic);
        return report;
    }
    report.coverage.evaluated_prompts = n;
    if n < cfg.min_evaluated_prompts {
        report.reasons.push(OutcomeReason::TooFewEvaluatedPrompts {
            required: cfg.min_evaluated_prompts,
            observed: n,
        });
    }
    if statistics.lower_bound <= 0.0 {
        report.reasons.push(OutcomeReason::NonPositiveLowerBound);
    }
    report.statistics = Some(statistics);
    report.status = if report.reasons.is_empty() {
        OutcomeStatus::Qualified
    } else {
        OutcomeStatus::Inconclusive
    };
    report
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn empty_report(
    scanned_records: usize,
    evidence: Option<&OutcomeEvidence>,
    cfg: &OutcomeConfig,
) -> OutcomeReport {
    let alpha = 1.0 - cfg.confidence_level;
    OutcomeReport {
        status: OutcomeStatus::InsufficientEvidence,
        policy: SelectionPolicy::JudgeArgmaxLowestCompletionIndexV1,
        method: BoundMethod::OneSidedHoeffdingPairedGap,
        confidence_level: (cfg.confidence_level.is_finite()
            && cfg.confidence_level > 0.0
            && cfg.confidence_level < 1.0
            && alpha > 0.0
            && alpha < 1.0)
            .then_some(cfg.confidence_level),
        min_evaluated_prompts: cfg.min_evaluated_prompts,
        run_id: evidence.map(|e| e.run_id.clone()),
        training_area: evidence.map(|e| e.training_area.clone()),
        metric: evidence.map(|e| e.metric.clone()),
        provenance: evidence.map(|e| e.provenance.clone()),
        sampling_assumption: evidence.map(|e| e.sampling_assumption),
        coverage: OutcomeCoverage {
            scanned_records,
            declared_records: evidence.map_or(0, |e| e.corpus.len()),
            ..Default::default()
        },
        statistics: None,
        reasons: Vec::new(),
    }
}

fn invalid(report: &mut OutcomeReport, reason: OutcomeReason) {
    report.status = OutcomeStatus::InvalidEvidence;
    report.reasons.push(reason);
}

fn full_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_contract(evidence: &OutcomeEvidence, report: &mut OutcomeReport) {
    if evidence.version != OUTCOME_EVIDENCE_VERSION {
        invalid(
            report,
            OutcomeReason::UnsupportedVersion {
                version: evidence.version,
            },
        );
    }
    for (field, value) in [
        ("run_id", evidence.run_id.as_str()),
        ("training_area", evidence.training_area.as_str()),
        ("metric.name", evidence.metric.name.as_str()),
        ("metric.version", evidence.metric.version.as_str()),
        (
            "provenance.protocol_revision",
            evidence.provenance.protocol_revision.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            invalid(
                report,
                OutcomeReason::InvalidContract {
                    field: field.into(),
                },
            );
        }
    }
    if !full_digest(&evidence.provenance.reference_artifact_digest.hex) {
        invalid(
            report,
            OutcomeReason::InvalidContract {
                field: "provenance.reference_artifact_digest.hex".into(),
            },
        );
    }
}

fn declared_members<'a>(
    evidence: &'a OutcomeEvidence,
    report: &mut OutcomeReport,
) -> BTreeMap<&'a str, &'a CandidateBinding> {
    let mut declared = BTreeMap::new();
    for member in &evidence.corpus {
        if declared.insert(member.record_id.as_str(), member).is_some() {
            invalid(
                report,
                OutcomeReason::DuplicateCorpusMember {
                    record_id: member.record_id.clone(),
                },
            );
        }
        for (field, valid) in [
            ("record_id", !member.record_id.trim().is_empty()),
            ("run_id", member.run_id == evidence.run_id),
            (
                "training_area",
                member.training_area == evidence.training_area,
            ),
            ("prompt_hash", full_digest(&member.prompt_hash)),
            ("record_hash", full_digest(&member.record_hash)),
        ] {
            if !valid {
                invalid(
                    report,
                    OutcomeReason::IdentityMismatch {
                        record_id: member.record_id.clone(),
                        field: field.into(),
                    },
                );
            }
        }
    }
    declared
}

fn mismatch(
    expected: &CandidateBinding,
    actual: &CandidateBinding,
    report: &mut OutcomeReport,
) -> bool {
    let mut equal = true;
    for (field, valid) in [
        ("record_id", expected.record_id == actual.record_id),
        ("run_id", expected.run_id == actual.run_id),
        (
            "training_area",
            expected.training_area == actual.training_area,
        ),
        ("prompt_hash", expected.prompt_hash == actual.prompt_hash),
        ("record_hash", expected.record_hash == actual.record_hash),
    ] {
        if !valid {
            equal = false;
            invalid(
                report,
                OutcomeReason::IdentityMismatch {
                    record_id: expected.record_id.clone(),
                    field: field.into(),
                },
            );
        }
    }
    !equal
}

fn match_records<'a>(
    records: &'a [TrainingRecord],
    evidence: &OutcomeEvidence,
    declared: &BTreeMap<&str, &CandidateBinding>,
    report: &mut OutcomeReport,
) -> Vec<&'a TrainingRecord> {
    let mut available = BTreeMap::new();
    for record in records {
        if available
            .insert(record.record_id.as_str(), record)
            .is_some()
        {
            invalid(
                report,
                OutcomeReason::DuplicateRecord {
                    record_id: record.record_id.clone(),
                },
            );
        }
    }
    let mut matched = Vec::with_capacity(evidence.corpus.len());
    for (&id, member) in declared {
        let Some(&record) = available.get(id) else {
            invalid(
                report,
                OutcomeReason::MissingRecord {
                    record_id: id.into(),
                },
            );
            continue;
        };
        let Ok(actual) = gw_storage::capture_candidate_binding(record) else {
            invalid(
                report,
                OutcomeReason::RecordHashFailed {
                    record_id: id.into(),
                },
            );
            continue;
        };
        if !mismatch(member, &actual, report) {
            matched.push(record);
        }
    }
    report.coverage.matched_records = matched.len();
    matched
}

fn match_outcomes(
    evidence: &OutcomeEvidence,
    declared: &BTreeMap<&str, &CandidateBinding>,
    report: &mut OutcomeReport,
) -> BTreeMap<String, f64> {
    let mut labels = BTreeMap::new();
    for label in &evidence.outcomes {
        let id = label.candidate.record_id.as_str();
        if labels.insert(id, label).is_some() {
            invalid(
                report,
                OutcomeReason::DuplicateOutcome {
                    record_id: id.into(),
                },
            );
        }
        match declared.get(id) {
            Some(member) => {
                mismatch(member, &label.candidate, report);
            }
            None => invalid(
                report,
                OutcomeReason::UnexpectedOutcome {
                    record_id: id.into(),
                },
            ),
        }
    }
    let mut values = BTreeMap::new();
    for &id in declared.keys() {
        let Some(label) = labels.get(id) else {
            report.coverage.missing_outcomes += 1;
            report.reasons.push(OutcomeReason::MissingOutcome {
                record_id: id.into(),
            });
            continue;
        };
        match &label.outcome {
            ReferenceOutcome::Known { value }
                if value.is_finite() && (0.0..=1.0).contains(value) =>
            {
                report.coverage.known_outcomes += 1;
                values.insert(id.to_string(), *value);
            }
            ReferenceOutcome::Known { .. } => invalid(
                report,
                OutcomeReason::InvalidOutcome {
                    record_id: id.into(),
                },
            ),
            ReferenceOutcome::Unknown { reason } => {
                report.coverage.unknown_outcomes += 1;
                report.reasons.push(OutcomeReason::UnknownOutcome {
                    record_id: id.into(),
                });
                if reason.trim().is_empty() {
                    invalid(
                        report,
                        OutcomeReason::InvalidContract {
                            field: "outcome.unknown.reason".into(),
                        },
                    );
                }
            }
        }
    }
    values
}

fn eligible_groups<'a>(
    records: &[&'a TrainingRecord],
    report: &mut OutcomeReport,
) -> Vec<Vec<&'a TrainingRecord>> {
    let mut by_prompt: BTreeMap<String, Vec<&TrainingRecord>> = BTreeMap::new();
    for &record in records {
        // Match validation already recomputed this key; grouping never trusts a stale stored hash.
        let prompt =
            gw_storage::prompt_hash(&record.messages).expect("validated prompt serialization");
        by_prompt.entry(prompt).or_default().push(record);
        if record
            .judging
            .aggregate
            .is_some_and(|score| !score.is_finite() || !(0.0..=1.0).contains(&score))
        {
            invalid(
                report,
                OutcomeReason::InvalidJudgeScore {
                    record_id: record.record_id.clone(),
                },
            );
        }
    }
    let mut eligible = Vec::new();
    for (prompt_hash, group) in by_prompt {
        if !group.iter().all(|record| record.verification.all_passed) {
            continue;
        }
        let scored: Vec<_> = group
            .into_iter()
            .filter(|record| {
                record
                    .judging
                    .aggregate
                    .is_some_and(|score| score.is_finite() && (0.0..=1.0).contains(&score))
            })
            .collect();
        if scored.len() < 2 {
            continue;
        }
        let mut indices = BTreeSet::new();
        for record in &scored {
            match record.generation.completion_index {
                None => invalid(
                    report,
                    OutcomeReason::MissingCompletionIndex {
                        record_id: record.record_id.clone(),
                    },
                ),
                Some(index) if !indices.insert(index) => invalid(
                    report,
                    OutcomeReason::DuplicateCompletionIndex {
                        prompt_hash: prompt_hash.clone(),
                        completion_index: index,
                    },
                ),
                Some(_) => {}
            }
        }
        eligible.push(scored);
    }
    eligible
}
