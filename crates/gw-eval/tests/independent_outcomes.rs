//! Independent hand-assigned outcome fixtures exercise the algorithm, not model quality.

mod common;

use common::*;
use gw_eval::outcomes::*;
use gw_eval::{SeparationConfig, analyze};
use gw_schema::Content;

#[test]
fn predictive_and_inverted_scores_have_opposite_independent_outcome_gaps() {
    for (predictive, expected) in [
        (true, OutcomeStatus::Qualified),
        (false, OutcomeStatus::Inconclusive),
    ] {
        let (records, evidence) = corpus(100, predictive);
        let report = analyze(&records, &SeparationConfig::default(), Some(&evidence));
        assert_eq!(report.outcome_evaluation.status, expected);
        assert_eq!(report.passed(), predictive);
        let stats = report.outcome_evaluation.statistics.unwrap();
        assert_eq!(stats.mean_gap, if predictive { 0.5 } else { -0.5 });
        // Hand-derived from n=100, alpha=.05 and a gap range of width 2.
        let expected_bound = if predictive {
            0.255_225_316_931_918_35
        } else {
            -0.744_774_683_068_081_6
        };
        assert!((stats.lower_bound - expected_bound).abs() < 1e-12);
        assert_eq!(report.outcome_evaluation.coverage.evaluated_prompts, 100);
        assert_eq!(
            report.outcome_evaluation.method,
            BoundMethod::OneSidedHoeffdingPairedGap
        );
    }
}

#[test]
fn balanced_constant_scores_tie_and_lowest_index_is_order_independent() {
    let (mut records, mut evidence) = corpus(100, true);
    for (i, record) in records.iter_mut().enumerate() {
        record.judging.aggregate = Some(0.5);
        if i / 2 >= 50 {
            evidence.outcomes[i].outcome = ReferenceOutcome::Known {
                value: if i % 2 == 0 { 0.0 } else { 1.0 },
            };
        }
    }
    let expected = analyze(&records, &SeparationConfig::default(), Some(&evidence));
    assert_eq!(
        expected.outcome_evaluation.status,
        OutcomeStatus::Inconclusive
    );
    assert_eq!(
        expected
            .outcome_evaluation
            .statistics
            .as_ref()
            .unwrap()
            .mean_gap,
        0.0
    );
    records.reverse();
    evidence.corpus.reverse();
    evidence.outcomes.reverse();
    assert_eq!(
        analyze(&records, &SeparationConfig::default(), Some(&evidence)),
        expected
    );
}

#[test]
fn a_small_positive_sample_is_inconclusive() {
    let (records, evidence) = corpus(1, true);
    let report =
        analyze(&records, &SeparationConfig::default(), Some(&evidence)).outcome_evaluation;
    assert_eq!(report.status, OutcomeStatus::Inconclusive);
    let stats = report.statistics.unwrap();
    assert_eq!(stats.mean_gap, 0.5);
    assert!(stats.lower_bound < 0.0);
    assert!(
        report
            .reasons
            .contains(&OutcomeReason::TooFewEvaluatedPrompts {
                required: 30,
                observed: 1
            })
    );
    assert!(
        report
            .reasons
            .contains(&OutcomeReason::NonPositiveLowerBound)
    );
}

#[test]
fn repeated_prompt_occurrences_never_inflate_the_independent_count() {
    let (mut records, _) = corpus(100, true);
    for (index, record) in records.iter_mut().enumerate() {
        record.messages[0].content = Content::Text("one repeated prompt".into());
        record.generation.completion_index = Some(index as u32);
    }
    let values: Vec<_> = (0..records.len())
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let evidence = evidence(&records, &values);
    let report =
        analyze(&records, &SeparationConfig::default(), Some(&evidence)).outcome_evaluation;
    assert_eq!(report.coverage.evaluated_prompts, 1);
    assert_eq!(report.status, OutcomeStatus::Inconclusive);
}

#[test]
fn prompt_groups_receive_equal_weight_even_with_different_candidate_counts() {
    let records = vec![
        candidate(0, 0, Some(0.9)),
        candidate(0, 1, Some(0.1)),
        candidate(1, 0, Some(0.9)),
        candidate(1, 1, Some(0.6)),
        candidate(1, 2, Some(0.4)),
        candidate(1, 3, Some(0.1)),
    ];
    let evidence = evidence(&records, &[1.0, 0.0, 0.8, 0.8, 0.8, 0.8]);
    let report =
        analyze(&records, &SeparationConfig::default(), Some(&evidence)).outcome_evaluation;
    let stats = report.statistics.unwrap();
    assert!((stats.mean_gap - 0.25).abs() < 1e-12);
    assert!((stats.selector_mean - 0.9).abs() < 1e-12);
    assert!((stats.random_mean - 0.65).abs() < 1e-12);
}

#[test]
fn any_missing_or_unknown_outcome_blocks_the_full_declared_corpus() {
    for unknown in [false, true] {
        let (records, mut evidence) = corpus(100, true);
        if unknown {
            evidence.outcomes[1].outcome = ReferenceOutcome::Unknown {
                reason: "reference unavailable".into(),
            };
        } else {
            evidence.outcomes.remove(1);
        }
        let report =
            analyze(&records, &SeparationConfig::default(), Some(&evidence)).outcome_evaluation;
        assert_eq!(report.status, OutcomeStatus::InsufficientEvidence);
        assert_eq!(report.coverage.declared_records, 200);
        assert_eq!(report.coverage.known_outcomes, 199);
        assert_eq!(
            report.coverage.unknown_outcomes + report.coverage.missing_outcomes,
            1
        );
        assert_eq!(report.coverage.eligible_prompts, 100);
        assert_eq!(report.coverage.evaluated_prompts, 0);
        assert!(
            report.statistics.is_none(),
            "never evaluate only the labeled survivors"
        );
        assert!(serde_json::to_value(report).unwrap()["statistics"].is_null());
    }
}

#[test]
fn missing_outcomes_on_ineligible_members_still_block_the_declared_corpus() {
    let (mut records, _) = corpus(100, true);
    records.push(candidate(101, 0, None));
    let mut evidence = evidence(&records, &vec![1.0; records.len()]);
    evidence.outcomes.pop();
    let report =
        analyze(&records, &SeparationConfig::default(), Some(&evidence)).outcome_evaluation;
    assert_eq!(report.status, OutcomeStatus::InsufficientEvidence);
    assert_eq!(report.coverage.missing_outcomes, 1);
    assert!(report.statistics.is_none());
}

#[test]
fn all_pass_and_scored_population_is_explicit_and_excludes_mixed_groups() {
    let (mut records, _) = corpus(2, true);
    records[0].verification.all_passed = false;
    records.push(candidate(1, 2, None));
    let evidence = evidence(&records, &[1.0, 0.0, 1.0, 0.0, 0.2]);
    let report =
        analyze(&records, &SeparationConfig::default(), Some(&evidence)).outcome_evaluation;
    assert_eq!(report.coverage.evaluated_prompts, 1);
    assert_eq!(report.statistics.unwrap().mean_gap, 0.5);
    assert_eq!(
        report.policy,
        SelectionPolicy::JudgeArgmaxLowestCompletionIndexV1
    );
}

#[test]
fn absent_evidence_and_empty_declared_corpus_have_no_numerical_result() {
    let (records, _) = corpus(100, true);
    let report = analyze(&records, &SeparationConfig::default(), None).outcome_evaluation;
    assert_eq!(report.status, OutcomeStatus::InsufficientEvidence);
    assert_eq!(report.reasons, [OutcomeReason::MissingEvidence]);
    assert!(report.statistics.is_none());
    let empty = evidence(&[], &[]);
    let report = analyze(&records, &SeparationConfig::default(), Some(&empty)).outcome_evaluation;
    assert_eq!(report.status, OutcomeStatus::InsufficientEvidence);
    assert_eq!(report.reasons, [OutcomeReason::NoEligiblePrompts]);
    assert!(report.statistics.is_none());
}
