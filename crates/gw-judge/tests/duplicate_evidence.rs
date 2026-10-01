//! Audit aliases and cloned collector positions cannot become independent consensus evidence.

use gw_judge::{
    AreaThresholds, CorrelationMatrix, Decision, Grade, HybridGrader, JudgeError, JudgeSampling,
    PanelFailure, PanelJudge, Verdict, grade_one, grade_one_cached, grade_panel_cached,
    validate_judge_panel,
};
use gw_providers::{ChatRequest, DeltaStream, Provider, StreamChatFuture, StreamDelta};
use gw_schema::AdmissionIntent;
use gw_storage::Store;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct CountingJudge(AtomicUsize);

impl Provider for CountingJudge {
    fn stream_chat(&self, _: ChatRequest) -> StreamChatFuture<'_> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            let stream: DeltaStream = Box::pin(futures::stream::iter([Ok(StreamDelta {
                content: Some(r#"{"score":0.9,"verdict":"accept"}"#.into()),
                finish_reason: Some("stop".into()),
                ..Default::default()
            })]));
            Ok(stream)
        })
    }
}

async fn duplicate_rejected(second: PanelJudge, expected_calls: usize) {
    let store = Store::open_in_memory().await.unwrap();
    let provider = CountingJudge::default();
    let judges = [PanelJudge::new("judge-a", "family-a"), second];
    for _ in 0..2 {
        let grades = grade_panel_cached(
            &store,
            &provider,
            &judges,
            "rubric",
            "candidate",
            "candidate-hash",
            |_| PanelFailure::Fatal,
        )
        .await
        .unwrap();
        assert_eq!(grades.len(), 2, "collector keeps original positions");
        assert_eq!(provider.0.load(Ordering::SeqCst), expected_calls);
        for intent in [AdmissionIntent::Automatic, AdmissionIntent::ReviewOnly] {
            let result = HybridGrader::new(AreaThresholds::default())
                .with_admission_intent(intent)
                .grade(
                    None,
                    &grades,
                    &[],
                    None,
                    &CorrelationMatrix::uniform_offdiagonal(2, 0.3),
                );
            assert!(
                matches!(
                    result,
                    Err(JudgeError::DuplicateJudgeEvidence {
                        first: 0,
                        duplicate: 1
                    })
                ),
                "one effective request cannot satisfy n_eff >= 1.5: {result:?}"
            );
        }
    }
}

#[tokio::test]
async fn identical_collector_positions_cannot_admit_one_observation_twice() {
    duplicate_rejected(PanelJudge::new("judge-a", "family-a"), 1).await;
}

#[tokio::test]
async fn family_aliases_cannot_admit_one_observation_twice() {
    duplicate_rejected(PanelJudge::new("judge-a", "family-alias"), 1).await;
}

#[tokio::test]
async fn rubric_audit_aliases_cannot_manufacture_independent_evidence() {
    duplicate_rejected(
        PanelJudge::new("judge-a", "family-a").with_rubric("audit-alias"),
        2,
    )
    .await;
}

#[tokio::test]
async fn clamped_completion_caps_do_not_change_effective_evidence() {
    duplicate_rejected(PanelJudge::new("judge-a", "family-a").with_max_tokens(1), 1).await;
}

#[tokio::test]
async fn signed_zero_does_not_create_another_effective_setting() {
    for top_p in [false, true] {
        let first = PanelJudge::new("judge", "family").with_sampling(JudgeSampling {
            top_p: top_p.then_some(0.0),
            ..Default::default()
        });
        let mut second = first.clone();
        if top_p {
            second.sampling.top_p = Some(-0.0);
        } else {
            second.sampling.temperature = -0.0;
        }
        let panel = [first, second];
        assert!(matches!(
            validate_judge_panel(&panel, "rubric"),
            Err(JudgeError::DuplicateJudgeEvidence {
                first: 0,
                duplicate: 1
            })
        ));
        let provider = CountingJudge::default();
        let store = Store::open_in_memory().await.unwrap();
        let grades = grade_panel_cached(
            &store,
            &provider,
            &panel,
            "rubric",
            "candidate",
            "hash",
            |_| PanelFailure::Fatal,
        )
        .await
        .unwrap();
        assert_eq!(
            provider.0.load(Ordering::SeqCst),
            2,
            "the audited cache still preserves sampling bits"
        );
        assert!(matches!(
            HybridGrader::new(AreaThresholds::default()).grade(
                None,
                &grades,
                &[],
                None,
                &CorrelationMatrix::uniform_offdiagonal(2, 0.3)
            ),
            Err(JudgeError::DuplicateJudgeEvidence {
                first: 0,
                duplicate: 1
            })
        ));
    }
}

#[tokio::test]
async fn actual_model_sampling_and_reasoning_differences_remain_usable() {
    let first = PanelJudge::new("judge", "family");
    let alternatives = [
        PanelJudge::new("other-judge", "family"),
        first.clone().with_sampling(JudgeSampling {
            temperature: 0.1,
            ..Default::default()
        }),
        first.clone().with_sampling(JudgeSampling {
            top_p: Some(0.9),
            ..Default::default()
        }),
        first.clone().with_sampling(JudgeSampling {
            seed: Some(7),
            ..Default::default()
        }),
        first.clone().with_max_tokens(4_000),
        first.clone().with_reasoning_max_tokens(1_000),
        first
            .clone()
            .with_reasoning_effort(gw_schema::ReasoningEffort::High),
        first.clone().with_reasoning(None),
    ];
    for alternative in alternatives {
        let panel = [first.clone(), alternative];
        validate_judge_panel(&panel, "rubric").unwrap();
        let provider = CountingJudge::default();
        let store = Store::open_in_memory().await.unwrap();
        for _ in 0..2 {
            let grades = grade_panel_cached(
                &store,
                &provider,
                &panel,
                "rubric",
                "candidate",
                "hash",
                |_| PanelFailure::Fatal,
            )
            .await
            .unwrap();
            assert_eq!(provider.0.load(Ordering::SeqCst), 2);
            let outcome = HybridGrader::new(AreaThresholds::default())
                .grade(
                    None,
                    &grades,
                    &[],
                    None,
                    &CorrelationMatrix::uniform_offdiagonal(2, 0.3),
                )
                .unwrap();
            assert!(matches!(outcome.decision, Decision::Accept { .. }));
            assert!((outcome.judging.n_eff.unwrap() - 20.0 / 13.0).abs() < 1e-12);
        }
    }
}

#[tokio::test]
async fn distinct_actual_rubric_prompts_survive_the_direct_grading_boundary() {
    let provider = CountingJudge::default();
    let judge = PanelJudge::new("judge", "family");
    let first = grade_one(&provider, &judge, "check arithmetic", "candidate")
        .await
        .unwrap();
    let second = grade_one(&provider, &judge, "check units", "candidate")
        .await
        .unwrap();
    let outcome = HybridGrader::new(AreaThresholds::default())
        .grade(
            None,
            &[first, second],
            &[],
            None,
            &CorrelationMatrix::uniform_offdiagonal(2, 0.3),
        )
        .unwrap();
    assert!(matches!(outcome.decision, Decision::Accept { .. }));
    assert_eq!(provider.0.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn precise_sampling_remains_valid_after_the_existing_cache_json_roundtrip() {
    for sampling in [
        JudgeSampling {
            temperature: 20.0 / 13.0,
            ..Default::default()
        },
        JudgeSampling {
            top_p: Some(10.0 / 13.0),
            ..Default::default()
        },
    ] {
        let judge = PanelJudge::new("judge", "family").with_sampling(sampling);
        let store = Store::open_in_memory().await.unwrap();
        let provider = CountingJudge::default();
        for cached in [false, true] {
            let grade = grade_one_cached(&store, &provider, &judge, "rubric", "candidate", "hash")
                .await
                .unwrap();
            let result = HybridGrader::new(AreaThresholds {
                min_n_eff: 1.0,
                ..Default::default()
            })
            .grade(None, &[grade], &[], None, &CorrelationMatrix::identity(1));
            assert!(
                result.is_ok(),
                "valid request sampling, cached={cached}: {result:?}"
            );
            assert_eq!(provider.0.load(Ordering::SeqCst), 1);
        }
    }
}

#[tokio::test]
async fn missing_or_inconsistent_live_contracts_fail_closed_even_for_review_only() {
    let provider = CountingJudge::default();
    let judge = PanelJudge::new("judge", "family");
    let original = grade_one(&provider, &judge, "rubric", "candidate")
        .await
        .unwrap();
    let changes: [fn(&mut Grade); 7] = [
        |grade| grade.effective_contract = None,
        |grade| grade.judge_model.push_str("-alias"),
        |grade| grade.temperature = 0.5,
        |grade| grade.top_p = Some(0.9),
        |grade| grade.seed = Some(9),
        |grade| grade.raw["scoring_used"] = serde_json::json!("other"),
        |grade| grade.raw["interpretation_version"] = serde_json::json!(2),
    ];
    for change in changes {
        let mut grade = original.clone();
        change(&mut grade);
        for intent in [AdmissionIntent::Automatic, AdmissionIntent::ReviewOnly] {
            let result = HybridGrader::new(AreaThresholds {
                min_n_eff: 1.0,
                ..Default::default()
            })
            .with_admission_intent(intent)
            .grade(
                None,
                &[grade.clone()],
                &[],
                None,
                &CorrelationMatrix::identity(1),
            );
            assert!(
                matches!(result, Err(JudgeError::Invariant(ref message)) if message.contains("effective judge evidence"))
            );
        }
    }
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn duplicates_are_invalid_before_uncertain_filtering_but_verifier_rejection_still_wins() {
    let provider = CountingJudge::default();
    let judge = PanelJudge::new("judge", "family");
    let first = grade_one(&provider, &judge, "rubric", "candidate")
        .await
        .unwrap();
    let mut second = first.clone();
    second.verdict = Verdict::Uncertain;
    let mut grades = [first, second];
    let grader = HybridGrader::new(AreaThresholds::default());
    let correlation = CorrelationMatrix::uniform_offdiagonal(2, 0.3);
    assert!(matches!(
        grader.grade(None, &grades, &[], None, &correlation),
        Err(JudgeError::DuplicateJudgeEvidence { .. })
    ));
    grades[0].effective_contract = None;
    let verifier = gw_judge::VerifierGrade {
        verdict: Verdict::Reject,
        verification: gw_schema::Verification {
            all_passed: false,
            ..Default::default()
        },
    };
    let rejected = grader
        .grade(Some(&verifier), &grades, &[], None, &correlation)
        .unwrap();
    assert_eq!(rejected.judging.verdict, Some(gw_schema::Verdict::Reject));
    assert_eq!(rejected.judging.n_eff, None);
}
