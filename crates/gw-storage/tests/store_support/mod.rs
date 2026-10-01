//! Shared complete generated record fixture.
use gw_schema::*;

/// Build a minimal-but-valid record for `run_id` with the given id and verdict/aggregate.
pub(super) fn record(
    record_id: &str,
    run_id: &str,
    verdict: Option<Verdict>,
    agg: Option<f64>,
) -> TrainingRecord {
    let mut judging = Judging {
        verdict,
        aggregate: agg,
        ..Default::default()
    };
    if let Some(a) = agg {
        judging.panel.push(JudgeVote {
            judge_model: "judge-x".into(),
            rubric_id: Some("rubric-1".into()),
            temperature: Some(0.0),
            top_p: None,
            seed: None,
            score: a,
            dimensions: None,
            rationale: None,
            raw_response: None,
        });
    }
    TrainingRecord {
        record_id: record_id.into(),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "rust-async".into(),
        tags: vec!["tokio".into()],
        messages: vec![
            Message {
                role: gw_schema::Role::User,
                content: Content::Text("What is 12*8?".into()),
                reasoning: None,
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: gw_schema::Role::Assistant,
                content: Content::Text("96".into()),
                reasoning: Some("12*8 = 96".into()),
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ],
        tools: None,
        origin: gw_schema::RecordOrigin::Generated(Box::new(gw_schema::GeneratedOrigin {
            provenance: Provenance {
                run_id: run_id.into(),
                parent_ids: vec![],
                teacher: TeacherRef {
                    provider: "openrouter".into(),
                    slug: "z-ai/glm-5.2".into(),
                    served_by: Some("Parasail".into()),
                    model_card_revision: None,
                },
                user_synth_model: None,
                user_turn_kind: None,
                in_scope_safe: Some(true),
                judge_models: vec![],
                harness_version: "0.1.0".into(),
                git_commit: None,
            },
            generation: Generation {
                reasoning_effort: Some(ReasoningEffort::Xhigh),
                ..Default::default()
            },
        })),
        task_provenance: None,
        verification_contract: None,
        execution_evidence: None,
        verification: Default::default(),
        judging,
        reasoning_quality: None,
        lifecycle: Lifecycle::default(),
        hashes: Hashes::default(),
        cost: Default::default(),
    }
}
