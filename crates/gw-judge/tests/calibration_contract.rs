//! Production rendering/interpretation parity and candidate-specific applicability.
mod calibration_support;
use calibration_support::*;
use gw_judge::*;
use gw_providers::{ChatRequest, DeltaStream, Provider, StreamChatFuture, StreamDelta};
use gw_schema::*;
use std::sync::Mutex;

struct TextProvider {
    text: String,
    finish: &'static str,
    requests: Mutex<Vec<ChatRequest>>,
}
impl Provider for TextProvider {
    fn stream_chat(&self, request: ChatRequest) -> StreamChatFuture<'_> {
        self.requests.lock().unwrap().push(request);
        Box::pin(async {
            let stream: DeltaStream = Box::pin(futures::stream::iter([Ok(StreamDelta {
                content: Some(self.text.clone()),
                finish_reason: Some(self.finish.into()),
                ..Default::default()
            })]));
            Ok(stream)
        })
    }
}

#[tokio::test]
async fn shared_interpreter_preserves_production_optional_fields_errors_and_length_behavior() {
    let judge = PanelJudge::new("judge", "family").with_rubric("rubric-v1");
    for text in [
        "```json\n{\"score\":9,\"verdict\":\"ADMIT\",\"confidence\":2,\"meta_prediction\":8,\"dimensions\":{\"binary64\":0.3},\"rationale\":\"text\"}\n```",
        "prose {\"score\":-2,\"verdict\":\"not-a-verdict\"} suffix",
    ] {
        let expected = interpret_judge_response(text).unwrap();
        let provider = TextProvider {
            text: text.into(),
            finish: "length",
            requests: Mutex::new(vec![]),
        };
        let grade = grade_one(&provider, &judge, "rubric", "candidate")
            .await
            .unwrap();
        assert_eq!(grade.score.to_bits(), expected.score.to_bits());
        assert_eq!(grade.verdict, expected.verdict);
        assert_eq!(grade.confidence.to_bits(), expected.confidence.to_bits());
        assert_eq!(grade.meta_prediction, expected.meta_prediction);
        assert_eq!(grade.dimensions, expected.dimensions);
        assert_eq!(grade.rationale, expected.rationale);
        assert_eq!(grade.raw["response"], text);
        assert_eq!(
            grade.raw["interpretation_version"],
            JUDGE_INTERPRETATION_VERSION
        );
        assert_eq!(grade.rubric_id.as_deref(), Some("rubric-v1"));
        assert_eq!(provider.requests.lock().unwrap().len(), 1);
    }
    let provider = TextProvider {
        text: "malformed".into(),
        finish: "stop",
        requests: Mutex::new(vec![]),
    };
    let parse_error = interpret_judge_response("malformed").unwrap_err();
    let live_error = grade_one(&provider, &judge, "rubric", "candidate")
        .await
        .unwrap_err();
    assert!(matches!(live_error, JudgeError::JudgeParse(ref text)
        if text == &format!("judge judge returned an unparseable response: {parse_error}")));
    let capped = TextProvider {
        text: "malformed".into(),
        finish: "length",
        requests: Mutex::new(vec![]),
    };
    let error = grade_one(&capped, &judge, "rubric", "candidate")
        .await
        .unwrap_err();
    assert!(
        matches!(error, JudgeError::JudgeParse(ref text) if text.contains("was truncated at max_tokens"))
    );
}

#[test]
fn candidates_share_one_definition_but_have_distinct_actual_request_identities() {
    let fixture = Fixture::new();
    let a = &fixture.evidence.fit[0].observations[0];
    let b = &fixture.evidence.fit[1].observations[0];
    assert_eq!(a.column_identity, b.column_identity);
    assert_ne!(a.request.identity, b.request.identity);
    assert_ne!(a.request.projection_json, b.request.projection_json);
    let legacy_render = gw_format::render(
        &fixture.records[0].messages,
        TrlFormat::OpenAiMessages,
        CotPolicy::Supervised,
    )
    .unwrap();
    assert_eq!(
        render_judge_candidate(&fixture.records[0].messages).unwrap(),
        legacy_render
    );
    let projected: serde_json::Value = serde_json::from_str(&a.request.projection_json).unwrap();
    let production = build_judge_request(
        &PanelJudge::new("judge-a", "a"),
        "synthetic rubric",
        &legacy_render,
    );
    assert_eq!(
        projected["request"],
        serde_json::to_value(production).unwrap()
    );
    assert_eq!(
        fixture.panel.declaration().render_contract,
        JUDGE_CANDIDATE_RENDER_VERSION
    );
}

#[test]
fn complete_message_content_reasoning_order_and_tool_changes_invalidate_old_cells() {
    let changes: [fn(&mut TrainingRecord); 7] = [
        |record| record.messages[1].content = Content::Text("different answer".into()),
        |record| record.messages[1].reasoning = Some("different reasoning".into()),
        |record| record.messages.reverse(),
        |record| record.messages[1].name = Some("different speaker".into()),
        |record| {
            record.messages[1].reasoning_details = Some(vec![ReasoningDetail::Text {
                text: "different detail".into(),
                signature: None,
                id: None,
                format: None,
                index: 0,
            }])
        },
        |record| {
            record.messages[1].tool_calls = Some(vec![ToolCall {
                id: Some("call-1".into()),
                function: FunctionCall {
                    name: "lookup".into(),
                    arguments: serde_json::json!({"binary64":"literal", "x":0.125}),
                    raw_arguments: None,
                },
            }])
        },
        |record| {
            record.tools = Some(vec![
                serde_json::json!({"type":"function", "name":"different-tool"}),
            ])
        },
    ];
    for change in changes {
        let mut fixture = Fixture::new();
        change(&mut fixture.records[0]);
        let report = fixture.report();
        assert_eq!(
            report.status,
            CalibrationStatus::InvalidEvidence,
            "{:?}",
            report.reasons
        );
        assert!(report.snapshot.is_none());
    }
    let mut fixture = Fixture::new();
    fixture.records[0].lifecycle.state = LifecycleState::NeedsReview;
    fixture.records[0].hashes.record_hash = "do-not-trust-stored-hashes".into();
    fixture.snapshot();
}

#[test]
fn rubric_sampling_reasoning_and_interpretation_changes_reject_old_evidence() {
    let mut fixture = Fixture::new();
    let changes: [fn(&mut PanelJudge); 4] = [
        |judge| judge.sampling.temperature = 0.25,
        |judge| judge.sampling.top_p = Some(0.9),
        |judge| judge.sampling.seed = Some(42),
        |judge| judge.reasoning = Some(JudgeReasoning::MaxTokens(1000)),
    ];
    for change in changes {
        let mut first = PanelJudge::new("judge-a", "a");
        change(&mut first);
        fixture.panel = ResolvedCalibrationPanel::new(
            "synthetic-area",
            &[first, PanelJudge::new("judge-b", "b")],
            "synthetic rubric",
        )
        .unwrap();
        assert_eq!(fixture.report().status, CalibrationStatus::InvalidEvidence);
    }
    fixture = Fixture::new();
    fixture.evidence.fit[0].observations[0].interpretation_version += 1;
    rehash(&mut fixture.evidence.fit[0].observations[0]);
    assert_eq!(fixture.report().status, CalibrationStatus::InvalidEvidence);
}
