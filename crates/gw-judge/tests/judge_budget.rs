use std::sync::Mutex;

use futures::stream;
use gw_judge::{
    DEFAULT_JUDGE_MAX_TOKENS, DEFAULT_JUDGE_REASONING_MAX_TOKENS, JudgeError, PanelJudge, Verdict,
    grade_one,
};
use gw_providers::{
    ChatRequest, DeltaStream, Provider, ProviderError, ReasoningParam, StreamChatFuture,
    StreamDelta,
};

struct BudgetAwareProvider {
    seen: Mutex<Vec<ChatRequest>>,
}

impl BudgetAwareProvider {
    fn new() -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
        }
    }
}

impl Provider for BudgetAwareProvider {
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        let long_trace = serde_json::to_string(&req.messages)
            .expect("request messages serialize")
            .len()
            > 4_000;
        let max_tokens = req.max_tokens.unwrap_or_default();
        let reasoning = req.reasoning;
        self.seen.lock().unwrap().push(req);

        Box::pin(async move {
            let starved = match reasoning {
                Some(ReasoningParam::MaxTokens { max_tokens: r }) => max_tokens <= r,
                Some(ReasoningParam::Effort { .. }) => long_trace,
                None => false,
            };
            let deltas = if starved {
                vec![Ok(StreamDelta {
                    reasoning: Some("reasoning consumed the whole cap".into()),
                    finish_reason: Some("length".into()),
                    ..Default::default()
                })]
            } else {
                vec![
                    Ok(StreamDelta {
                        reasoning: Some("bounded judge reasoning".into()),
                        ..Default::default()
                    }),
                    Ok(StreamDelta {
                        content: Some(
                            "{\"score\":0.9,\"verdict\":\"accept\",\"confidence\":0.8}".into(),
                        ),
                        finish_reason: Some("stop".into()),
                        ..Default::default()
                    }),
                ]
            };
            let stream: DeltaStream = Box::pin(stream::iter(deltas));
            Ok(stream)
        })
    }
}

#[tokio::test]
async fn long_trace_judge_request_preserves_verdict_headroom() {
    let provider = BudgetAwareProvider::new();
    let judge = PanelJudge::new("deepseek/deepseek-v4-pro", "deepseek");
    let long_trace = "HVAC diagnostic step. ".repeat(4_500);

    let grade = grade_one(&provider, &judge, "rubric", &long_trace)
        .await
        .expect("bounded judge reasoning leaves room for verdict JSON");

    assert_eq!(grade.verdict, Verdict::Accept);
    let seen = provider.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].max_tokens, Some(DEFAULT_JUDGE_MAX_TOKENS));
    assert_eq!(
        seen[0].reasoning,
        Some(ReasoningParam::max_tokens(
            DEFAULT_JUDGE_REASONING_MAX_TOKENS
        ))
    );
}

struct AlwaysLengthProvider;

impl Provider for AlwaysLengthProvider {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        Box::pin(async move {
            let stream: DeltaStream =
                Box::pin(stream::iter(vec![Ok::<StreamDelta, ProviderError>(
                    StreamDelta {
                        reasoning: Some("no visible verdict".into()),
                        finish_reason: Some("length".into()),
                        ..Default::default()
                    },
                )]));
            Ok(stream)
        })
    }
}

#[tokio::test]
async fn length_truncated_empty_judge_completion_names_budget_fix() {
    let judge = PanelJudge::new("deepseek/deepseek-v4-pro", "deepseek");
    let err = grade_one(&AlwaysLengthProvider, &judge, "rubric", "trace")
        .await
        .expect_err("length-truncated empty content errors");

    let JudgeError::JudgeParse(msg) = err else {
        panic!("expected JudgeParse");
    };
    assert!(msg.contains("truncated at max_tokens"), "got: {msg}");
    assert!(msg.contains("raise judge max_tokens"), "got: {msg}");
    assert!(
        msg.contains("lower the judge reasoning budget"),
        "got: {msg}"
    );
}
