//! Real local HTTP transmissions observed into SQLite before consumers parse output.
mod common;
use common::*;
use futures::StreamExt;
use gw_providers::{
    ChatRequest, EmbeddingsClient, OpenRouterProvider, Provider, ProviderError, RetryPolicy,
};
use gw_schema::{AttemptPurpose as Purpose, AttemptRole as Role, ReportedCost, TransportOutcome};
use gw_storage::Store;
use serde_json::json;
use std::{sync::atomic::Ordering, time::Duration};

fn chat(server: &Server, attempts: u32) -> OpenRouterProvider {
    OpenRouterProvider::builder()
        .base_url(&server.url)
        .rpm(60_000)
        .retry_policy(RetryPolicy {
            max_attempts: attempts,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
        })
        .build_with_key("fixture")
        .unwrap()
}
async fn drain(
    stream: gw_providers::DeltaStream,
) -> Result<Vec<gw_providers::StreamDelta>, ProviderError> {
    let items: Vec<_> = stream.collect().await;
    items.into_iter().collect()
}
#[tokio::test]
async fn failed_pre_send_persistence_prevents_chat_and_embedding_transmission() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, Role::Teacher, Purpose::Initial).await;
    sqlx::query("CREATE TRIGGER reject_attempt BEFORE INSERT ON model_attempts BEGIN SELECT RAISE(FAIL, 'intent failure'); END").execute(store.raw_pool()).await.unwrap();
    let server = Server::responses(vec![(200, String::new())]).await;
    let error = match chat(&server, 4)
        .stream_chat_observed(ChatRequest::new("model", vec![]), ctx.call())
        .await
    {
        Ok(_) => panic!("intent should fail"),
        Err(error) => error,
    };
    assert!(error.is_accounting() && !error.is_retryable());
    let client = EmbeddingsClient::builder()
        .base_url(&server.url)
        .dim(2)
        .build_with_key(None)
        .unwrap();
    assert!(
        client
            .embed_batch_observed(&["text"], ctx.call())
            .await
            .unwrap_err()
            .is_accounting()
    );
    assert_eq!(server.posts.load(Ordering::SeqCst), 0);
    assert!(store.model_attempts("run").await.unwrap().is_empty());
}
#[tokio::test]
async fn every_intentional_http_retry_has_its_own_pre_send_receipt() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, Role::Judge, Purpose::Grade).await;
    let server = Server::responses(vec![
        (503, json!({"usage":{"cost":0.1},"id":"failed"}).to_string()),
        (200, sse(json!({"choices":[{"delta":{"content":"yes"}}],"usage":{"cost":0.2},"id":"success"}), true)),
    ]).await;
    let request = ChatRequest::new("requested-model", vec![]).with_max_tokens(42);
    let expected = serde_json::to_vec(&request).unwrap();
    drain(
        chat(&server, 2)
            .stream_chat_observed(request, ctx.call())
            .await
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(server.posts.load(Ordering::SeqCst), 2);
    let mut receipts = store.model_attempts("run").await.unwrap();
    receipts.sort_by_key(|r| r.intent.retry_ordinal);
    assert_eq!(receipts.len(), 2);
    assert_ne!(receipts[0].attempt_id, receipts[1].attempt_id);
    for (index, receipt) in receipts.iter().enumerate() {
        assert_eq!(receipt.intent.retry_ordinal, index as u32);
        assert_eq!(receipt.intent.context, ctx.context);
        assert_eq!(
            receipt.intent.request_digest,
            blake3::hash(&expected).to_hex().to_string()
        );
        assert_eq!(receipt.intent.requested_model, "requested-model");
    }
    assert_eq!(receipts[0].metadata.cost_usd, ReportedCost::Known(0.1));
    assert_eq!(
        receipts[0].transport.as_ref().unwrap().outcome,
        TransportOutcome::HttpError
    );
    assert_eq!(receipts[1].metadata.cost_usd, ReportedCost::Known(0.2));
    assert_eq!(
        receipts[1].transport.as_ref().unwrap().outcome,
        TransportOutcome::Complete
    );
    assert_eq!(
        *server.requests.lock().unwrap(),
        vec![expected.clone(), expected]
    );
}
#[tokio::test]
async fn settlement_failure_is_fatal_nonretryable_and_retains_primary_http_error() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, Role::Teacher, Purpose::Initial).await;
    sqlx::query("CREATE TRIGGER reject_settlement BEFORE UPDATE ON model_attempts WHEN json_extract(NEW.receipt_json, '$.transport') IS NOT NULL BEGIN SELECT RAISE(FAIL, 'settlement failure'); END").execute(store.raw_pool()).await.unwrap();
    let server = Server::responses(vec![(503, json!({"usage":{"cost":0.3}}).to_string())]).await;
    let error = match chat(&server, 4)
        .stream_chat_observed(ChatRequest::new("model", vec![]), ctx.call())
        .await
    {
        Ok(_) => panic!("settlement must fail"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, ProviderError::Accounting { primary: Some(primary), .. } if primary.contains("503"))
    );
    assert!(!error.is_retryable());
    assert_eq!(server.posts.load(Ordering::SeqCst), 1);
    let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(receipt.metadata.cost_usd, ReportedCost::Known(0.3));
    assert!(receipt.transport.is_none());
}
#[tokio::test]
async fn malformed_typed_chunk_retains_metadata_before_delta_decoding() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, Role::Teacher, Purpose::Initial).await;
    let server = Server::responses(vec![(200, sse(json!({"id":17,"model":"actual","choices":"wrong-type","usage":{"prompt_tokens":"9","total_tokens":9,"cost":"0"}}), true))]).await;
    let error = drain(
        chat(&server, 1)
            .stream_chat_observed(ChatRequest::new("model", vec![]), ctx.call())
            .await
            .unwrap(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, ProviderError::Decode(_)));
    let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(receipt.metadata.cost_usd, ReportedCost::Known(0.0));
    assert_eq!(receipt.metadata.prompt_tokens, Some(9));
    assert_eq!(receipt.metadata.model.as_deref(), Some("actual"));
    assert!(receipt.metadata.invalid_fields.contains(&"id".into()));
    assert_eq!(
        receipt.transport.unwrap().outcome,
        TransportOutcome::Complete
    );
}
#[tokio::test]
async fn incomplete_stream_retains_usage_and_drop_remains_unresolved() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, Role::Teacher, Purpose::Initial).await;
    let payload = sse(
        json!({"choices":[{"delta":{"content":"partial"}}],"usage":{"cost":0.75,"completion_tokens":12}}),
        false,
    );
    let server = Server::responses(vec![(200, payload)]).await;
    let provider = chat(&server, 1);
    let error = drain(
        provider
            .stream_chat_observed(ChatRequest::new("model", vec![]), ctx.call())
            .await
            .unwrap(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, ProviderError::StreamReset(_)));
    let first = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(first.metadata.cost_usd, ReportedCost::Known(0.75));
    assert_eq!(first.transport.unwrap().outcome, TransportOutcome::Failed);
    let stream = provider
        .stream_chat_observed(ChatRequest::new("model", vec![]), ctx.call())
        .await
        .unwrap();
    drop(stream);
    let receipts = store.model_attempts("run").await.unwrap();
    assert_eq!(receipts.iter().filter(|r| r.transport.is_none()).count(), 1);
}
#[tokio::test]
async fn malformed_embeddings_preserve_usage_and_distinguish_invalid_prices() {
    for cost in [json!(-1), json!("NaN"), json!("inf"), json!(true)] {
        let store = Store::open_in_memory().await.unwrap();
        let ctx = context(&store, Role::Embedding, Purpose::CandidateQc).await;
        let server = Server::responses(vec![(200, json!({"model":"resolved","id":"embedding-id","usage":{"prompt_tokens":4.5,"completion_tokens":"18446744073709551616","total_tokens":4,"cost":cost},"data":[{"index":0,"embedding":[1]}]}).to_string())]).await;
        let client = EmbeddingsClient::builder()
            .base_url(&server.url)
            .dim(2)
            .build_with_key(None)
            .unwrap();
        assert!(matches!(
            client
                .embed_batch_observed(&["text"], ctx.call())
                .await
                .unwrap_err(),
            ProviderError::Decode(_)
        ));
        let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
        assert_eq!(receipt.metadata.cost_usd, ReportedCost::Invalid);
        assert_eq!(receipt.metadata.prompt_tokens, None);
        assert_eq!(receipt.metadata.completion_tokens, None);
        assert_eq!(receipt.metadata.total_tokens, Some(4));
        assert_eq!(receipt.metadata.model.as_deref(), Some("resolved"));
        assert_eq!(
            receipt.transport.unwrap().outcome,
            TransportOutcome::Complete
        );
        assert_eq!(
            receipt.interpretation,
            Some(gw_schema::OutputInterpretation::Invalid)
        );
    }
}
#[tokio::test]
async fn repeated_cumulative_sse_metadata_writes_once_and_tokens_do_not_trigger_writes() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, Role::Teacher, Purpose::Initial).await;
    let metadata = json!({"id":"one","model":"model","usage":{"cost":0,"total_tokens":5}});
    let mut body = sse(metadata.clone(), false);
    for _ in 0..30 {
        body.push_str(&sse(
            json!({"id":"one","model":"model","choices":[{"delta":{"content":"a"}}]}),
            false,
        ));
    }
    body.push_str(&sse(metadata, true));
    let server = Server::responses(vec![(200, body)]).await;
    sqlx::query("CREATE TABLE writes (n INTEGER)")
        .execute(store.raw_pool())
        .await
        .unwrap();
    sqlx::query("CREATE TRIGGER count_updates AFTER UPDATE ON model_attempts BEGIN INSERT INTO writes VALUES (1); END").execute(store.raw_pool()).await.unwrap();
    drain(
        chat(&server, 1)
            .stream_chat_observed(ChatRequest::new("model", vec![]), ctx.call())
            .await
            .unwrap(),
    )
    .await
    .unwrap();
    let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(receipt.metadata.total_tokens, Some(5));
    assert_eq!(receipt.metadata.cost_usd, ReportedCost::Known(0.0));
    assert_eq!(receipt.observations.len(), 1);
    let writes: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM writes")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(
        writes, 2,
        "one metadata change plus one settlement; no token writes"
    );
}

#[tokio::test]
async fn an_out_of_range_cost_number_does_not_erase_valid_sibling_usage_fields() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, Role::Embedding, Purpose::ResumePrior).await;
    let body = r#"{"usage":{"prompt_tokens":7,"cost":1e999},"model":"known","data":[{"index":0,"embedding":[1,0]}]}"#;
    let server = Server::responses(vec![(200, body.into())]).await;
    let client = EmbeddingsClient::builder()
        .base_url(&server.url)
        .dim(2)
        .build_with_key(None)
        .unwrap();
    let _ = client.embed_batch_observed(&["text"], ctx.call()).await;
    let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(receipt.metadata.prompt_tokens, Some(7));
    assert_eq!(receipt.metadata.cost_usd, ReportedCost::Invalid);
    assert_eq!(receipt.metadata.model.as_deref(), Some("known"));
    assert_eq!(
        receipt.transport.unwrap().outcome,
        TransportOutcome::Complete
    );
}

#[tokio::test]
async fn later_usage_is_drained_after_an_earlier_output_decode_failure() {
    for done in [true, false] {
        let store = Store::open_in_memory().await.unwrap();
        let ctx = context(&store, Role::Judge, Purpose::Grade).await;
        let mut body = sse(json!({"choices":"malformed"}), false);
        body.push_str(&sse(
            json!({"id":"terminal","usage":{"cost":0.75,"total_tokens":90}}),
            done,
        ));
        let server = Server::responses(vec![(200, body)]).await;
        let result = drain(
            chat(&server, 1)
                .stream_chat_observed(ChatRequest::new("judge", vec![]), ctx.call())
                .await
                .unwrap(),
        )
        .await;
        assert!(matches!(result, Err(ProviderError::Decode(_))));
        let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
        assert_eq!(receipt.metadata.cost_usd, ReportedCost::Known(0.75));
        assert_eq!(receipt.metadata.total_tokens, Some(90));
        assert_eq!(receipt.metadata.response_id.as_deref(), Some("terminal"));
        assert_eq!(
            receipt.transport.unwrap().outcome,
            if done {
                TransportOutcome::Complete
            } else {
                TransportOutcome::Failed
            }
        );
    }
}
