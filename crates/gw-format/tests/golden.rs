//! Byte-exact golden-file tests for every render target, plus the render→ingest round-trip.
//!
//! Fixtures live in `tests/golden/<target>.txt` and are author-written to the `_research` spec;
//! each test asserts `render(...) == include_str!(fixture)` byte-for-byte. The Gemma-4 token
//! bytes here are the ones that MUST be diff-verified against the official pinned
//! `google/gemma-4-12B-it` `chat_template.jinja` before production SFT (no network in any test).

use gw_format::{ingest_openrouter, render};
use gw_schema::{Content, CotPolicy, Message, Role, TrlFormat};
use serde_json::json;

/// A clean assistant/user message (reasoning optional). `content` is always leak-free.
fn m(role: Role, content: &str, reasoning: Option<&str>) -> Message {
    Message {
        role,
        content: Content::Text(content.into()),
        reasoning: reasoning.map(str::to_owned),
        reasoning_details: None,
        tool_calls: None,
        name: None,
    }
}

/// The canonical single-turn conversation used by the per-target golden fixtures.
fn canonical() -> Vec<Message> {
    vec![
        m(Role::System, "You are a precise math tutor.", None),
        m(Role::User, "What is 12 * 8, and why?", None),
        m(
            Role::Assistant,
            "96. Break it into 10*8=80 and 2*8=16, then add.",
            Some("12 * 8: 10*8=80, 2*8=16, 80+16=96."),
        ),
    ]
}

/// A two-round conversation for the multi-turn Gemma-4 golden fixture.
fn multiturn() -> Vec<Message> {
    vec![
        m(Role::User, "Hi", None),
        m(
            Role::Assistant,
            "Hello! How can I help?",
            Some("Greet the user."),
        ),
        m(Role::User, "What is 2+2?", None),
        m(Role::Assistant, "4", Some("2+2=4.")),
    ]
}

// ----------------------------- byte-exact golden tests -----------------------------

#[test]
fn gemma4_supervised_golden() {
    let out = render(&canonical(), TrlFormat::Gemma4, CotPolicy::Supervised).unwrap();
    assert_eq!(out, include_str!("golden/gemma4_supervised.txt"));
    // Spot-check the asymmetric channel framing the spec pins.
    assert!(out.starts_with("<bos><|turn>user\n"));
    assert!(out.contains("<|channel>thought\n12 * 8: 10*8=80, 2*8=16, 80+16=96.\n<channel|>96."));
    assert!(out.trim_end().ends_with("<turn|>"));
}

#[test]
fn gemma4_stripped_golden() {
    let out = render(&canonical(), TrlFormat::Gemma4, CotPolicy::Stripped).unwrap();
    assert_eq!(out, include_str!("golden/gemma4_stripped.txt"));
    // Stripped => EMPTY thought wrapper, reasoning text gone. The needle `80+16` appears ONLY
    // in the reasoning (the clean content says "then add", not "80+16").
    assert!(out.contains("<|channel>thought\n<channel|>96."));
    assert!(!out.contains("80+16"));
}

#[test]
fn gemma4_multiturn_golden() {
    let out = render(&multiturn(), TrlFormat::Gemma4, CotPolicy::Supervised).unwrap();
    assert_eq!(out, include_str!("golden/gemma4_multiturn.txt"));
    // Two model turns, each with its own thought channel.
    assert_eq!(out.matches("<|turn>model\n").count(), 2);
    assert_eq!(out.matches("<channel|>").count(), 2);
}

#[test]
fn chatml_supervised_golden() {
    let out = render(&canonical(), TrlFormat::ChatML, CotPolicy::Supervised).unwrap();
    assert_eq!(out, include_str!("golden/chatml_supervised.txt"));
    assert!(out.contains("<|im_start|>assistant\n<think>"));
}

#[test]
fn sharegpt_supervised_golden() {
    let out = render(&canonical(), TrlFormat::ShareGpt, CotPolicy::Supervised).unwrap();
    assert_eq!(out, include_str!("golden/sharegpt_supervised.txt"));
    // `value` stays clean; reasoning is a sibling key.
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let gpt = &v["conversations"][2];
    assert_eq!(gpt["from"], "gpt");
    assert!(!gpt["value"].as_str().unwrap().contains("80+16"));
    assert!(gpt["reasoning"].is_string());
}

#[test]
fn openai_supervised_golden() {
    let out = render(
        &canonical(),
        TrlFormat::OpenAiMessages,
        CotPolicy::Supervised,
    )
    .unwrap();
    assert_eq!(out, include_str!("golden/openai_supervised.txt"));
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["messages"][2]["role"], "assistant");
    assert!(v["messages"][2]["reasoning"].is_string());
}

#[test]
fn harmony_supervised_golden() {
    let out = render(&canonical(), TrlFormat::Harmony, CotPolicy::Supervised).unwrap();
    assert_eq!(out, include_str!("golden/harmony_supervised.txt"));
    assert!(out.contains("<|channel|>analysis<|message|>"));
    assert!(out.contains("<|channel|>final<|message|>"));
    assert!(out.ends_with("<|return|>"));
}

#[test]
fn prompt_completion_supervised_golden() {
    let out = render(
        &canonical(),
        TrlFormat::TrlPromptCompletion,
        CotPolicy::Supervised,
    )
    .unwrap();
    assert_eq!(out, include_str!("golden/prompt_completion_supervised.txt"));
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    // Split at the last assistant turn: prompt = [system, user], completion = [assistant].
    assert_eq!(v["prompt"].as_array().unwrap().len(), 2);
    assert_eq!(v["completion"].as_array().unwrap().len(), 1);
    assert_eq!(v["completion"][0]["role"], "assistant");
}

// ------------------------------- round-trip property -------------------------------

#[test]
fn round_trip_chatml_recovers_messages_and_reasoning() {
    // Render the assistant turn to ChatML, then re-ingest the *content region* and confirm the
    // clean content + reasoning are recovered (INVARIANT-a: render→ingest is the identity on
    // {clean content, reasoning}).
    let assistant = m(
        Role::Assistant,
        "The answer is 42.",
        Some("Sum the digits, carry the one."),
    );
    let chatml = render(
        std::slice::from_ref(&assistant),
        TrlFormat::ChatML,
        CotPolicy::Supervised,
    )
    .unwrap();
    // Peel the ChatML frame to the bare content region the model would emit.
    let body = chatml
        .strip_prefix("<|im_start|>assistant\n")
        .unwrap()
        .strip_suffix("<|im_end|>\n")
        .unwrap();
    let ingested = ingest_openrouter(&json!({"role": "assistant", "content": body})).unwrap();
    assert_eq!(ingested.content, assistant.content);
    assert_eq!(ingested.reasoning, assistant.reasoning);
}

#[test]
fn round_trip_gemma4_recovers_messages_and_reasoning() {
    let assistant = m(Role::Assistant, "96", Some("12*8=96"));
    let out = render(
        std::slice::from_ref(&assistant),
        TrlFormat::Gemma4,
        CotPolicy::Supervised,
    )
    .unwrap();
    // The model turn's content region (between the thought OPEN and the turn CLOSE).
    let body = out
        .split("<|turn>model\n")
        .nth(1)
        .unwrap()
        .strip_suffix("<turn|>\n")
        .unwrap();
    let ingested = ingest_openrouter(&json!({"role": "assistant", "content": body})).unwrap();
    assert_eq!(ingested.content, Content::Text("96".into()));
    assert_eq!(ingested.reasoning.as_deref(), Some("12*8=96"));
}

#[test]
fn round_trip_harmony_recovers_messages_and_reasoning() {
    let assistant = m(Role::Assistant, "42", Some("deep thought"));
    let out = render(
        std::slice::from_ref(&assistant),
        TrlFormat::Harmony,
        CotPolicy::Supervised,
    )
    .unwrap();
    let ingested = ingest_openrouter(&json!({"role": "assistant", "content": out})).unwrap();
    assert_eq!(ingested.content, Content::Text("42".into()));
    assert_eq!(ingested.reasoning.as_deref(), Some("deep thought"));
}

#[test]
fn stripped_drops_reasoning_every_target() {
    let convo = canonical();
    let needle = "80+16"; // appears ONLY in the reasoning (content says "then add")
    for target in [
        TrlFormat::Gemma4,
        TrlFormat::ChatML,
        TrlFormat::ShareGpt,
        TrlFormat::OpenAiMessages,
        TrlFormat::Harmony,
        TrlFormat::TrlPromptCompletion,
    ] {
        let out = render(&convo, target, CotPolicy::Stripped).unwrap();
        assert!(
            !out.contains(needle),
            "stripped render for {target:?} leaked reasoning"
        );
    }
}
