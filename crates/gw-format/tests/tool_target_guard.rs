//! The render-boundary tool guard: a tool trajectory FAILS CLOSED on every target that has no tool
//! representation, and the two tool-faithful targets still emit the calls and the result links.
//!
//! The fixture is the same hand-authored SYNTHETIC conversation the identity tests use
//! (`tests/fixtures/tool_trajectory.json` — two same-name `read_file` calls, a null-content
//! tool-calling turn, results in reversed order). The expectations are written here by hand from
//! the fixture's semantics, not read back out of it, so a regression cannot pass by comparing the
//! fixture against itself. Nothing here renders for a model, fetches a template, or substitutes a
//! 12B checkpoint: the guard is a pure function of the canonical `Message[]`.

use gw_format::{FormatError, ToolCallRecovery, ToolSignal, ingest_openrouter, render};
use gw_schema::{Content, CotPolicy, Message, Role, TrlFormat};
use serde_json::Value;

/// The targets with no tool representation: Gemma-4, ChatML, ShareGPT, Harmony.
const DROPPING: [TrlFormat; 4] = [
    TrlFormat::Gemma4,
    TrlFormat::ChatML,
    TrlFormat::ShareGpt,
    TrlFormat::Harmony,
];

/// The targets that carry the signals verbatim: the OpenAI wire shape and the TRL shape built from
/// it.
const PRESERVING: [TrlFormat; 2] = [TrlFormat::OpenAiMessages, TrlFormat::TrlPromptCompletion];

/// Every target, for the text-only non-regression.
const ALL: [TrlFormat; 6] = [
    TrlFormat::Gemma4,
    TrlFormat::ChatML,
    TrlFormat::ShareGpt,
    TrlFormat::OpenAiMessages,
    TrlFormat::Harmony,
    TrlFormat::TrlPromptCompletion,
];

/// The tool-bearing fixture conversation as canonical messages.
fn tool_trajectory() -> Vec<Message> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/tool_trajectory.json"))
        .expect("fixture parses");
    fixture["messages"]
        .as_array()
        .expect("messages array")
        .iter()
        .map(|wire| ingest_openrouter(wire).expect("fixture wire message ingests"))
        .collect()
}

/// A clean text-only conversation: the historical shape that must keep working everywhere.
fn text_only() -> Vec<Message> {
    vec![
        Message {
            role: Role::User,
            content: Content::Text("What is 12 * 8?".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        Message {
            role: Role::Assistant,
            content: Content::Text("96".into()),
            reasoning: Some("10*8=80, 2*8=16, 80+16=96".into()),
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ]
}

// ------------------------------ fail closed, per route --------------------------------

#[test]
fn every_dropping_target_refuses_the_tool_trajectory_with_a_precise_diagnostic() {
    for target in DROPPING {
        let err = render(&tool_trajectory(), target, CotPolicy::Supervised)
            .expect_err("a target with no tool representation must not render a tool trajectory");
        match err {
            FormatError::UnsupportedToolCalls {
                target: refused,
                signals,
                index,
                recovery,
            } => {
                // The route: the caller is told WHICH target refused, not just "some target".
                assert_eq!(refused, target);
                // The position: the assistant turn that declares the calls is messages[2].
                assert_eq!(index, 2, "{target:?} named the wrong message");
                // The signal set: all three carriers are present in this fixture, and the error
                // says which ones would have been dropped.
                assert_eq!(
                    signals,
                    vec![
                        ToolSignal::ToolCalls,
                        ToolSignal::ToolCallId,
                        ToolSignal::ToolRole,
                    ],
                    "{target:?}"
                );
                // The recovery: the canonical export + an external official-template consumer.
                assert_eq!(
                    recovery,
                    ToolCallRecovery::CanonicalExportAndOfficialTemplate
                );
            }
            other => panic!("{target:?} returned the wrong error class: {other:?}"),
        }
    }
}

#[test]
fn the_refusal_does_not_depend_on_the_cot_policy() {
    // Stripping the reasoning does not make a tool trajectory representable, so the guard is
    // policy-independent: it must not be something a caller can turn off with `--cot stripped`.
    for cot in [
        CotPolicy::Supervised,
        CotPolicy::Masked,
        CotPolicy::Stripped,
    ] {
        let err = render(&tool_trajectory(), TrlFormat::Gemma4, cot).unwrap_err();
        assert!(
            matches!(
                err,
                FormatError::UnsupportedToolCalls {
                    target: TrlFormat::Gemma4,
                    ..
                }
            ),
            "{cot:?}: {err:?}"
        );
    }
}

#[test]
fn a_bare_tool_result_turn_is_refused_even_with_no_calls_and_no_link() {
    // The case a "does it have tool_calls?" check would miss: a result turn with neither a declared
    // call nor a link still has no representation here — it would be emitted as an ordinary text
    // turn, so the pairing would be unrecoverable.
    let msgs = vec![
        Message {
            role: Role::User,
            content: Content::Text("read it".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        Message {
            role: Role::Tool,
            content: Content::Text("ok".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        Message {
            role: Role::Assistant,
            content: Content::Text("done".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];
    for target in DROPPING {
        let err = render(&msgs, target, CotPolicy::Supervised).unwrap_err();
        match err {
            FormatError::UnsupportedToolCalls {
                target: refused,
                signals,
                index,
                ..
            } => {
                assert_eq!(refused, target);
                assert_eq!(index, 1);
                assert_eq!(signals, vec![ToolSignal::ToolRole], "{target:?}");
            }
            other => panic!("{target:?} returned the wrong error class: {other:?}"),
        }
    }
}

#[test]
fn the_control_token_leak_is_still_reported_ahead_of_the_tool_guard() {
    // Precedence is deterministic: a leaked control token is a data-integrity fault on EVERY
    // target, so it is reported first. (The tool guard runs after `validate_clean`.)
    let mut msgs = tool_trajectory();
    msgs[5].reasoning = Some("leaked <think> tag".into());
    let err = render(&msgs, TrlFormat::Gemma4, CotPolicy::Supervised).unwrap_err();
    assert!(matches!(
        err,
        FormatError::ControlTokenInContent {
            token: "<think>",
            role: Role::Assistant,
        }
    ));
}

#[test]
fn the_human_message_names_the_route_the_position_and_the_way_out() {
    // The structured fields above are the contract; this is the operator-facing string, which must
    // still be actionable on its own (it is what a log line shows).
    let err = render(&tool_trajectory(), TrlFormat::Gemma4, CotPolicy::Supervised).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Gemma4"), "{msg}");
    assert!(msg.contains("messages[2]"), "{msg}");
    assert!(msg.contains("ToolCalls"), "{msg}");
    assert!(msg.contains("messages_json"), "{msg}");
}

// ------------------------- the tool-faithful targets still work -----------------------

#[test]
fn both_tool_faithful_targets_render_the_trajectory_with_the_call_ids_present() {
    // A guard applied to every route by mistake would make the trajectory unexportable anywhere;
    // asserting the ids appear in the bytes of BOTH preserving routes pins the exemption.
    for target in PRESERVING {
        let out = render(&tool_trajectory(), target, CotPolicy::Supervised)
            .unwrap_or_else(|e| panic!("{target:?} must render a tool trajectory: {e}"));
        for id in ["read-a", "read-b"] {
            assert!(out.contains(id), "{target:?} dropped call id {id}");
        }
    }
}

#[test]
fn openai_messages_keeps_the_calls_and_the_result_links() {
    let out = render(
        &tool_trajectory(),
        TrlFormat::OpenAiMessages,
        CotPolicy::Supervised,
    )
    .expect("a tool-faithful target is never refused");
    let v: Value = serde_json::from_str(&out).unwrap();
    let msgs = v["messages"].as_array().unwrap();

    // Both declared calls, in order, with their distinct ids.
    assert_eq!(msgs[2]["tool_calls"][0]["id"], "read-a");
    assert_eq!(msgs[2]["tool_calls"][1]["id"], "read-b");
    assert_eq!(
        msgs[2]["tool_calls"][1]["function"]["arguments"]["start_line"],
        20
    );
    // Each result still names the call it answers, in the fixture's reversed arrival order.
    assert_eq!(msgs[3]["tool_call_id"], "read-b");
    assert_eq!(msgs[3]["name"], "read_file");
    assert_eq!(msgs[4]["tool_call_id"], "read-a");
    // The null-content tool turn renders an empty body (documented policy) with the calls beside it.
    assert_eq!(msgs[2]["content"], "");
}

#[test]
fn trl_prompt_completion_keeps_the_tool_trajectory_across_the_split() {
    // The fixture ends on an assistant turn, so it IS a prompt/completion pair. The split at that
    // final turn must not disturb the tool region living in the prompt.
    let out = render(
        &tool_trajectory(),
        TrlFormat::TrlPromptCompletion,
        CotPolicy::Supervised,
    )
    .expect("a tool-faithful target is never refused");
    let v: Value = serde_json::from_str(&out).unwrap();
    let prompt = v["prompt"].as_array().unwrap();
    let completion = v["completion"].as_array().unwrap();

    assert_eq!(
        prompt.len(),
        5,
        "everything before the final assistant turn"
    );
    assert_eq!(completion.len(), 1);
    assert_eq!(prompt[2]["tool_calls"][0]["id"], "read-a");
    assert_eq!(prompt[2]["tool_calls"][1]["id"], "read-b");
    assert_eq!(prompt[3]["tool_call_id"], "read-b");
    assert_eq!(prompt[4]["tool_call_id"], "read-a");
    // The supervised turn itself carries no tool keys (absent, not null).
    assert!(completion[0].get("tool_calls").is_none());
    assert!(completion[0].get("tool_call_id").is_none());
    assert_eq!(completion[0]["role"], "assistant");
}

// ----------------------------- text-only non-regression -------------------------------

#[test]
fn a_text_only_conversation_still_renders_on_every_target() {
    let msgs = text_only();
    for target in ALL {
        let out = render(&msgs, target, CotPolicy::Supervised)
            .unwrap_or_else(|e| panic!("{target:?} must still render text-only: {e}"));
        assert!(!out.is_empty(), "{target:?} produced nothing");
    }
}

#[test]
fn a_text_only_conversation_matches_the_golden_gemma4_bytes() {
    // The fail-closed guard must not have moved a single byte of the initial fine-tune target.
    // (The same bytes are asserted against the golden file in tests/golden.rs.)
    let out = render(&text_only(), TrlFormat::Gemma4, CotPolicy::Supervised).unwrap();
    assert!(
        out.starts_with("<bos><|turn>user\nWhat is 12 * 8?<turn|>"),
        "{out}"
    );
    assert!(
        out.contains("<|channel>thought\n10*8=80, 2*8=16, 80+16=96\n<channel|>96<turn|>"),
        "{out}"
    );
}

#[test]
fn an_empty_conversation_and_a_single_assistant_turn_are_unaffected() {
    assert!(render(&[], TrlFormat::Gemma4, CotPolicy::Stripped).is_ok());
    let single = vec![Message {
        role: Role::Assistant,
        content: Content::Parts(vec![gw_schema::ContentPart::Text {
            text: "parts flatten".into(),
        }]),
        reasoning: None,
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];
    for target in ALL {
        assert!(
            render(&single, target, CotPolicy::Stripped).is_ok(),
            "{target:?}"
        );
    }
}
