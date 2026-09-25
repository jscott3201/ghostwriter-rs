//! Tool-trajectory fidelity regression: wire → canonical → wire, plus the identity negatives.
//!
//! The fixture (`tests/fixtures/tool_trajectory.json`) is a hand-authored SYNTHETIC conversation
//! with two same-name `read_file` calls, distinct arguments (one string-encoded JSON object), a null
//! assistant `content`, unicode / newline / quote escapes, and the two results in REVERSED order.
//! It is ported from a small external reference suite; it is not competition data, an official
//! schema, or a claimed provider capture.
//!
//! The expectations here are written by hand from the fixture's semantics rather than duplicated
//! in the fixture file, so a regression cannot pass by comparing the fixture against itself.

use gw_format::{FormatError, ingest_openrouter, render, validate_tool_links};
use gw_schema::{Content, CotPolicy, Message, ReasoningDetail, Role, TrlFormat};
use serde_json::{Value, json};

/// The fixture as a parsed value.
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/tool_trajectory.json")).expect("fixture parses")
}

/// The fixture's wire conversation, in order.
fn wire_conversation() -> Vec<Value> {
    fixture()["messages"]
        .as_array()
        .expect("messages array")
        .clone()
}

/// Ingest every wire message into the canonical type.
fn ingest_conversation() -> Vec<Message> {
    wire_conversation()
        .iter()
        .map(|wire| ingest_openrouter(wire).expect("fixture wire message ingests"))
        .collect()
}

// --- positives -------------------------------------------------------------------------------

#[test]
fn fixture_ingests_into_the_expected_canonical_conversation() {
    let msgs = ingest_conversation();
    assert_eq!(msgs.len(), 6);

    assert_eq!(msgs[0].role, Role::System);
    assert_eq!(
        msgs[0].content,
        Content::Text("Inspect a toy module and preserve tool-result identity.".into())
    );

    // Unicode + escaped-quote + newline content survives verbatim.
    assert_eq!(
        msgs[1].content,
        Content::Text("Read two regions; quoted text: \"λ\"; newline follows\nend.".into())
    );

    // The tool-calling assistant turn: NULL content (not empty text), reasoning a sibling, both
    // calls present with their distinct ids and arguments.
    let calls_turn = &msgs[2];
    assert_eq!(calls_turn.role, Role::Assistant);
    assert_eq!(
        calls_turn.content,
        Content::Null,
        "null content must not become \"\""
    );
    assert_eq!(
        calls_turn.reasoning.as_deref(),
        Some("Two regions; the ids distinguish them.")
    );
    let calls = calls_turn
        .tool_calls
        .as_ref()
        .expect("tool_calls populated at ingest");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id.as_deref(), Some("read-a"));
    assert_eq!(calls[1].id.as_deref(), Some("read-b"));
    assert_eq!(calls[0].function.name, "read_file");
    assert_eq!(calls[1].function.name, "read_file");
    // Distinct arguments — the whole point of keeping the calls separate.
    assert_eq!(calls[0].function.arguments["start_line"], 1);
    assert_eq!(calls[1].function.arguments["start_line"], 20);
    // The object payload was already canonical: no raw duplicate.
    assert_eq!(calls[0].function.raw_arguments, None);
    // The string payload was normalized ONCE, and its original wire text is retained.
    assert_eq!(
        calls[1].function.raw_arguments.as_deref(),
        Some("{\"filepath\": \"toy.py\", \"start_line\": 20, \"end_line\": 22}")
    );

    // The results, in REVERSED arrival order, each keeping its EXPLICIT link.
    assert_eq!(msgs[3].role, Role::Tool);
    assert_eq!(msgs[3].tool_call_id.as_deref(), Some("read-b"));
    assert_eq!(msgs[3].name.as_deref(), Some("read_file"));
    assert_eq!(msgs[4].role, Role::Tool);
    assert_eq!(msgs[4].tool_call_id.as_deref(), Some("read-a"));
    assert_eq!(msgs[4].name.as_deref(), Some("read_file"));
    assert_ne!(
        msgs[3], msgs[4],
        "reversed results must stay distinguishable"
    );

    // The closing turn is plain text, and the escaped payload in the result is intact.
    assert_eq!(
        msgs[5].content,
        Content::Text("The second read failed; no repair or verification is claimed.".into())
    );
    assert!(
        matches!(&msgs[4].content, Content::Text(t) if t.contains("λ")),
        "the escaped payload inside the read-a result survives ingest"
    );
}

#[test]
fn identity_survives_a_serde_round_trip_of_the_whole_conversation() {
    let msgs = ingest_conversation();
    let encoded = serde_json::to_string(&msgs).unwrap();
    let decoded: Vec<Message> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, msgs);
    // And the bytes themselves carry the link, so a store-and-reload is lossless.
    let v: Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(v[3]["tool_call_id"], "read-b");
    assert_eq!(v[4]["tool_call_id"], "read-a");
    assert_eq!(v[2]["content"], Value::Null);
}

#[test]
fn swapping_the_two_links_produces_a_different_conversation() {
    // The identity case, stated as an equality: the messages are byte-identical except for which
    // call each result answers, so a faithful type MUST keep them apart.
    let msgs = ingest_conversation();
    let mut swapped = msgs.clone();
    let (a, b) = (
        swapped[3].tool_call_id.take(),
        swapped[4].tool_call_id.take(),
    );
    swapped[3].tool_call_id = b;
    swapped[4].tool_call_id = a;
    assert_ne!(swapped, msgs);
}

#[test]
fn the_whole_trajectory_passes_identity_admission() {
    assert!(validate_tool_links(&ingest_conversation()).is_ok());
}

#[test]
fn openai_messages_render_emits_the_result_link_for_each_tool_turn() {
    let out = render(
        &ingest_conversation(),
        TrlFormat::OpenAiMessages,
        CotPolicy::Supervised,
    )
    .unwrap();
    let v: Value = serde_json::from_str(&out).unwrap();
    let msgs = v["messages"].as_array().unwrap();

    let assistant = &msgs[2];
    assert_eq!(assistant["role"], "assistant");
    // Documented render policy: every render target frames content as a string, so a tool-calling
    // turn (whose only output is `tool_calls`) renders an empty body. The null-vs-empty distinction
    // is preserved in the STORED record — asserted by the serde round-trip above — not in the
    // rendered bytes. Whether the structured 1:1 OpenAI target should instead emit `null` is a
    // renderer-boundary decision, not part of the message contract.
    assert_eq!(assistant["content"], "");
    assert_eq!(assistant["tool_calls"][0]["id"], "read-a");
    assert_eq!(assistant["tool_calls"][1]["id"], "read-b");
    // The normalized object is what rides the wire, with the retained raw text beside it.
    assert_eq!(
        assistant["tool_calls"][1]["function"]["arguments"]["start_line"],
        20
    );
    assert_eq!(
        assistant["tool_calls"][1]["function"]["raw_arguments"],
        "{\"filepath\": \"toy.py\", \"start_line\": 20, \"end_line\": 22}"
    );

    // Each result names the call it answers; a consumer never has to re-infer it from the name.
    assert_eq!(msgs[3]["role"], "tool");
    assert_eq!(msgs[3]["tool_call_id"], "read-b");
    assert_eq!(msgs[3]["name"], "read_file");
    assert_eq!(msgs[4]["tool_call_id"], "read-a");

    // A non-tool turn emits no link key at all (absent, not null).
    assert!(msgs[1].get("tool_call_id").is_none());
    assert!(msgs[0].get("tool_call_id").is_none());
}

#[test]
fn reasoning_kinds_stay_distinct_on_a_null_content_tool_turn() {
    let variants = &fixture()["reasoning_variants"];
    for key in ["absent", "exposed", "summary"] {
        let m = ingest_openrouter(&variants[key]).expect("variant ingests");
        assert_eq!(
            m.content,
            Content::Null,
            "{key}: tool turn keeps null content"
        );
        assert_eq!(
            m.tool_call_id, None,
            "{key}: an assistant turn declares no result link"
        );
    }

    let absent = ingest_openrouter(&variants["absent"]).unwrap();
    assert_eq!(absent.reasoning, None);
    assert_eq!(absent.reasoning_details, None);

    let exposed = ingest_openrouter(&variants["exposed"]).unwrap();
    assert_eq!(
        exposed.reasoning.as_deref(),
        Some("exposed plaintext chain of thought")
    );
    assert!(matches!(
        exposed.reasoning_details.as_ref().unwrap()[0],
        ReasoningDetail::Text { .. }
    ));

    // A summary-only turn is NOT upgraded into exposed full thought: the kind observed is kept.
    let summary = ingest_openrouter(&variants["summary"]).unwrap();
    assert_eq!(summary.reasoning, None, "no plaintext CoT was exposed");
    assert!(matches!(
        summary.reasoning_details.as_ref().unwrap()[0],
        ReasoningDetail::Summary { .. }
    ));
}

// --- negatives -------------------------------------------------------------------------------

/// Ingest the fixture conversation with `mutate` applied to the wire `messages` array.
fn ingest_mutated(mutate: impl FnOnce(&mut Vec<Value>)) -> Result<Vec<Message>, FormatError> {
    let mut wire = wire_conversation();
    mutate(&mut wire);
    wire.iter().map(ingest_openrouter).collect()
}

#[test]
fn a_result_linking_an_undeclared_call_is_rejected() {
    let msgs = ingest_mutated(|wire| {
        wire[3]["tool_call_id"] = json!("read-z");
    })
    .expect("ingest itself is fine — the wire is well formed");
    let err = validate_tool_links(&msgs).unwrap_err();
    assert!(matches!(err, FormatError::ToolIdentity(_)), "{err:?}");
    assert!(err.to_string().contains("read-z"), "{err}");
}

#[test]
fn two_declared_calls_sharing_one_id_are_rejected() {
    let msgs = ingest_mutated(|wire| {
        wire[2]["tool_calls"][1]["id"] = json!("read-a");
    })
    .expect("ingest is fine");
    let err = validate_tool_links(&msgs).unwrap_err();
    assert!(err.to_string().contains("duplicate tool call id"), "{err}");
}

#[test]
fn a_result_with_no_link_among_repeated_names_is_rejected_rather_than_guessed() {
    let msgs = ingest_mutated(|wire| {
        wire[3]["tool_call_id"] = Value::Null;
    })
    .expect("ingest is fine");
    let err = validate_tool_links(&msgs).unwrap_err();
    assert!(err.to_string().contains("refusing to guess"), "{err}");
}

#[test]
fn a_result_whose_only_call_declares_no_id_is_rejected() {
    let mut wire = vec![json!({
        "role": "assistant", "content": null,
        "tool_calls": [{"id": null, "type": "function", "function": {"name": "read_file", "arguments": {}}}]
    })];
    wire.push(json!({"role": "tool", "name": "read_file", "content": "ok"}));
    let msgs = wire
        .iter()
        .map(ingest_openrouter)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let err = validate_tool_links(&msgs).unwrap_err();
    assert!(err.to_string().contains("declares no id"), "{err}");
}

#[test]
fn a_result_arriving_before_its_call_is_rejected_as_partial() {
    let msgs = ingest_mutated(|wire| {
        // Move the second result ahead of the turn that declares its call.
        let result = wire.remove(3);
        wire.insert(2, result);
    })
    .expect("ingest is fine");
    let err = validate_tool_links(&msgs).unwrap_err();
    assert!(err.to_string().contains("partial"), "{err}");
}

#[test]
fn two_results_answering_one_call_are_rejected() {
    let msgs = ingest_mutated(|wire| {
        let dup = wire[3].clone();
        wire.insert(4, dup);
    })
    .expect("ingest is fine");
    let err = validate_tool_links(&msgs).unwrap_err();
    assert!(err.to_string().contains("second result"), "{err}");
}

#[test]
fn malformed_tool_arguments_are_rejected_at_ingest() {
    for bad in ["{not json", "[1, 2]", "7", "null"] {
        let wire = json!({
            "role": "assistant", "content": null,
            "tool_calls": [{"id": "x", "type": "function",
                            "function": {"name": "read_file", "arguments": bad}}]
        });
        let err = ingest_openrouter(&wire).unwrap_err();
        assert!(matches!(err, FormatError::Ingest(_)), "{bad}: {err:?}");
    }
    // A non-object, non-string `arguments` is rejected too.
    let wire = json!({
        "role": "assistant", "content": null,
        "tool_calls": [{"id": "x", "type": "function",
                        "function": {"name": "read_file", "arguments": 7}}]
    });
    assert!(ingest_openrouter(&wire).is_err());
}

#[test]
fn a_non_string_result_link_is_rejected_rather_than_dropped() {
    let mut wire = wire_conversation();
    wire[3]["tool_call_id"] = json!(7);
    let err = ingest_openrouter(&wire[3]).unwrap_err();
    assert!(err.to_string().contains("tool_call_id"), "{err}");
}

// --- the text-only workflow must be untouched --------------------------------------------------

#[test]
fn a_text_only_conversation_is_unchanged_and_still_ingests_and_renders() {
    let wire = [
        json!({"role": "user", "content": "What is 12 * 8?"}),
        json!({"role": "assistant", "content": "<think>10*8+2*8</think>96"}),
    ];
    let msgs: Vec<Message> = wire
        .iter()
        .map(ingest_openrouter)
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(msgs[1].content, Content::Text("96".into()));
    assert_eq!(msgs[1].reasoning.as_deref(), Some("10*8+2*8"));
    assert!(msgs.iter().all(|m| m.tool_calls.is_none()));
    assert!(msgs.iter().all(|m| m.tool_call_id.is_none()));
    // The identity check is a no-op for a conversation that declares no tool fields.
    assert!(validate_tool_links(&msgs).is_ok());
    // And the OpenAI-messages render emits no link keys at all.
    let out = render(&msgs, TrlFormat::OpenAiMessages, CotPolicy::Supervised).unwrap();
    let v: Value = serde_json::from_str(&out).unwrap();
    for m in v["messages"].as_array().unwrap() {
        assert!(m.get("tool_call_id").is_none());
    }
}
