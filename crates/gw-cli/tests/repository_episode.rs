//! Authored declarations exercise identity and authority independently of any external evaluator.
use gw_judge::{capture_repository_episode, verify_repository_episode};
use gw_schema::{RepositoryEpisodeArtifact, RepositoryEpisodeRequest, strict_repository_json};
use serde_json::{Value, json};

const FIXTURE: &[u8] = include_bytes!("fixtures/repository-episode-request.json");
fn request() -> Value {
    strict_repository_json(FIXTURE).unwrap()
}
fn capture(value: &Value) -> RepositoryEpisodeArtifact {
    capture_repository_episode(&serde_json::to_vec(value).unwrap()).unwrap()
}
fn wire(value: &impl serde::Serialize) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}
fn passing_request() -> Value {
    let mut value = request();
    let artifact = capture(&value);
    value["report"] = json!({
        "publisher":"authored-evaluator", "source_ref":"never-open:this-report",
        "raw":{"status":"publisher-green", "cases":[{"id":"case-a","status":"OK"},{"id":"case-b","status":"OK"}]},
        "execution":{
            "outcome":"passed", "required_tests":["publisher-only"],
            "cases":[{"node":"case-a","status":"passed"},{"node":"case-b","status":"passed"}],
            "exit_code":0, "errors":[], "source_ref":"never-open:normalized",
            "binding":{"task":artifact.identities.task,"attempt":"attempt-1","patch_hash":artifact.identities.candidate}
        }
    });
    value
}

#[test]
fn repository_identity_domains_and_capture_roundtrip() {
    let value = request();
    let artifact = capture(&value);
    assert!(artifact.identities.pending.is_empty());
    assert!(artifact.identities.task.is_some());
    assert!(artifact.identities.candidate.is_some());
    assert_ne!(artifact.identities.task, artifact.identities.candidate);
    let report = verify_repository_episode(&wire(&artifact)).unwrap();
    assert_eq!(report.identities, artifact.identities);
    assert_eq!(
        artifact,
        capture(&serde_json::to_value(&artifact.request).unwrap())
    );
    assert_eq!(
        artifact
            .request
            .candidate
            .delta
            .changes
            .iter()
            .map(|c| c.path.as_str())
            .collect::<Vec<_>>(),
        ["bin/run", "gone.txt", "new.txt", "src/a.py"]
    );
    assert_eq!(
        artifact.request.candidate.messages[2]
            .tool_call_id
            .as_deref(),
        Some("b")
    );
    assert_eq!(
        artifact.request.candidate.messages[3]
            .tool_call_id
            .as_deref(),
        Some("a")
    );
    let encoded = String::from_utf8(wire(&artifact)).unwrap();
    assert!(encoded.contains("old\\r\\n"));
    assert!(encoded.contains("\"text\":\"new\""));
    assert!(encoded.contains("\"negative_zero\":-0.0"));
}
#[test]
fn repository_identity_binds_tools_results_diff_attempt_and_all_task_semantics() {
    let base = request();
    let original = capture(&base).identities;
    for (pointer, new, task_changed) in [
        (
            "/candidate/messages/4/content",
            json!("different prose"),
            false,
        ),
        (
            "/candidate/messages/1/reasoning",
            json!("different reasoning"),
            false,
        ),
        (
            "/candidate/messages/1/tool_calls/0/function/arguments/fraction",
            json!(0.75),
            false,
        ),
        (
            "/candidate/messages/1/tool_calls/0/function/raw_arguments",
            json!("raw spacing changed"),
            false,
        ),
        (
            "/candidate/messages/2/content",
            json!("different result"),
            false,
        ),
        (
            "/candidate/tools/0/function/parameters/description",
            json!("not present"),
            false,
        ),
        (
            "/candidate/delta/changes/0/after/text",
            json!("new\n"),
            false,
        ),
        (
            "/candidate/delta/changes/0/after/mode",
            json!("100755"),
            false,
        ),
        ("/candidate/attempt", json!("attempt-2"), false),
        ("/task/problem", json!("Changed task"), true),
        ("/task/source/revision", json!("fixture-v2"), true),
        (
            "/task/upstream_base/hex",
            json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            true,
        ),
        (
            "/task/actor_baseline/hex",
            json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
            true,
        ),
        ("/task/environment/platform", json!("linux/arm64"), true),
        ("/task/environment/recipe", json!("recipe-v2"), true),
        ("/task/environment/digest/hex", json!("a".repeat(64)), true),
        (
            "/task/private_contract/digest/hex",
            json!("b".repeat(64)),
            true,
        ),
        (
            "/task/private_contract/required_test_ids/0",
            json!("new-case"),
            true,
        ),
    ] {
        let mut changed = base.clone();
        if pointer.ends_with("/description") {
            changed["candidate"]["tools"][0]["function"]["parameters"]["description"] = new;
        } else {
            *changed.pointer_mut(pointer).unwrap() = new;
        }
        let actual = capture(&changed).identities;
        assert_ne!(original.candidate, actual.candidate, "{pointer}");
        assert_ne!(original.capture, actual.capture, "{pointer}");
        assert_eq!(original.task != actual.task, task_changed, "{pointer}");
    }
}
#[test]
fn repository_rights_split_usage_generation_and_report_only_change_capture() {
    let base = passing_request();
    let original = capture(&base).identities;
    for (pointer, new) in [
        ("/task/task_id", json!("renamed-label")),
        ("/task/group/id", json!("other-group")),
        ("/task/rights/reviewer", json!("other-reviewer")),
        ("/task/split/role", json!("train")),
        ("/generation/usage/input_tokens", json!(0)),
        ("/generation/usage/cost_usd", json!(0.0)),
        ("/generation/settings/temperature", json!(0.6)),
        (
            "/generation/model",
            json!({"status":"declared","value":{"model":"fixture"}}),
        ),
        (
            "/generation/serving/value/implementation",
            json!("other-server"),
        ),
        ("/report/raw/status", json!("publisher-red")),
        (
            "/report/execution/required_tests",
            json!(["different publisher coverage"]),
        ),
    ] {
        let mut changed = base.clone();
        *changed.pointer_mut(pointer).unwrap() = new;
        let actual = capture(&changed).identities;
        assert_eq!(original.task, actual.task, "{pointer}");
        assert_eq!(original.candidate, actual.candidate, "{pointer}");
        assert_ne!(original.capture, actual.capture, "{pointer}");
    }
    let mut changed = base.clone();
    changed["candidate"]["delta"]["changes"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert_eq!(capture(&changed).identities, original);
    changed["candidate"]["messages"]
        .as_array_mut()
        .unwrap()
        .swap(2, 3);
    assert_ne!(capture(&changed).identities.candidate, original.candidate);
}
#[test]
fn repository_missing_immutable_inputs_remain_inspectable() {
    for pointer in [
        "/task/upstream_base",
        "/task/actor_baseline",
        "/task/environment",
        "/task/private_contract",
    ] {
        let mut value = passing_request();
        *value.pointer_mut(pointer).unwrap() = Value::Null;
        let artifact = capture(&value);
        assert!(artifact.identities.task.is_none(), "{pointer}");
        assert!(artifact.identities.candidate.is_none());
        assert!(!artifact.identities.pending.is_empty());
        assert_eq!(
            artifact.assessment.declared_outcome,
            gw_schema::VerificationOutcome::Unknown
        );
        assert_eq!(
            artifact.assessment.report_binding,
            gw_schema::RepositoryReportBinding::Pending
        );
        verify_repository_episode(&wire(&artifact)).unwrap();
    }
    for (pointer, new) in [
        ("/task/source/revision", json!("")),
        ("/task/private_contract/required_test_ids", json!([])),
    ] {
        let mut value = passing_request();
        *value.pointer_mut(pointer).unwrap() = new;
        let artifact = capture(&value);
        assert!(artifact.identities.task.is_none());
        assert_eq!(
            artifact.assessment.declared_outcome,
            gw_schema::VerificationOutcome::Unknown
        );
    }
}
#[test]
fn repository_incomplete_or_unsupported_candidate_never_gets_complete_identity() {
    for (pointer, new) in [
        ("/candidate/trajectory_complete", json!(false)),
        ("/candidate/tools", Value::Null),
        ("/candidate/messages", json!([])),
        ("/candidate/delta/enumeration", json!("incomplete")),
        ("/candidate/delta/unsupported", json!(["binary"])),
        ("/candidate/delta/unsupported", json!(["submodule"])),
        (
            "/candidate/delta/unsupported",
            json!(["symlink", "non_utf8_path"]),
        ),
    ] {
        let mut value = passing_request();
        *value.pointer_mut(pointer).unwrap() = new;
        let artifact = capture(&value);
        assert!(artifact.identities.task.is_some());
        assert!(artifact.identities.candidate.is_none(), "{pointer}");
        assert_eq!(
            serde_json::to_value(&artifact.assessment).unwrap(),
            json!({"report_binding":"pending","declared_outcome":"unknown","observed_execution":"unknown","training_eligible":false})
        );
        verify_repository_episode(&wire(&artifact)).unwrap();
    }
}
#[test]
fn repository_explicit_links_reject_guessing_duplicates_and_partial_complete_claims() {
    for (pointer, new) in [
        ("/candidate/messages/2/tool_call_id", Value::Null),
        ("/candidate/messages/2/tool_call_id", json!("missing")),
        ("/candidate/messages/2/tool_call_id", json!("a")),
        ("/candidate/messages/2/name", json!("wrong-name")),
        ("/candidate/messages/1/tool_calls/0/id", Value::Null),
        ("/candidate/messages/1/tool_calls/0/id", json!("b")),
        (
            "/candidate/messages/1/tool_calls/0/function/arguments",
            json!("{}"),
        ),
        ("/candidate/tools", json!([])),
    ] {
        let mut value = request();
        *value.pointer_mut(pointer).unwrap() = new;
        assert!(
            capture_repository_episode(&wire(&value)).is_err(),
            "{pointer}"
        );
    }
    let mut value = request();
    value["candidate"]["messages"]
        .as_array_mut()
        .unwrap()
        .swap(1, 2);
    assert!(capture_repository_episode(&wire(&value)).is_err());
    let mut value = request();
    value["candidate"]["messages"]
        .as_array_mut()
        .unwrap()
        .remove(2);
    assert!(capture_repository_episode(&wire(&value)).is_err());
    value["candidate"]["trajectory_complete"] = json!(false);
    assert!(capture(&value).identities.candidate.is_none());
    let mut value = request();
    value["candidate"]["messages"]
        .as_array_mut()
        .unwrap()
        .remove(2);
    value["candidate"]["messages"][1]["tool_calls"]
        .as_array_mut()
        .unwrap()
        .remove(1);
    assert!(
        capture(&value).identities.candidate.is_some(),
        "serial explicit calls also work"
    );
}
#[test]
fn repository_report_declared_pass_never_becomes_observed_or_eligible() {
    let value = passing_request();
    let artifact = capture(&value);
    assert_eq!(
        serde_json::to_value(&artifact.assessment).unwrap(),
        json!({"report_binding":"matched","declared_outcome":"pass","observed_execution":"unknown","training_eligible":false})
    );
    assert_eq!(
        serde_json::to_value(&artifact.request.report).unwrap(),
        value["report"]
    );
    for (pointer, new, expected) in [
        ("/report/execution/cases/0/status", json!("failed"), "fail"),
        ("/report/execution/cases/0/status", json!("skipped"), "fail"),
        (
            "/report/execution/cases/0/node",
            json!("not-required"),
            "fail",
        ),
        ("/report/execution/cases/0/node", json!("case-b"), "unknown"),
        ("/report/execution/cases", json!([]), "unknown"),
        ("/report/execution/exit_code", Value::Null, "unknown"),
        ("/report/execution/exit_code", json!(1), "fail"),
        (
            "/report/execution/errors",
            json!(["publisher error"]),
            "fail",
        ),
        ("/report/execution/outcome", json!("unknown"), "unknown"),
        ("/report/execution/outcome", json!("failed"), "fail"),
        ("/report/execution", Value::Null, "unknown"),
    ] {
        let mut changed = value.clone();
        *changed.pointer_mut(pointer).unwrap() = new;
        let artifact = capture(&changed);
        let assessment = serde_json::to_value(&artifact.assessment).unwrap();
        assert_eq!(assessment["declared_outcome"], expected, "{pointer}");
        assert_eq!(assessment["observed_execution"], "unknown");
        assert_eq!(assessment["training_eligible"], false);
        verify_repository_episode(&wire(&artifact)).unwrap();
    }
}
#[test]
fn repository_report_binding_changes_are_stale_or_foreign_before_interpretation() {
    for (pointer, new, expected) in [
        (
            "/candidate/messages/2/content",
            json!("changed result"),
            "stale",
        ),
        (
            "/candidate/delta/changes/0/after/text",
            json!("changed patch"),
            "stale",
        ),
        (
            "/candidate/tools/0/function/parameters/required/0",
            json!("fraction"),
            "stale",
        ),
        ("/task/upstream_base/hex", json!("b".repeat(40)), "foreign"),
        ("/candidate/attempt", json!("second attempt"), "foreign"),
        ("/task/actor_baseline/hex", json!("a".repeat(40)), "foreign"),
        (
            "/report/execution/binding/task",
            json!("another task"),
            "foreign",
        ),
        (
            "/report/execution/binding/patch_hash",
            json!("another candidate"),
            "stale",
        ),
    ] {
        let mut value = passing_request();
        *value.pointer_mut(pointer).unwrap() = new;
        let artifact = capture(&value);
        assert_eq!(
            serde_json::to_value(&artifact.request.report).unwrap(),
            value["report"]
        );
        let assessment = serde_json::to_value(artifact.assessment).unwrap();
        assert_eq!(assessment["report_binding"], expected);
        assert_eq!(assessment["declared_outcome"], "unknown");
    }
}
#[test]
fn repository_verification_rejects_rehashed_or_unhashed_authority_and_content_tampering() {
    let original = serde_json::to_value(capture(&passing_request())).unwrap();
    for (pointer, new) in [
        (
            "/request/candidate/messages/2/content",
            json!("substituted"),
        ),
        ("/identities/task", json!("a".repeat(64))),
        ("/identities/candidate", Value::Null),
        ("/identities/capture", json!("a".repeat(64))),
        ("/identities/pending", json!(["invented"])),
        ("/assessment/declared_outcome", json!("fail")),
        ("/assessment/observed_execution", json!("pass")),
        ("/assessment/training_eligible", json!(true)),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(pointer).unwrap() = new;
        assert!(
            verify_repository_episode(&wire(&changed)).is_err(),
            "{pointer}"
        );
        changed["identities"] =
            serde_json::to_value(capture(&changed["request"]).identities).unwrap();
        if pointer.starts_with("/assessment/") {
            assert!(
                verify_repository_episode(&wire(&changed)).is_err(),
                "rehashed {pointer}"
            );
        }
    }
}
#[test]
fn repository_numbers_keep_fractional_nested_values_and_explicit_negative_zero() {
    for raw in [
        r#"{"a":-0,"b":-0.0,"c":-0e0,"d":0,"e":0.125,"f":1e3,"g":[true,null,{"x":1.2345678901234567}]}"#,
        r#"{"min":-9223372036854775808,"max":18446744073709551615,"tiny":5e-324,"big":1.7976931348623157e308}"#,
    ] {
        let value = strict_repository_json(raw.as_bytes()).unwrap();
        let first = wire(&value);
        assert_eq!(wire(&strict_repository_json(&first).unwrap()), first);
    }
    let value = strict_repository_json(br#"[-0,-0.0,-0e0,0,0.0]"#).unwrap();
    assert_eq!(
        String::from_utf8(wire(&value)).unwrap(),
        "[-0.0,-0.0,-0.0,0,0.0]"
    );
    let source = String::from_utf8(FIXTURE.to_vec())
        .unwrap()
        .replace("\"negative_zero\": -0.0", "\"negative_zero\": -0");
    assert_eq!(
        capture_repository_episode(source.as_bytes())
            .unwrap()
            .identities,
        capture(&request()).identities
    );
    for raw in [
        "18446744073709551616",
        "-9223372036854775809",
        "1e999",
        "1e-999",
        "NaN",
        "Infinity",
        "01",
        "[1,]",
        r#"{"x":1,"x":2}"#,
        r#"{"x":{"a":1,"a":2}}"#,
    ] {
        assert!(strict_repository_json(raw.as_bytes()).is_err(), "{raw}");
    }
}
#[test]
fn repository_strict_fields_reject_private_contract_content_and_nested_unknowns() {
    for pointer in [
        "",
        "/task",
        "/task/private_contract",
        "/candidate",
        "/candidate/messages/1",
        "/candidate/messages/1/tool_calls/0",
        "/candidate/messages/1/tool_calls/0/function",
        "/candidate/delta/changes/0",
        "/generation/usage",
        "/report/execution",
        "/report/execution/binding",
        "/report/execution/cases/0",
    ] {
        let mut value = passing_request();
        value
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("private_test_contents".into(), json!("DO_NOT_ECHO_PRIVATE"));
        let error = capture_repository_episode(&wire(&value))
            .unwrap_err()
            .to_string();
        assert!(!error.contains("DO_NOT_ECHO_PRIVATE"));
        assert!(!error.contains("private_test_contents"));
    }
    let mut value = request();
    value["candidate"]["messages"][1]["reasoning_details"] =
        json!([{"type":"reasoning.text","text":"reasoning","index":0,"unexpected":"secret"}]);
    assert!(capture_repository_episode(&wire(&value)).is_err());
}
#[test]
fn repository_paths_reject_ambiguity_and_preserve_supported_unicode_text() {
    for invalid in [
        "/absolute",
        "../traverse",
        "a/../b",
        "a//b",
        "a/./b",
        "C:/drive",
        "a\\b",
        ".git/config",
        "NUL.txt",
        "a.",
        "a ",
        "cafe\u{301}.txt",
    ] {
        let mut value = request();
        value["candidate"]["delta"]["changes"][0]["path"] = json!(invalid);
        assert!(
            capture_repository_episode(&wire(&value)).is_err(),
            "{invalid}"
        );
    }
    for duplicate in ["bin/run", "BIN/RUN"] {
        let mut value = request();
        value["candidate"]["delta"]["changes"][0]["path"] = json!(duplicate);
        assert!(capture_repository_episode(&wire(&value)).is_err());
    }
    let mut value = request();
    value["candidate"]["delta"]["changes"][0]["path"] = json!("café/中.txt");
    value["candidate"]["delta"]["changes"][0]["after"]["text"] = json!("cafe\u{301} 中 🙂\r\n");
    assert!(capture(&value).identities.candidate.is_some());
    value["candidate"]["delta"]["changes"][0]["after"]["text"] = json!("binary\0bytes");
    assert!(capture_repository_episode(&wire(&value)).is_err());
    let mut value = request();
    value["candidate"]["delta"]["changes"][0]["path"] = json!("bin");
    assert!(capture_repository_episode(&wire(&value)).is_err());
    let mut value = request();
    value["candidate"]["delta"]["changes"][0]["after"]["mode"] = json!("120000");
    assert!(capture_repository_episode(&wire(&value)).is_err());
}
#[test]
fn repository_rejects_credentialed_locator_and_preserves_opaque_refs_without_opening() {
    for locator in [
        "https://user:password@example.org/repo",
        "https://example.org/repo?token=secret",
        "file:///private/test",
        "https://example.org/../repo",
    ] {
        let mut value = request();
        value["task"]["repository"] = json!(locator);
        assert!(capture_repository_episode(&wire(&value)).is_err());
    }
    let value = passing_request();
    let artifact = capture(&value);
    assert_eq!(
        artifact.request.report.unwrap().source_ref.as_deref(),
        Some("never-open:this-report")
    );
}
#[test]
fn repository_canonical_request_parser_does_not_change_legacy_integer_decoder() {
    assert!(gw_schema::strict_coding_json(br#"{"fraction":0.125}"#).is_err());
    assert!(RepositoryEpisodeRequest::from_json(FIXTURE).is_ok());
}

#[test]
fn repository_unicode_aliases_and_file_to_directory_transition_are_explicit() {
    for (first, second) in [("café.txt", "CAFÉ.txt"), ("straße.txt", "STRASSE.txt")] {
        let mut value = request();
        value["candidate"]["delta"]["changes"][0]["path"] = json!(first);
        value["candidate"]["delta"]["changes"][1]["path"] = json!(second);
        assert!(capture_repository_episode(&wire(&value)).is_err());
    }
    let mut value = request();
    value["candidate"]["delta"]["changes"] = json!([
        {"path":"old","before":{"kind":"text","text":"was a file","mode":"100644"},"after":{"kind":"absent"}},
        {"path":"old/child","before":{"kind":"absent"},"after":{"kind":"text","text":"now a child","mode":"100644"}}
    ]);
    assert!(capture(&value).identities.candidate.is_some());
}
#[test]
fn repository_structured_reasoning_and_unknown_usage_remain_distinct() {
    let mut value = request();
    value["candidate"]["messages"][1]["reasoning_details"] = json!([
        {"type":"reasoning.text","text":"Separate reasoning 中","index":0,"signature":"declared-signature"},
        {"type":"reasoning.encrypted","data":"opaque","index":1}
    ]);
    let artifact = capture(&value);
    assert_eq!(
        serde_json::to_value(&artifact.request.candidate.messages[1]).unwrap(),
        value["candidate"]["messages"][1]
    );
    let mut zero = value.clone();
    zero["generation"]["usage"] =
        json!({"input_tokens":0,"output_tokens":0,"reasoning_tokens":0,"cost_usd":0.0});
    let zero = capture(&zero);
    assert_ne!(artifact.identities.capture, zero.identities.capture);
    assert_eq!(artifact.identities.candidate, zero.identities.candidate);
    value["candidate"]["messages"][0]["content"] =
        json!([{"type":"image_url","image_url":"never-open:image"}]);
    assert!(capture(&value).identities.candidate.is_none());
    let mut unknown = request();
    unknown["candidate"]["tools"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type":"other","payload":{"fraction":0.25}}));
    let artifact = capture(&unknown);
    assert!(artifact.identities.candidate.is_none());
    assert_eq!(
        artifact.request.candidate.tools.unwrap()[1],
        unknown["candidate"]["tools"][1]
    );
}
