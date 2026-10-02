//! Literal serial-tool source for independent official-tokenizer and native-reader checks.
use super::*;
use gw_schema::{CotPolicy, TrlFormat};
use serde_json::json;

fn source() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../gw-format/tests/fixtures/gemma31b-serial-source.json"
    ))
    .unwrap()
}
fn record(source: &serde_json::Value, id: &str) -> TrainingRecord {
    serde_json::from_value(json!({
        "record_id":id,"schema_version":"1.0.0","training_area":"tools",
        "messages":source["messages"],"tools":source["tools"],
        "provenance":{"run_id":"gemma31b-fixture","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"0.1.0"},
        "generation":{},"lifecycle":{"state":"formatted"},"judging":{"panel":[],"verdict":"admit","aggregate":0.9}
    })).unwrap()
}
fn fixture(records: &[TrainingRecord], filename: &str) {
    let mut plan = ExportPlan::prepare(
        records,
        ExportOptions {
            target: TrlFormat::OpenAiMessages,
            cot_policy: CotPolicy::Masked,
            dataset_version: None,
            scope: ExportScope::Records,
        },
    )
    .unwrap();
    if filename == "v5-gemma31b-negative-zero.parquet" {
        let raw = &mut plan.rows[0].messages_json;
        assert!(raw.contains("\"negative\":-0.0"));
        *raw = raw.replace("\"negative\":-0.0", "\"negative\":-0");
        plan.artifact.artifact_id = artifact_identity(&plan.artifact, &plan.rows).unwrap();
    }
    let mut bytes = Vec::new();
    crate::export::write_parquet(&plan.rows, &plan.artifact, &mut bytes).unwrap();
    let report = verify_artifact_snapshot(bytes.clone()).unwrap();
    assert_eq!(
        report.artifact.manifest.column_schema_version,
        ExportSchemaVersion::ToolDefinitions
    );
    if let Some(directory) = std::env::var_os("GW_REGENERATE_GEMMA31B_FIXTURES") {
        std::fs::write(Path::new(&directory).join(filename), &bytes).unwrap();
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../adapters/trl/tests/fixtures")
        .join(filename);
    assert_eq!(
        verify_artifact_snapshot(std::fs::read(path).unwrap())
            .unwrap()
            .artifact,
        report.artifact
    );
}
#[test]
fn serial_gemma31b_source_fixture_is_a_complete_v5_artifact() {
    fixture(
        &[record(&source(), "gemma31b-serial")],
        "v5-gemma31b-serial.parquet",
    );
}
#[test]
fn serial_gemma31b_canonical_capture_retains_unsupported_complete_sources() {
    let mut incomplete = source();
    incomplete["messages"].as_array_mut().unwrap().truncate(5);
    let mut scientific = source();
    scientific["messages"][2]["tool_calls"][0]["function"]["arguments"]["small"] = json!(0.00001);
    let mut parallel = source();
    let mut extra = parallel["messages"][2]["tool_calls"][0].clone();
    extra["id"] = json!("parallel-call");
    parallel["messages"][2]["tool_calls"]
        .as_array_mut()
        .unwrap()
        .push(extra);
    fixture(
        &[
            record(&incomplete, "incomplete-suffix"),
            record(&scientific, "scientific-domain"),
            record(&parallel, "parallel-calls"),
        ],
        "v5-gemma31b-rejected.parquet",
    );
}

#[test]
fn serial_gemma31b_mixed_schema_types_remain_captured_records() {
    let mut records = vec![record(&source(), "valid-source")];
    for (name, kind) in [
        ("array-type", json!([])),
        ("object-type", json!({})),
        ("union-type", json!(["string", "null"])),
    ] {
        let mut invalid = source();
        invalid["tools"][0]["function"]["parameters"]["properties"]["query"]["type"] = kind;
        records.push(record(&invalid, name));
    }
    fixture(&records, "v5-gemma31b-mixed-schema.parquet");
}
#[test]
fn serial_gemma31b_generated_messages_retain_bare_negative_zero() {
    let mut values = source();
    values["messages"][2]["tool_calls"][0]["function"]["arguments"]["negative"] = json!(-0.0);
    fixture(
        &[record(&values, "negative-zero")],
        "v5-gemma31b-negative-zero.parquet",
    );
}
