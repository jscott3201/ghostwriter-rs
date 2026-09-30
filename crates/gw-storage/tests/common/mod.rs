//! Explicit immutable fixture contract for transaction-only tests.
#![allow(dead_code)]
use gw_schema::*;
pub fn manifest() -> RunManifest {
    let d = SemanticDeclaration::new(
        "test/transaction-fixture",
        "1",
        serde_json::json!({"behavior":"no-client-execution"}),
    );
    RunManifest {
        version: RUN_MANIFEST_VERSION,
        input_plan: InputPlanIdentity {
            content_hash: "a".repeat(64),
            shard_items: vec![1],
        },
        execution: d.clone(),
        clients: ClientSemantics {
            teacher: d.clone(),
            judge: d.clone(),
            embedding: d.clone(),
            sandbox: d.clone(),
            execution_evidence: d,
        },
        unattested_deployment: UnattestedDeployment::default(),
    }
}
