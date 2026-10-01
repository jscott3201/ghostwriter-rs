#![allow(dead_code)]
use gw_eval::screening::validate_screening_plan;
use gw_schema::*;
use gw_storage::{StorageError, Store};
use std::sync::atomic::{AtomicU64, Ordering};
pub struct Temp(pub std::path::PathBuf);
impl Temp {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gw-screened-publish-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    pub fn artifact(&self) -> std::path::PathBuf {
        self.0.join("dataset.parquet")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
pub async fn setup(rows: &[TrainingRecord], out: &Temp) -> Store {
    let store = Store::open(out.0.join("store.sqlite")).await.unwrap();
    let runs: std::collections::BTreeSet<_> =
        rows.iter().map(|row| &row.provenance.run_id).collect();
    for run in runs {
        store.insert_historical_run(run, "{}", None).await.unwrap();
    }
    for row in rows {
        store.replace_record_for_import(row).await.unwrap();
    }
    store
}
pub fn options(plan: &FrozenScreeningPlan) -> ExportOptions {
    ExportOptions {
        target: plan.declaration.policy.target,
        cot_policy: plan.declaration.policy.cot_policy,
        dataset_version: None,
        scope: ExportScope::Run {
            run_id: plan.declaration.output.run_id.clone(),
        },
    }
}
pub fn validate(
    sets: Vec<ProtectedScreeningSet>,
) -> impl FnOnce(&[TrainingRecord], &FrozenScreeningPlan) -> gw_storage::Result<()> {
    move |rows, plan| {
        validate_screening_plan(rows, &sets, plan)
            .map_err(|error| StorageError::Export(error.to_string()))
    }
}
pub fn hash(domain: &str, value: &impl serde::Serialize) -> String {
    gw_storage::canonical_json_hash(&serde_json::json!([domain, value])).unwrap()
}
