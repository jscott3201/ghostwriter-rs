//! Synthetic evidence only. The loss oracle is hand-derived from unequal groups, not fitted output.
#![allow(dead_code)]
use gw_judge::{PanelJudge, ResolvedCalibrationPanel};
use gw_schema::*;

pub struct Fixture {
    pub panel: ResolvedCalibrationPanel,
    pub records: Vec<TrainingRecord>,
    pub evidence: CalibrationEvidence,
}

pub fn candidate(id: &str, prompt: &str) -> TrainingRecord {
    serde_json::from_value(serde_json::json!({
        "record_id": id, "schema_version":"1.0.0", "training_area":"synthetic-area",
        "messages":[{"role":"user", "content":prompt},
            {"role":"assistant", "content":format!("answer {id}"), "reasoning":"independent reasoning"}],
        "provenance":{"run_id":"run-1", "teacher":{"provider":"fixture", "slug":"synthetic"},
            "harness_version":"fixture"}, "generation":{}, "lifecycle":{"state":"seeded"}
    })).unwrap()
}

impl Fixture {
    pub fn new() -> Self {
        Self::with_judges(&[
            PanelJudge::new("judge-a", "a"),
            PanelJudge::new("judge-b", "b"),
        ])
    }
    pub fn with_judges(judges: &[PanelJudge]) -> Self {
        let panel =
            ResolvedCalibrationPanel::new("synthetic-area", judges, "synthetic rubric").unwrap();
        let records = vec![
            candidate("a1", "first prompt"),
            candidate("a2", "first prompt"),
            candidate("a3", "first prompt"),
            candidate("b1", "second prompt"),
            candidate("held", "held-out prompt"),
        ];
        let fit = records[..4]
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let (group, scores) = if index < 3 {
                    ("fit-a", [0.0, 0.5])
                } else {
                    ("fit-b", [1.0, 0.0])
                };
                row(&panel, record, group, &scores)
            })
            .collect();
        let assessment = vec![row(&panel, &records[4], "assessment-a", &[0.25, 0.75])];
        let evidence = CalibrationEvidence {
            version: CALIBRATION_VERSION,
            group_map_version: 1,
            method: CALIBRATION_METHOD.into(),
            numerical_recipe: CALIBRATION_NUMERICAL_RECIPE.into(),
            target: CalibrationTarget {
                name: "synthetic quality".into(),
                version: "v1".into(),
                protocol_revision: "synthetic independent reference v1".into(),
                semantics: CalibrationTargetSemantics::HigherIsBetterUnitInterval,
            },
            provenance: CalibrationProvenance {
                source: "hand-authored synthetic fixture".into(),
                reference_digest: "fixture-v1".into(),
                independence_blinding: "synthetic declaration, not empirical independence".into(),
                runtime: CalibrationRuntimeProvenance::SuppliedUnverified,
            },
            panel: panel.declaration().clone(),
            panel_identity: panel.identity().into(),
            beta: 8.0,
            correlation: CalibrationCorrelation::AssumedConstantRho { rho: 0.25 },
            fit,
            assessment,
        };
        Self {
            panel,
            records,
            evidence,
        }
    }
    pub fn report(&self) -> CalibrationReport {
        gw_judge::fit_calibration(&self.panel, &self.records, &self.evidence)
    }
    pub fn snapshot(&self) -> CalibrationSnapshot {
        let report = self.report();
        assert_eq!(
            report.status,
            CalibrationStatus::ComputedUnqualified,
            "{:?}",
            report.reasons
        );
        report.snapshot.unwrap()
    }
}

pub fn row(
    panel: &ResolvedCalibrationPanel,
    record: &TrainingRecord,
    group: &str,
    scores: &[f64],
) -> CalibrationRow {
    CalibrationRow {
        candidate: gw_storage::capture_candidate_binding(record).unwrap(),
        prompt_group: group.into(),
        label: CalibrationLabel::Known { value: 0.0 },
        observations: scores
            .iter()
            .enumerate()
            .map(|(index, score)| {
                panel
                    .observation(
                        record,
                        index,
                        CalibrationCollection::Observed,
                        Some(serde_json::json!({"score":score, "verdict":"accept"}).to_string()),
                    )
                    .unwrap()
            })
            .collect(),
    }
}

pub fn rehash(cell: &mut CalibrationObservation) {
    cell.payload_identity = gw_judge::calibration_payload_identity(&cell.raw).unwrap();
    cell.identity = gw_judge::calibration_observation_identity(cell).unwrap();
}
