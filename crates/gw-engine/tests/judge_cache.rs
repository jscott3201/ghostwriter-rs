//! Individual judge requests reuse their grades across runs and downstream admission settings.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{Engine, EventSink};
use gw_storage::{RecordFilter, Store};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn unchanged_judge_request_reuses_its_grade_across_runs() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![good_cot(0.01), good_cot(0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let mut panels = Vec::new();

    for (run_id, accept_threshold, family) in [
        ("cache-run-a", 0.80, "family-a"),
        ("cache-run-b", 0.90, "family-b"),
    ] {
        let mut judges = one_judge();
        judges[0].family = family.into();
        let mut thresholds = lenient_thresholds();
        thresholds.accept_threshold = accept_threshold;
        let engine = Engine::new(
            clients(
                store.clone(),
                teacher.clone(),
                judge.clone(),
                EventSink::disconnected(),
            ),
            area_k1(judges, thresholds),
            1,
        );
        let report = engine
            .run(run_id, &one_item_source(), CancellationToken::new())
            .await
            .unwrap();
        assert!(report.completed);
        assert_eq!(report.admitted, 1);
        let records = store
            .scan(&RecordFilter::new().run_id(run_id))
            .await
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].provenance.run_id, run_id);
        let panel = records[0].judging.panel.clone();
        assert_eq!(panel.len(), 1);
        let audit: serde_json::Value =
            serde_json::from_str(panel[0].raw_response.as_ref().unwrap()).unwrap();
        assert_eq!(audit["scoring_used"], "json_score");
        assert_eq!(audit["interpretation_version"], 1);
        panels.push(panel);
    }

    assert_eq!(
        teacher.call_count(),
        2,
        "separate runs each generated their candidate"
    );
    assert_eq!(
        judge.call_count(),
        1,
        "identical grading requests reuse one grade across runs"
    );
    assert_eq!(
        panels[0], panels[1],
        "cached vote and raw audit bytes are preserved"
    );
}
