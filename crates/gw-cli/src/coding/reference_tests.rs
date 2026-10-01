//! Synthetic final-member controls for the opaque fresh batch boundary, plus owned Docker controls.
use super::*;
use crate::coding::{CodingCaseObservation, CodingCaseReason, TestStatus};
#[path = "../../../gw-schema/tests/reference_support/mod.rs"]
mod support;

fn observed(member: &ValidatedReferenceMember) -> ObservedCodingRun {
    let document = CodingTaskDocument {
        version: 1,
        tasks: vec![member.task.clone()],
    };
    let input =
        CapturedCodingInput::new(&document, &member.task.task_id, member.code.clone()).unwrap();
    let cases = input
        .suite
        .case_ids
        .iter()
        .map(|id| CodingCaseObservation {
            case_id: id.clone(),
            status: TestStatus::Passed,
            reason: CodingCaseReason::Matched,
            result_id: Some("0".repeat(64)),
            exit_code: Some(0),
            settled: true,
            container_id: Some("1".repeat(64)),
            elapsed_ms: 1,
        })
        .collect();
    ObservedCodingRun {
        input,
        run_id: "0".repeat(32),
        cases,
    }
}
#[tokio::test]
async fn final_fail_unknown_cancel_or_unsettled_member_never_creates_partial_batch() {
    let store = Store::open_in_memory().await.unwrap();
    let registered = store
        .register_reference_catalogue(&support::capture())
        .await
        .unwrap();
    for scenario in 0..4 {
        let mut batch = FreshBatch { members: vec![] };
        for member in &registered.population().members()[..111] {
            batch.push(member, observed(member)).unwrap();
        }
        let member = &registered.population().members()[111];
        let mut last = observed(member);
        match scenario {
            0 => {
                last.cases[0].status = TestStatus::Failed;
                last.cases[0].reason = CodingCaseReason::WrongResult;
            }
            1 => {
                last.cases[0].status = TestStatus::Skipped;
                last.cases[0].reason = CodingCaseReason::Infrastructure;
            }
            2 => {
                last.cases[0].status = TestStatus::Skipped;
                last.cases[0].reason = CodingCaseReason::Cancelled;
            }
            _ => {
                last.cases[0].settled = false;
            }
        }
        assert!(batch.push(member, last).is_err(), "scenario {scenario}");
        assert_eq!(batch.members.len(), 111);
        assert!(
            store
                .commit_reference_import(&registered, &batch.members, std::future::pending())
                .await
                .is_err()
        );
        assert!(
            store
                .scan(&gw_storage::RecordFilter::new())
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .committed_reference_import(&registered)
                .await
                .unwrap()
                .is_none()
        );
    }
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(import(&store, &registered, false, cancel).await.is_err());
}
#[test]
fn builder_checks_actual_captured_member_even_when_observation_is_positive() {
    let captured = support::capture().validate().unwrap();
    let mut batch = FreshBatch { members: vec![] };
    assert!(
        batch
            .push(&captured.members()[1], observed(&captured.members()[0]))
            .is_err()
    );
    assert!(batch.members.is_empty());
}
#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls"]
async fn owned_runtime_positive_negative_and_syntax_controls_cross_fresh_builder() {
    for (code, pass) in [
        ("def probe():\n    return True\n", true),
        ("def probe():\n    return False\n", false),
        ("def probe(:\n", false),
    ] {
        let input = crate::coding::containment_tests::probe_input(code);
        let member = ValidatedReferenceMember {
            member_id: "a".repeat(64),
            task: input.task.clone(),
            code: input.code.clone(),
            reference_code_id: input.code_id.clone(),
            component: input.task.group.clone(),
            authorship: gw_schema::ReferenceAuthorship {
                author: gw_schema::ReferenceActorKind::Agent,
                reviewer: gw_schema::ReferenceActorKind::Agent,
            },
        };
        let fresh = observe_coding(input, CancellationToken::new())
            .await
            .unwrap();
        let mut batch = FreshBatch { members: vec![] };
        assert_eq!(batch.push(&member, fresh).is_ok(), pass);
        assert_eq!(batch.members.len(), usize::from(pass));
    }
}

#[path = "../../../gw-eval/tests/screening_support/mod.rs"]
mod screening_support;
#[tokio::test]
async fn reference_screening_binds_origins_without_inventing_teachers_or_siblings() {
    let store = Store::open_in_memory().await.unwrap();
    let registered = store
        .register_reference_catalogue(&support::capture())
        .await
        .unwrap();
    let mut batch = FreshBatch { members: vec![] };
    for member in registered.population().members() {
        batch.push(member, observed(member)).unwrap();
    }
    let imported = store
        .commit_reference_import(&registered, &batch.members, std::future::pending())
        .await
        .unwrap();
    let declaration = screening_support::declaration(&imported.records);
    assert!(declaration.siblings.is_empty());
    let protected = screening_support::protected();
    let plan =
        gw_eval::screening::prepare_screening(&imported.records, &declaration, &protected, None)
            .unwrap();
    assert!(plan.incomplete.is_empty(), "{:?}", plan.incomplete);
    assert_eq!(plan.eligible_output.len(), 64);
    assert!(plan.strata.iter().all(|stratum| stratum.teacher.is_none()));
    assert!(
        plan.edges
            .iter()
            .any(|edge| edge.kind == "reference_component")
    );
    let options = gw_schema::ExportOptions {
        target: declaration.policy.target,
        cot_policy: declaration.policy.cot_policy,
        dataset_version: None,
        scope: gw_schema::ExportScope::Run {
            run_id: registered.batch_id().into(),
        },
    };
    let path = std::env::temp_dir().join(format!(
        "gw-reference-screened-{}.parquet",
        std::process::id()
    ));
    let published = store
        .publish_screened_export(
            options,
            plan,
            &path,
            gw_storage::ExportPurpose::Reference,
            move |records, plan| {
                gw_eval::screening::validate_screening_plan(records, &protected, plan)
                    .map_err(|error| gw_storage::StorageError::Export(error.to_string()))
            },
        )
        .await
        .unwrap();
    assert_eq!(published.advanced_record_ids.len(), 64);
    if let Some(destination) = std::env::var_os("GW_REFERENCE_SCREENED_FIXTURE_OUT") {
        std::fs::copy(&path, destination).unwrap();
    }
    std::fs::remove_file(path).unwrap();
}
