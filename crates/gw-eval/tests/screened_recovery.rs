//! Full-population integrity and recovery use genuine prepared database receipts.
mod screened_publication_support;
mod screening_support;
use gw_eval::screening::prepare_screening;
use gw_schema::*;
use gw_storage::{ExportPurpose, Store};
use screened_publication_support::*;
use screening_support::*;

fn rows() -> Vec<TrainingRecord> {
    let a = record("a", "ordinary source question");
    let mut excluded = record("excluded", "separate excluded question");
    excluded.lifecycle.state = LifecycleState::Rejected;
    excluded.judging.verdict = Some(Verdict::Reject);
    vec![a, excluded]
}
async fn fail_ack(store: &Store) {
    sqlx::query("CREATE TRIGGER fail_ack BEFORE UPDATE ON export_receipts WHEN NEW.state = 'acknowledged' BEGIN SELECT RAISE(FAIL, 'injected acknowledgment failure'); END").execute(store.raw_pool()).await.unwrap();
}
async fn receipts(store: &Store) -> Vec<(String, String)> {
    sqlx::query_as("SELECT publication_id,state FROM export_receipts ORDER BY rowid")
        .fetch_all(store.raw_pool())
        .await
        .unwrap()
}
async fn publish(
    store: &Store,
    out: &Temp,
    records: &[TrainingRecord],
) -> gw_storage::Result<gw_storage::ExportPublication> {
    let sets = protected();
    let plan = prepare_screening(records, &declaration(records), &sets, None).unwrap();
    store
        .publish_screened_export(
            options(&plan),
            plan,
            out.artifact(),
            ExportPurpose::Engine,
            validate(sets),
        )
        .await
}

#[tokio::test]
async fn excluded_inputs_and_all_in_scope_membership_changes_invalidate_prepared_and_acknowledged_receipts()
 {
    for acknowledged in [false, true] {
        for mutation in [
            "parent",
            "split",
            "verdict",
            "eligibility",
            "message",
            "delete",
            "add_rejected",
            "damaged_projection",
            "damaged_history",
        ] {
            let out = Temp::new();
            let original = rows();
            let store = setup(&original, &out).await;
            if !acknowledged {
                fail_ack(&store).await;
            }
            let result = publish(&store, &out, &original).await;
            assert_eq!(result.is_ok(), acknowledged);
            if !acknowledged {
                sqlx::query("DROP TRIGGER fail_ack")
                    .execute(store.raw_pool())
                    .await
                    .unwrap();
            }
            let before = receipts(&store).await;
            let bytes = std::fs::read(out.artifact()).unwrap();
            let history = store.lifecycle_history("a").await.unwrap();
            let mut changed = original[1].clone();
            match mutation {
                "parent" => changed
                    .origin
                    .generated_mut()
                    .expect("generated fixture")
                    .provenance
                    .parent_ids
                    .push("a".into()),
                "split" => {
                    changed.task_provenance.as_mut().unwrap().split.role = TaskSplitRole::Test
                }
                "verdict" => changed.judging.verdict = Some(Verdict::NeedsReview),
                "eligibility" => {
                    changed.lifecycle.state = LifecycleState::Admitted;
                    changed.judging.verdict = Some(Verdict::Admit);
                }
                "message" => changed.messages[1].reasoning = Some("new excluded reasoning".into()),
                "delete" => {
                    sqlx::query("DELETE FROM records WHERE record_id='excluded'")
                        .execute(store.raw_pool())
                        .await
                        .unwrap();
                }
                "add_rejected" => {
                    let mut added = changed.clone();
                    added.record_id = "new-sibling".into();
                    store.replace_record_for_import(&added).await.unwrap();
                }
                "damaged_projection" => {
                    sqlx::query(
                        "UPDATE records SET judge_aggregate=0.2 WHERE record_id='excluded'",
                    )
                    .execute(store.raw_pool())
                    .await
                    .unwrap();
                }
                "damaged_history" => {
                    sqlx::query("INSERT INTO lifecycle_history(record_id,state,at,history_ordinal,attempt) VALUES('excluded','exported','bad',0,0)").execute(store.raw_pool()).await.unwrap();
                }
                _ => unreachable!(),
            }
            if matches!(
                mutation,
                "parent" | "split" | "verdict" | "eligibility" | "message"
            ) {
                store.replace_record_for_import(&changed).await.unwrap();
            }
            let error = store.resume_export(&before[0].0).await.unwrap_err();
            assert!(
                error.to_string().contains("publication"),
                "{mutation}: {error}"
            );
            assert_eq!(receipts(&store).await, before, "{mutation}");
            assert_eq!(
                store.lifecycle_history("a").await.unwrap(),
                history,
                "{mutation}"
            );
            assert_eq!(std::fs::read(out.artifact()).unwrap(), bytes, "{mutation}");
        }
    }
}

#[tokio::test]
async fn excluded_aggregate_and_outside_run_addition_preserve_exact_prepared_recovery() {
    let out = Temp::new();
    let original = rows();
    let store = setup(&original, &out).await;
    fail_ack(&store).await;
    publish(&store, &out, &original).await.unwrap_err();
    sqlx::query("DROP TRIGGER fail_ack")
        .execute(store.raw_pool())
        .await
        .unwrap();
    let id = receipts(&store).await[0].0.clone();
    let bytes = std::fs::read(out.artifact()).unwrap();
    let mut changed = original[1].clone();
    changed.judging.aggregate = Some(0.2);
    store.replace_record_for_import(&changed).await.unwrap();
    let outside = record("outside", "irrelevant run question");
    store
        .insert_historical_run(outside.run_id(), "{}", None)
        .await
        .unwrap();
    store.replace_record_for_import(&outside).await.unwrap();
    let current = store.get("a").await.unwrap();
    store
        .advance_lifecycle(&current, LifecycleState::Formatted, None)
        .await
        .unwrap();
    for content in [None, Some(b"replaced".as_slice()), Some(bytes.as_slice())] {
        match content {
            None => std::fs::remove_file(out.artifact()).unwrap(),
            Some(value) => std::fs::write(out.artifact(), value).unwrap(),
        }
        store.resume_export(&id).await.unwrap();
        assert_eq!(std::fs::read(out.artifact()).unwrap(), bytes);
    }
    assert_eq!(
        store
            .lifecycle_history("a")
            .await
            .unwrap()
            .iter()
            .filter(|h| h.0 == "exported")
            .count(),
        1
    );
}

#[tokio::test]
async fn raw_screened_and_different_plan_pending_paths_never_implicitly_cross() {
    for raw_first in [false, true] {
        let out = Temp::new();
        let original = rows();
        let store = setup(&original, &out).await;
        let sets = protected();
        let plan = prepare_screening(&original, &declaration(&original), &sets, None).unwrap();
        let opts = options(&plan);
        fail_ack(&store).await;
        if raw_first {
            store
                .publish_export(opts.clone(), out.artifact(), ExportPurpose::Engine)
                .await
                .unwrap_err();
        } else {
            publish(&store, &out, &original).await.unwrap_err();
        }
        sqlx::query("DROP TRIGGER fail_ack")
            .execute(store.raw_pool())
            .await
            .unwrap();
        let before = receipts(&store).await;
        let bytes = std::fs::read(out.artifact()).unwrap();
        let result = if raw_first {
            store
                .publish_screened_export(
                    opts,
                    plan.clone(),
                    out.artifact(),
                    ExportPurpose::Engine,
                    validate(sets.clone()),
                )
                .await
        } else {
            store
                .publish_export(opts, out.artifact(), ExportPurpose::Engine)
                .await
        };
        assert!(result.unwrap_err().to_string().contains("flavor or plan"));
        if !raw_first {
            let mut declaration = plan.declaration;
            declaration.output.record_ids.clear();
            let other = prepare_screening(&original, &declaration, &sets, None).unwrap();
            let error = store
                .publish_screened_export(
                    options(&other),
                    other,
                    out.artifact(),
                    ExportPurpose::Engine,
                    validate(sets),
                )
                .await
                .unwrap_err();
            assert!(error.to_string().contains("flavor or plan"));
        }
        assert_eq!(receipts(&store).await, before);
        assert_eq!(std::fs::read(out.artifact()).unwrap(), bytes);
        store.resume_export(&before[0].0).await.unwrap();
    }
}

#[tokio::test]
async fn changed_selected_projection_remains_invalid_even_when_screening_binding_is_stable() {
    let out = Temp::new();
    let original = rows();
    let store = setup(&original, &out).await;
    let publication = publish(&store, &out, &original).await.unwrap();
    let mut changed = store.get("a").await.unwrap();
    changed.judging.aggregate = Some(0.8);
    store.replace_record_for_import(&changed).await.unwrap();
    assert!(
        store
            .resume_export(&publication.publication_id)
            .await
            .unwrap_err()
            .to_string()
            .contains("selected export record changed")
    );
}
