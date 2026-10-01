//! Controllable callback barriers and SQLite write ordering protect the captured population.
mod screened_publication_support;
mod screening_support;
use gw_eval::screening::{prepare_screening, validate_screening_plan};
use gw_schema::*;
use gw_storage::{ExportPurpose, StorageError};
use screened_publication_support::*;
use screening_support::*;

#[tokio::test]
async fn membership_change_after_full_rerun_before_prepare_has_no_receipt_or_file_effect() {
    let original = vec![record("a", "training source")];
    let out = Temp::new();
    let store = setup(&original, &out).await;
    let sets = protected();
    let plan = prepare_screening(&original, &declaration(&original), &sets, None).unwrap();
    let (entered, waiting) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let publisher = store.clone();
    let dst = out.artifact();
    let task = tokio::spawn(async move {
        publisher
            .publish_screened_export(
                options(&plan),
                plan,
                dst,
                ExportPurpose::Engine,
                move |rows, candidate| {
                    validate_screening_plan(rows, &sets, candidate)
                        .map_err(|e| StorageError::Export(e.to_string()))?;
                    entered.send(()).unwrap();
                    blocked.recv().unwrap();
                    Ok(())
                },
            )
            .await
    });
    waiting.await.unwrap();
    let mut added = original[0].clone();
    added.record_id = "late-rejected".into();
    added.lifecycle.state = LifecycleState::Rejected;
    added.judging.verdict = Some(Verdict::Reject);
    // This write succeeds while validation is paused: the initial read transaction is closed.
    store.replace_record_for_import(&added).await.unwrap();
    release.send(()).unwrap();
    assert!(
        task.await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("population membership or inputs changed")
    );
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM export_receipts")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(count, 0);
    assert!(!out.artifact().exists());
    assert!(store.lifecycle_history("a").await.unwrap().is_empty());
}

#[tokio::test]
async fn held_writer_committing_a_new_sibling_before_ack_prevents_any_partial_transition() {
    let original = vec![record("a", "training source")];
    let out = Temp::new();
    let store = setup(&original, &out).await;
    let sets = protected();
    let plan = prepare_screening(&original, &declaration(&original), &sets, None).unwrap();
    sqlx::query("CREATE TRIGGER fail_ack BEFORE UPDATE ON export_receipts WHEN NEW.state = 'acknowledged' BEGIN SELECT RAISE(FAIL, 'pause receipt'); END").execute(store.raw_pool()).await.unwrap();
    store
        .publish_screened_export(
            options(&plan),
            plan,
            out.artifact(),
            ExportPurpose::Engine,
            validate(sets),
        )
        .await
        .unwrap_err();
    sqlx::query("DROP TRIGGER fail_ack")
        .execute(store.raw_pool())
        .await
        .unwrap();
    let id: String = sqlx::query_scalar("SELECT publication_id FROM export_receipts")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    let connection = sqlx::SqlitePool::connect(&format!(
        "sqlite://{}",
        out.0.join("store.sqlite").display()
    ))
    .await
    .unwrap();
    let mut writer = connection.begin_with("BEGIN IMMEDIATE").await.unwrap();
    // Duplicate a real row with a consistent new rejected envelope and indexes in the second connection.
    let mut added = store.get("a").await.unwrap();
    added.record_id = "new-rejected-sibling".into();
    added.lifecycle.state = LifecycleState::Rejected;
    added.judging.verdict = Some(Verdict::Reject);
    sqlx::query("INSERT INTO records(record_id,run_id,lifecycle_state,verdict,judge_aggregate,record_hash,prompt_hash,record_json,updated_at) VALUES(?,?,'rejected','reject',NULL,?,?,?,'fixture')")
        .bind(&added.record_id).bind(&added.provenance.run_id).bind(&added.hashes.record_hash).bind(&added.hashes.prompt_hash).bind(serde_json::to_string(&added).unwrap()).execute(&mut *writer).await.unwrap();
    let publisher = store.clone();
    let resume_id = id.clone();
    let task = tokio::spawn(async move { publisher.resume_export(&resume_id).await });
    tokio::task::yield_now().await;
    writer.commit().await.unwrap();
    assert!(
        task.await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("population membership or inputs changed")
    );
    assert!(store.lifecycle_history("a").await.unwrap().is_empty());
    assert_eq!(
        store.get("a").await.unwrap().lifecycle.state,
        LifecycleState::Admitted
    );
    let state: String =
        sqlx::query_scalar("SELECT state FROM export_receipts WHERE publication_id=?")
            .bind(&id)
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
    assert_eq!(state, "prepared");
    connection.close().await;
}

#[tokio::test]
async fn incomplete_empty_plans_and_missing_or_changed_protected_inputs_have_no_effect() {
    for mutation in [
        "missing",
        "rights",
        "coverage",
        "contents",
        "incomplete_empty",
    ] {
        let original = vec![record("a", "training source")];
        let out = Temp::new();
        let store = setup(&original, &out).await;
        let mut sets = protected();
        let mut declared = declaration(&original);
        if mutation == "incomplete_empty" {
            declared.output.record_ids.clear();
            sets[0].coverage.complete = false;
        }
        let plan = prepare_screening(&original, &declared, &sets, None).unwrap();
        match mutation {
            "missing" => {
                sets.pop();
            }
            "rights" => sets[0].rights = None,
            "coverage" => sets[0].coverage.complete = false,
            "contents" => {
                sets[0].items[0].responses[0].content =
                    Content::Text("changed protected text".into());
                sets[0].content_digest =
                    gw_eval::screening::protected_screening_content_digest(&sets[0].items).unwrap();
            }
            _ => {}
        }
        assert!(
            store
                .publish_screened_export(
                    options(&plan),
                    plan,
                    out.artifact(),
                    ExportPurpose::Engine,
                    validate(sets)
                )
                .await
                .is_err(),
            "{mutation}"
        );
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM export_receipts")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
        assert_eq!(count, 0);
        assert!(!out.artifact().exists());
        assert!(store.lifecycle_history("a").await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn captured_protected_contents_survive_path_replacement_and_receipt_recovery_needs_no_files()
{
    let original = vec![record("a", "training source")];
    let out = Temp::new();
    let store = setup(&original, &out).await;
    let path = out.0.join("protected.json");
    std::fs::write(&path, serde_json::to_vec(&protected()).unwrap()).unwrap();
    let captured: Vec<ProtectedScreeningSet> =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let plan = prepare_screening(&original, &declaration(&original), &captured, None).unwrap();
    std::fs::write(&path, b"replacement invalid bytes").unwrap();
    sqlx::query("CREATE TRIGGER fail_ack BEFORE UPDATE ON export_receipts WHEN NEW.state = 'acknowledged' BEGIN SELECT RAISE(FAIL, 'pause receipt'); END").execute(store.raw_pool()).await.unwrap();
    store
        .publish_screened_export(
            options(&plan),
            plan,
            out.artifact(),
            ExportPurpose::Engine,
            validate(captured),
        )
        .await
        .unwrap_err();
    std::fs::remove_file(path).unwrap();
    sqlx::query("DROP TRIGGER fail_ack")
        .execute(store.raw_pool())
        .await
        .unwrap();
    let id: String = sqlx::query_scalar("SELECT publication_id FROM export_receipts")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    std::fs::remove_file(out.artifact()).unwrap();
    let result = store.resume_export(&id).await.unwrap();
    assert_eq!(result.advanced_record_ids, ["a"]);
}

#[tokio::test]
async fn membership_change_at_receipt_commit_is_detected_again_at_acknowledgment() {
    let original = vec![record("a", "training source")];
    let out = Temp::new();
    let store = setup(&original, &out).await;
    let sets = protected();
    let plan = prepare_screening(&original, &declaration(&original), &sets, None).unwrap();
    // Inject a consistent new excluded member only after prepare's population check. This models
    // the publication/ack gap without timing sleeps; the same shared check must run again at ACK.
    sqlx::query("CREATE TRIGGER late_member AFTER INSERT ON export_receipts BEGIN INSERT INTO records(record_id,run_id,lifecycle_state,verdict,judge_aggregate,record_hash,prompt_hash,record_json,updated_at) SELECT record_id || '-late',run_id,'rejected','reject',judge_aggregate,record_hash,prompt_hash,json_set(record_json,'$.record_id',record_id || '-late','$.lifecycle.state','rejected','$.judging.verdict','reject'),updated_at FROM records WHERE record_id='a'; END").execute(store.raw_pool()).await.unwrap();
    let error = store
        .publish_screened_export(
            options(&plan),
            plan,
            out.artifact(),
            ExportPurpose::Engine,
            validate(sets),
        )
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("population membership or inputs changed")
    );
    assert!(out.artifact().is_file());
    assert!(store.lifecycle_history("a").await.unwrap().is_empty());
    assert_eq!(
        store.get("a").await.unwrap().lifecycle.state,
        LifecycleState::Admitted
    );
    let state: String = sqlx::query_scalar("SELECT state FROM export_receipts")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(state, "prepared");
}
