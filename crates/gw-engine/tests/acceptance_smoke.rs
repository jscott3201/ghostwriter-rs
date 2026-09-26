//! PR05 acceptance smoke: ONE synthetic tool trajectory walked through the whole shipped chain, and
//! the negatives that must not survive it.
//!
//! The per-stage suites own the detail (`tool_identity` for ingest, `tool_trajectory` /
//! `export_tool_trajectory` for the store + shard, `execution_evidence` for the evidence axis,
//! `tool_target_guard` for the render guard, `gw-cli`'s `export.rs` for the handler). This file
//! deliberately re-walks those stages on ONE corpus to assert the HAND-OFFS the per-stage suites
//! cannot see: that the conversation ingest produces is the one the store persists, the resumed run
//! exports, a consumer decodes, and the render guard refuses; and that a record the evidence axis
//! declined never reaches the artifact. It adds no corpus, no new dependency and no production code.
//!
//! The chain is walked in dependency order — admission BEFORE export, because the storage export
//! gate *is* the judge verdict, so an export stage ahead of admission would be asserting a shard
//! whose admission the chain never established. The same eight stages are covered either way.
//!
//! # What is exercised, and at what level
//!
//! SYNTHETIC ONLY. The corpus is the hand-authored `gw-format` wire fixture (two same-name
//! `read_file` calls, one string-encoded, results in reversed order) — not a real provider capture,
//! not a task trace, not competition data. The execution reports are hand-authored
//! (`common::passing_report`): the harness executes nothing, so this proves the ADMISSION wiring, not
//! that any code ran.
//!
//! # READY (this file's scope)
//!
//! The integration gate: ingest → store → advance/resume → evidence admission → Parquet export →
//! independent decode → render guard → `gen export` parity, with the stale and cross-attempt
//! negatives denied and absent from the shard, all hermetic and offline (scripted providers, in-memory
//! `Clients`, no network, no paid call).
//!
//! # NOT READY (and what is missing)
//!
//! **1. The canonical export/consumer route to use.** Read `ExportManifest::column_schema_version`
//! first, then take the SINGLE `messages_json` column and `serde_json`-decode each cell straight into
//! `Vec<gw_schema::Message>`; there is no transform to reverse. The manifest is written with
//! `ExportSchemaVersion::CURRENT` (`CanonicalMessages`). A tool trajectory has no representation in
//! this repo's own renderers, which is exactly why the route ends at the canonical column: the
//! refusal's own recovery is `ToolCallRecovery::CanonicalExportAndOfficialTemplate` — a downstream
//! consumer using the OFFICIAL template. A target template is NOT fetched, pinned or diff-verified
//! here, so no rendered tool trajectory is proven.
//!
//! **2. Unsupported legacy records and routes.** A v1 (`RoleContentText`) shard is a lossy
//! `{role, content}` pair plus a parallel `reasoning_json`: `content: null` collapsed to `""`, parts
//! re-encoded as a string inside a string, and `tool_calls` / `name` / the `tool_call_id` result
//! link DROPPED. It still DESERIALIZES, so it fails silently rather than loudly, and it cannot tell
//! two results of the same function apart. A manifest with no `column_schema_version` key reads back
//! as v1, so branch on the version. `reasoning_json` is removed in v2 and is not carried forward.
//! On the render side, Gemma4 / ChatML / ShareGPT / Harmony FAIL CLOSED on a tool trajectory
//! (`FormatError::UnsupportedToolCalls`); the Gemma-4 route additionally folds a `system` /
//! `developer` turn into the next `user` turn (a documented divergence from the upstream conditional
//! system turn), so its `Role::Tool` arm is unreachable through `render()` — defensive mapping, not
//! a data path. Provider wire fields outside the canonical type (e.g. `source_extension`, the OpenAI
//! `tool_calls[].type`) have nowhere to live and are absent from every export.
//!
//! **3. Missing real-trace and template evidence.** No real provider trace and no official
//! `chat_template` is present in-repo, so the byte shapes are pinned to transcribed spec bytes plus
//! golden files, NOT diff-verified against the pinned upstream template (the `gemma4-verify` TODO in
//! `render/gemma4.rs`); the render-guard golden path is text-only and is not proof of tool fidelity.
//! Adjudication, clustering and separation tooling exercised elsewhere is not a Ghostwriter test
//! either. Nothing here is evidence about any external benchmark or corpus.
//!
//! **4. Unrun here.** The live provider tests are `#[ignore]`d and env-gated (`OPENROUTER_API_KEY`,
//! `GW_EMBEDDINGS_LIVE=1`) and were NOT run — no key, no network, no paid call. The OWNER's
//! target-template / mask CONSUMER test (does a real trainer reproduce the intended loss region from
//! this shard?) is not in this repository and was not run; this crate asserts the shard's bytes, not
//! a downstream trainer's interpretation of them.
//!
//! **5. Bulk generation stays disabled.** This smoke adds no generation capability and relaxes no
//! gate. Bulk generation remains owner-gated: baseline + verifier in place, permissions, split /
//! lineage accounting, and a tiny fitment run first. No RL loop and no classifier/annotation SDK are
//! introduced or implied here.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::{Array, StringArray};
use common::*;
use gw_engine::{EventSink, drive, evidence_key};
use gw_format::{FormatError, ToolCallRecovery, ToolSignal, ingest_openrouter, render};
use gw_schema::{
    Content, CotPolicy, ExportSchemaVersion, Lifecycle, LifecycleState, Message, Provenance,
    TeacherRef, TrainingRecord, TrlFormat, Verdict,
};
use gw_storage::{RecordFilter, Store, export_parquet, export_parquet_bytes};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

const RUN: &str = "run-smoke";
/// The record whose evidence is bound to it: the one that must reach the shard.
const ADMITTED: &str = "rec-smoke-admitted";
/// The same conversation under a report bound to a MOVED patch (stale).
const STALE: &str = "rec-smoke-stale";
/// The same conversation under a report bound to ANOTHER attempt.
const CROSS: &str = "rec-smoke-cross";

/// The shared synthetic wire corpus, by include rather than by copy: the smoke's ingest input is
/// byte-identical to the fixture the identity suite asserts field-for-field, so the two cannot drift.
/// A move of that fixture is a loud compile error here, not a silent divergence.
fn wire_conversation() -> Vec<Value> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../gw-format/tests/fixtures/tool_trajectory.json"
    ))
    .expect("the shared synthetic fixture parses");
    fixture["messages"]
        .as_array()
        .expect("fixture carries a messages array")
        .clone()
}

/// Stage 1 — ingest. The provider wire shape normalized into canonical turns, losslessly.
fn ingested() -> Vec<Message> {
    wire_conversation()
        .iter()
        .map(|wire| ingest_openrouter(wire).expect("a fixture wire message ingests"))
        .collect()
}

/// The chain record for `record_id`, entering the pipeline at `AssistantGenerated` (the record is
/// already generated, so no teacher call is owed — asserted by the `ExplodingTeacher` below).
fn chain_record(record_id: &str, messages: &[Message]) -> TrainingRecord {
    TrainingRecord {
        record_id: record_id.into(),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "swe-agents".into(),
        tags: vec!["tool-use".into()],
        messages: messages.to_vec(),
        tools: None,
        provenance: Provenance {
            run_id: RUN.into(),
            parent_ids: vec![],
            teacher: TeacherRef {
                provider: "openrouter".into(),
                slug: "z-ai/glm-5.2".into(),
                served_by: None,
                model_card_revision: None,
            },
            user_synth_model: None,
            user_turn_kind: None,
            in_scope_safe: Some(true),
            judge_models: vec![],
            harness_version: "0.1.0-smoke".into(),
            git_commit: None,
        },
        generation: Default::default(),
        verification_contract: None,
        execution_evidence: None,
        verification: Default::default(),
        judging: Default::default(),
        reasoning_quality: None,
        lifecycle: Lifecycle {
            state: LifecycleState::AssistantGenerated,
            ..Default::default()
        },
        hashes: Default::default(),
        cost: Default::default(),
    }
}

fn temp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("gw-smoke-{}-{n}-{suffix}", std::process::id()));
    path
}

/// Remove a temp artifact and, for a SQLite file, its WAL/SHM siblings.
fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
    for ext in ["-wal", "-shm"] {
        let mut sibling = path.as_os_str().to_owned();
        sibling.push(ext);
        let _ = std::fs::remove_file(PathBuf::from(sibling));
    }
}

/// Decode a shard on disk with a REAL Parquet reader — an independent read path, not the writer's
/// own types — returning each row's `record_id` and its canonical conversation.
fn decode_shard(path: &Path) -> (Vec<String>, Vec<Message>) {
    let file = std::fs::File::open(path).expect("a readable shard file");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("a Parquet file")
        .build()
        .expect("a readable Parquet reader");
    let mut ids = Vec::new();
    let mut turns = Vec::new();
    for batch in reader {
        let batch = batch.expect("a readable Parquet batch");
        let id_col = batch
            .column_by_name("record_id")
            .expect("the shard carries record_id")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("record_id is a string column");
        let messages = batch
            .column_by_name("messages_json")
            .expect("the shard carries the canonical conversation column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("messages_json is a string column");
        for i in 0..id_col.len() {
            ids.push(id_col.value(i).to_string());
            turns.extend(
                serde_json::from_str::<Vec<Message>>(messages.value(i))
                    .expect("the canonical column decodes into Message[]"),
            );
        }
    }
    (ids, turns)
}

/// One keyed evidence source serving all three records, exactly as an out-of-process evaluator
/// would: the well-behaved attempt gets the report it earned, and the two adversarial records get a
/// report that is deliberately bound elsewhere.
///
/// A negative here NEVER FACES A PANEL, and this file does not claim it does. `src/step.rs`
/// short-circuits the `Verified → Judged` edge when the re-derived `VerifierGrade`
/// `is_hard_reject()` (proven failure) or `blocks_admission()` (an undecidable deterministic axis),
/// grading against an EMPTY panel — and `src/grade.rs::verifier_grade_from_verification` re-derives
/// that hold from the persisted `verification` block, so it survives the edge without a re-run.
///
/// What this file therefore proves is the WIRING, and it is the load-bearing half: a
/// verifier-held or verifier-failed record spends NO judge token, so no score — however glowing —
/// can rescue or sink it (asserted by `judge.call_count()` below). Panel-vs-verifier PRECEDENCE, in
/// which a glowing panel sits *alongside* a verifier fail or an undecided axis, is owned elsewhere
/// because no pipeline path here can deliver one: `tests/execution_evidence.rs` owns the no-spend
/// proof per record (`judge_calls == 0`), and `gw-judge`'s `grader.rs` unit tests own the precedence
/// by injecting a panel straight into `HybridGrader` — the only place a glowing panel can sit
/// beside a verifier fail.
fn evidence_source() -> ScriptedEvidence {
    ScriptedEvidence::new(|key| {
        let mut report = passing_report(key);
        match key.attempt.as_str() {
            STALE => report.binding.patch_hash = "patch-from-a-superseded-candidate".into(),
            CROSS => report.binding.attempt = "rec-some-other-attempt".into(),
            _ => {}
        }
        Some(report)
    })
}

/// The full chain on one corpus, with both negatives asserted to be denied AND absent from the
/// artifact. Stages are labelled in the order they are walked.
#[tokio::test]
async fn one_tool_trajectory_survives_the_whole_chain_and_the_negatives_do_not() {
    let turns = ingested();

    // --- stage 1: ingest (the wire fixture → canonical turns) ---------------------------------
    // Only the hand-off facts are asserted here; field-for-field fidelity is `tool_identity`'s job.
    assert_eq!(turns.len(), 6, "six wire messages ingest to six turns");
    assert_eq!(
        turns[2].content,
        Content::Null,
        "a tool-calling turn's null body must not become \"\""
    );
    let calls = turns[2]
        .tool_calls
        .as_ref()
        .expect("both calls survive ingest");
    assert_eq!(calls.len(), 2, "two calls of the SAME function");
    assert_eq!(calls[0].id.as_deref(), Some("read-a"));
    assert_eq!(calls[1].id.as_deref(), Some("read-b"));
    assert!(
        calls[1].function.raw_arguments.is_some(),
        "the string-encoded wire text is retained beside the normalized object"
    );
    assert_eq!(turns[3].tool_call_id.as_deref(), Some("read-b"));
    assert_eq!(turns[4].tool_call_id.as_deref(), Some("read-a"));
    assert_ne!(
        turns[3].tool_call_id, turns[4].tool_call_id,
        "the reversed results stay distinguishable"
    );

    // --- stage 2: store (file-backed, so the export route below is the real one) --------------
    let db = temp_path("smoke.sqlite");
    cleanup(&db);
    let store = Store::open(&db).await.expect("open store");
    store
        .create_run(RUN, "{}", Some(25.0))
        .await
        .expect("create run");

    let ids = [ADMITTED, STALE, CROSS];
    for id in ids {
        store
            .put(&chain_record(id, &turns))
            .await
            .expect("put the chain record");
    }
    let stored = store.get(ADMITTED).await.expect("get the chain record");
    assert_eq!(
        stored.messages, turns,
        "the store persists the ingested conversation unchanged"
    );
    assert_eq!(
        stored.lifecycle.state,
        LifecycleState::AssistantGenerated,
        "the record enters the pipeline post-generation, so no teacher call is owed"
    );

    // --- stage 3: advance / resume (the run-ledger cursor + the record's own edge history) ----
    store
        .checkpoint(
            RUN,
            0,
            "assistant_generated",
            &serde_json::json!({"seed_offset": 8}),
        )
        .await
        .expect("checkpoint the shard");
    let point = store
        .resume_cursor(RUN, 0)
        .await
        .expect("read the resume cursor")
        .expect("the shard checkpointed");
    assert_eq!(point.state, "assistant_generated");
    assert_eq!(point.cursor["seed_offset"], 8);

    // --- stage 4: admission (the execution-evidence axis decides, with both negatives) ---------
    // A non-CoT area on purpose: this corpus's teacher turn carries flat `reasoning` with no
    // `reasoning.text` DETAIL, so under a CoT-required area the reasoning-present hard gate would
    // reject it before the evidence axis is even consulted. The area's job here is the evidence
    // axis, and `with_cot_required(false)` makes that gate inert rather than bypassed.
    // The panel is k=1, so the ADMITTED record spends exactly ONE judge call. The three bodies are
    // deliberately OVER-PROVISIONED for that reason: `ScriptedJudge` falls back to a neutral
    // `0.5/uncertain` once its script is exhausted, so a second record reaching the provider would draw
    // another glowing body rather than a masked one.
    //
    // What the count does and does not measure, precisely: `ScriptedJudge::call_count` increments in
    // `stream_chat`, so it counts PROVIDER calls, not panel consultations. All three records share ONE
    // `record_hash` — the content projection behind `gw_storage::cache::record_hash` is a positive
    // allowlist (`schema_version` / `training_area` / sorted `tags` / per-turn content / `tools`) that
    // deliberately EXCLUDES `record_id`, and `chain_record` varies nothing else — so the judge cache
    // key `(content_hash, "judge", model, rubric_id)` is IDENTICAL for all three. The ADMITTED record
    // populates that entry first, so a second record reaching the panel would read it as a cache HIT
    // (`grade_one_cached` returns before any provider call) and would draw NO body at all.
    //
    // Consequence: the count below is SPEND evidence (one provider call across the loop), and nothing
    // more. It cannot distinguish "the negatives never reached a panel" from "the negatives reached a
    // panel and were elided by the cache". The EMPTY-PANEL assertions further down are the
    // discriminating observable for the short-circuit.
    let glowing: Vec<String> = (0..3).map(|_| judge_body(0.99, "accept")).collect();
    let judge = Arc::new(ScriptedJudge::new(
        glowing.iter().map(String::as_str).collect::<Vec<&str>>(),
    ));
    let cl = clients(
        store.clone(),
        Arc::new(ExplodingTeacher),
        // A clone, so this test keeps a handle to count the judge's spend below.
        judge.clone(),
        25.0,
        EventSink::disconnected(),
    )
    .with_execution_evidence_source(Arc::new(evidence_source()));
    let area = area_k1(one_judge(), lenient_thresholds()).with_cot_required(false);

    // The engine-derived key IS the candidate's identity, so a well-behaved report binds to it.
    let key = evidence_key(&stored).expect("the chain record is keyable");
    assert_eq!(key.task, RUN);
    assert_eq!(key.attempt, ADMITTED);
    assert!(!key.patch_hash.is_empty());

    let mut outcomes = Vec::new();
    for id in ids {
        let rec = store.get(id).await.expect("reload for the drive");
        let done = drive(rec, &cl, &area, &CancellationToken::new())
            .await
            .expect("drive the record to a terminal state");
        outcomes.push((id, done.lifecycle.state, done.judging.verdict));
    }

    // Across all three records exactly ONE provider call is spent: the k=1 panel on the ADMITTED one.
    // Both negatives short-circuit in `src/step.rs` before the panel is consulted, so the two
    // over-provisioned glowing bodies above are still unspent. This is what makes the negatives
    // evidence-axis decisions rather than score decisions — no score was ever available to them.
    //
    // SCOPE OF THIS ASSERTION: it is spend evidence only. Because the three records are cache-elided
    // against one shared `record_hash` (see the stage-4 comment), this count would still read 1 if the
    // `src/step.rs` short-circuit were deleted and every record were handed to the panel. Do not read it
    // as short-circuit evidence — the empty-panel assertions below carry that, and they fail loudly
    // without it.
    assert_eq!(
        judge.call_count(),
        1,
        "the whole loop must spend exactly one judge token: a verifier-held or verifier-failed \
         record never reaches a panel, so it cannot be rescued or sunk by a score"
    );

    // Bound + corroborated pass → admitted, all the way to the exported terminal.
    assert_eq!(
        outcomes[0],
        (ADMITTED, LifecycleState::Exported, Some(Verdict::Admit)),
        "a bound, corroborated pass must reach the exported terminal"
    );
    // A STALE report (same attempt, moved content) is never followed and never sinks the record.
    assert_eq!(
        outcomes[1],
        (
            STALE,
            LifecycleState::NeedsReview,
            Some(Verdict::NeedsReview)
        ),
        "a stale pass is held by the evidence axis BEFORE the panel is consulted, so no judge score \
         can admit it"
    );
    // A report bound to ANOTHER attempt describes another candidate: a definite failure.
    assert_eq!(
        outcomes[2],
        (CROSS, LifecycleState::Rejected, Some(Verdict::Reject)),
        "another attempt's pass must never admit this record"
    );
    // The two negatives are held/rejected for the RIGHT reason: only the cross-attempt one is
    // allowed to claim a proven failure, because a stale report proves nothing about this content.
    let stale = store.get(STALE).await.expect("get stale");
    let cross = store.get(CROSS).await.expect("get cross");
    assert!(
        stale.verification.all_passed,
        "a stale report must not reject on content it never saw"
    );
    assert!(stale.verification.needs_review.is_some());
    assert!(
        !cross.verification.all_passed,
        "another attempt's report is a hard failure of this candidate"
    );

    // --- the DISCRIMINATING observable: an empty panel on both negatives ---------------------
    // The provider-call count above is elidable and therefore cannot detect the short-circuit
    // (all three records share one content hash, so a panel consultation on a negative would be a
    // cache HIT and spend nothing). The persisted `judging.panel` CAN: `src/step.rs` short-circuits
    // into `HybridGrader::grade` with an EMPTY `&[]` panel, and the grader maps that slice straight
    // into the block (`panel: panel.iter().map(Grade::to_vote).collect()`, gw-judge grader.rs:139
    // for the hard-reject arm and :170 for the undecidable arm). So a record the panel never saw
    // persists ZERO votes, and a record that did reach the panel persists one vote per judge.
    //
    // Non-vacuity: the ADMITTED record's k=1 panel DID run and persisted its single vote. The
    // contrast is asserted here too, so a bug that emptied the panel everywhere (or a grader that
    // stopped recording votes) fails this block instead of passing it by accident.
    assert_eq!(
        stale.judging.panel.len(),
        0,
        "the stale record is held by the evidence axis, so the panel was never consulted and no \
         judge vote may exist; a nonempty panel would mean a score was consulted"
    );
    assert_eq!(
        cross.judging.panel.len(),
        0,
        "the cross-attempt record is rejected by the evidence axis, so the panel was never \
         consulted and no judge vote may exist; a nonempty panel would mean a score was consulted"
    );
    let admitted = store.get(ADMITTED).await.expect("get admitted");
    assert_eq!(
        admitted.judging.panel.len(),
        1,
        "the admitted record DID reach its k=1 panel and recorded that one vote — the contrast that \
         makes the two empty-panel assertions above non-vacuous"
    );

    // The advance is an event log, and the resume cursor follows the furthest committed state.
    let history: Vec<String> = store
        .lifecycle_history(ADMITTED)
        .await
        .expect("lifecycle history")
        .into_iter()
        .map(|(state, _, _)| state)
        .collect();
    assert_eq!(
        history,
        vec!["verified", "judged", "admitted", "formatted", "exported"],
        "the admitted record's edges are the ones the chain claims"
    );
    for (id, terminal, _) in &outcomes[1..] {
        let states: Vec<String> = store
            .lifecycle_history(id)
            .await
            .expect("lifecycle history")
            .into_iter()
            .map(|(state, _, _)| state)
            .collect();
        assert_eq!(
            states.last().map(String::as_str),
            Some(match terminal {
                LifecycleState::NeedsReview => "needs_review",
                _ => "rejected",
            }),
            "{id} stopped at its terminal state"
        );
        assert!(
            !states.iter().any(|s| s == "admitted"),
            "{id} must never have been admitted"
        );
    }
    store
        .checkpoint(RUN, 0, "exported", &serde_json::json!({"seed_offset": 8}))
        .await
        .expect("re-checkpoint the shard");
    assert_eq!(
        store
            .resume_cursor(RUN, 0)
            .await
            .expect("re-read the resume cursor")
            .expect("a cursor exists")
            .state,
        "exported",
        "a relaunch would resume past the whole chain, not re-spend the teacher"
    );

    // --- stage 5: export → independent decode -------------------------------------------------
    let scanned = store
        .scan(&RecordFilter::new().run_id(RUN))
        .await
        .expect("scan the run");
    assert_eq!(scanned.len(), 3, "all three records are in the run");
    let (bytes, manifest) = export_parquet_bytes(&scanned, TrlFormat::Gemma4, CotPolicy::Masked)
        .await
        .expect("export the shard");
    assert_eq!(manifest.n_records, 3, "n_records counts the input set");
    assert_eq!(
        manifest.n_admitted, 1,
        "the two declined records must not be written"
    );
    assert_eq!(
        manifest.column_schema_version,
        ExportSchemaVersion::CanonicalMessages
    );
    assert!(!manifest.build_inputs_hash.is_empty());

    let shard = temp_path("chain.parquet");
    cleanup(&shard);
    std::fs::write(&shard, &bytes).expect("write the shard artifact");
    let (shard_ids, shard_turns) = decode_shard(&shard);
    assert_eq!(
        shard_ids,
        vec![ADMITTED.to_string()],
        "only the admitted record reaches the artifact"
    );
    assert_eq!(
        shard_turns, turns,
        "a consumer decodes the conversation ingest produced, field for field and in order"
    );
    assert_eq!(shard_turns[3].tool_call_id.as_deref(), Some("read-b"));
    assert_eq!(shard_turns[4].tool_call_id.as_deref(), Some("read-a"));

    // --- stage 6: render guard, on the DECODED (consumer-side) conversation -------------------
    let err = render(&shard_turns, TrlFormat::Gemma4, CotPolicy::Masked)
        .expect_err("a target with no tool representation must refuse the trajectory");
    let diagnostic = err.to_string();
    for expected in ["Gemma4", "messages[2]", "messages_json"] {
        assert!(
            diagnostic.contains(expected),
            "the refusal must stay actionable on its own: {diagnostic}"
        );
    }
    match err {
        FormatError::UnsupportedToolCalls {
            target,
            signals,
            index,
            recovery,
        } => {
            assert_eq!(target, TrlFormat::Gemma4, "the route that refused is named");
            assert_eq!(index, 2, "the tool-calling turn is located");
            assert!(
                signals.contains(&ToolSignal::ToolCalls)
                    && signals.contains(&ToolSignal::ToolCallId),
                "the carriers this trajectory would lose are named: {signals:?}"
            );
            assert_eq!(
                recovery,
                ToolCallRecovery::CanonicalExportAndOfficialTemplate,
                "the way out is the canonical export, not a lossy render"
            );
        }
        other => panic!("wrong error class: {other:?}"),
    }

    // The tool-faithful route keeps every id and every result link, so the trajectory is not
    // unexportable anywhere — only unrepresentable on the dropping routes.
    let openai = render(&shard_turns, TrlFormat::OpenAiMessages, CotPolicy::Masked)
        .expect("a tool-faithful target is never refused");
    let wire: Value = serde_json::from_str(&openai).expect("OpenAI-shaped JSON");
    let msgs = wire["messages"].as_array().expect("a messages array");
    assert_eq!(msgs[2]["tool_calls"][0]["id"], "read-a");
    assert_eq!(msgs[2]["tool_calls"][1]["id"], "read-b");
    assert_eq!(msgs[3]["tool_call_id"], "read-b");
    assert_eq!(msgs[4]["tool_call_id"], "read-a");

    // --- stage 7: `gen export` parity ---------------------------------------------------------
    // The handler is exactly `Store::open` → `scan` → `export_parquet`; this walks that sequence
    // over the same database and requires the artifact to be the same shard. The handler function
    // itself is invoked (over its own corpus) by `gw-cli`'s `export.rs`, which the workspace gate
    // runs: `gw-cli` is DOWNSTREAM of `gw-engine`, so a `gw-engine` test cannot call it.
    drop(cl);
    drop(store);
    let reopened = Store::open(&db)
        .await
        .expect("reopen the store as the handler does");
    let handler_scan = reopened
        .scan(&RecordFilter::new().run_id(RUN))
        .await
        .expect("the handler's scan");
    let out = temp_path("cli.parquet");
    cleanup(&out);
    let handler_manifest =
        export_parquet(&handler_scan, TrlFormat::Gemma4, CotPolicy::Masked, &out)
            .await
            .expect("the handler's export");
    assert_eq!(
        handler_manifest, manifest,
        "the handler's manifest must agree with the in-memory export field for field"
    );
    let written = std::fs::read(&out).expect("read the handler's artifact");
    assert_eq!(&written[..4], b"PAR1", "the artifact is a Parquet file");
    assert_eq!(
        written, bytes,
        "the handler's file must be the same shard the in-memory exporter encodes"
    );
    let (handler_ids, handler_turns) = decode_shard(&out);
    assert_eq!(handler_ids, shard_ids, "same row");
    assert_eq!(handler_turns, shard_turns, "same conversation");
    drop(reopened);

    cleanup(&out);
    cleanup(&shard);
    cleanup(&db);
}
