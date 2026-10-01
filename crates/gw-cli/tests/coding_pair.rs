//! Synthetic storage setup for paired software qualification; prior reference execution is declared.
#[path = "../../gw-schema/tests/reference_support/mod.rs"]
mod support;
use gw_schema::*;
use gw_storage::{ReferenceMemberObservation, Store};

fn capture() -> ReferenceCapture {
    let mut capture = support::capture();
    let mut ordinal = 0;
    let metadata = std::env::var("GW_PAIR_TEST_LARGE_METADATA").ok();
    for document in &mut capture.task_documents {
        let mut tasks = CodingTaskDocument::from_json(document.as_bytes()).unwrap();
        for task in &mut tasks.tasks {
            task.description = format!("Owned comparison control {ordinal}: return integer zero.");
            if ordinal == 81 && std::env::var_os("GW_PAIR_TEST_REJECTED_PROMPT").is_some() {
                task.description.push_str(" <bos>");
            }
            if metadata.as_deref() == Some("train") && ordinal < 64 {
                task.source.citation = "é".repeat(40 * 1024);
            } else if metadata.as_deref() == Some("heldout") && ordinal >= 80 {
                task.source.citation = "é".repeat(80 * 1024);
            }
            task.function.entry_point = "probe".into();
            task.function.parameters.clear();
            task.visible_examples.clear();
            task.train_cases.clear();
            task.protected_cases.clear();
            let case = CodingCase {
                label: "PRIVATE_ZERO_ORACLE".into(),
                arguments: vec![],
                expected: CodingValue::Integer(0),
            };
            if ordinal < 64 {
                task.train_cases.push(case);
            } else {
                task.protected_cases.push(case);
            }
            capture.modules[ordinal] =
                format!("# Synthetic comparison fixture {ordinal}\ndef probe():\n    return 0\n");
            let mut review: ReferenceReview =
                serde_json::from_str(&capture.reviews[ordinal]).unwrap();
            review.task_digest = reference_task_digest(task);
            review.reference_code_id = coding_digest(
                "ghostwriter.coding-module.v1",
                capture.modules[ordinal].as_bytes(),
            );
            capture.reviews[ordinal] = serde_json::to_string(&review).unwrap();
            ordinal += 1;
        }
        *document = serde_json::to_string(&tasks).unwrap();
    }
    if metadata.is_some() {
        let tasks = capture
            .task_documents
            .iter()
            .flat_map(|document| {
                serde_json::from_str::<CodingTaskDocument>(document)
                    .unwrap()
                    .tasks
            })
            .collect::<Vec<_>>();
        capture.task_documents = tasks
            .into_iter()
            .map(|task| {
                serde_json::to_string(&CodingTaskDocument {
                    version: 1,
                    tasks: vec![task],
                })
                .unwrap()
            })
            .collect();
        let mut catalogue: ReferenceCatalogue = serde_json::from_str(&capture.catalogue).unwrap();
        catalogue.task_documents = (0..112).map(|i| format!("member-{i}.json")).collect();
        for (index, member) in catalogue.members.iter_mut().enumerate() {
            member.task_document = index;
        }
        capture.catalogue = serde_json::to_string(&catalogue).unwrap();
    }
    capture
}
#[tokio::test]
async fn paired_population_is_complete_current_redacted_and_registered() {
    let output = std::env::var_os("GW_PAIR_TEST_DIRECTORY").map(std::path::PathBuf::from);
    let store = if let Some(directory) = &output {
        std::fs::create_dir_all(directory).unwrap();
        Store::open(directory.join("reference.sqlite"))
            .await
            .unwrap()
    } else {
        Store::open_in_memory().await.unwrap()
    };
    let registered = store
        .register_reference_catalogue(&capture())
        .await
        .unwrap();
    assert!(
        store
            .capture_coding_population(registered.registration_id(), TaskSplitRole::Test)
            .await
            .is_err()
    );
    let observations = registered.population().members().iter().map(|member| {
        let absent = VerificationAxis { policy: VerificationPolicy::Absent, observation: None };
        let verification = VerificationInterpretation { version: 2, reasoning: absent.clone(), answer: absent,
            execution: VerificationAxis { policy: VerificationPolicy::Authoritative,
                observation: Some(VerificationObservation { outcome: VerificationOutcome::Pass, reason: "synthetic native adapter fixture".into() }) } };
        let result_id = coding_digest("synthetic-test-only", member.member_id.as_bytes());
        let evidence = serde_json::json!({"artifact_id":result_id,"input":{"task":member.task,"code":member.code,
            "code_id":member.reference_code_id,"suite":member.task.suite_binding()},"report":{"native_verification":verification}});
        ReferenceMemberObservation::from_native_observation(member.member_id.clone(), result_id, verification, evidence).unwrap()
    }).collect::<Vec<_>>();
    store
        .commit_reference_import(&registered, &observations, std::future::pending())
        .await
        .unwrap();
    for (split, count, start) in [
        (TaskSplitRole::Validation, 16, 64),
        (TaskSplitRole::Test, 32, 80),
    ] {
        let captured = store
            .capture_coding_population(registered.registration_id(), split)
            .await
            .unwrap();
        assert_eq!(captured.private_tasks().len(), count);
        let public = captured.public();
        assert_eq!(public.members.len(), count);
        assert_eq!(public.training_members.len(), 64);
        assert_eq!(public.members[0].ordinal, start);
        let json = serde_json::to_string(public).unwrap();
        for private in [
            "PRIVATE_REVIEW_CANARY",
            "PRIVATE_ZERO_ORACLE",
            "\"expected\":",
        ] {
            assert!(!json.contains(private), "{private}");
        }
        assert!(
            serde_json::to_string(&public.members)
                .unwrap()
                .find("def probe():")
                .is_none()
        );
        let mut changed_train = public.clone();
        changed_train.training_members[0].record_id =
            changed_train.training_members[1].record_id.clone();
        changed_train.population_id = changed_train.computed_id();
        assert!(changed_train.validate().is_err());
        let mut omitted = public.clone();
        omitted.members.pop();
        omitted.population_id = omitted.computed_id();
        assert!(omitted.validate().is_err());
        let mut swapped = public.clone();
        swapped.members.swap(0, 1);
        swapped.population_id = swapped.computed_id();
        assert!(swapped.validate().is_err());
    }
    assert!(
        store
            .capture_coding_population(registered.registration_id(), TaskSplitRole::Train)
            .await
            .is_err()
    );
    assert!(
        store
            .capture_coding_population(&"0".repeat(64), TaskSplitRole::Test)
            .await
            .is_err()
    );
    if let Some(directory) = output {
        std::fs::write(directory.join("registration.json"),serde_json::to_vec(&serde_json::json!({"registration_id":registered.registration_id(),"prior_reference_execution":"synthetic_test_fixture"})).unwrap()).unwrap();
    }
}
