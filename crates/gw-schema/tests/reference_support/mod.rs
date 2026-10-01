//! Explicitly synthetic population for software contracts; not a reviewed reference corpus.
use gw_schema::*;
pub fn capture() -> ReferenceCapture {
    let seed = CodingTaskDocument::from_json(include_bytes!(
        "../../../../examples/reviewed-coding-tasks.json"
    ))
    .unwrap()
    .tasks
    .remove(0);
    let mut documents = vec![
        CodingTaskDocument {
            version: 1,
            tasks: vec![]
        };
        2
    ];
    let mut catalogue = ReferenceCatalogue {
        version: 1,
        training_area: "synthetic".into(),
        task_documents: vec!["a.json".into(), "b.json".into()],
        members: vec![],
    };
    let mut modules = vec![];
    let mut reviews = vec![];
    for index in 0..112 {
        let mut task = seed.clone();
        let label = format!("synthetic-{index}");
        task.task_id = label.clone();
        task.source.item = label.clone();
        task.group.id = format!("family-{}", index / 4);
        task.split.role = if index < 64 {
            TaskSplitRole::Train
        } else if index < 80 {
            TaskSplitRole::Validation
        } else {
            TaskSplitRole::Test
        };
        task.rights.permitted_uses = vec![TaskPermittedUse::Training, TaskPermittedUse::Evaluation];
        if index >= 64 {
            task.protected_cases = std::mem::take(&mut task.train_cases);
        }
        let code = format!(
            "# synthetic member {index}\ndef {}({}):\n    return 0\n",
            task.function.entry_point,
            task.function.parameters.join(", ")
        );
        let review = ReferenceReview {
            version: 1,
            task_digest: reference_task_digest(&task),
            reference_code_id: coding_digest("ghostwriter.coding-module.v1", code.as_bytes()),
            author: ReferenceActor {
                kind: ReferenceActorKind::Agent,
                label: "synthetic author".into(),
            },
            reviewer: ReferenceActor {
                kind: ReferenceActorKind::Agent,
                label: "synthetic reviewer".into(),
            },
            independence: "Synthetic test assertion".into(),
            correctness: "PRIVATE_REVIEW_CANARY".into(),
            oracle: "Synthetic partition review".into(),
            rights: "Synthetic rights review".into(),
            permitted_use: if index < 64 {
                TaskPermittedUse::Training
            } else {
                TaskPermittedUse::Evaluation
            },
        };
        let component = task.group.clone();
        documents[index / 64].tasks.push(task);
        catalogue.members.push(ReferenceMemberDeclaration {
            task_document: index / 64,
            task_id: label.clone(),
            module_path: format!("{label}.py"),
            review_path: format!("{label}-review.json"),
            component,
        });
        modules.push(code);
        reviews.push(serde_json::to_string(&review).unwrap());
    }
    ReferenceCapture {
        catalogue: serde_json::to_string(&catalogue).unwrap(),
        task_documents: documents
            .iter()
            .map(|d| serde_json::to_string(d).unwrap())
            .collect(),
        modules,
        reviews,
    }
}
