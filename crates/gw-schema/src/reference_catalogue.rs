//! Complete captured reference populations. Validation is pure and confers no operator authority.
use crate::coding_value::coding_json_digest;
use crate::{
    CodingTaskDocument, NamespacedTaskId, ReferenceActorKind, ReferenceAuthorship,
    ReviewedCodingTask, TaskDeclarations, TaskPermittedUse, TaskProvenance, TaskSplit,
    TaskSplitRole,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Strict catalogue for the supported 28-family, four-members-per-family population.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceCatalogue {
    /// Exactly one.
    pub version: u32,
    /// Public training area for the 64 Train members.
    pub training_area: String,
    /// Ordered relative document paths. Each document retains its 64-task bound.
    pub task_documents: Vec<String>,
    /// Complete ordered membership across all documents.
    pub members: Vec<ReferenceMemberDeclaration>,
}
/// One captured member's sources; no field declares approval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceMemberDeclaration {
    /// Index in the catalogue's ordered document list.
    pub task_document: usize,
    /// Exact source task label.
    pub task_id: String,
    /// Relative complete Python module path.
    pub module_path: String,
    /// Relative review-evidence path.
    pub review_path: String,
    /// Whole related component, which must never cross split assignments.
    pub component: NamespacedTaskId,
}
/// An operator's declared actor, retained privately. This is not external authentication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceActor {
    /// Truthful declared category.
    pub kind: ReferenceActorKind,
    /// Nonempty private actor label.
    pub label: String,
}
/// Private review coverage tied to exact task and code semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceReview {
    /// Exactly one.
    pub version: u32,
    /// Digest of the entire reviewed task, including private cases and declarations.
    pub task_digest: String,
    /// Digest of the exact complete module bytes.
    pub reference_code_id: String,
    /// Declared author.
    pub author: ReferenceActor,
    /// Declared reviewer.
    pub reviewer: ReferenceActor,
    /// How the review was independent of authorship, without a certification claim.
    pub independence: String,
    /// Specific correctness review evidence.
    pub correctness: String,
    /// Specific oracle and partition review evidence.
    pub oracle: String,
    /// Specific rights and source review evidence.
    pub rights: String,
    /// Required use for the member's assigned split.
    pub permitted_use: TaskPermittedUse,
}
/// Exact once-captured UTF-8 input bytes, retained in the private registration ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceCapture {
    /// Exact original catalogue JSON, including whitespace.
    pub catalogue: String,
    /// Exact task document JSON in catalogue order.
    pub task_documents: Vec<String>,
    /// Exact modules in member order.
    pub modules: Vec<String>,
    /// Exact review JSON in member order.
    pub reviews: Vec<String>,
}
/// One pure validated member; it has no fresh-execution or registration authority.
#[derive(Debug, Clone)]
pub struct ValidatedReferenceMember {
    /// Ordered identity binding every captured member source and its declarations.
    pub member_id: String,
    /// Complete private reviewed task.
    pub task: ReviewedCodingTask,
    /// Exact captured module.
    pub code: String,
    /// Exact module content identity.
    pub reference_code_id: String,
    /// Whole component declaration.
    pub component: NamespacedTaskId,
    /// Public truthful authorship categories.
    pub authorship: ReferenceAuthorship,
}
/// Validated complete population. Only explicit Store registration creates operator authority.
#[derive(Debug, Clone)]
pub struct ValidatedReferenceCatalogue {
    catalogue_id: String,
    catalogue: ReferenceCatalogue,
    members: Vec<ValidatedReferenceMember>,
}
impl ValidatedReferenceCatalogue {
    /// Exact byte capture identity.
    #[must_use]
    pub fn catalogue_id(&self) -> &str {
        &self.catalogue_id
    }
    /// Public catalogue declarations.
    #[must_use]
    pub fn catalogue(&self) -> &ReferenceCatalogue {
        &self.catalogue
    }
    /// Complete validated members in their original order.
    #[must_use]
    pub fn members(&self) -> &[ValidatedReferenceMember] {
        &self.members
    }
}
/// Exact complete reviewed-task identity used by private review evidence.
#[must_use]
pub fn reference_task_digest(task: &ReviewedCodingTask) -> String {
    coding_json_digest("ghostwriter.reference-task.v1", task)
}
fn parse<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, String> {
    serde_json::from_value(crate::strict_coding_json(text.as_bytes())?).map_err(|e| e.to_string())
}
fn nonblank(text: &str) -> Result<(), String> {
    if text.trim().is_empty() {
        Err("empty reference declaration".into())
    } else {
        Ok(())
    }
}
fn path(text: &str) -> Result<(), String> {
    if text.is_empty()
        || text.contains('\\')
        || text.contains('\0')
        || text.split('/').any(|part| matches!(part, "" | "." | ".."))
    {
        return Err("reference paths must be bounded normal relative paths".into());
    }
    Ok(())
}
impl ReferenceCapture {
    /// Validate all bytes, coverage, identities, family/component splits and rights before execution.
    ///
    /// # Errors
    /// Rejects malformed/oversized input, incomplete reviews, changed semantics and any population
    /// other than 16 Train, four Validation and eight Test families of four unique members each.
    pub fn validate(&self) -> Result<ValidatedReferenceCatalogue, String> {
        if self.catalogue.len() > 1024 * 1024
            || self.task_documents.len() > 112
            || self.modules.len() != 112
            || self.reviews.len() != 112
            || self.task_documents.iter().any(|s| s.len() > 1024 * 1024)
            || self.modules.iter().any(|s| s.is_empty() || s.len() > 65536)
            || self.reviews.iter().any(|s| s.len() > 65536)
            || self.catalogue.len()
                + self.task_documents.iter().map(String::len).sum::<usize>()
                + self.modules.iter().map(String::len).sum::<usize>()
                + self.reviews.iter().map(String::len).sum::<usize>()
                > 32 * 1024 * 1024
        {
            return Err("reference capture size or member-count bound exceeded".into());
        }
        let catalogue: ReferenceCatalogue = parse(&self.catalogue)?;
        if catalogue.version != 1
            || catalogue.members.len() != 112
            || catalogue.task_documents.len() != self.task_documents.len()
        {
            return Err("unsupported or incomplete reference catalogue".into());
        }
        nonblank(&catalogue.training_area)?;
        let mut paths = HashSet::new();
        for name in &catalogue.task_documents {
            path(name)?;
            if !paths.insert(name) {
                return Err("duplicate task document path".into());
            }
        }
        let mut declarations = TaskDeclarations::default();
        let mut tasks = HashMap::new();
        for (index, bytes) in self.task_documents.iter().enumerate() {
            let document = CodingTaskDocument::from_json(bytes.as_bytes())?;
            for task in document.tasks {
                declarations.insert(&TaskProvenance::from_coding_task(&task)?)?;
                tasks.insert((index, task.task_id.clone()), task);
            }
        }
        if tasks.len() != 112 {
            return Err("catalogue requires exactly 112 unique source tasks".into());
        }
        let catalogue_id = coding_json_digest("ghostwriter.reference-capture.v1", self);
        let mut families = HashMap::<NamespacedTaskId, (TaskSplit, usize)>::new();
        let mut components = HashMap::<NamespacedTaskId, TaskSplit>::new();
        let mut members = Vec::with_capacity(112);
        for (index, member) in catalogue.members.iter().enumerate() {
            path(&member.module_path)?;
            path(&member.review_path)?;
            for name in [&member.module_path, &member.review_path] {
                if !paths.insert(name) {
                    return Err("duplicate catalogue input path".into());
                }
            }
            nonblank(&member.component.namespace)?;
            nonblank(&member.component.id)?;
            let task = tasks
                .remove(&(member.task_document, member.task_id.clone()))
                .ok_or("duplicate or missing catalogue task membership")?;
            let family = families
                .entry(task.group.clone())
                .or_insert((task.split.clone(), 0));
            if family.0 != task.split {
                return Err("family crosses split assignments".into());
            }
            family.1 += 1;
            if let Some(previous) = components.insert(member.component.clone(), task.split.clone())
                && previous != task.split
            {
                return Err("component crosses split assignments".into());
            }
            let required_use = match task.split.role {
                TaskSplitRole::Train => TaskPermittedUse::Training,
                TaskSplitRole::Validation | TaskSplitRole::Test => TaskPermittedUse::Evaluation,
            };
            let review: ReferenceReview = parse(&self.reviews[index])?;
            let reference_code_id = crate::coding_digest(
                "ghostwriter.coding-module.v1",
                self.modules[index].as_bytes(),
            );
            if review.version != 1
                || review.task_digest != reference_task_digest(&task)
                || review.reference_code_id != reference_code_id
                || review.permitted_use != required_use
                || !task.rights.permitted_uses.contains(&required_use)
            {
                return Err("review does not cover exact task, code or required use".into());
            }
            for text in [
                &review.author.label,
                &review.reviewer.label,
                &review.independence,
                &review.correctness,
                &review.oracle,
                &review.rights,
            ] {
                nonblank(text)?;
            }
            let member_id = coding_json_digest(
                "ghostwriter.reference-member.v1",
                &(
                    &catalogue_id,
                    index,
                    member,
                    &review.task_digest,
                    &reference_code_id,
                ),
            );
            members.push(ValidatedReferenceMember {
                member_id,
                task,
                code: self.modules[index].clone(),
                reference_code_id,
                component: member.component.clone(),
                authorship: ReferenceAuthorship {
                    author: review.author.kind,
                    reviewer: review.reviewer.kind,
                },
            });
        }
        let counts = [
            TaskSplitRole::Train,
            TaskSplitRole::Validation,
            TaskSplitRole::Test,
        ]
        .map(|role| {
            families
                .values()
                .filter(|(split, _)| split.role == role)
                .count()
        });
        if !tasks.is_empty()
            || families.len() != 28
            || counts != [16, 4, 8]
            || families.values().any(|(_, count)| *count != 4)
        {
            return Err(
                "reference population requires 28 four-member families split 16/4/8".into(),
            );
        }
        Ok(ValidatedReferenceCatalogue {
            catalogue_id,
            catalogue,
            members,
        })
    }
}
