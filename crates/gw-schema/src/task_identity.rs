//! Pure validation and canonical identity for reviewed numeric tasks.
use crate::{
    Content, Message, NamespacedTaskId, NumericTaskDocument, Oracle, ReviewedNumericTask,
    ReviewedTaskRights, Role, TASK_DOCUMENT_VERSION, TaskObservations, TaskSource, TaskSplit,
    VerificationContract, VerificationKind, VerificationPolicy, parse_finite_decimal,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Version of the canonical source/prompt/numeric-semantics identity.
pub const TASK_IDENTITY_VERSION: u32 = 1;

/// Derived identity. Intake accepts no caller-supplied digest; persisted identities are rechecked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticTaskIdentity {
    /// Canonical encoding version.
    pub version: u32,
    /// Lowercase BLAKE3 over the canonical semantic projection.
    pub digest: String,
}

/// Typed task provenance retained through generation, retries, storage, and export.
/// Prompt and answer settings remain authoritative in messages/verification_contract, avoiding
/// duplicate answer definitions. The identity binds their actual values to this source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskProvenance {
    /// Source task label; never interpreted as a content digest.
    pub task_id: String,
    /// Identity derived from actual source, prompt, and numeric answer semantics.
    pub identity: SemanticTaskIdentity,
    /// Reviewed immutable source and citation.
    pub source: TaskSource,
    /// Reviewed rights declarations.
    pub rights: ReviewedTaskRights,
    /// Declared corpus group; distinct from a prompt-hash sibling group.
    pub group: NamespacedTaskId,
    /// Declared split manifest/revision/role, outside the semantic digest.
    pub split: TaskSplit,
    /// Reviewed domain/difficulty/QC observations.
    pub observations: TaskObservations,
}

/// Self-contained task block in a v3 export row. The conversation column supplies the actual prompt;
/// this block retains its reviewed provenance and the record's sole numeric answer contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportTaskProjection {
    /// Reviewed declarations and derived semantic identity.
    pub provenance: TaskProvenance,
    /// Exact answer/extraction/tolerance/policy contract used by verification.
    pub verification_contract: VerificationContract,
}
impl ExportTaskProjection {
    /// Validate the projected task against its row's canonical conversation.
    ///
    /// # Errors
    /// Rejects a missing prompt, malformed declarations, or substituted answer/source identity.
    pub fn validate(&self, messages: &[Message]) -> Result<(), &'static str> {
        let prompt = messages
            .first()
            .ok_or("exported numeric task lacks its user prompt")?;
        self.provenance
            .validate_for(prompt, &self.verification_contract)
    }
}

fn nonblank(text: &str) -> Result<(), &'static str> {
    if text.trim().is_empty() {
        Err("task identities, text, evidence, and observations must be nonempty")
    } else {
        Ok(())
    }
}
fn references(values: &[String]) -> Result<(), &'static str> {
    if values.is_empty() {
        return Err("reviewed task evidence must be nonempty");
    }
    for value in values {
        nonblank(value)?;
    }
    Ok(())
}
fn namespace(id: &NamespacedTaskId) -> Result<(), &'static str> {
    nonblank(&id.namespace)?;
    nonblank(&id.id)
}

impl NumericTaskDocument {
    /// Parse and validate one strict JSON document. No file, network, or model access occurs.
    ///
    /// # Errors
    /// Rejects malformed/unknown fields and versions, invalid tasks, duplicate identities, or
    /// conflicting split declarations for a corpus group.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let document: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        document.validate().map_err(str::to_owned)?;
        Ok(document)
    }

    /// Validate the entire ordered document before any task can be materialized or dispatched.
    ///
    /// # Errors
    /// Rejects unsupported/empty documents, invalid tasks, duplicates, and group split conflicts.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != TASK_DOCUMENT_VERSION {
            return Err("unsupported numeric task document version");
        }
        if self.tasks.is_empty() {
            return Err("numeric task document must contain tasks");
        }
        let mut declarations = TaskDeclarations::default();
        for task in &self.tasks {
            task.validate()?;
            declarations.insert(&TaskProvenance::from_task(task)?)?;
        }
        Ok(())
    }
}

impl ReviewedNumericTask {
    /// Validate reviewed fields and the supported literal numeric contract.
    ///
    /// # Errors
    /// Rejects incomplete declarations, malformed numeric semantics, and active execution policy.
    pub fn validate(&self) -> Result<(), &'static str> {
        nonblank(self.prompt.text())?;
        validate_numeric_contract(&self.verification.contract())?;
        let provenance = TaskProvenance {
            task_id: self.task_id.clone(),
            identity: SemanticTaskIdentity {
                version: TASK_IDENTITY_VERSION,
                digest: String::new(),
            },
            source: self.source.clone(),
            rights: self.rights.clone(),
            group: self.group.clone(),
            split: self.split.clone(),
            observations: self.observations.clone(),
        };
        provenance.validate_declarations()
    }
}

/// Validate the supported task family, also used when checking persisted provenance.
fn validate_numeric_contract(contract: &VerificationContract) -> Result<(), &'static str> {
    contract.validate()?;
    if contract.kind != VerificationKind::NumericMatch
        || !matches!(contract.oracle, Oracle::Literal { .. })
        || contract.execution_policy != Some(VerificationPolicy::Absent)
        || !contract.required_tests.is_empty()
    {
        return Err(
            "reviewed numeric tasks require a literal numeric oracle and absent execution policy",
        );
    }
    Ok(())
}

impl TaskProvenance {
    /// Construct provenance using a derived digest, never a caller's claimed hash.
    ///
    /// # Errors
    /// Rejects incomplete declarations or unsupported numeric semantics.
    pub fn from_task(task: &ReviewedNumericTask) -> Result<Self, &'static str> {
        task.validate()?;
        Ok(Self {
            task_id: task.task_id.clone(),
            identity: semantic_identity(
                &task.source,
                task.prompt.text(),
                &task.verification.contract(),
            )?,
            source: task.source.clone(),
            rights: task.rights.clone(),
            group: task.group.clone(),
            split: task.split.clone(),
            observations: task.observations.clone(),
        })
    }

    fn validate_declarations(&self) -> Result<(), &'static str> {
        for value in [
            &self.task_id,
            &self.source.namespace,
            &self.source.item,
            &self.source.revision,
            &self.source.citation,
            &self.rights.reviewer,
            &self.split.revision,
            &self.observations.domain,
            &self.observations.difficulty.label,
            &self.observations.difficulty.basis,
            &self.observations.qc.reviewer,
        ] {
            nonblank(value)?;
        }
        namespace(&self.group)?;
        namespace(&self.split.manifest)?;
        references(&self.rights.evidence)?;
        references(&self.observations.qc.evidence)?;
        let uses: HashSet<_> = self.rights.permitted_uses.iter().collect();
        if uses.is_empty() || uses.len() != self.rights.permitted_uses.len() {
            return Err("reviewed permitted uses must be nonempty and unique");
        }
        Ok(())
    }

    /// Check a persisted/caller-constructed identity against the actual prompt and answer contract.
    ///
    /// # Errors
    /// Rejects unsupported message structures, declarations, or a substituted semantic digest.
    pub fn validate_for(
        &self,
        message: &Message,
        contract: &VerificationContract,
    ) -> Result<(), &'static str> {
        self.validate_declarations()?;
        let Content::Text(prompt) = &message.content else {
            return Err("numeric task prompt must be user text");
        };
        if message.role != Role::User
            || message.reasoning.is_some()
            || message.reasoning_details.is_some()
            || message.tool_calls.is_some()
            || message.tool_call_id.is_some()
            || message.name.is_some()
        {
            return Err("numeric task prompt must be one plain user text message");
        }
        nonblank(prompt)?;
        if self.identity != semantic_identity(&self.source, prompt, contract)? {
            return Err(
                "task semantic identity disagrees with actual source, prompt, or numeric answer",
            );
        }
        Ok(())
    }
}

/// Pure whole-plan duplicate and group/split validation, reused for arbitrary library sources.
#[derive(Debug, Default)]
pub struct TaskDeclarations {
    labels: HashSet<String>,
    identities: HashSet<String>,
    sources: HashSet<(String, String, String)>,
    groups: HashMap<NamespacedTaskId, TaskSplit>,
}
impl TaskDeclarations {
    /// Add one validated source item to an ordered plan.
    ///
    /// # Errors
    /// Rejects duplicate labels, semantic/source identities, or conflicting group splits.
    pub fn insert(&mut self, task: &TaskProvenance) -> Result<(), &'static str> {
        if !self.labels.insert(task.task_id.clone())
            || !self.identities.insert(task.identity.digest.clone())
            || !self.sources.insert((
                task.source.namespace.clone(),
                task.source.item.clone(),
                task.source.revision.clone(),
            ))
        {
            return Err("duplicate task label, source item, or semantic identity");
        }
        if let Some(split) = self.groups.insert(task.group.clone(), task.split.clone())
            && split != task.split
        {
            return Err("conflicting split assignments for one declared corpus group");
        }
        Ok(())
    }
}

fn semantic_identity(
    source: &TaskSource,
    prompt: &str,
    contract: &VerificationContract,
) -> Result<SemanticTaskIdentity, &'static str> {
    validate_numeric_contract(contract)?;
    let Oracle::Literal { expected } = &contract.oracle else {
        unreachable!("validated literal oracle")
    };
    let expected =
        parse_finite_decimal(expected).ok_or("numeric expected answer must be finite")?;
    let numeric = contract
        .numeric
        .as_ref()
        .expect("validated numeric settings");
    // serde_json::Value sorts object keys; arrays and exact prompt/source text retain their order.
    // Normalize signed zero so equivalent finite numeric semantics have the same identity.
    let normalize = |v: f64| if v == 0.0 { 0.0 } else { v };
    let value = serde_json::json!({"encoding":"reviewed-numeric-task-v1", "source":source, "prompt":prompt,
        "answer":{"expected": normalize(expected), "extraction":numeric.extraction,
            "tolerance":{"absolute":normalize(numeric.tolerance.absolute),"relative":normalize(numeric.tolerance.relative)}}});
    let bytes = serde_json::to_vec(&value).map_err(|_| "task identity serialization failed")?;
    Ok(SemanticTaskIdentity {
        version: TASK_IDENTITY_VERSION,
        digest: blake3::hash(&bytes).to_hex().to_string(),
    })
}
