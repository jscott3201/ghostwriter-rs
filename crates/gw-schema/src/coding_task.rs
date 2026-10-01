//! Reviewed pure Python function tasks. Private oracles never enter a prompt or export projection.
use crate::coding_value::{coding_hash_valid, coding_json_digest};
use crate::{
    CodingValue, Content, Message, NamespacedTaskId, Oracle, ReviewedTaskRights, Role,
    SemanticTaskIdentity, TaskDeclarations, TaskObservations, TaskProvenance, TaskSource,
    TaskSplit, TaskSplitRole, VerificationContract, VerificationKind, VerificationPolicy,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Supported captured document version; semantic coding identities use their own v2 domain.
pub const CODING_TASK_DOCUMENT_VERSION: u32 = 1;
/// Exact immutable runtime recipe supported by the first local execution controller.
pub const CODING_RUNTIME_RECIPE: &str = "docker_linux_arm64_cpython_3_12_14_v1";

/// Strict ordered reviewed task input. Fields are declarations, not rights certification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingTaskDocument {
    /// Must equal [`CODING_TASK_DOCUMENT_VERSION`].
    pub version: u32,
    /// Tasks with unique labels, source identities, and consistent group splits.
    pub tasks: Vec<ReviewedCodingTask>,
}

/// Exact candidate representation; no Markdown fence stripping or implicit snippet extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingRepresentation {
    /// Complete UTF-8 Python module defining the declared function.
    CompleteUtf8PythonModuleV1,
}
/// Exact recursively typed JSON comparison, computed outside the candidate process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingComparison {
    /// The [`CodingValue`] contract.
    TypedJsonExactV1,
}
/// Private oracle population retained as a redacted semantic requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingPrivatePartition {
    /// Private grading cases for a training task.
    Training,
    /// Protected cases for a validation or test task.
    Protected,
}
/// Fixed positional signature; defaults, variadics, async functions, and keyword-only arguments
/// are unsupported. Parameter names and order are part of the task identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingFunction {
    /// Plain ASCII Python identifier at module scope.
    pub entry_point: String,
    /// Ordered parameter names, at most eight.
    pub parameters: Vec<String>,
    /// Complete module representation.
    pub representation: CodingRepresentation,
}
/// One literal reviewed test. Expected values remain in the external controller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingCase {
    /// Nonblank unique suite-local human label; not a trusted test identity.
    pub label: String,
    /// Ordered function arguments. Only the current invocation is sent to the candidate.
    pub arguments: Vec<CodingValue>,
    /// Exact externally compared result.
    pub expected: CodingValue,
}
/// One reviewed coding task, including private payload that must never be exported as labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedCodingTask {
    /// Source-local label.
    pub task_id: String,
    /// Reviewed source identity and immutable revision.
    pub source: TaskSource,
    /// Operator-declared reviewed rights assertions.
    pub rights: ReviewedTaskRights,
    /// Whole-family grouping for split disjointness.
    pub group: NamespacedTaskId,
    /// Declared role for the entire group.
    pub split: TaskSplit,
    /// Public problem statement; private cases are not appended here.
    pub description: String,
    /// Exact callable and module representation.
    pub function: CodingFunction,
    /// Examples deliberately visible in the prompt, also executed for coverage.
    pub visible_examples: Vec<CodingCase>,
    /// Nonempty for training tasks only; never exposed in prompts or training artifacts.
    pub train_cases: Vec<CodingCase>,
    /// Nonempty for test/validation tasks only; expected results stay outside the container.
    pub protected_cases: Vec<CodingCase>,
    /// Externally applied exact comparison rule.
    pub comparison: CodingComparison,
    /// Required pinned runtime, verified against actual local execution before consumption.
    pub runtime_recipe: String,
    /// Domain, difficulty, and QC review assertions.
    pub observations: TaskObservations,
}
/// Redacted execution contract retained by generation/export. It contains neither private inputs
/// nor expected outputs. Hashes bind the complete suite but do not authenticate observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingSuiteBinding {
    /// Version-one suite canonicalization.
    pub version: u32,
    /// Public callable contract.
    pub function: CodingFunction,
    /// Exact comparison semantics.
    pub comparison: CodingComparison,
    /// Required runtime recipe.
    pub runtime_recipe: String,
    /// Complete suite digest including partitions and every input/expected pair.
    pub suite_id: String,
    /// Private population category; prevents relabeling protected suites as training exports.
    pub private_partition: CodingPrivatePartition,
    /// Number of leading public examples; all remaining case IDs belong to the private partition.
    pub visible_case_count: u32,
    /// Ordered content-derived case identities, including visible examples.
    pub case_ids: Vec<String>,
}

impl CodingFunction {
    /// Validate the supported exact signature before any process is started.
    ///
    /// # Errors
    /// Rejects unsupported identifiers, duplicate parameters, or more than eight parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        let mut seen = HashSet::new();
        if !identifier(&self.entry_point)
            || self.parameters.len() > 8
            || self
                .parameters
                .iter()
                .any(|name| !identifier(name) || !seen.insert(name))
        {
            return Err("coding function requires unique plain ASCII Python identifiers");
        }
        Ok(())
    }
}
fn identifier(name: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
        "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
        "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
        "try", "while", "with", "yield",
    ];
    !name.is_empty()
        && name.len() <= 64
        && !KEYWORDS.contains(&name)
        && name
            .bytes()
            .enumerate()
            .all(|(i, b)| b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit()))
}
impl CodingTaskDocument {
    /// Capture caller-provided bytes as strict JSON, preserving duplicate-field evidence.
    ///
    /// # Errors
    /// Rejects unknown fields, oversized documents, unsupported versions, or invalid declarations.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > 1024 * 1024 {
            return Err("coding task document exceeds 1 MiB".into());
        }
        let raw = crate::strict_coding_json(bytes)?;
        let document: Self =
            serde_json::from_value(raw.clone()).map_err(|error| error.to_string())?;
        if serde_json::to_value(&document).map_err(|error| error.to_string())? != raw {
            return Err("coding task document contains unsupported or omitted fields".into());
        }
        document.validate().map_err(str::to_owned)?;
        Ok(document)
    }
    /// Validate every task before selecting one, preventing hidden split conflicts.
    ///
    /// # Errors
    /// Rejects unsupported versions, invalid tasks, duplicate labels/sources, or group conflicts.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != CODING_TASK_DOCUMENT_VERSION
            || self.tasks.is_empty()
            || self.tasks.len() > 64
        {
            return Err("unsupported or oversized coding task document");
        }
        let mut declarations = TaskDeclarations::default();
        for task in &self.tasks {
            declarations.insert(&TaskProvenance::from_coding_task(task)?)?;
        }
        Ok(())
    }
}
impl ReviewedCodingTask {
    /// Check declarations, case bounds, exact arity, and split-specific private oracle partitions.
    ///
    /// # Errors
    /// Rejects incomplete review declarations, invalid values, or overlapping private partitions.
    pub fn validate(&self) -> Result<(), &'static str> {
        self.function.validate()?;
        if self.description.trim().is_empty()
            || self.description.len() > 32 * 1024
            || self.runtime_recipe != CODING_RUNTIME_RECIPE
        {
            return Err("coding task requires bounded text and the supported runtime recipe");
        }
        match self.split.role {
            TaskSplitRole::Train
                if self.train_cases.is_empty() || !self.protected_cases.is_empty() =>
            {
                return Err("training coding tasks require train cases and no protected cases");
            }
            TaskSplitRole::Test | TaskSplitRole::Validation
                if self.protected_cases.is_empty() || !self.train_cases.is_empty() =>
            {
                return Err("held-out coding tasks require protected cases and no train cases");
            }
            _ => {}
        }
        let mut seen = HashSet::new();
        let cases = self.cases();
        if cases.len() > 64 {
            return Err("coding suite exceeds 64 cases");
        }
        for case in cases {
            if case.label.trim().is_empty()
                || case.label.len() > 128
                || !seen.insert(&case.label)
                || case.arguments.len() != self.function.parameters.len()
            {
                return Err("coding cases require unique labels and exact declared arity");
            }
            for argument in &case.arguments {
                argument.validate()?;
            }
            case.expected.validate()?;
        }
        self.provenance_with_identity(SemanticTaskIdentity {
            version: 2,
            digest: String::new(),
        })
        .validate_declarations()
    }

    /// Every required case in stable visible, training-private, held-out order.
    #[must_use]
    pub fn cases(&self) -> Vec<&CodingCase> {
        self.visible_examples
            .iter()
            .chain(&self.train_cases)
            .chain(&self.protected_cases)
            .collect()
    }
    /// Redacted public contract; no private case values are copied into this structure.
    #[must_use]
    pub fn suite_binding(&self) -> CodingSuiteBinding {
        let suite = serde_json::json!({"function":self.function,"comparison":self.comparison,
            "runtime_recipe":self.runtime_recipe,"visible":self.visible_examples,
            "train":self.train_cases,"protected":self.protected_cases});
        let case_ids = [
            ("visible", &self.visible_examples),
            ("train", &self.train_cases),
            ("protected", &self.protected_cases),
        ]
        .into_iter()
        .flat_map(|(partition, cases)| {
            cases.iter().map(move |case| {
                coding_json_digest("ghostwriter.coding-case.v1", &(partition, case))
            })
        })
        .collect();
        CodingSuiteBinding {
            version: 1,
            function: self.function.clone(),
            comparison: self.comparison,
            runtime_recipe: self.runtime_recipe.clone(),
            suite_id: coding_json_digest("ghostwriter.coding-suite.v1", &suite),
            private_partition: if self.split.role == TaskSplitRole::Train {
                CodingPrivatePartition::Training
            } else {
                CodingPrivatePartition::Protected
            },
            visible_case_count: self.visible_examples.len().try_into().unwrap_or(u32::MAX),
            case_ids,
        }
    }
    /// Deterministic prompt containing only the description, signature, and public examples.
    #[must_use]
    pub fn prompt(&self) -> Message {
        let text = format!(
            "{}\n\nReturn one complete UTF-8 Python module defining {}({}). Use only the Python standard library. Return exact typed JSON-compatible values: None, bool, signed 64-bit int, str, list, or dict with string keys. No Markdown fences. No stdin/stdout protocol or top-level output.\n\nVisible examples (typed JSON):\n{}",
            self.description,
            self.function.entry_point,
            self.function.parameters.join(", "),
            serde_json::to_string(&self.visible_examples).expect("coding examples")
        );
        Message {
            role: Role::User,
            content: Content::Text(text),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
    /// Sole execution contract used by the native verifier and exported task projection.
    #[must_use]
    pub fn contract(&self) -> VerificationContract {
        let suite = self.suite_binding();
        VerificationContract {
            answer_policy: Some(VerificationPolicy::Absent),
            execution_policy: Some(VerificationPolicy::Authoritative),
            required_tests: suite.case_ids.clone(),
            kind: VerificationKind::None,
            oracle: Oracle::CodingSuite { suite },
            numeric: None,
        }
    }
    fn provenance_with_identity(&self, identity: SemanticTaskIdentity) -> TaskProvenance {
        TaskProvenance {
            task_id: self.task_id.clone(),
            identity,
            source: self.source.clone(),
            rights: self.rights.clone(),
            group: self.group.clone(),
            split: self.split.clone(),
            observations: self.observations.clone(),
        }
    }
}
impl CodingSuiteBinding {
    /// Validate a redacted persisted binding without treating its hashes as observed execution.
    ///
    /// # Errors
    /// Rejects unsupported recipes, malformed hashes, empty coverage, or duplicate case identities.
    pub fn validate(&self) -> Result<(), &'static str> {
        self.function.validate()?;
        let mut seen = HashSet::new();
        if self.version != 1
            || self.runtime_recipe != CODING_RUNTIME_RECIPE
            || !coding_hash_valid(&self.suite_id)
            || self.case_ids.is_empty()
            || self.case_ids.len() > 64
            || self.visible_case_count as usize >= self.case_ids.len()
            || self
                .case_ids
                .iter()
                .any(|id| !coding_hash_valid(id) || !seen.insert(id))
        {
            return Err("invalid or unsupported coding suite binding");
        }
        Ok(())
    }
    pub(crate) fn validate_split(&self, role: TaskSplitRole) -> Result<(), &'static str> {
        if (self.private_partition == CodingPrivatePartition::Training)
            != (role == TaskSplitRole::Train)
        {
            return Err("coding private oracle partition disagrees with declared task split");
        }
        Ok(())
    }
}
impl TaskProvenance {
    /// Derive a coding identity from the source, public prompt, and redacted complete-suite binding.
    ///
    /// # Errors
    /// Rejects invalid reviewed declarations, unsupported tasks, or mixed oracle partitions.
    pub fn from_coding_task(task: &ReviewedCodingTask) -> Result<Self, &'static str> {
        task.validate()?;
        let Content::Text(prompt) = task.prompt().content else {
            unreachable!("coding prompt")
        };
        Ok(task.provenance_with_identity(coding_semantic_identity(
            &task.source,
            &prompt,
            &task.contract(),
        )?))
    }
}
pub(crate) fn coding_semantic_identity(
    source: &TaskSource,
    prompt: &str,
    contract: &VerificationContract,
) -> Result<SemanticTaskIdentity, &'static str> {
    contract.validate()?;
    let Oracle::CodingSuite { suite } = &contract.oracle else {
        return Err("expected coding suite");
    };
    Ok(SemanticTaskIdentity {
        version: 2,
        digest: coding_json_digest(
            "ghostwriter.reviewed-coding-task.v2",
            &serde_json::json!({"source":source,"prompt":prompt,"suite":suite}),
        ),
    })
}
