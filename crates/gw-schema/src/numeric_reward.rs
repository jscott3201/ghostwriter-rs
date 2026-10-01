//! Self-contained reviewed training tasks for fresh numeric reward evaluation. No teacher data,
//! model execution, I/O, split qualification, or corpus-quality claim is part of this contract.
use crate::{
    NumericTaskDocument, ReviewedNumericTask, SemanticTaskIdentity, TaskDeclarations,
    TaskPermittedUse, TaskProvenance, TaskSplitRole, VERIFICATION_INTERPRETATION_VERSION,
    VerificationPolicy,
};
use serde::{Deserialize, Serialize};

/// Current portable numeric corpus and factual reward contract version.
pub const NUMERIC_REWARD_VERSION: u32 = 1;

/// Versioned numeric semantics: the existing finite binary64 comparator and content-only
/// extraction yield factual Pass=1, decisive Fail=0, and Unknown without a numeric reward.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRewardContract {
    /// Must equal [`NUMERIC_REWARD_VERSION`].
    pub version: u32,
    /// The factual interpretation revision shared with the verifier.
    pub verification_interpretation_version: u32,
}
impl Default for NumericRewardContract {
    fn default() -> Self {
        Self {
            version: NUMERIC_REWARD_VERSION,
            verification_interpretation_version: VERIFICATION_INTERPRETATION_VERSION,
        }
    }
}

/// One unique declared training task, with its recomputable semantic identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRewardTask {
    /// Complete reviewed source, rights, split, prompt, oracle, and numeric settings.
    /// Reward snapshots encode both finite tolerances as exact binary64 bit objects.
    #[serde(with = "crate::numeric_reward_task::Task")]
    pub task: ReviewedNumericTask,
    /// Binds the actual source, prompt, and numeric semantics; never trusted without validation.
    pub task_identity: SemanticTaskIdentity,
}

/// Ordered, immutable corpus identity. Train selection is explicit and does not establish split
/// screening or authorize a later training experiment. Teacher siblings never enter this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRewardArtifact {
    /// Supported artifact encoding version.
    pub artifact_version: u32,
    /// Domain-separated canonical BLAKE3 of the complete ordered payload.
    pub artifact_id: String,
    /// Exact interpretation and reward mapping.
    pub reward_contract: NumericRewardContract,
    /// Domain-separated identity of [`Self::reward_contract`].
    pub reward_contract_id: String,
    /// Unique declared Train tasks in document and source order.
    pub tasks: Vec<NumericRewardTask>,
}

/// Raw captured-byte binding, distinct from an artifact's canonical semantic identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardSnapshotIdentity {
    /// Exact number of captured bytes.
    pub byte_length: u64,
    /// Lowercase BLAKE3 over those bytes, including whitespace.
    pub blake3: String,
}
impl RewardSnapshotIdentity {
    /// Bind exactly one immutable byte slice without I/O.
    #[must_use]
    pub fn for_bytes(bytes: &[u8]) -> Self {
        Self {
            byte_length: bytes.len() as u64,
            blake3: blake3::hash(bytes).to_hex().to_string(),
        }
    }
}

/// Successful verification of one complete raw artifact snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRewardArtifactReport {
    /// Supported report version.
    pub report_version: u32,
    /// The exact bytes validated.
    pub snapshot: RewardSnapshotIdentity,
    /// Recomputed corpus identity.
    pub artifact_id: String,
    /// Recomputed factual reward contract identity.
    pub reward_contract_id: String,
    /// Number of unique declared Train tasks.
    pub task_count: u64,
}

pub(crate) fn reward_identity<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<String, &'static str> {
    // Value sorts every object recursively; array order and strings remain exact.
    let value = serde_json::to_value(value).map_err(|_| "invalid reward identity payload")?;
    let bytes = serde_json::to_vec(&value).map_err(|_| "invalid reward identity encoding")?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(&[0]);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

impl NumericRewardArtifact {
    /// Validate every supplied document and cross-document declaration, then select Train tasks.
    ///
    /// # Errors
    /// Rejects duplicate/conflicting tasks, unsupported contracts, empty training selection,
    /// missing reviewed training permission, and failed reviewed QC assertions.
    pub fn from_documents(documents: Vec<NumericTaskDocument>) -> Result<Self, &'static str> {
        let mut declarations = TaskDeclarations::default();
        let mut tasks = Vec::new();
        for document in documents {
            document.validate()?;
            for task in document.tasks {
                let provenance = TaskProvenance::from_task(&task)?;
                declarations.insert(&provenance)?;
                if task.split.role == TaskSplitRole::Train {
                    tasks.push(NumericRewardTask {
                        task,
                        task_identity: provenance.identity,
                    });
                }
            }
        }
        let reward_contract = NumericRewardContract::default();
        let mut artifact = Self {
            artifact_version: NUMERIC_REWARD_VERSION,
            artifact_id: String::new(),
            reward_contract_id: reward_identity("gw-numeric-reward-contract-v1", &reward_contract)?,
            reward_contract,
            tasks,
        };
        artifact.artifact_id = artifact.canonical_id()?;
        artifact.validate()?;
        Ok(artifact)
    }

    fn canonical_id(&self) -> Result<String, &'static str> {
        reward_identity(
            "gw-numeric-reward-artifact-v1",
            &serde_json::json!({
                "artifact_version": self.artifact_version,
                "reward_contract": self.reward_contract,
                "reward_contract_id": self.reward_contract_id,
                "tasks": self.tasks,
            }),
        )
    }

    /// Recompute identities and check all declared training-task constraints without I/O.
    ///
    /// # Errors
    /// Rejects unsupported versions, policies, duplicate tasks, changed payloads, or held-out rows.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.artifact_version != NUMERIC_REWARD_VERSION
            || self.reward_contract != NumericRewardContract::default()
        {
            return Err("unsupported numeric reward artifact or interpretation version");
        }
        if self.tasks.is_empty() {
            return Err("numeric reward artifact needs at least one declared Train task");
        }
        let mut declarations = TaskDeclarations::default();
        for entry in &self.tasks {
            let task = &entry.task;
            let provenance = TaskProvenance::from_task(task)?;
            declarations.insert(&provenance)?;
            if entry.task_identity != provenance.identity {
                return Err("numeric reward task semantic identity mismatch");
            }
            if task.split.role != TaskSplitRole::Train
                || task.verification.answer_policy != VerificationPolicy::Authoritative
                || task.verification.execution_policy != VerificationPolicy::Absent
                || !task
                    .rights
                    .permitted_uses
                    .contains(&TaskPermittedUse::Training)
                || !task.observations.qc.answerable
                || !task.observations.qc.difficulty_targeted
                || !task.observations.qc.in_scope
            {
                return Err(
                    "numeric rewards require reviewed Train tasks with training rights, passing QC, authoritative answers, and absent execution policy",
                );
            }
        }
        if self.reward_contract_id
            != reward_identity("gw-numeric-reward-contract-v1", &self.reward_contract)?
            || self.artifact_id != self.canonical_id()?
        {
            return Err("numeric reward artifact or contract identity mismatch");
        }
        Ok(())
    }

    /// Parse strict raw JSON, rejecting duplicate fields before any map conversion, then validate.
    ///
    /// # Errors
    /// Rejects malformed JSON, unknown/duplicate fields, invalid scalar types, and invalid identity.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let artifact: Self = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        artifact.validate().map_err(str::to_owned)?;
        Ok(artifact)
    }

    /// Verify one captured snapshot and return a receipt for exactly those bytes.
    ///
    /// # Errors
    /// Returns the same validation errors as [`Self::from_json`].
    pub fn verify_snapshot(bytes: &[u8]) -> Result<NumericRewardArtifactReport, String> {
        let artifact = Self::from_json(bytes)?;
        Ok(NumericRewardArtifactReport {
            report_version: NUMERIC_REWARD_VERSION,
            snapshot: RewardSnapshotIdentity::for_bytes(bytes),
            artifact_id: artifact.artifact_id,
            reward_contract_id: artifact.reward_contract_id,
            task_count: artifact.tasks.len() as u64,
        })
    }
}
