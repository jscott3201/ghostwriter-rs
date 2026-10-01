//! Portable fresh-completion bindings. Tokenizer declarations bind caller-supplied decoding
//! evidence; the qualified external adapter additionally verifies actual token-to-text decoding.
use crate::numeric_reward::reward_identity;
use crate::{
    NUMERIC_REWARD_VERSION, NumericRewardArtifact, RewardSnapshotIdentity, VerificationOutcome,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// One declared control token, including tokens hidden by skip-special decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardControlToken {
    /// Token ID in the declared immutable tokenizer vocabulary.
    pub id: u32,
    /// Literal decoded control syntax, also forbidden when produced through ordinary token IDs.
    pub literal: String,
}

/// The first supported plain assistant decoding contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardDecodePolicy {
    /// Skip special tokens, disable cleanup, reject structured/thinking/tool/media output.
    PlainAssistantSkipSpecialNoCleanupV1,
}

/// Immutable tokenizer/decode declaration supplied by the adapter, separate from task semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardCompletionPolicy {
    /// Supported policy version.
    pub version: u32,
    /// Identity of the captured tokenizer bytes and runtime wrapper policy.
    pub tokenizer_id: String,
    /// Exact decoding and assistant-content interpretation.
    pub decode: RewardDecodePolicy,
    /// Valid IDs are strictly less than this bound.
    pub vocab_size: u32,
    /// Only this control ID is allowed, once at the end of the raw unpadded completion.
    pub eos_token_id: u32,
    /// All reserved/control tokens in strictly ascending ID order.
    pub control_tokens: Vec<RewardControlToken>,
}

/// Raw unpadded generated IDs and exactly decoded UTF-8 assistant content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardCompletion {
    /// The full unpadded sequence, including a terminal EOS when present.
    pub token_ids: Vec<u32>,
    /// Exact plain assistant content; prompt and reasoning are not evaluator inputs.
    pub text: String,
}

/// Fresh reward evaluation identity. It does not establish model-generation/checkpoint lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardAttempt {
    /// Fresh callback instance namespace, represented as 32 lowercase hexadecimal characters.
    pub callback_run_id: String,
    /// Monotonic callback sequence; retries and failed batches consume a sequence too.
    pub batch_sequence: u64,
    /// Zero-based position in this exact submitted batch, including repeated task rows.
    pub position: u64,
}

/// Complete correlation key. Responses must preserve every field, order, and cardinality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRewardBinding {
    /// Frozen corpus identity.
    pub artifact_id: String,
    /// Selected task label inside that corpus.
    pub task_id: String,
    /// Recomputed semantic task identity.
    pub semantic_task_digest: String,
    /// Exact factual reward mapping and interpretation identity.
    pub reward_contract_id: String,
    /// Fresh callback run/batch/position identity.
    pub attempt: RewardAttempt,
    /// Domain-separated digest of the completion IDs, exact text, and full decoding declaration.
    pub completion_digest: String,
}

/// One fresh completion and the correlation key the evaluator must verify.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRewardInput {
    /// Caller-supplied fields, recomputed before any result is emitted.
    pub binding: NumericRewardBinding,
    /// Actual fresh completion evidence.
    pub completion: RewardCompletion,
}

/// One atomic evaluation batch. Invalid bindings reject the whole request before numeric work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRewardBatch {
    /// Supported request version.
    pub version: u32,
    /// Full self-contained verified corpus; no mutable path or teacher evidence is consulted.
    pub artifact: NumericRewardArtifact,
    /// Actual tokenizer/decode declaration validated by the adapter.
    pub completion_policy: RewardCompletionPolicy,
    /// Effective trainer setting, preserved without inferring or changing completion truncation.
    pub mask_truncated_completions: bool,
    /// Ordered completions. Repeated tasks are allowed, with distinct attempts/positions.
    pub items: Vec<NumericRewardInput>,
}

/// Descriptive termination evidence available at the reward callback boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardTermination {
    /// The last unpadded raw ID is the declared terminal EOS.
    ObservedEos,
    /// No authoritative finish reason is available. This is not evidence of truncation.
    Unknown,
}

/// One fresh factual result, separate from the trainer's batch-abort policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRewardResult {
    /// Exact verified input correlation key.
    pub binding: NumericRewardBinding,
    /// Pass, decisive Fail, or Unknown; never an admission-compatible boolean.
    pub outcome: VerificationOutcome,
    /// Pass=1, Fail=0, Unknown=null. Unknown cannot enter the first trainer adapter's tensor.
    pub reward: Option<f64>,
    /// Independent descriptive stop evidence, never inferred from sequence length.
    pub termination: RewardTermination,
}

/// Complete response to one raw byte request. A caller must validate every result before use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericRewardBatchReport {
    /// Supported report version.
    pub report_version: u32,
    /// Exact input bytes, including the fresh run/batch/position identities.
    pub request: RewardSnapshotIdentity,
    /// Exact tokenizer/decode declaration identity.
    pub completion_policy_id: String,
    /// Unmodified trainer setting from the request.
    pub mask_truncated_completions: bool,
    /// One ordered result per submitted completion, including factual Unknown values.
    pub results: Vec<NumericRewardResult>,
}

fn lowercase_hex(text: &str, length: usize) -> bool {
    text.len() == length
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl RewardCompletionPolicy {
    /// Validate the complete declared vocabulary/control/decode policy without loading a tokenizer.
    ///
    /// # Errors
    /// Rejects unsupported versions, malformed identities, duplicate controls, and invalid IDs.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != NUMERIC_REWARD_VERSION
            || !lowercase_hex(&self.tokenizer_id, 64)
            || self.vocab_size == 0
            || self.eos_token_id >= self.vocab_size
        {
            return Err("invalid numeric reward tokenizer/decode declaration");
        }
        let mut previous = None;
        let mut literals = HashSet::new();
        for token in &self.control_tokens {
            if token.id >= self.vocab_size
                || previous.is_some_and(|id| token.id <= id)
                || token.literal.is_empty()
                || !literals.insert(&token.literal)
            {
                return Err("invalid or duplicate reward control-token declaration");
            }
            previous = Some(token.id);
        }
        if !self
            .control_tokens
            .iter()
            .any(|token| token.id == self.eos_token_id)
        {
            return Err("reward control-token declaration must include terminal EOS");
        }
        Ok(())
    }

    /// Identity of the complete decoding declaration.
    ///
    /// # Errors
    /// Rejects an invalid declaration before hashing.
    pub fn identity(&self) -> Result<String, &'static str> {
        self.validate()?;
        reward_identity("gw-numeric-reward-policy-v1", self)
    }

    /// Check raw IDs before any skip-special decoding, and reject raw control syntax in text.
    /// Exact token-to-text decoding is additionally checked by the qualified external adapter.
    ///
    /// # Errors
    /// Rejects invalid IDs, nonterminal/repeated EOS, reserved controls, and control literals.
    pub fn validate_completion(&self, completion: &RewardCompletion) -> Result<(), &'static str> {
        self.validate()?;
        for (position, id) in completion.token_ids.iter().enumerate() {
            if *id >= self.vocab_size
                || (self.control_tokens.iter().any(|token| token.id == *id)
                    && !(*id == self.eos_token_id && position + 1 == completion.token_ids.len()))
            {
                return Err(
                    "numeric reward completion contains an invalid or unsupported control ID",
                );
            }
        }
        if self
            .control_tokens
            .iter()
            .any(|token| completion.text.contains(&token.literal))
        {
            return Err("numeric reward completion contains unsupported control syntax");
        }
        Ok(())
    }

    /// Bind the exact UTF-8 content, raw IDs, and complete tokenizer/decode declaration.
    ///
    /// # Errors
    /// Rejects unsupported completion or policy shapes before hashing.
    pub fn completion_digest(&self, completion: &RewardCompletion) -> Result<String, &'static str> {
        self.validate_completion(completion)?;
        reward_identity("gw-numeric-reward-completion-v1", &(self, completion))
    }
}

impl NumericRewardBatch {
    /// Validate every artifact, attempt, position, token declaration, and completion binding.
    ///
    /// # Errors
    /// Rejects unsupported/empty batches, stale bindings, shuffled positions, or mixed attempts.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != NUMERIC_REWARD_VERSION || self.items.is_empty() {
            return Err("unsupported or empty numeric reward batch");
        }
        self.artifact.validate()?;
        self.completion_policy.validate()?;
        let tasks: HashMap<_, _> = self
            .artifact
            .tasks
            .iter()
            .map(|task| (&task.task.task_id, task))
            .collect();
        let first = &self.items[0].binding.attempt;
        if !lowercase_hex(&first.callback_run_id, 32) {
            return Err("invalid reward callback run namespace");
        }
        for (position, item) in self.items.iter().enumerate() {
            let binding = &item.binding;
            let task = tasks
                .get(&binding.task_id)
                .ok_or("reward task is absent from corpus")?;
            if binding.artifact_id != self.artifact.artifact_id
                || binding.reward_contract_id != self.artifact.reward_contract_id
                || binding.semantic_task_digest != task.task_identity.digest
                || binding.attempt.callback_run_id != first.callback_run_id
                || binding.attempt.batch_sequence != first.batch_sequence
                || binding.attempt.position != position as u64
                || binding.completion_digest
                    != self.completion_policy.completion_digest(&item.completion)?
            {
                return Err(
                    "numeric reward artifact, task, attempt, position, or completion binding mismatch",
                );
            }
        }
        Ok(())
    }

    /// Parse strict original JSON bytes without losing duplicate-field evidence, then validate.
    ///
    /// # Errors
    /// Rejects invalid wire shapes, versions, or correlation bindings.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let request: Self = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        request.validate().map_err(str::to_owned)?;
        Ok(request)
    }
}
