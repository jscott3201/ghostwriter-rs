//! Pure prepared materialization of reviewed numeric tasks. File I/O belongs to the CLI.
use crate::{CapturedSeedPlan, EngineError, InMemorySeedSource, Result, SeedItem, SeedSource};
use gw_generate::{UserSeed, UserTurnCandidate, user_message};
use gw_schema::{NumericTaskDocument, TaskProvenance};

/// Validated numeric task source, with the same seed/shard/offset mapping as plain prompts.
#[derive(Debug, Clone)]
pub struct NumericTaskSource {
    source: InMemorySeedSource,
    len: usize,
}
impl NumericTaskSource {
    /// Parse and validate a complete strict task document before constructing any runtime resource.
    ///
    /// # Errors
    /// Rejects malformed/unsupported JSON, declarations, duplicates, or numeric contracts.
    pub fn from_json(text: &str, shard_count: usize) -> Result<Self> {
        let document = NumericTaskDocument::from_json(text).map_err(EngineError::Invariant)?;
        Self::from_document(document, shard_count)
    }

    /// Materialize a typed document using the same validation as JSON/CLI input.
    /// No caller-supplied identity is accepted; every task digest is derived from actual semantics.
    ///
    /// # Errors
    /// Rejects invalid declarations, conflicting identities, or inconsistent candidate contracts.
    pub fn from_document(document: NumericTaskDocument, shard_count: usize) -> Result<Self> {
        document
            .validate()
            .map_err(|reason| EngineError::Invariant(reason.into()))?;
        let len = document.tasks.len();
        let candidates = document
            .tasks
            .into_iter()
            .map(|task| {
                let provenance = TaskProvenance::from_task(&task)
                    .map_err(|reason| EngineError::Invariant(reason.into()))?;
                let qc = &task.observations.qc;
                let candidate = UserTurnCandidate {
                    message: user_message(task.prompt.text()),
                    seed: UserSeed {
                        taxonomy_node: Some(task.observations.domain.clone()),
                        difficulty: Some(task.observations.difficulty.label.clone()),
                        ..Default::default()
                    },
                    contract: task.verification.contract(),
                    task_provenance: Some(provenance),
                    answerable: qc.answerable,
                    difficulty_targeted: qc.difficulty_targeted,
                    in_scope: qc.in_scope,
                };
                candidate.validate_contract()?;
                candidate.validate_framing()?;
                Ok(candidate)
            })
            .collect::<Result<Vec<_>>>()?;
        let source = InMemorySeedSource::new(candidates, shard_count);
        // Validate the exact same complete materialized plan used by arbitrary library sources.
        CapturedSeedPlan::capture(&source)?;
        Ok(Self { source, len })
    }

    /// Number of validated tasks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Successful construction always has at least one task.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
impl SeedSource for NumericTaskSource {
    fn shard_count(&self) -> usize {
        self.source.shard_count()
    }
    fn items_for_shard(&self, shard: i64) -> Vec<SeedItem> {
        self.source.items_for_shard(shard)
    }
}
