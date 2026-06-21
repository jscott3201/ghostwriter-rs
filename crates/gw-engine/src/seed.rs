//! The [`SeedSource`] seam + deterministic record-id minting (ARCHITECTURE §5 sharding/idempotency).
//!
//! ## The `SeedSource` seam (v1 boundary — see the crate report)
//!
//! v1 does NOT implement the MAGPIE user-turn LLM elicitation (a tracked follow-up). Instead the
//! engine takes an injected [`SeedSource`]: a trait that yields already-elicited candidate USER turns
//! ([`gw_generate::UserTurnCandidate`]) per shard. A real run injects a JSONL/list-backed source; a
//! test injects a deterministic in-memory source ([`InMemorySeedSource`]). The engine gates each
//! candidate through `gw-generate`'s four-bool QC gate before any teacher spend — the source only
//! PROPOSES, the gate DISPOSES.
//!
//! ## Sharding: partition the seed space by index
//!
//! The seed space is partitioned into `N` shards by a stable rule: seed `i` belongs to shard
//! `i % N`. So a [`SeedSource`] exposes its candidates for ONE shard as an ordered list, and the
//! executor runs shards concurrently. Each candidate carries a `seed: i64` (the reproducibility seed,
//! threaded into best-of-k sibling sampling) and a 0-based `offset` within the shard (the resume
//! cursor index).
//!
//! ## Deterministic record id (idempotency, INVARIANT 4)
//!
//! `record_id = f(run_id, shard, seed, attempt, completion_index)` — a pure function, so the SAME
//! seed re-processed (a retry, a crash-restart, an at-least-once redelivery) mints the SAME id, and
//! every transition is an idempotent UPSERT keyed by it. [`record_id`] is that function; it is the
//! single place the id is minted so the formula can never drift between the producer and the resume
//! path.

use gw_generate::UserTurnCandidate;
use gw_schema::Content;

use crate::Result;

/// One unit of seed work for a shard: an already-elicited candidate USER turn plus its
/// reproducibility seed and shard offset. The [`SeedSource`] yields these in order; the engine gates,
/// generates, and judges each.
#[derive(Debug, Clone, PartialEq)]
pub struct SeedItem {
    /// The reproducibility seed (threaded into best-of-k sibling sampling + the record id). Distinct
    /// seeds give distinct records; the SAME seed re-processed mints the SAME record id (idempotency).
    pub seed: i64,
    /// The 0-based offset of this item within its shard — the resume cursor index. After committing
    /// item `offset`, the shard checkpoint advances the cursor past it, so a relaunch resumes at the
    /// first un-committed offset.
    pub offset: u64,
    /// The candidate USER turn to gate + generate from.
    pub candidate: UserTurnCandidate,
}

/// A source of candidate USER turns, partitioned by shard (the v1 stand-in for MAGPIE elicitation).
///
/// `gw-engine` is generic over this seam so it stays HERMETIC: unit tests inject an
/// [`InMemorySeedSource`] (deterministic, no network); a real run injects a JSONL/list-backed source.
/// The contract: [`shard_count`](SeedSource::shard_count) is stable across a run, and
/// [`items_for_shard`](SeedSource::items_for_shard) returns the SAME ordered items for the SAME shard
/// on every call (so a crash-restart re-derives the identical work plan and resumes by offset).
pub trait SeedSource: Send + Sync {
    /// The number of shards the seed space is partitioned into (stable for the life of the run).
    fn shard_count(&self) -> usize;

    /// A stable hash of the ordered prompt list whose indices define shard assignment.
    ///
    /// Implementations must hash the same post-filter prompt sequence that
    /// [`items_for_shard`](Self::items_for_shard) partitions. The shard count is intentionally not
    /// folded into this hash; the run ledger persists it in its own column.
    ///
    /// # Errors
    /// Returns [`EngineError`](crate::EngineError) if the source cannot derive or serialize the
    /// manifest hash.
    fn prompts_hash(&self) -> Result<String>;

    /// The ordered [`SeedItem`]s belonging to `shard` (0-based). MUST be deterministic — the same
    /// shard yields the same items in the same order on every call, so resume-by-offset is sound.
    fn items_for_shard(&self, shard: i64) -> Vec<SeedItem>;
}

/// A deterministic, in-memory [`SeedSource`] for tests and small literal runs.
///
/// Candidates are supplied as a flat ordered list and partitioned across `shard_count` shards by
/// `index % shard_count` (so item `i` lands in shard `i % N` at the offset it appears within that
/// shard). The seed for item `i` is `base_seed + i`, deterministic and distinct. Re-querying any
/// shard returns the identical list (the resume-soundness contract).
#[derive(Debug, Clone)]
pub struct InMemorySeedSource {
    items: Vec<UserTurnCandidate>,
    shard_count: usize,
    base_seed: i64,
}

impl InMemorySeedSource {
    /// Build a source from a flat candidate list partitioned across `shard_count` shards. A
    /// `shard_count` of 0 is clamped to 1 (a single shard); seeds start at `0`.
    #[must_use]
    pub fn new(items: Vec<UserTurnCandidate>, shard_count: usize) -> Self {
        Self {
            items,
            shard_count: shard_count.max(1),
            base_seed: 0,
        }
    }

    /// Build a source with an explicit `base_seed` (item `i` gets seed `base_seed + i`).
    #[must_use]
    pub fn with_base_seed(
        items: Vec<UserTurnCandidate>,
        shard_count: usize,
        base_seed: i64,
    ) -> Self {
        Self {
            items,
            shard_count: shard_count.max(1),
            base_seed,
        }
    }
}

impl SeedSource for InMemorySeedSource {
    fn shard_count(&self) -> usize {
        self.shard_count
    }

    fn prompts_hash(&self) -> Result<String> {
        let prompts = self
            .items
            .iter()
            .map(candidate_prompt_text)
            .collect::<Result<Vec<_>>>()?;
        Ok(gw_storage::prompts_hash(&prompts)?)
    }

    fn items_for_shard(&self, shard: i64) -> Vec<SeedItem> {
        let n = self.shard_count as i64;
        // Partition by `global_index % shard_count`; the per-shard offset is the position within the
        // shard's own ordered subsequence. Deterministic on every call.
        self.items
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i as i64).rem_euclid(n) == shard.rem_euclid(n))
            .enumerate()
            .map(|(offset, (global_index, candidate))| SeedItem {
                seed: self.base_seed.wrapping_add(global_index as i64),
                offset: offset as u64,
                candidate: candidate.clone(),
            })
            .collect()
    }
}

fn candidate_prompt_text(candidate: &UserTurnCandidate) -> Result<String> {
    match &candidate.message.content {
        Content::Text(text) => Ok(text.trim().to_string()),
        Content::Parts(_) => Ok(serde_json::to_string(&candidate.message.content)?),
    }
}

/// Mint the deterministic record id `f(run_id, shard, seed, attempt, completion_index)` (INVARIANT 4).
///
/// The id is a stable, human-greppable composite — NOT a hash — so it is debuggable and the formula
/// is obvious. The SAME inputs always mint the SAME id, so a retry / crash-restart / at-least-once
/// redelivery upserts the same row rather than duplicating it. `attempt` distinguishes the original
/// generation from a bounded revise retry; `completion_index` distinguishes best-of-k siblings.
#[must_use]
pub fn record_id(
    run_id: &str,
    shard: i64,
    seed: i64,
    attempt: u32,
    completion_index: u32,
) -> String {
    format!("{run_id}-s{shard}-seed{seed}-a{attempt}-c{completion_index}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_generate::{UserSeed, user_message};
    use gw_schema::{ContentPart, Oracle, VerificationContract, VerificationKind};

    fn candidate(text: &str) -> UserTurnCandidate {
        UserTurnCandidate {
            message: user_message(text),
            seed: UserSeed::default(),
            contract: VerificationContract {
                kind: VerificationKind::NumericMatch,
                oracle: Oracle::None,
                answer_marker: None,
            },
            answerable: true,
            difficulty_targeted: true,
            in_scope: true,
        }
    }

    fn candidate_with_content(content: Content) -> UserTurnCandidate {
        let mut candidate = candidate("");
        candidate.message.content = content;
        candidate
    }

    #[test]
    fn record_id_is_deterministic_and_distinct_per_axis() {
        let a = record_id("run-1", 0, 7, 0, 0);
        // Same inputs → same id (idempotency).
        assert_eq!(a, record_id("run-1", 0, 7, 0, 0));
        // Each axis changes the id.
        assert_ne!(a, record_id("run-1", 1, 7, 0, 0));
        assert_ne!(a, record_id("run-1", 0, 8, 0, 0));
        assert_ne!(a, record_id("run-1", 0, 7, 1, 0));
        assert_ne!(a, record_id("run-1", 0, 7, 0, 1));
    }

    #[test]
    fn record_id_shape_is_greppable() {
        assert_eq!(record_id("run-x", 2, 100, 1, 3), "run-x-s2-seed100-a1-c3");
    }

    #[test]
    fn in_memory_partitions_round_robin_by_index() {
        let items: Vec<_> = (0..6).map(|i| candidate(&format!("q{i}"))).collect();
        let src = InMemorySeedSource::new(items, 2);
        assert_eq!(src.shard_count(), 2);
        // Shard 0 gets indices 0,2,4; shard 1 gets 1,3,5.
        let s0 = src.items_for_shard(0);
        let s1 = src.items_for_shard(1);
        assert_eq!(s0.len(), 3);
        assert_eq!(s1.len(), 3);
        // Offsets within each shard are 0,1,2.
        assert_eq!(
            s0.iter().map(|i| i.offset).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        // Seeds are the GLOBAL index (base 0): shard 0 → seeds 0,2,4.
        assert_eq!(s0.iter().map(|i| i.seed).collect::<Vec<_>>(), vec![0, 2, 4]);
        assert_eq!(s1.iter().map(|i| i.seed).collect::<Vec<_>>(), vec![1, 3, 5]);
    }

    #[test]
    fn in_memory_is_deterministic_across_calls() {
        let items: Vec<_> = (0..5).map(|i| candidate(&format!("q{i}"))).collect();
        let src = InMemorySeedSource::new(items, 3);
        // Re-querying a shard returns the identical plan (resume soundness).
        for shard in 0..3 {
            assert_eq!(src.items_for_shard(shard), src.items_for_shard(shard));
        }
    }

    #[test]
    fn zero_shard_count_clamps_to_one() {
        let src = InMemorySeedSource::new(vec![candidate("q")], 0);
        assert_eq!(src.shard_count(), 1);
        assert_eq!(src.items_for_shard(0).len(), 1);
    }

    #[test]
    fn prompts_hash_tracks_ordered_prompt_text() {
        let a = InMemorySeedSource::new(vec![candidate("q1"), candidate("q2")], 1);
        let b = InMemorySeedSource::new(vec![candidate(" q1 "), candidate("q2")], 1);
        let c = InMemorySeedSource::new(vec![candidate("q2"), candidate("q1")], 1);
        assert_eq!(a.prompts_hash().unwrap(), b.prompts_hash().unwrap());
        assert_ne!(a.prompts_hash().unwrap(), c.prompts_hash().unwrap());
    }

    #[test]
    fn prompts_hash_distinguishes_part_boundaries_and_non_text_parts() {
        let joined = InMemorySeedSource::new(
            vec![candidate_with_content(Content::Parts(vec![
                ContentPart::Text { text: "ab".into() },
            ]))],
            1,
        );
        let split = InMemorySeedSource::new(
            vec![candidate_with_content(Content::Parts(vec![
                ContentPart::Text { text: "a".into() },
                ContentPart::Text { text: "b".into() },
            ]))],
            1,
        );
        let image_only = InMemorySeedSource::new(
            vec![candidate_with_content(Content::Parts(vec![
                ContentPart::ImageUrl {
                    image_url: "https://example.test/image.png".into(),
                },
            ]))],
            1,
        );
        let empty_parts =
            InMemorySeedSource::new(vec![candidate_with_content(Content::Parts(vec![]))], 1);

        let joined_hash = joined.prompts_hash().unwrap();
        let split_hash = split.prompts_hash().unwrap();
        let image_hash = image_only.prompts_hash().unwrap();
        let empty_hash = empty_parts.prompts_hash().unwrap();

        assert_ne!(joined_hash, split_hash);
        assert_ne!(image_hash, empty_hash);
        assert_ne!(split_hash, image_hash);
    }

    #[test]
    fn base_seed_offsets_the_seeds() {
        let items: Vec<_> = (0..3).map(|i| candidate(&format!("q{i}"))).collect();
        let src = InMemorySeedSource::with_base_seed(items, 1, 1000);
        let s0 = src.items_for_shard(0);
        assert_eq!(
            s0.iter().map(|i| i.seed).collect::<Vec<_>>(),
            vec![1000, 1001, 1002]
        );
    }
}
