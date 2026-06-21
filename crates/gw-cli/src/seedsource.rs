//! A file-backed [`SeedSource`] for the v1 live run.
//!
//! The engine takes seeds through the [`SeedSource`] seam — its own docs name "a JSONL/list-backed
//! source" as the real-run injectee, but the engine ships only the in-memory test source. This module
//! provides the CLI's minimal real source: a newline-delimited PROMPTS file (one user turn per line),
//! mapped to [`UserTurnCandidate`]s via the public [`user_message`] helper.
//!
//! ## v1 seam scope (documented deferral — see the crate report)
//!
//! `gw-generate`'s [`UserTurnCandidate`] does not derive `serde`, and there is no published JSONL
//! schema for the four QC bools / the per-turn [`VerificationContract`]. So v1 reads PLAIN TEXT, one
//! prompt per line, and synthesizes a judge-only candidate: a [`VerificationKind::None`] /
//! [`Oracle::None`] contract (admission is judge-only, no deterministic oracle) with the three
//! engine-side QC bools set true (`answerable` / `difficulty_targeted` / `in_scope`) — the candidate
//! is a hand-supplied prompt the operator vouches for. A RICHER seed format (per-turn contracts,
//! oracles, difficulty bands) and the engine-driven MAGPIE elicitation are tracked follow-ups; they
//! would arrive as a new `SeedSource` impl, not a change here. This source is enough to drive a real
//! end-to-end run from a curated prompt list.

use std::path::Path;

use gw_engine::{SeedItem, SeedSource};
use gw_generate::{UserSeed, UserTurnCandidate, user_message};
use gw_schema::{Oracle, VerificationContract, VerificationKind};

/// A [`SeedSource`] over an ordered list of plain-text prompts, partitioned across `shard_count`
/// shards by index (matching [`gw_engine::InMemorySeedSource`]'s round-robin rule).
#[derive(Debug, Clone)]
pub struct FileSeedSource {
    candidates: Vec<UserTurnCandidate>,
    prompts: Vec<String>,
    shard_count: usize,
    base_seed: i64,
}

impl FileSeedSource {
    /// Read a newline-delimited prompts file into a source over `shard_count` shards.
    ///
    /// Blank lines and lines whose first non-whitespace char is `#` (comments) are skipped; every
    /// other line becomes one judge-only candidate. `shard_count` is clamped to `>= 1`.
    ///
    /// # Errors
    /// Returns an `std::io::Error` (via `anyhow`) if the file cannot be read, or an error if the file
    /// yields zero usable prompts (an empty run is almost certainly a mistake — fail loud).
    pub fn from_prompts_file(path: &Path, shard_count: usize) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading prompts file {}: {e}", path.display()))?;
        Self::from_prompts_str(&text, shard_count)
            .map_err(|e| anyhow::anyhow!("{e} (in {})", path.display()))
    }

    /// Build a source from an in-memory newline-delimited prompts blob (the testable core of
    /// [`from_prompts_file`](Self::from_prompts_file)).
    ///
    /// # Errors
    /// Returns an error if the blob yields zero usable prompts.
    pub fn from_prompts_str(text: &str, shard_count: usize) -> anyhow::Result<Self> {
        let prompts: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_owned)
            .collect();
        if prompts.is_empty() {
            anyhow::bail!("no usable prompts (all lines blank or comments)");
        }
        let candidates = prompts.iter().map(|p| judge_only_candidate(p)).collect();
        Ok(Self {
            candidates,
            prompts,
            shard_count: shard_count.max(1),
            base_seed: 0,
        })
    }

    /// The number of candidate prompts loaded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Whether the source has no candidates (never true after a successful construction, which
    /// rejects an empty set — present for the standard `len`/`is_empty` pair).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

/// Build a judge-only candidate from one prompt line: a clean user message, a default seed, a
/// [`VerificationKind::None`] / [`Oracle::None`] contract, and the three engine-side QC bools true.
fn judge_only_candidate(prompt: &str) -> UserTurnCandidate {
    UserTurnCandidate {
        message: user_message(prompt),
        seed: UserSeed::default(),
        contract: VerificationContract {
            kind: VerificationKind::None,
            oracle: Oracle::None,
            answer_marker: None,
        },
        answerable: true,
        difficulty_targeted: true,
        in_scope: true,
    }
}

impl SeedSource for FileSeedSource {
    fn shard_count(&self) -> usize {
        self.shard_count
    }

    fn prompts_hash(&self) -> gw_engine::Result<String> {
        Ok(gw_storage::prompts_hash(&self.prompts)?)
    }

    fn items_for_shard(&self, shard: i64) -> Vec<SeedItem> {
        let n = self.shard_count as i64;
        self.candidates
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prompts_skipping_blanks_and_comments() {
        let text = "  what is 2+2?\n\n# a comment\nexplain ownership in rust\n";
        let src = FileSeedSource::from_prompts_str(text, 2).expect("parses");
        assert_eq!(src.len(), 2);
        assert_eq!(src.shard_count(), 2);
        // Round-robin partition: index 0 -> shard 0, index 1 -> shard 1.
        assert_eq!(src.items_for_shard(0).len(), 1);
        assert_eq!(src.items_for_shard(1).len(), 1);
    }

    #[test]
    fn empty_or_comment_only_blob_is_an_error() {
        let err = FileSeedSource::from_prompts_str("\n# only comments\n   \n", 1)
            .expect_err("must reject an empty seed set");
        assert!(format!("{err}").contains("no usable prompts"));
    }

    #[test]
    fn candidate_is_judge_only_and_in_scope() {
        let src = FileSeedSource::from_prompts_str("a question", 1).expect("parses");
        let items = src.items_for_shard(0);
        let c = &items[0].candidate;
        assert_eq!(c.contract.kind, VerificationKind::None);
        assert!(matches!(c.contract.oracle, Oracle::None));
        assert!(c.answerable && c.difficulty_targeted && c.in_scope);
    }

    #[test]
    fn shard_count_clamps_to_one() {
        let src = FileSeedSource::from_prompts_str("q1\nq2", 0).expect("parses");
        assert_eq!(src.shard_count(), 1);
        assert_eq!(src.items_for_shard(0).len(), 2);
    }

    #[test]
    fn prompts_hash_is_over_post_filter_trimmed_prompts() {
        let a = FileSeedSource::from_prompts_str(" q1 \n# comment\nq2\n", 2).expect("parses");
        let b = FileSeedSource::from_prompts_str("q1\n\n  # other comment\n q2  \n", 2)
            .expect("parses");
        let reordered = FileSeedSource::from_prompts_str("q2\nq1\n", 2).expect("parses");
        let edited = FileSeedSource::from_prompts_str("q1\nq2 edited\n", 2).expect("parses");

        assert_eq!(a.prompts_hash().unwrap(), b.prompts_hash().unwrap());
        assert_ne!(a.prompts_hash().unwrap(), reordered.prompts_hash().unwrap());
        assert_ne!(a.prompts_hash().unwrap(), edited.prompts_hash().unwrap());
    }
}
