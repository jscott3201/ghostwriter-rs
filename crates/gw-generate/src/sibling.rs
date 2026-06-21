//! Best-of-k fan-out: the per-sibling sampling plan for a shared prompt (ARCHITECTURE D-BESTOFK).
//!
//! A best-of-k group draws `k` independent assistant completions from the SAME synthesized USER
//! turn, so the engine can admit the best (by `judging.aggregate`) and retain the rejected siblings
//! for DPO + audit. This module computes the FAN-OUT SHAPE — one [`SiblingPlan`] per index — that
//! the orchestrator turns into `k` teacher calls.
//!
//! ## Invariants enforced here
//!
//! - Siblings carry `completion_index` in `0..k` and `n_completions == k` ([`plan_group`]).
//! - **Per-sibling sampling MUST vary** — we never emit `k` identical greedy samples. Each
//!   sibling's [`SamplingPreset`] gets a DISTINCT seed derived deterministically from the group's
//!   `base_seed + completion_index` (so the fan-out is reproducible AND diverse). [`SiblingPlan::sampling`]
//!   guarantees no two siblings in a group share a seed. This is checked by
//!   [`seeds_are_distinct`] and the unit tests.
//!
//! ## The `sibling_group_id` / `prompt_hash` seam (DOCUMENTED CHOICE)
//!
//! Per `gw-schema::Hashes`, `prompt_hash` is "blake3 of canonicalized prompt messages (also the
//! DPO pairing key / sibling group id)", and `gw-storage` is AUTHORITATIVE for content hashes —
//! it recomputes them on `put`. `gw-generate` has NO storage dependency and MUST NOT compute
//! content hashes, so it CANNOT mint the canonical `prompt_hash` here. We therefore set
//! `completion_index` / `n_completions` on every sibling and LEAVE `sibling_group_id == None` for
//! `gw-engine`/`gw-storage` to fill from the canonical `prompt_hash` once the prompt is hashed.
//! The k siblings are linked structurally instead: they are produced together from one
//! [`plan_group`] call against one prompt, so the engine knows the grouping without us guessing a
//! fingerprint. (Flagged in the crate report.)

use crate::request::SamplingPreset;

/// The default best-of-k for a single-trace run: `k = 1` (no fan-out). RFT runs use `4..=16`.
pub const DEFAULT_K: u32 = 1;

/// The sampling plan for ONE sibling of a best-of-k group: its 0-based index, the group size, and
/// the seed-varied [`SamplingPreset`] to sample it with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiblingPlan {
    /// 0-based index within the sibling group (`0..n_completions`).
    pub completion_index: u32,
    /// The group size `k` (`>= 1`).
    pub n_completions: u32,
    /// The sampling preset for this sibling — seed-distinct from every other sibling in the group.
    pub sampling: SamplingPreset,
}

/// Plan a best-of-k group of size `k` over one prompt, varying the seed per sibling from `base`.
///
/// Returns `k` [`SiblingPlan`]s with `completion_index` `0..k` and `n_completions == k`. Each
/// sibling's seed is `base_seed + completion_index` (using `base.seed`, or `0` if `base` has no
/// seed) so the samples are reproducible yet DISTINCT — never `k` identical greedy draws. `k == 0`
/// is clamped to `1` (a degenerate group of one).
#[must_use]
pub fn plan_group(base: SamplingPreset, k: u32) -> Vec<SiblingPlan> {
    let n = k.max(1);
    let base_seed = base.seed.unwrap_or(0);
    (0..n)
        .map(|i| SiblingPlan {
            completion_index: i,
            n_completions: n,
            // Distinct, deterministic per-sibling seed: base + index.
            sampling: SamplingPreset {
                seed: Some(base_seed.wrapping_add(i64::from(i))),
                ..base
            },
        })
        .collect()
}

/// `true` iff every sibling in `plans` carries a DISTINCT seed (the anti-"k identical samples"
/// guard). Used in tests and available to a defensive caller.
#[must_use]
pub fn seeds_are_distinct(plans: &[SiblingPlan]) -> bool {
    let mut seeds: Vec<Option<i64>> = plans.iter().map(|p| p.sampling.seed).collect();
    seeds.sort_unstable();
    let before = seeds.len();
    seeds.dedup();
    seeds.len() == before
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k1_is_a_single_plan_indexed_zero() {
        let plans = plan_group(SamplingPreset::official(), 1);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].completion_index, 0);
        assert_eq!(plans[0].n_completions, 1);
    }

    #[test]
    fn k_produces_k_indexed_siblings() {
        let plans = plan_group(SamplingPreset::official(), 8);
        assert_eq!(plans.len(), 8);
        for (i, p) in plans.iter().enumerate() {
            assert_eq!(p.completion_index, i as u32);
            assert_eq!(p.n_completions, 8);
        }
    }

    #[test]
    fn per_sibling_seeds_are_distinct_never_identical() {
        // The crux: best-of-k must NOT emit k identical greedy samples.
        let plans = plan_group(SamplingPreset::official(), 16);
        assert!(
            seeds_are_distinct(&plans),
            "siblings must have distinct seeds"
        );
        // And they are deterministic: base 0 → 0,1,2,...
        let seeds: Vec<i64> = plans.iter().map(|p| p.sampling.seed.unwrap()).collect();
        assert_eq!(seeds, (0..16).collect::<Vec<i64>>());
    }

    #[test]
    fn base_seed_offsets_the_group() {
        let base = SamplingPreset::official().with_seed(1000);
        let plans = plan_group(base, 4);
        let seeds: Vec<i64> = plans.iter().map(|p| p.sampling.seed.unwrap()).collect();
        assert_eq!(seeds, vec![1000, 1001, 1002, 1003]);
        assert!(seeds_are_distinct(&plans));
    }

    #[test]
    fn k0_is_clamped_to_one() {
        let plans = plan_group(SamplingPreset::official(), 0);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].n_completions, 1);
    }

    #[test]
    fn temperature_is_shared_only_the_seed_varies() {
        // Sampling preset (temp/top_p) is shared; only the seed differs across siblings.
        let plans = plan_group(SamplingPreset::precise(), 3);
        for p in &plans {
            assert_eq!(p.sampling.temperature, 0.6);
            assert_eq!(p.sampling.top_p, Some(0.95));
        }
        assert!(seeds_are_distinct(&plans));
    }
}
