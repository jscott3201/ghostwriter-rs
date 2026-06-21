//! A1 — the headless selector-vs-random **separation diagnostic** + the `decidable_fraction`
//! judge-budget skip lever (`_research/2026-06-20-rlhf-book-applicability.md` §A1).
//!
//! ## What this proves (and what it deliberately does NOT)
//!
//! The harness keeps rich per-judge calibration machinery but never proves, model-free, that
//! picking the argmax-aggregate sibling beats a coin flip at equal budget. This module is that
//! proof. It is **read-only**, GPU-free, makes **no model calls**, and reads only fields the
//! harness already persists.
//!
//! Two distinct things are reported:
//!
//! 1. **`decidable_fraction`** — the cost lever. Group best-of-k siblings by their shared
//!    [`prompt_hash`] (`== sibling_group_id`). Classify each group by the DETERMINISTIC verifier
//!    ([`Verification::all_passed`]) into `all_pass` / `all_fail` / `mixed`. On all-pass and
//!    all-fail groups every admissible sibling is verifier-equivalent, so the panel judge cannot
//!    separate them on answer-correctness — its tokens are skippable.
//!    `decidable_fraction = mixed / groups_with_≥2_siblings`.
//!
//! 2. **selector-vs-random winrate**, on the **reasoning-quality axis ONLY**
//!    ([`Judging::aggregate`]), **restricted to all-pass groups**. Scoring answer-correctness
//!    argmax-vs-random would be circular: the verifier hard gate pins that winrate at ~1.0 by
//!    construction (the panel cannot admit a verifier-failing sibling). Among all-pass groups
//!    where ≥2 siblings carry an aggregate, the selector picks the argmax aggregate; we compare
//!    it to the two controls the spec names ([`random_per_prompt`] and [`random_k_overall`]).
//!
//! ## Why closed-form, not a sampled RNG
//!
//! To stay DETERMINISTIC and exactly reproducible (no seed to thread, no flaky test), the random
//! controls use the **closed-form expectation** of a uniform random pick rather than a sampled
//! draw: `E[random_per_prompt]` over a group is the group's MEAN aggregate, and
//! `E[random_k_overall]` is the global mean aggregate across all eligible siblings. The selector
//! "wins" a group iff its argmax `>=` that expectation. This is documented on each field.
//!
//! [`prompt_hash`]: gw_schema::Hashes::prompt_hash
//! [`Verification::all_passed`]: gw_schema::Verification::all_passed
//! [`Judging::aggregate`]: gw_schema::Judging::aggregate
//! [`random_per_prompt`]: SeparationReport::control_per_prompt_mean
//! [`random_k_overall`]: SeparationReport::control_k_overall_mean

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use gw_schema::TrainingRecord;
use gw_storage::{RecordFilter, Store};

use crate::error::Result;

/// Tunables for [`analyze`]. All have spec-aligned defaults via [`Default`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeparationConfig {
    /// Minimum number of DECIDABLE (mixed) groups required before the selector-vs-random signal
    /// is trusted. Below this the report sets [`SeparationReport::low_data`]. Default `10`.
    pub min_decidable_groups: usize,
    /// Floor on [`SeparationReport::decidable_fraction`] below which the corpus is treated as a
    /// DATA CEILING (the verifier rarely separates siblings), not a grader failure. Jointly with
    /// `min_decidable_groups` this gates [`SeparationReport::low_data`]. Default `0.05`.
    pub min_decidable_fraction: f64,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            min_decidable_groups: 10,
            min_decidable_fraction: 0.05,
        }
    }
}

/// The serde-serializable output of the separation diagnostic.
///
/// Every count is re-derivable from the input corpus; nothing here is randomized. See the module
/// docs for the meaning of the selector-vs-random fields and the `low_data` flag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeparationReport {
    /// Distinct sibling groups (distinct `prompt_hash` values) seen, including singletons.
    pub n_groups: usize,
    /// Groups with exactly one sibling — excluded from every fraction below.
    pub n_singletons: usize,
    /// Groups (with `>= 2` siblings) where EVERY sibling passed the deterministic verifier.
    pub n_allpass: usize,
    /// Groups (with `>= 2` siblings) where EVERY sibling FAILED the deterministic verifier.
    pub n_allfail: usize,
    /// Groups (with `>= 2` siblings) with a mix of pass/fail — the verifier-DECIDABLE groups.
    pub n_mixed: usize,
    /// Alias for [`Self::n_mixed`]; the count the budget lever skips judge tokens AGAINST.
    pub n_decidable: usize,
    /// `n_mixed / (n_allpass + n_allfail + n_mixed)`. `0.0` when there are no multi-sibling
    /// groups. LOW value ⇒ the panel can be skipped on most groups (a budget saving), and ALSO
    /// means a weak selector-vs-random signal is a data ceiling, not a grader failure.
    pub decidable_fraction: f64,
    /// All-pass groups carrying `>= 2` siblings WITH a present `judging.aggregate` — the eligible
    /// population for the reasoning-axis selector-vs-random control.
    pub n_selector_eligible: usize,
    /// Fraction of eligible groups where the selector (argmax aggregate) `>=` the
    /// `random_per_prompt` expectation (the group MEAN). `1.0` when every group ties-or-wins;
    /// `0.0` when there are no eligible groups.
    pub selector_winrate: f64,
    /// Mean over eligible groups of `selector_argmax - E[random_per_prompt]` (the group mean).
    /// `>= 0` always (argmax `>=` mean), `== 0` exactly when every eligible group has a constant
    /// aggregate. `0.0` when there are no eligible groups.
    pub selector_mean_gap: f64,
    /// `E[random_per_prompt]` averaged over eligible groups = the mean of per-group mean
    /// aggregates. The matched per-prompt control level.
    pub control_per_prompt_mean: f64,
    /// `E[random_k_overall]` = the GLOBAL mean aggregate across all eligible siblings (one draw
    /// from the whole pool, ignoring group boundaries). The matched overall control level.
    pub control_k_overall_mean: f64,
    /// Mean selector argmax across eligible groups — the level the controls are measured against.
    pub selector_mean: f64,
    /// WARN flag: too little decidable data to trust the selector signal. Set iff
    /// `n_decidable < cfg.min_decidable_groups` OR
    /// `decidable_fraction < cfg.min_decidable_fraction`. A low gap UNDER this flag is a DATA
    /// CEILING (the verifier rarely separates), NOT evidence the grader fails to separate.
    pub low_data: bool,
}

/// One sibling group's per-sibling facts, reduced from the raw records.
struct Group {
    /// `verification.all_passed` for each sibling in the group.
    passed: Vec<bool>,
    /// `judging.aggregate` for each sibling that carries one (others dropped).
    aggregates: Vec<f64>,
}

impl Group {
    fn n_siblings(&self) -> usize {
        self.passed.len()
    }
    fn all_pass(&self) -> bool {
        self.passed.iter().all(|&p| p)
    }
    fn all_fail(&self) -> bool {
        self.passed.iter().all(|&p| !p)
    }
}

/// Group a slice of records by `prompt_hash` into [`Group`]s, in a deterministic key order.
fn group_records(records: &[TrainingRecord]) -> Vec<Group> {
    let mut by_hash: BTreeMap<&str, Group> = BTreeMap::new();
    for rec in records {
        let key = rec.hashes.prompt_hash.as_str();
        let g = by_hash.entry(key).or_insert_with(|| Group {
            passed: Vec::new(),
            aggregates: Vec::new(),
        });
        g.passed.push(rec.verification.all_passed);
        if let Some(agg) = rec.judging.aggregate {
            g.aggregates.push(agg);
        }
    }
    by_hash.into_values().collect()
}

/// Mean of a non-empty slice; `0.0` for an empty slice (callers guard emptiness where it matters).
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Run the separation diagnostic over an already-loaded corpus — the PURE, infallible core.
///
/// This does no I/O and is fully deterministic, so it is the unit-testable heart of the module.
/// The async [`analyze_store`] is a thin wrapper that scans a [`Store`] and calls this.
///
/// Records with an EMPTY `prompt_hash` are still grouped (the empty string is one key); callers
/// that scan a real [`Store`] never see empty hashes because [`Store::put`] recomputes them.
#[must_use]
pub fn analyze(records: &[TrainingRecord], cfg: &SeparationConfig) -> SeparationReport {
    let groups = group_records(records);
    let n_groups = groups.len();

    let mut n_singletons = 0usize;
    let mut n_allpass = 0usize;
    let mut n_allfail = 0usize;
    let mut n_mixed = 0usize;

    // Per-eligible-group selector argmax and the per-prompt control expectation (group mean).
    let mut selector_argmaxes: Vec<f64> = Vec::new();
    let mut per_prompt_expectations: Vec<f64> = Vec::new();
    // The flat pool of every eligible sibling aggregate (for E[random_k_overall]).
    let mut overall_pool: Vec<f64> = Vec::new();

    for g in &groups {
        if g.n_siblings() < 2 {
            n_singletons += 1;
            continue;
        }
        if g.all_pass() {
            n_allpass += 1;
            // Selector control is RESTRICTED to all-pass groups with >= 2 scored siblings.
            if g.aggregates.len() >= 2 {
                let argmax = g
                    .aggregates
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                selector_argmaxes.push(argmax);
                per_prompt_expectations.push(mean(&g.aggregates));
                overall_pool.extend(g.aggregates.iter().copied());
            }
        } else if g.all_fail() {
            n_allfail += 1;
        } else {
            n_mixed += 1;
        }
    }

    let multi = n_allpass + n_allfail + n_mixed;
    let decidable_fraction = if multi == 0 {
        0.0
    } else {
        n_mixed as f64 / multi as f64
    };

    let n_selector_eligible = selector_argmaxes.len();
    let control_k_overall_mean = mean(&overall_pool);
    let control_per_prompt_mean = mean(&per_prompt_expectations);
    let selector_mean = mean(&selector_argmaxes);

    // Winrate + gap are measured against the per-prompt expectation (the matched control the spec
    // names first). The overall-pool level is reported alongside for context.
    let (selector_winrate, selector_mean_gap) = if n_selector_eligible == 0 {
        (0.0, 0.0)
    } else {
        let wins = selector_argmaxes
            .iter()
            .zip(&per_prompt_expectations)
            .filter(|(s, e)| *s >= *e)
            .count();
        let gaps: Vec<f64> = selector_argmaxes
            .iter()
            .zip(&per_prompt_expectations)
            .map(|(s, e)| s - e)
            .collect();
        (wins as f64 / n_selector_eligible as f64, mean(&gaps))
    };

    let low_data =
        n_mixed < cfg.min_decidable_groups || decidable_fraction < cfg.min_decidable_fraction;

    SeparationReport {
        n_groups,
        n_singletons,
        n_allpass,
        n_allfail,
        n_mixed,
        n_decidable: n_mixed,
        decidable_fraction,
        n_selector_eligible,
        selector_winrate,
        selector_mean_gap,
        control_per_prompt_mean,
        control_k_overall_mean,
        selector_mean,
        low_data,
    }
}

/// Scan a [`Store`] for the records matching `filter`, then run [`analyze`] on them.
///
/// The thin async wrapper. It performs the ONLY fallible step (the storage read); the math is
/// delegated to the pure core. Pass a [`RecordFilter`] to restrict to one run / lifecycle state
/// (e.g. `RecordFilter::new().run_id("run-1")`).
///
/// # Errors
/// Returns [`crate::EvalError::Storage`] if the underlying scan fails (SQL fault or a stored
/// envelope that fails to decode).
pub async fn analyze_store(
    store: &Store,
    filter: &RecordFilter,
    cfg: &SeparationConfig,
) -> Result<SeparationReport> {
    let records = store.scan(filter).await?;
    Ok(analyze(&records, cfg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{group, scored_sibling};

    /// A corpus where every multi-sibling group is mixed ⇒ `decidable_fraction == 1.0`.
    #[test]
    fn fully_decidable_corpus() {
        let mut recs = Vec::new();
        // 3 mixed groups: one pass + one fail each.
        for h in ["g0", "g1", "g2"] {
            recs.push(scored_sibling(h, true, None));
            recs.push(scored_sibling(h, false, None));
        }
        let rep = analyze(&recs, &SeparationConfig::default());
        assert_eq!(rep.n_mixed, 3);
        assert_eq!(rep.n_allpass, 0);
        assert_eq!(rep.n_allfail, 0);
        assert!((rep.decidable_fraction - 1.0).abs() < 1e-12);
    }

    /// A corpus with only all-pass / all-fail groups ⇒ `decidable_fraction == 0.0` + WARN.
    #[test]
    fn fully_undecidable_corpus_warns() {
        let recs = vec![
            // all-pass group (2 siblings)
            scored_sibling("p", true, Some(0.8)),
            scored_sibling("p", true, Some(0.6)),
            // all-fail group (2 siblings)
            scored_sibling("f", false, None),
            scored_sibling("f", false, None),
        ];
        let rep = analyze(&recs, &SeparationConfig::default());
        assert_eq!(rep.n_mixed, 0);
        assert_eq!(rep.n_allpass, 1);
        assert_eq!(rep.n_allfail, 1);
        assert!((rep.decidable_fraction - 0.0).abs() < 1e-12);
        assert!(rep.low_data, "0 decidable groups must trip the WARN flag");
    }

    /// Selector beats random when the argmax aggregate is the genuinely-best all-pass sibling.
    #[test]
    fn selector_beats_random_on_reasoning_axis() {
        // One all-pass group, aggregates {0.9, 0.5, 0.1}. argmax = 0.9, group mean = 0.5.
        let recs = group("ap", true, &[0.9, 0.5, 0.1]);
        let rep = analyze(&recs, &SeparationConfig::default());
        assert_eq!(rep.n_selector_eligible, 1);
        assert!((rep.selector_mean - 0.9).abs() < 1e-12);
        assert!((rep.control_per_prompt_mean - 0.5).abs() < 1e-12);
        assert!((rep.control_k_overall_mean - 0.5).abs() < 1e-12);
        assert!((rep.selector_winrate - 1.0).abs() < 1e-12, "argmax >= mean");
        assert!(
            (rep.selector_mean_gap - 0.4).abs() < 1e-12,
            "0.9 - 0.5 == 0.4"
        );
    }

    /// Selector TIES random when the aggregate is constant across the group.
    #[test]
    fn selector_ties_random_when_aggregate_constant() {
        let recs = group("flat", true, &[0.7, 0.7, 0.7]);
        let rep = analyze(&recs, &SeparationConfig::default());
        assert_eq!(rep.n_selector_eligible, 1);
        // argmax == mean == 0.7 ⇒ a tie still counts as a (non-strict) win, zero gap.
        assert!((rep.selector_mean_gap - 0.0).abs() < 1e-12);
        assert!((rep.selector_winrate - 1.0).abs() < 1e-12);
    }

    /// Singletons are counted but excluded from every fraction.
    #[test]
    fn singletons_excluded_from_fractions() {
        let recs = vec![
            scored_sibling("solo", true, Some(0.9)),
            // a real mixed group alongside
            scored_sibling("m", true, None),
            scored_sibling("m", false, None),
        ];
        let rep = analyze(&recs, &SeparationConfig::default());
        assert_eq!(rep.n_groups, 2);
        assert_eq!(rep.n_singletons, 1);
        assert_eq!(rep.n_mixed, 1);
        assert!((rep.decidable_fraction - 1.0).abs() < 1e-12);
    }

    /// An all-pass group with only ONE scored sibling is NOT selector-eligible (needs >= 2).
    #[test]
    fn allpass_group_with_one_scored_sibling_is_not_eligible() {
        let recs = vec![
            scored_sibling("ap", true, Some(0.9)),
            scored_sibling("ap", true, None),
        ];
        let rep = analyze(&recs, &SeparationConfig::default());
        assert_eq!(rep.n_allpass, 1);
        assert_eq!(rep.n_selector_eligible, 0);
        assert!((rep.selector_winrate - 0.0).abs() < 1e-12);
    }

    /// `min_decidable_fraction` alone can trip `low_data` even with enough decidable groups.
    #[test]
    fn low_fraction_trips_warn_even_with_enough_groups() {
        let mut recs = Vec::new();
        // 12 mixed groups (>= default min_decidable_groups of 10) ...
        for i in 0..12 {
            let h = format!("mix{i}");
            recs.push(scored_sibling(&h, true, None));
            recs.push(scored_sibling(&h, false, None));
        }
        // ... but 400 all-pass groups drown the decidable fraction far below 0.05.
        for i in 0..400 {
            let h = format!("ap{i}");
            recs.push(scored_sibling(&h, true, Some(0.5)));
            recs.push(scored_sibling(&h, true, Some(0.5)));
        }
        let rep = analyze(&recs, &SeparationConfig::default());
        assert_eq!(rep.n_mixed, 12);
        assert!(rep.decidable_fraction < 0.05);
        assert!(
            rep.low_data,
            "a low decidable_fraction is a data ceiling and must WARN"
        );
    }
}
