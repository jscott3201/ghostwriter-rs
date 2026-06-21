//! A3 + ITEM 9 — the variance-aware **promotion gate**
//! (`_research/2026-06-20-rlhf-book-applicability.md` §A3,
//! `_research/REMEDIATION-DECISIONS.md` §ITEM 9).
//!
//! The gate decides promote / no-promote for a candidate fine-tune against a baseline. It is
//! headless, GPU-free, and imports no TRL/PEFT: it consumes already-computed B2 `eval_results`
//! and a capability-drift probe **exit code**, and emits a binary decision.
//!
//! ## Two gates — BOTH must pass (ITEM 9 truth table)
//!
//! 1. **Drift gate (hard).** If [`PromoteConfig::drift_must_pass`] and the drift probe exit code
//!    is non-zero, the candidate is REJECTED regardless of any A/B delta. A capability-drift
//!    regression is never bought back by a benchmark win.
//! 2. **A/B gate (variance-aware).** A benchmark "Wins" iff its delta exceeds the NOISE BAND
//!    `ab_min_delta + ab_sigma_k * sigma`, and "Regresses" iff its delta is below the negative
//!    band. The overall A/B gate passes iff **NO benchmark regresses AND at least one benchmark
//!    Wins** (spec A3: all-probes-must-not-regress + ≥1 Win — NEVER any-single-Win, which invites
//!    cherry-picking a lucky benchmark while another silently regresses).
//!
//! ## Re-derivability (sealed-verdict invariant)
//!
//! [`PromotionReport`] stores per-benchmark `{baseline, candidate, delta, sigma, noise_band, win,
//! regress}` plus the config knobs used (`ab_sigma_k`, `ab_min_delta`) and `drift_exit`, so the
//! decision is RECOMPUTABLE at a different `k` WITHOUT re-running eval — mirroring the
//! JUDGE-DESIGN sealed-verdict / re-derive-threshold invariant. [`PromotionReport::rederive`]
//! does exactly that.
//!
//! ## Config homing (deferred)
//!
//! `gw-schema` already carries a SIMPLER `PromoteConfig` (`drift_probe`, `drift_must_pass`,
//! `ab_metric`, `ab_min_delta`, `ab_baseline`) with no variance machinery. The A3 gate needs
//! `ab_sigma_k` / `ab_benchmark_sigma` / `ab_avg_n`, so for v1 this crate defines its OWN
//! self-contained [`PromoteConfig`] superset rather than making a large cross-crate schema edit.
//! CONFIG.md eventually homes the variance fields in `gw-schema::PromoteConfig`; that merge is a
//! reported follow-up, not part of this crate.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A B2 `eval_results.json` artifact: per-benchmark scores plus a headline aggregate.
///
/// Deserialized straight from the JSON a candidate or baseline adapter's eval run emits. The
/// `aggregate` is carried explicitly (it is the default headline [`PromoteConfig::ab_metric`])
/// AND is also addressable through [`Self::score`] under the dotted key `"eval_results.aggregate"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalResults {
    /// Per-benchmark scores, e.g. `{"gsm8k": 0.71, "ifeval": 0.63, ...}`. Ordered for stable
    /// iteration/serialization.
    #[serde(default)]
    pub benchmarks: BTreeMap<String, f64>,
    /// The headline rolled-up score (the default A/B metric).
    pub aggregate: f64,
}

impl EvalResults {
    /// Resolve a metric name to a score. The dotted sentinel `"eval_results.aggregate"` (and the
    /// bare `"aggregate"`) map to [`Self::aggregate`]; any other name indexes [`Self::benchmarks`].
    #[must_use]
    pub fn score(&self, metric: &str) -> Option<f64> {
        if metric == "eval_results.aggregate" || metric == "aggregate" {
            Some(self.aggregate)
        } else {
            self.benchmarks.get(metric).copied()
        }
    }

    /// Parse an `eval_results.json` byte buffer.
    ///
    /// # Errors
    /// Returns [`crate::EvalError::EvalResultsParse`] if the bytes are not valid `EvalResults`
    /// JSON.
    pub fn from_json(bytes: &[u8]) -> crate::Result<Self> {
        serde_json::from_slice(bytes).map_err(crate::EvalError::EvalResultsParse)
    }
}

/// The variance-aware promotion-gate config (A3 superset of `gw-schema::PromoteConfig`).
///
/// See the module docs for why this lives in `gw-eval` for v1. [`Default`] matches the spec
/// defaults (`ab_min_delta = 0.0`, `ab_sigma_k = 1.0`, `drift_must_pass = true`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromoteConfig {
    /// Which metric in `eval_results.json` is the HEADLINE benchmark, always evaluated and always
    /// included in the report. Default `"eval_results.aggregate"`.
    pub ab_metric: String,
    /// The fixed noise floor added to every benchmark's band (the spec's named-but-unmeasured
    /// `ab_min_delta`). Default `0.0`.
    pub ab_min_delta: f64,
    /// The σ multiplier: a benchmark must clear `ab_min_delta + ab_sigma_k * sigma` to Win.
    /// Default `1.0`. Re-tunable WITHOUT re-running eval via [`PromotionReport::rederive`].
    pub ab_sigma_k: f64,
    /// Per-benchmark σ priors (seeded from the Olmo 3 table, e.g. `gpqa 1.48`, `ifeval 0.88`).
    /// A benchmark absent here uses σ = `0.0` (band collapses to `ab_min_delta`) — documented and
    /// intentional so an un-prior'd benchmark is not silently given a free pass.
    #[serde(default)]
    pub ab_benchmark_sigma: BTreeMap<String, f64>,
    /// Number of measured runs averaged into each benchmark score. Once `>= 3`, a MEASURED σ
    /// should override the prior. v1 consumes only the configured prior map; this field is the
    /// documented OVERRIDE HOOK and does not yet change the band. Default `0`.
    pub ab_avg_n: u32,
    /// When true, a non-zero drift exit code hard-REJECTS regardless of the A/B delta. Default
    /// `true`.
    pub drift_must_pass: bool,
}

impl Default for PromoteConfig {
    fn default() -> Self {
        Self {
            ab_metric: "eval_results.aggregate".to_string(),
            ab_min_delta: 0.0,
            ab_sigma_k: 1.0,
            ab_benchmark_sigma: BTreeMap::new(),
            ab_avg_n: 0,
            drift_must_pass: true,
        }
    }
}

impl PromoteConfig {
    /// The σ to use for `benchmark`: the configured prior, or `0.0` if un-prior'd.
    ///
    /// The measured-σ override (`ab_avg_n >= 3`) is a documented v1 follow-up; see the field docs.
    #[must_use]
    fn sigma_for(&self, benchmark: &str) -> f64 {
        self.ab_benchmark_sigma
            .get(benchmark)
            .copied()
            .unwrap_or(0.0)
    }
}

/// One benchmark's re-derivable A/B verdict. `win` and `regress` are mutually exclusive; both
/// false means the delta sat inside the `±noise_band` (a NOISE result, neither a win nor a loss).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkOutcome {
    /// Benchmark name (e.g. `"gsm8k"`, or the headline `"eval_results.aggregate"`).
    pub name: String,
    /// Baseline score.
    pub baseline: f64,
    /// Candidate score.
    pub candidate: f64,
    /// `candidate - baseline`.
    pub delta: f64,
    /// σ used for this benchmark (prior, or `0.0` if un-prior'd).
    pub sigma: f64,
    /// `ab_min_delta + ab_sigma_k * sigma` — the half-width of the symmetric noise band.
    pub noise_band: f64,
    /// `delta > noise_band`.
    pub win: bool,
    /// `delta < -noise_band`.
    pub regress: bool,
}

/// The serde-serializable promotion decision + everything needed to recompute it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionReport {
    /// Final decision: promote iff BOTH gates pass.
    pub promote: bool,
    /// The drift probe exit code that was fed in (`0` == clean).
    pub drift_exit: i32,
    /// Whether the drift gate passed (`!drift_must_pass || drift_exit == 0`).
    pub drift_pass: bool,
    /// Whether the A/B gate passed (no regression AND `>= 1` win).
    pub ab_pass: bool,
    /// Per-benchmark outcomes, headline metric first, then the rest in name order.
    pub benchmarks: Vec<BenchmarkOutcome>,
    /// `ab_min_delta` used (stored so the decision is re-derivable).
    pub ab_min_delta: f64,
    /// `ab_sigma_k` used (stored so the decision is re-derivable at a different `k`).
    pub ab_sigma_k: f64,
    /// A human-readable one-line reason for the decision.
    pub reason: String,
}

impl PromotionReport {
    /// Recompute the promote decision from the SEALED per-benchmark fields at a (possibly) new
    /// `(ab_min_delta, ab_sigma_k)` — WITHOUT re-running eval.
    ///
    /// Each benchmark's `delta` and `sigma` are sealed in the report, so a fresh band
    /// `ab_min_delta + ab_sigma_k * sigma` re-classifies win/regress and re-applies the
    /// no-regression + ≥1-win rule. The drift gate is sealed too (`drift_pass`). This is the A3
    /// "k is re-tunable without re-running eval" guarantee, executable.
    #[must_use]
    pub fn rederive(&self, ab_min_delta: f64, ab_sigma_k: f64) -> bool {
        let mut any_win = false;
        let mut any_regress = false;
        for b in &self.benchmarks {
            let band = ab_min_delta + ab_sigma_k * b.sigma;
            if b.delta > band {
                any_win = true;
            } else if b.delta < -band {
                any_regress = true;
            }
        }
        self.drift_pass && !any_regress && any_win
    }
}

/// Run the variance-aware promotion gate — the PURE, infallible core.
///
/// `drift_exit` is the capability-drift probe's process exit code (`0` == clean). The benchmarks
/// compared are the UNION of `cfg.ab_metric` (always) and every key present in BOTH `baseline`
/// and `candidate` benchmark maps; a benchmark present in only one side is skipped (no comparable
/// delta) — that skip is intentional and does not fail the gate by itself.
///
/// See the module docs for the full two-gate truth table.
#[must_use]
pub fn promote(
    baseline: &EvalResults,
    candidate: &EvalResults,
    drift_exit: i32,
    cfg: &PromoteConfig,
) -> PromotionReport {
    // Build the comparable benchmark set: the headline metric first, then every shared key.
    let mut names: Vec<String> = vec![cfg.ab_metric.clone()];
    for k in candidate.benchmarks.keys() {
        if baseline.benchmarks.contains_key(k) && *k != cfg.ab_metric {
            names.push(k.clone());
        }
    }

    let mut benchmarks: Vec<BenchmarkOutcome> = Vec::new();
    for name in names {
        let (Some(b), Some(c)) = (baseline.score(&name), candidate.score(&name)) else {
            // Headline metric missing on a side, or a key that turned out non-shared: skip it.
            continue;
        };
        let delta = c - b;
        let sigma = cfg.sigma_for(&name);
        let noise_band = cfg.ab_min_delta + cfg.ab_sigma_k * sigma;
        benchmarks.push(BenchmarkOutcome {
            name,
            baseline: b,
            candidate: c,
            delta,
            sigma,
            noise_band,
            win: delta > noise_band,
            regress: delta < -noise_band,
        });
    }

    let any_win = benchmarks.iter().any(|b| b.win);
    let any_regress = benchmarks.iter().any(|b| b.regress);
    let ab_pass = !any_regress && any_win;

    let drift_pass = !cfg.drift_must_pass || drift_exit == 0;
    let promote = drift_pass && ab_pass;

    let reason = if !drift_pass {
        format!("REJECT: drift probe exit {drift_exit} != 0 (hard gate)")
    } else if any_regress {
        "REJECT: at least one benchmark regressed beyond the noise band".to_string()
    } else if !any_win {
        "REJECT: no benchmark cleared its noise band (all deltas are noise)".to_string()
    } else {
        "PROMOTE: drift clean, no regression, >= 1 benchmark win".to_string()
    };

    PromotionReport {
        promote,
        drift_exit,
        drift_pass,
        ab_pass,
        benchmarks,
        ab_min_delta: cfg.ab_min_delta,
        ab_sigma_k: cfg.ab_sigma_k,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an [`EvalResults`] from an aggregate + benchmark pairs.
    fn results(aggregate: f64, pairs: &[(&str, f64)]) -> EvalResults {
        EvalResults {
            aggregate,
            benchmarks: pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
        }
    }

    /// A config with a σ prior for `gsm8k` and a k of 1.0.
    fn cfg_with_sigma() -> PromoteConfig {
        let mut c = PromoteConfig::default();
        c.ab_benchmark_sigma.insert("gsm8k".to_string(), 0.5);
        c
    }

    /// Quadrant (drift_pass=Y, ab_pass=Y): Δ above floor+kσ on the headline ⇒ PROMOTE.
    #[test]
    fn promote_when_drift_clean_and_delta_clears_band() {
        let base = results(0.50, &[]);
        let cand = results(0.60, &[]);
        let rep = promote(&base, &cand, 0, &PromoteConfig::default());
        assert!(rep.promote);
        assert!(rep.drift_pass && rep.ab_pass);
        assert_eq!(rep.benchmarks[0].name, "eval_results.aggregate");
        assert!(rep.benchmarks[0].win);
    }

    /// Quadrant (drift_pass=N, ab_pass=Y): a huge Δ cannot buy back a drift failure ⇒ REJECT.
    #[test]
    fn reject_when_drift_fails_despite_huge_delta() {
        let base = results(0.10, &[]);
        let cand = results(0.99, &[]);
        let rep = promote(&base, &cand, 1, &PromoteConfig::default());
        assert!(!rep.promote, "drift exit != 0 is a hard reject");
        assert!(!rep.drift_pass);
        assert!(rep.ab_pass, "the A/B side still passed on its own");
        assert!(rep.reason.contains("drift"));
    }

    /// Quadrant (drift_pass=Y, ab_pass=N): Δ inside the noise band ⇒ REJECT (promote-on-noise).
    #[test]
    fn reject_when_delta_within_noise_band() {
        // gsm8k σ=0.5, k=1 ⇒ band 0.5. A +0.30 gsm8k delta is NOISE; aggregate flat.
        let base = results(0.50, &[("gsm8k", 0.50)]);
        let cand = results(0.50, &[("gsm8k", 0.80)]);
        let rep = promote(&base, &cand, 0, &cfg_with_sigma());
        assert!(!rep.promote);
        assert!(rep.drift_pass);
        assert!(!rep.ab_pass, "no benchmark cleared its band");
        let gsm = rep.benchmarks.iter().find(|b| b.name == "gsm8k").unwrap();
        assert!(!gsm.win && !gsm.regress, "delta sat inside +/- band");
    }

    /// No cherry-pick: one benchmark Wins while ANOTHER regresses ⇒ REJECT.
    #[test]
    fn reject_when_one_wins_but_another_regresses() {
        // aggregate wins big (+0.20, band 0.0); gsm8k regresses past its band (-0.80 < -0.5).
        let base = results(0.50, &[("gsm8k", 0.90)]);
        let cand = results(0.70, &[("gsm8k", 0.10)]);
        let rep = promote(&base, &cand, 0, &cfg_with_sigma());
        assert!(
            !rep.promote,
            "a regression blocks promotion even with a win"
        );
        let agg = rep
            .benchmarks
            .iter()
            .find(|b| b.name == "eval_results.aggregate")
            .unwrap();
        let gsm = rep.benchmarks.iter().find(|b| b.name == "gsm8k").unwrap();
        assert!(agg.win);
        assert!(gsm.regress);
        assert!(!rep.ab_pass);
    }

    /// Quadrant (drift_pass=Y, ab_pass=Y) with σ priors: Δ must clear floor+kσ, not just floor.
    #[test]
    fn win_requires_clearing_sigma_band_not_just_floor() {
        // gsm8k σ=0.5, k=1 ⇒ band 0.5. +0.60 CLEARS it; aggregate flat (no regression elsewhere).
        let base = results(0.50, &[("gsm8k", 0.20)]);
        let cand = results(0.50, &[("gsm8k", 0.80)]);
        let rep = promote(&base, &cand, 0, &cfg_with_sigma());
        let gsm = rep.benchmarks.iter().find(|b| b.name == "gsm8k").unwrap();
        assert!(gsm.win, "+0.60 > 0.5 band");
        assert!(rep.promote);
    }

    /// Re-derivability: a decision recomputes from sealed fields at a HIGHER k that flips it.
    #[test]
    fn report_is_rederivable_at_a_different_k() {
        // gsm8k σ=0.5: at k=1 a +0.60 delta WINS (band 0.5) and promotes ...
        let base = results(0.50, &[("gsm8k", 0.20)]);
        let cand = results(0.50, &[("gsm8k", 0.80)]);
        let rep = promote(&base, &cand, 0, &cfg_with_sigma());
        assert!(rep.promote);
        // recomputing at k=1 from sealed fields reproduces the original decision ...
        assert_eq!(rep.rederive(0.0, 1.0), rep.promote);
        // ... and at k=2 the band widens to 1.0, so +0.60 is no longer a win ⇒ no promote.
        assert!(
            !rep.rederive(0.0, 2.0),
            "wider band at higher k must demote the win to noise"
        );
    }

    /// Non-shared benchmark keys are skipped (no comparable delta), so only the headline
    /// aggregate is compared; a flat aggregate then yields no win ⇒ no promote.
    #[test]
    fn non_shared_benchmarks_are_skipped() {
        // Each side carries a benchmark the OTHER lacks; neither is comparable.
        let base = results(0.50, &[("only_base", 0.9)]);
        let cand = results(0.50, &[("only_cand", 0.9)]);
        let rep = promote(&base, &cand, 0, &PromoteConfig::default());
        // Only the (flat) aggregate is comparable ⇒ no win ⇒ no promote.
        assert_eq!(rep.benchmarks.len(), 1);
        assert_eq!(rep.benchmarks[0].name, "eval_results.aggregate");
        assert!(!rep.promote);
    }

    /// `EvalResults::score` resolves the dotted aggregate sentinel and bare benchmark names.
    #[test]
    fn eval_results_score_resolves_metric_names() {
        let r = results(0.42, &[("gsm8k", 0.7)]);
        assert_eq!(r.score("eval_results.aggregate"), Some(0.42));
        assert_eq!(r.score("aggregate"), Some(0.42));
        assert_eq!(r.score("gsm8k"), Some(0.7));
        assert_eq!(r.score("nope"), None);
    }

    /// `EvalResults` parses from a B2 `eval_results.json` byte buffer.
    #[test]
    fn eval_results_parses_from_json() {
        let json = br#"{"aggregate": 0.66, "benchmarks": {"gsm8k": 0.71, "ifeval": 0.6}}"#;
        let r = EvalResults::from_json(json).unwrap();
        assert!((r.aggregate - 0.66).abs() < 1e-12);
        assert_eq!(r.benchmarks.get("gsm8k"), Some(&0.71));
    }
}
