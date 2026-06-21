//! The live rail verdict types that live in `gw-judge` (not `gw-schema`) and their mapping DOWN to
//! the persisted envelope (ARCHITECTURE §5 "Verdict reconciliation"; REMEDIATION ITEM 6).
//!
//! Three vocabularies, one mapping:
//!
//! | type | variants | persists? |
//! |---|---|---|
//! | `Grade.verdict` ([`Verdict`]) | `Accept` `Revise` `Reject` `Uncertain` | per-judge, NOT directly |
//! | [`Decision`] | `Accept` `Revise` `Reject` `Escalate{Human\|Verifier}` | maps to lifecycle |
//! | `gw_schema::Verdict` | `Admit` `Reject` `NeedsReview` | YES (the envelope) |
//!
//! [`Verdict`] is the richer **per-grade** enum: it carries `Revise` (transient, drives the bounded
//! retry) and `Uncertain` (a single judge could not decide — e.g. a position-swap disagreement or
//! an unparseable score). Neither persists in the envelope: `Revise` is transient and `Uncertain`
//! is per-judge only. [`Decision`] is the **panel-level** outcome; [`Decision::to_schema_verdict`]
//! maps it to the 3-variant persisted `gw_schema::Verdict` and [`Decision::to_lifecycle`] to the
//! `LifecycleState` the engine drives the record to.

use gw_schema::{LifecycleState, Verdict as SchemaVerdict};

/// One judge's per-grade verdict (the 4-variant live enum). Distinct from the persisted 3-variant
/// `gw_schema::Verdict`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Verdict {
    /// The trace is good enough to admit on this judge's view.
    Accept,
    /// The trace is salvageable — route to a bounded single revise (transient; never persisted).
    Revise,
    /// The trace is bad; reject it.
    Reject,
    /// This judge could not decide (position-swap disagreement, unparseable score). Per-judge only;
    /// never persisted. Carries no weight as an `Accept` or `Reject` in the tally.
    Uncertain,
}

/// Who a record escalates to when the panel cannot be trusted (correlated judges) or a soft
/// minority-veto fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EscalateTo {
    /// Park for human adjudication.
    Human,
    /// Defer to a deterministic verifier / oracle that the automated panel could not stand in for.
    Verifier,
}

/// Why a [`Decision`] was reached — a structured reason folded into `Judging.verdict_reason` for
/// audit. Keeps the seam machine-readable rather than only a free-text string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionReason {
    /// A deterministic verifier hard-failed (`Verification.all_passed == false`); authoritative over
    /// any panel score.
    VerifierReject,
    /// The aggregate cleared the area accept threshold.
    AboveThreshold,
    /// The aggregate fell in the revise band `[reject_below, accept_threshold)`.
    ReviseBand,
    /// The aggregate fell below `reject_below`.
    BelowThreshold,
    /// `n_eff/k < min_n_eff_ratio` or `n_eff < min_n_eff` — the panel is too correlated to trust.
    CorrelatedJudges,
    /// A soft minority-veto / SP-override routed the record to review against the majority.
    MinorityVeto,
    /// The panel produced no usable verdict (all `Uncertain`); conservative default.
    ConservativeTie,
}

impl DecisionReason {
    /// A stable, human-readable token for [`Judging.verdict_reason`](gw_schema::Judging).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            DecisionReason::VerifierReject => "verifier_reject",
            DecisionReason::AboveThreshold => "above_threshold",
            DecisionReason::ReviseBand => "revise_band",
            DecisionReason::BelowThreshold => "below_threshold",
            DecisionReason::CorrelatedJudges => "correlated_judges",
            DecisionReason::MinorityVeto => "minority_veto",
            DecisionReason::ConservativeTie => "conservative_tie",
        }
    }
}

#[cfg(test)]
#[test]
fn every_reason_token_round_trips() {
    // A guard so a new DecisionReason variant cannot ship without a token.
    for r in [
        DecisionReason::VerifierReject,
        DecisionReason::AboveThreshold,
        DecisionReason::ReviseBand,
        DecisionReason::BelowThreshold,
        DecisionReason::CorrelatedJudges,
        DecisionReason::MinorityVeto,
        DecisionReason::ConservativeTie,
    ] {
        assert!(!r.as_str().is_empty());
    }
}

/// The panel-level outcome of grading one candidate. Maps to the lifecycle and the persisted
/// verdict via [`to_lifecycle`](Decision::to_lifecycle) / [`to_schema_verdict`](Decision::to_schema_verdict).
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Admit the trace.
    Accept {
        /// Why (always [`DecisionReason::AboveThreshold`] for a clean accept).
        reason: DecisionReason,
    },
    /// Route to a bounded single revise (transient).
    Revise {
        /// Why (always [`DecisionReason::ReviseBand`]).
        reason: DecisionReason,
    },
    /// Reject the trace.
    Reject {
        /// Why — a verifier hard fail, a below-threshold aggregate, or the conservative tie.
        reason: DecisionReason,
    },
    /// Park the record for human/verifier resolution (correlated panel, or a soft minority-veto).
    Escalate {
        /// Where the record escalates.
        to: EscalateTo,
        /// Why.
        reason: DecisionReason,
    },
}

impl Decision {
    /// The [`DecisionReason`] carried by this decision.
    #[must_use]
    pub fn reason(&self) -> DecisionReason {
        match self {
            Decision::Accept { reason }
            | Decision::Revise { reason }
            | Decision::Reject { reason }
            | Decision::Escalate { reason, .. } => reason.clone(),
        }
    }

    /// Map to the persisted 3-variant `gw_schema::Verdict`. `Revise` collapses to `Reject`'s sibling
    /// `NeedsReview`? — NO: a `Revise` is **transient** and does NOT persist a final verdict yet, so
    /// it returns `None` (the record is mid-retry, `judging.verdict` stays unset). `Accept → Admit`,
    /// `Reject → Reject`, `Escalate → NeedsReview`.
    #[must_use]
    pub fn to_schema_verdict(&self) -> Option<SchemaVerdict> {
        match self {
            Decision::Accept { .. } => Some(SchemaVerdict::Admit),
            Decision::Reject { .. } => Some(SchemaVerdict::Reject),
            Decision::Escalate { .. } => Some(SchemaVerdict::NeedsReview),
            // Revise is transient — no terminal verdict is persisted while the retry is in flight.
            Decision::Revise { .. } => None,
        }
    }

    /// Map to the `LifecycleState` the engine drives the record to (ARCHITECTURE §5): `Accept →
    /// Admitted`, `Reject → Rejected`, `Revise → Revising`, `Escalate → NeedsReview`.
    #[must_use]
    pub fn to_lifecycle(&self) -> LifecycleState {
        match self {
            Decision::Accept { .. } => LifecycleState::Admitted,
            Decision::Reject { .. } => LifecycleState::Rejected,
            Decision::Revise { .. } => LifecycleState::Revising,
            Decision::Escalate { .. } => LifecycleState::NeedsReview,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_maps_to_admit_and_admitted() {
        let d = Decision::Accept {
            reason: DecisionReason::AboveThreshold,
        };
        assert_eq!(d.to_schema_verdict(), Some(SchemaVerdict::Admit));
        assert_eq!(d.to_lifecycle(), LifecycleState::Admitted);
    }

    #[test]
    fn reject_maps_to_reject_and_rejected() {
        let d = Decision::Reject {
            reason: DecisionReason::BelowThreshold,
        };
        assert_eq!(d.to_schema_verdict(), Some(SchemaVerdict::Reject));
        assert_eq!(d.to_lifecycle(), LifecycleState::Rejected);
    }

    #[test]
    fn escalate_maps_to_needs_review() {
        let d = Decision::Escalate {
            to: EscalateTo::Human,
            reason: DecisionReason::CorrelatedJudges,
        };
        assert_eq!(d.to_schema_verdict(), Some(SchemaVerdict::NeedsReview));
        assert_eq!(d.to_lifecycle(), LifecycleState::NeedsReview);
    }

    #[test]
    fn revise_is_transient_and_persists_no_verdict() {
        let d = Decision::Revise {
            reason: DecisionReason::ReviseBand,
        };
        // Transient: no terminal verdict while the bounded retry is in flight.
        assert_eq!(d.to_schema_verdict(), None);
        assert_eq!(d.to_lifecycle(), LifecycleState::Revising);
    }

    #[test]
    fn reason_token_is_stable() {
        assert_eq!(DecisionReason::VerifierReject.as_str(), "verifier_reject");
        assert_eq!(
            DecisionReason::CorrelatedJudges.as_str(),
            "correlated_judges"
        );
    }
}
