//! User-synthesis verification contract + the pre-teacher-spend QC verdict
//! (USER-SYNTHESIS §8/§9).
//!
//! These types describe *what correctness means* for a synthesized USER turn and gate it
//! before any teacher tokens are spent. `gw-judge` reads `VerificationContract.kind` to gate
//! the `over_refusal` dimension; `Provenance.user_turn_kind` / `Provenance.in_scope_safe`
//! mirror the serialized variant + `UserTurnVerdict.in_scope_safe`.

use serde::{Deserialize, Serialize};

/// What the deterministic Verifier rail will check, and how the oracle answer is obtained
/// (USER-SYNTHESIS §8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationContract {
    /// Answer policy. Missing only on historical records; executable tasks must declare it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_policy: Option<VerificationPolicy>,
    /// Execution policy, independent of the answer comparator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_policy: Option<VerificationPolicy>,
    /// Exact test identifiers required by the task, never reduced by a report's own list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_tests: Vec<String>,
    /// The class of correctness check.
    pub kind: VerificationKind,
    /// How ground truth is computed.
    pub oracle: Oracle,
    /// Explicit numeric extraction/tolerances. Required for `NumericMatch`; absent for other kinds.
    /// Missing on historical numeric records, which remain readable but cannot execute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numeric: Option<crate::NumericComparison>,
}

/// How an observation affects admission, independently of its factual outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationPolicy {
    /// Do not run or interpret this axis.
    Absent,
    /// Preserve facts while allowing the quality panel to decide.
    Advisory,
    /// Fail rejects; unknown holds for review; pass still requires the quality panel.
    Authoritative,
}

impl VerificationContract {
    /// Validate an executable task without I/O. Historical missing policies remain readable.
    ///
    /// # Errors
    /// Rejects missing policies, unsupported active answer combinations, and invalid test IDs.
    pub fn validate(&self) -> Result<(), &'static str> {
        use VerificationPolicy::{Absent, Authoritative};
        let answer = self
            .answer_policy
            .ok_or("verification contract lacks explicit answer policy")?;
        let execution = self
            .execution_policy
            .ok_or("verification contract lacks explicit execution policy")?;
        if self.kind == VerificationKind::NumericMatch {
            let numeric = self
                .numeric
                .as_ref()
                .ok_or("numeric comparator requires explicit extraction and tolerances")?;
            numeric.validate()?;
            let expected = match &self.oracle {
                Oracle::Literal { expected } => Some(expected),
                Oracle::SandboxExecution { expected, .. } => expected.as_ref(),
                _ => None,
            };
            if let Some(expected) = expected {
                numeric.bound(crate::parse_finite_decimal(expected).ok_or(
                    "numeric expected answer must be a finite decimal/scientific token",
                )?)?;
            }
        } else if self.numeric.is_some() {
            return Err("numeric extraction/tolerances require the numeric comparator");
        }
        if answer != Absent {
            let supported = matches!(
                (&self.kind, &self.oracle),
                (
                    VerificationKind::NumericMatch
                        | VerificationKind::SetMatch
                        | VerificationKind::SqlResultMatch
                        | VerificationKind::SchemaShape,
                    Oracle::Literal { .. } | Oracle::SandboxExecution { .. }
                ) | (
                    VerificationKind::RefusalExpected,
                    Oracle::RefusalPolicy { .. }
                )
            );
            if !supported {
                return Err("active answer policy requires a supported comparator and oracle");
            }
        }
        if execution == Authoritative && self.required_tests.is_empty() {
            return Err("authoritative execution requires task-declared test identifiers");
        }
        let mut seen = std::collections::HashSet::new();
        for id in &self.required_tests {
            if id.trim().is_empty() || !seen.insert(id) {
                return Err("required test identifiers must be nonblank and unique");
            }
        }
        Ok(())
    }
}

/// The class of correctness check for a synthesized USER turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationKind {
    /// answer is a number / aggregate; compare to oracle within tolerance.
    NumericMatch,
    /// answer is a set/ranking; compare membership/order.
    SetMatch,
    /// answer derives from a SQL/tool result; compare to oracle execution.
    SqlResultMatch,
    /// adversarial-by-construction (seed-020): correct behavior is refusal.
    RefusalExpected,
    /// answer must match a described schema/columns.
    SchemaShape,
    /// No answer comparator; execution policy remains independent.
    None,
}

/// How ground truth is computed for a [`VerificationContract`]. Tagged on `oracle`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "oracle")]
pub enum Oracle {
    /// Oracle answer computed by executing a reference query/tool in the sandbox (REMEDIATION ITEM 3).
    SandboxExecution {
        tool_or_sql: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expected: Option<String>,
    },
    /// Oracle is a precomputed literal carried with the contract.
    Literal { expected: String },
    /// Refusal is the oracle: correct behavior is a safe refusal + explanation.
    RefusalPolicy { policy_id: String },
    /// No answer oracle; active answer policies cannot use this variant.
    None,
}

/// QC verdict on a candidate USER turn, BEFORE teacher spend (USER-SYNTHESIS §9). A candidate
/// advances to `user_synthesized` iff all four booleans are true.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserTurnVerdict {
    /// a competent teacher could answer it; not nonsense/contradictory.
    pub answerable: bool,
    /// matches the requested difficulty band (not trivially off-target).
    pub difficulty_targeted: bool,
    /// passes embedding-dedup: not a near-repeat.
    pub diverse: bool,
    /// within Taxonomy + decontaminated + safety-classified. Adversarial-by-construction
    /// prompts (seed-020) are `in_scope_safe = true` (wanted, tagged adversarial); only
    /// out-of-scope-unsafe prompts fail. Mirrored to `Provenance.in_scope_safe`.
    pub in_scope_safe: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}
