//! `verification{}` — the deterministic Verifier rail (DATA-SCHEMA §1.6).

use serde::{Deserialize, Serialize};

/// Persisted deterministic facts and the policy applied to them. Historical boolean-only blocks
/// remain readable but cannot authorize current executable replay.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Verification {
    #[serde(default)]
    pub checks: Vec<Check>,
    /// No authoritative hard failure. Advisory failures and unknowns are not factual passes.
    pub all_passed: bool,
    /// Versioned facts and policy; absent on historical records and before verification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpretation: Option<VerificationInterpretation>,
    /// Derived hold reason when an authoritative axis is Unknown and no authoritative failure
    /// exists. The supported interpretation is the authority for reconstruction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub needs_review: Option<String>,
}

/// One deterministic check result, e.g. `"rust_compiles"`, `"json_valid"`,
/// `"reasoning_present"`, `"decontam"`, `"language_consistency"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Check {
    pub name: String,
    pub kind: CheckKind,
    /// Factual pass only. Unknown and failed observations both serialize false.
    pub passed: bool,
    /// Optional factual score in `[0,1]`; Unknown has no score. This is audit data; admission
    /// follows the separately persisted policy and observation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The class of a deterministic [`Check`]. A safety/quality *rubric* score is NOT a
/// `CheckKind` — it is a JudgePanel dimension (DATA-SCHEMA §1.6/§1.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckKind {
    Compile,
    Exec,
    Regex,
    Schema,
    UnitTest,
    MathCheck,
    Decontam,
    ReasoningPresent,
    /// A sandboxed tool/SQL/code execution verdict (serializes `"sandbox"`; REMEDIATION ITEM 3).
    Sandbox,
    /// A deterministic language-consistency verdict (serializes `"language"`; check name
    /// `"language_consistency"`). INERT when the area's `target_language` is None (B12).
    Language,
}

/// Current policy/fact interpretation revision. Older or missing revisions cannot be re-certified.
pub const VERIFICATION_INTERPRETATION_VERSION: u32 = 1;

/// A factual result before admission policy is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationOutcome {
    /// The check established its passing condition.
    Pass,
    /// The check established a failure (including an evidence-contract failure).
    Fail,
    /// Missing, unavailable, ambiguous, or stale evidence established neither result.
    Unknown,
}

/// An auditable observation with a bounded explanation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationObservation {
    /// Factual result, never rewritten by policy.
    pub outcome: VerificationOutcome,
    /// Bounded explanation; at most 512 UTF-8 bytes in this interpretation.
    pub reason: String,
}

/// One axis's declared policy and its observation, absent only when the axis is inactive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationAxis {
    /// Applied policy, retained for replay independently of live configuration.
    pub policy: crate::VerificationPolicy,
    /// Facts for an active axis.
    pub observation: Option<VerificationObservation>,
}

/// Facts used to reconstruct the verifier without rerunning any oracle or evidence source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationInterpretation {
    /// Semantic revision, independent of the training-record schema version.
    pub version: u32,
    /// The existing reasoning-present requirement.
    pub reasoning: VerificationAxis,
    /// Answer correctness independent of execution.
    pub answer: VerificationAxis,
    /// Candidate execution with task-declared coverage.
    pub execution: VerificationAxis,
}

impl VerificationInterpretation {
    /// Reconstruct the admission gate from typed facts and applied policy.
    ///
    /// # Errors
    /// Rejects unsupported revisions, missing/unexpected observations, or unbounded reasons.
    pub fn gate(&self) -> Result<(bool, Option<String>), &'static str> {
        use crate::VerificationPolicy::{Absent, Authoritative};
        if self.version != VERIFICATION_INTERPRETATION_VERSION {
            return Err("unsupported verification interpretation version");
        }
        let mut hard_fail = false;
        let mut unknown = None;
        for (name, axis) in [
            ("reasoning", &self.reasoning),
            ("answer", &self.answer),
            ("execution", &self.execution),
        ] {
            match (axis.policy, &axis.observation) {
                (Absent, None) => (),
                (Absent, Some(_)) | (_, None) => {
                    return Err("verification policy and observation are inconsistent");
                }
                (policy, Some(observation)) => {
                    if observation.reason.len() > 512 {
                        return Err("verification reason exceeds interpretation limit");
                    }
                    if policy == Authoritative {
                        hard_fail |= observation.outcome == VerificationOutcome::Fail;
                        if observation.outcome == VerificationOutcome::Unknown {
                            unknown = Some(format!("{name}: {}", observation.reason));
                        }
                    }
                }
            }
        }
        Ok((!hard_fail, if hard_fail { None } else { unknown }))
    }
}
