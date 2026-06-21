//! `verification{}` — the deterministic Verifier rail (DATA-SCHEMA §1.6).

use serde::{Deserialize, Serialize};

/// Results of the deterministic checks. `all_passed` is the binary hard gate; a `false`
/// `ReasoningPresent`/`Decontam`/etc. check sinks the record.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Verification {
    #[serde(default)]
    pub checks: Vec<Check>,
    pub all_passed: bool,
}

/// One deterministic check result, e.g. `"rust_compiles"`, `"json_valid"`,
/// `"reasoning_present"`, `"decontam"`, `"language_consistency"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Check {
    pub name: String,
    pub kind: CheckKind,
    pub passed: bool,
    /// OPTIONAL partial-credit fraction in `[0,1]`; None for binary checks (B11). AUDIT +
    /// revise-routing trigger ONLY; does NOT demote the Accept/Reject hard gate (`passed`),
    /// and MUST NOT update Glicko-2/calibration except at the extremes 1.0/0.0.
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
