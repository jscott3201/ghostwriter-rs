//! Operational request admission and durable run evidence, independent of generation identity.
use crate::AccountingHistory;
use serde::{Deserialize, Serialize};

/// Explicit monetary admission policy. A finite limit controls dispatch, not the final invoice.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum AccountingPolicy {
    /// Record evidence while retaining ordinary bounded concurrency, irrespective of prices.
    ObservationOnly,
    /// Serialize physical requests and admit only while complete known spend is below the limit.
    FiniteUsd {
        /// Finite, nonnegative dispatch threshold in reported US dollars.
        limit_usd: f64,
    },
}
impl<'de> Deserialize<'de> for AccountingPolicy {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Empty struct variant is intentional: serde's internally tagged unit variants otherwise
        // ignore extra fields, even with deny_unknown_fields on the enum.
        #[derive(Deserialize)]
        #[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            ObservationOnly {},
            FiniteUsd { limit_usd: f64 },
        }
        let policy = match Wire::deserialize(deserializer)? {
            Wire::ObservationOnly {} => Self::ObservationOnly,
            Wire::FiniteUsd { limit_usd } => Self::FiniteUsd { limit_usd },
        };
        if !policy.is_valid() {
            return Err(serde::de::Error::custom(
                "finite_usd limit_usd must be finite and nonnegative",
            ));
        }
        Ok(policy)
    }
}
impl Default for AccountingPolicy {
    fn default() -> Self {
        Self::FiniteUsd { limit_usd: 5.0 }
    }
}
impl AccountingPolicy {
    /// Whether the policy's numerical domain is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        match self {
            Self::ObservationOnly => true,
            Self::FiniteUsd { limit_usd } => limit_usd.is_finite() && *limit_usd >= 0.0,
        }
    }
}

/// Durable operational authority captured by a launch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyState {
    /// Policy storage contract version, currently 1.
    pub version: u32,
    /// Monotonically increasing authority epoch within one run.
    pub epoch: u64,
    /// Policy effective in this epoch.
    pub policy: AccountingPolicy,
}

/// Why a still-needed physical request was not admitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum AdmissionDenial {
    /// A direct engine helper lacks a registered launch context.
    UnregisteredContext,
    /// A newer operational policy superseded the caller's epoch.
    PolicySuperseded,
    /// Historical execution cannot be certified from run creation.
    IncompleteHistory,
    /// At least one injected model lane has unknown physical-request coverage.
    UnknownCoverage,
    /// An unresolved intent is not owned by this live coordinator.
    UnresolvedAttempts,
    /// A settled request has no reported dollar cost.
    UnknownCost,
    /// Invalid, contradictory, or overflowing accounting evidence prevents admission.
    InvalidEvidence,
    /// The known reported subtotal reached the selected threshold.
    LimitReached {
        /// Known reported US dollars before the denied send.
        known_usd: f64,
        /// Configured US-dollar dispatch threshold.
        limit_usd: f64,
    },
}
impl std::fmt::Display for AdmissionDenial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnregisteredContext => {
                write!(f, "model dispatch requires registered launch context")
            }
            Self::PolicySuperseded => write!(f, "launch policy was superseded"),
            Self::IncompleteHistory => write!(f, "historical accounting is incomplete"),
            Self::UnknownCoverage => write!(f, "model client accounting coverage is unknown"),
            Self::UnresolvedAttempts => write!(
                f,
                "unresolved request is not owned by this live coordinator"
            ),
            Self::UnknownCost => write!(f, "settled request has unknown cost"),
            Self::InvalidEvidence => write!(
                f,
                "accounting evidence is invalid, conflicting, or overflowing"
            ),
            Self::LimitReached {
                known_usd,
                limit_usd,
            } => write!(
                f,
                "known spend ${known_usd:.4} reached dispatch threshold ${limit_usd:.4}"
            ),
        }
    }
}

/// Known token subtotal and its evidence limits. Token categories must not be added together.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenEvidence {
    /// Sum of valid reported values, or absent if the sum overflowed.
    pub known: Option<u64>,
    /// Attempts without this measurement.
    pub missing_attempts: u64,
    /// Attempts reporting a malformed value for this field.
    pub invalid_attempts: u64,
}
impl Default for TokenEvidence {
    fn default() -> Self {
        Self {
            known: Some(0),
            missing_attempts: 0,
            invalid_attempts: 0,
        }
    }
}

/// Absolute durable accounting snapshot. A newer revision may correct a subtotal downward.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountingSnapshot {
    /// Durable ordering within this run; do not order snapshots by their monetary value.
    pub revision: u64,
    /// Launch's configured policy/epoch when supplied by its coordinator.
    pub configured: Option<PolicyState>,
    /// Currently effective durable policy, which may have superseded the launch.
    pub effective: Option<PolicyState>,
    /// Whether run creation established complete historical coverage.
    pub history: AccountingHistory,
    /// Number of historical launch lanes with unknown coverage.
    pub unknown_coverage_lanes: u64,
    /// Physical attempt count, including unsuccessful outputs.
    pub attempts: u64,
    /// Known valid reported dollars, absent on arithmetic overflow.
    pub known_usd: Option<f64>,
    /// Attempts whose cost was omitted.
    pub unknown_cost_attempts: u64,
    /// Attempts whose cost was present but invalid.
    pub invalid_cost_attempts: u64,
    /// Attempts with contradictory accounting evidence.
    pub conflicting_attempts: u64,
    /// Attempts without durable transport settlement.
    pub unresolved_attempts: u64,
    /// Reported prompt/input tokens.
    pub prompt_tokens: TokenEvidence,
    /// Reported completion/output tokens.
    pub completion_tokens: TokenEvidence,
    /// Reported total tokens (not an additional category to sum with input/output).
    pub total_tokens: TokenEvidence,
    /// Reported reasoning tokens (may be a subset of completion tokens).
    pub reasoning_tokens: TokenEvidence,
    /// Sum of settled client wall-clock milliseconds, not GPU time; absent on overflow.
    pub elapsed_ms: Option<u64>,
}
