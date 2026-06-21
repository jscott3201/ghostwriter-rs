//! [`EngineError`] — the single typed error surfaced by every `gw-engine` operation.
//!
//! `gw-engine` is the convergence point of the pipeline, so its error type wraps every dependency
//! crate's typed error (`gw-generate`, `gw-judge`, `gw-storage`) plus the engine's own orchestration
//! faults. The wrapped variants carry their source via `#[from]`, so a caller can match on the
//! original error class (and consult retryability where the source preserves it). No `anyhow` — this
//! crate surfaces a typed error like the sibling crates.
//!
//! ## The unhappy path lives here
//!
//! `gw-engine` IS the unhappy path of the system. Every transition that could fail surfaces a
//! variant here rather than swallowing the fault: a budget breach
//! ([`BudgetExceeded`](EngineError::BudgetExceeded)) and a violated orchestration invariant
//! ([`Invariant`](EngineError::Invariant), e.g. an identity correlation matrix handed to a `k > 1`
//! panel grade, which is fail-loud by contract) are explicit, catchable errors. Cancellation is normal
//! run-control and returns `Ok(RunReport { completed: false, .. })`.

use thiserror::Error;

use gw_generate::GenerateError;
use gw_judge::JudgeError;
use gw_providers::ProviderError;
use gw_storage::StorageError;

/// Everything that can go wrong driving a record through the pipeline.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change. `Generate`, `Judge`,
/// and `Storage` carry their underlying source via `#[from]`; the rest are constructed directly with
/// a human-readable message. No secret (API key) is ever placed in a variant — the wrapped errors
/// inherit their crates' safe-to-log stance.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EngineError {
    /// A producer call (user-turn synthesis or assistant generation) failed. Carries the
    /// [`GenerateError`] verbatim so the engine's caller can consult the underlying
    /// [`ProviderError`] retryability.
    #[error("generate error: {0}")]
    Generate(#[from] GenerateError),

    /// A grading call (verifier or judge panel) or the consensus math failed. Carries the
    /// [`JudgeError`] verbatim.
    #[error("judge error: {0}")]
    Judge(#[from] JudgeError),

    /// A persistence operation (record `put` / `advance_lifecycle` / cache / checkpoint) failed.
    /// Carries the [`StorageError`] verbatim. A persistence fault is load-bearing — the engine never
    /// proceeds to the next transition on a failed persist (persist-after-every-transition).
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// A rendering / projection fault from the format layer (Formatted / Exported stages).
    #[error("format error: {0}")]
    Format(#[from] gw_format::FormatError),

    /// (De)serializing an engine-owned envelope (a resume cursor, a sibling-group id) failed.
    #[error("serde_json error: {0}")]
    Serde(#[from] serde_json::Error),

    /// The run-wide budget cap (`cap_usd`) was reached, so no new teacher work may be dispatched
    /// (ARCHITECTURE §3.5, the `Drain`/`Abort` breach). Carries the spent total for the run log.
    #[error("budget cap reached: spent ${spent:.4} of ${cap:.4}")]
    BudgetExceeded {
        /// The total USD spent at the point the cap tripped.
        spent: f64,
        /// The configured cap.
        cap: f64,
    },

    /// An orchestration INVARIANT was violated — a programmer/config fault surfaced LOUD at the seam
    /// rather than silently corrupting a record. The load-bearing cases: an identity correlation
    /// matrix handed to a `k > 1` panel grade (which would degrade the correlation guard to
    /// Kish-only), a step driven from a state it cannot advance from, or a sibling group with no
    /// admissible member while one was expected. Terminal — never retryable.
    #[error("engine invariant violated: {0}")]
    Invariant(String),

    /// A RECORD-SCOPED wrapper that attributes an underlying error to the SPECIFIC record it struck
    /// (F1). The best-of-k group drives siblings sequentially, so a fault on a LATER sibling (e.g.
    /// `c2`) must be attributed to THAT sibling's id — never blindly to `c0`, which may be a healthy
    /// already-judged record. `crate::run_group` wraps a record-level fault with the faulting
    /// `record_id` so the shard parks the RIGHT record at `Error` (see [`Self::attributed_record`]).
    /// Classification ([`Self::is_record_level`]) and any inner-provider inspection delegate to the
    /// wrapped `source`, so this wrapper is transparent to the fatal/record-level taxonomy.
    #[error("{source}")]
    Record {
        /// The id of the record the wrapped error is attributed to.
        record_id: String,
        /// The underlying error.
        source: Box<EngineError>,
    },
}

impl EngineError {
    /// `true` when this error is RECORD-LEVEL — a fault that pertains to ONE record's content and must
    /// NOT abort the whole run (E5). The shard parks the faulting record at
    /// [`LifecycleState::Error`](gw_schema::LifecycleState::Error), emits
    /// [`RecordErrored`](crate::EngineEvent::RecordErrored), and CONTINUES with the next item.
    ///
    /// Record-level: `Generate` / `Judge` carrying a CONTENT fault (truncated CoT, empty completion, a
    /// failed QC gate, a judge-parse fault, an empty panel, a one-off provider decode), and `Format` (a
    /// render/projection fault on this record's content). These are isolated to the one record.
    ///
    /// NOT record-level (INFRASTRUCTURE — fatal to the run): `Storage` (the data plane is down — every
    /// record would fail the same way), `Serde` (an engine-owned envelope is corrupt), `BudgetExceeded`
    /// (run-level control flow, handled separately), and `Invariant` (a programmer/config
    /// bug — fail loud rather than silently park record after record).
    ///
    /// SYSTEMIC PROVIDER FAULTS are the subtle case (F2/H-B). A `Generate`/`Judge` error WRAPS a
    /// [`ProviderError`]; a non-retryable AUTH/CONFIG-class provider fault — an invalid/revoked key
    /// (HTTP 401/403/407), a `MissingApiKey`, or a `Config` fault — is SYSTEMIC: it would fail EVERY
    /// record identically. Classifying it record-level would park the entire seed space at `Error` and
    /// laundered a misconfiguration into N wasted teacher passes that "complete". So those are treated
    /// as INFRASTRUCTURE (fatal-fast), while genuine per-record content faults stay record-level.
    #[must_use]
    pub fn is_record_level(&self) -> bool {
        match self {
            // A record-scoped wrapper is transparent: classify by the underlying error (F1).
            EngineError::Record { source, .. } => source.is_record_level(),
            // A judge DATA-PLANE fault (the never-re-spend cache / store) is INFRASTRUCTURE — every
            // record would fail the same way — symmetric with the top-level `Storage` arm (and unlike a
            // judge CONTENT fault). Matched BEFORE the Generate|Judge record-level fallthrough so a
            // recoverable store outage on the judge path is not laundered into per-record `Error`s.
            EngineError::Judge(JudgeError::Storage(_)) => false,
            // Generate/Judge are record-level UNLESS they wrap a SYSTEMIC (non-retryable auth/config)
            // provider fault, which would fail every record the same way → infrastructure-fatal (F2).
            EngineError::Generate(_) | EngineError::Judge(_) => !self.is_systemic_provider_fault(),
            EngineError::Format(_) => true,
            EngineError::Storage(_)
            | EngineError::Serde(_)
            | EngineError::BudgetExceeded { .. }
            | EngineError::Invariant(_) => false,
        }
    }

    /// `true` when this error wraps a SYSTEMIC provider fault — a misconfiguration or auth failure that
    /// would fail EVERY record identically, NOT a one-off per-record content fault (F2/H-B). The
    /// reachable runtime trigger is an invalid/revoked API key: the provider BUILDS fine (the builder
    /// only checks key PRESENCE) but the server returns a non-retryable `401`/`403`/`407` on every
    /// call, surfaced as `ProviderError::Status { retryable: false, .. }`. `MissingApiKey` and `Config`
    /// are construction-time faults included here as defense-in-depth. A RETRYABLE provider fault
    /// (transport, 429, 5xx, stream reset) is NOT systemic-fatal — it is transient and isolatable; and
    /// a non-retryable `Decode` is a per-record malformed payload, left record-level.
    #[must_use]
    fn is_systemic_provider_fault(&self) -> bool {
        let Some(pe) = self.provider_source() else {
            return false;
        };
        match pe {
            // Construction-time misconfiguration (defense-in-depth — unreachable at the call site, but
            // unambiguously systemic if it ever surfaces).
            ProviderError::MissingApiKey(_) | ProviderError::Config(_) => true,
            // The reachable trigger: a non-retryable AUTH status (invalid/revoked key) returned every
            // call. 401 Unauthorized, 403 Forbidden, 407 Proxy Authentication Required.
            ProviderError::Status {
                status,
                retryable: false,
                ..
            } => matches!(status, 401 | 403 | 407),
            // Retryable status / transport / 429 / stream reset → transient, not systemic.
            // Decode → per-record malformed payload, stays record-level.
            _ => false,
        }
    }

    /// The inner [`ProviderError`] this error wraps, if any — reached through the `Generate`/`Judge`
    /// `Provider` variant (both carry it via `#[from]`), unwrapping a record-scoped [`Self::Record`]
    /// wrapper first. Returns `None` for errors that do not originate at a provider call.
    #[must_use]
    fn provider_source(&self) -> Option<&ProviderError> {
        match self {
            EngineError::Record { source, .. } => source.provider_source(),
            EngineError::Generate(GenerateError::Provider(pe)) => Some(pe),
            EngineError::Judge(JudgeError::Provider(pe)) => Some(pe),
            _ => None,
        }
    }

    /// Attribute this error to a specific record id (F1): wrap it in [`Self::Record`] so the shard's
    /// error-park targets the ACTUAL faulting sibling, not a blindly-assumed `c0`. Idempotent — an
    /// error already attributed is returned unchanged (the innermost attribution wins, so a fault
    /// re-wrapped while unwinding keeps the id closest to the failure site).
    #[must_use]
    pub fn attribute_to(self, record_id: &str) -> Self {
        if matches!(self, EngineError::Record { .. }) {
            return self;
        }
        EngineError::Record {
            record_id: record_id.to_string(),
            source: Box::new(self),
        }
    }

    /// The record id this error was attributed to via [`Self::attribute_to`] (F1), or `None` if it was
    /// never record-scoped. The shard reads this to park the RIGHT record at `Error`; absent an
    /// attribution it falls back to the item's primary id.
    #[must_use]
    pub fn attributed_record(&self) -> Option<&str> {
        match self {
            EngineError::Record { record_id, .. } => Some(record_id),
            _ => None,
        }
    }
}

/// Convenience alias for results returned by `gw-engine` operations.
pub type Result<T> = std::result::Result<T, EngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_exceeded_renders_amounts() {
        let e = EngineError::BudgetExceeded {
            spent: 25.5,
            cap: 25.0,
        };
        let msg = e.to_string();
        assert!(msg.contains("25.5"));
        assert!(msg.contains("25.0"));
    }

    #[test]
    fn generate_error_converts() {
        let ge = GenerateError::Invariant("max_tokens".into());
        let e: EngineError = ge.into();
        assert!(matches!(e, EngineError::Generate(_)));
    }

    #[test]
    fn judge_error_converts() {
        let je = JudgeError::EmptyPanel("no judges".into());
        let e: EngineError = je.into();
        assert!(matches!(e, EngineError::Judge(_)));
    }

    #[test]
    fn invariant_renders_message() {
        let e = EngineError::Invariant("identity R for k>1 panel".into());
        assert!(e.to_string().contains("identity R"));
    }

    // ---- F2/H-B: systemic provider faults are infrastructure-fatal, content faults stay record-level.

    #[test]
    fn non_retryable_auth_status_is_systemic_not_record_level() {
        // An invalid/revoked key: the server returns a non-retryable 401 on EVERY call. This must abort
        // the run, NOT park every record at Error (which would "complete" a misconfigured run).
        for status in [401u16, 403, 407] {
            let pe = ProviderError::Status {
                status,
                retryable: false,
                body: Some("unauthorized".into()),
            };
            let e: EngineError = GenerateError::Provider(pe).into();
            assert!(
                !e.is_record_level(),
                "a non-retryable {status} auth status is systemic-fatal, not record-level"
            );
        }
    }

    #[test]
    fn missing_key_and_config_are_systemic_not_record_level() {
        // Defense-in-depth: construction-time misconfig, if it ever surfaces at the call site, is
        // unambiguously systemic.
        let e: EngineError =
            GenerateError::Provider(ProviderError::MissingApiKey("OPENROUTER_API_KEY".into()))
                .into();
        assert!(!e.is_record_level());
        let e: EngineError =
            GenerateError::Provider(ProviderError::Config("bad base url".into())).into();
        assert!(!e.is_record_level());
        // Same for a judge-side provider fault.
        let e: EngineError =
            JudgeError::Provider(ProviderError::Config("bad base url".into())).into();
        assert!(!e.is_record_level());
    }

    #[test]
    fn retryable_provider_fault_stays_record_level() {
        // A transient transport/429/5xx fault is isolatable per-record — NOT systemic-fatal.
        let e: EngineError =
            GenerateError::Provider(ProviderError::Transport("connreset".into())).into();
        assert!(e.is_record_level());
        let e: EngineError = GenerateError::Provider(ProviderError::from_status(503, None)).into();
        assert!(e.is_record_level());
        let e: EngineError =
            GenerateError::Provider(ProviderError::RateLimited { retry_after: None }).into();
        assert!(e.is_record_level());
    }

    #[test]
    fn content_faults_stay_record_level() {
        // Genuine per-record content faults must remain isolatable (don't regress E5's real purpose).
        let e: EngineError = GenerateError::TruncatedReasoning("len".into()).into();
        assert!(e.is_record_level());
        let e: EngineError = GenerateError::EmptyResponse("empty".into()).into();
        assert!(e.is_record_level());
        let e: EngineError = GenerateError::Invariant("qc gate failed".into()).into();
        assert!(e.is_record_level());
        let e: EngineError = JudgeError::JudgeParse("no score".into()).into();
        assert!(e.is_record_level());
        let e: EngineError = JudgeError::EmptyPanel("no judges".into()).into();
        assert!(e.is_record_level());
        // A non-429, non-auth, non-retryable provider Decode is a per-record malformed payload.
        let e: EngineError =
            GenerateError::Provider(ProviderError::Decode("bad json".into())).into();
        assert!(e.is_record_level());
    }

    #[test]
    fn infrastructure_errors_are_not_record_level() {
        assert!(!EngineError::Invariant("x".into()).is_record_level());
        assert!(
            !EngineError::BudgetExceeded {
                spent: 1.0,
                cap: 0.5
            }
            .is_record_level()
        );
    }

    #[test]
    fn judge_storage_fault_is_infrastructure_not_record_level() {
        // A judge DATA-PLANE fault (the never-re-spend cache / store) is infrastructure — symmetric with
        // the top-level Storage arm — so a recoverable store outage on the judge path aborts the run
        // rather than being laundered into a per-record Error on every subsequent record.
        let e: EngineError = JudgeError::Storage(StorageError::NotFound("rec".into())).into();
        assert!(!e.is_record_level());
        // A judge CONTENT fault stays record-level (isolated to the one record).
        let e: EngineError = JudgeError::JudgeParse("no score".into()).into();
        assert!(e.is_record_level());
    }

    // ---- F1/H-A: record attribution wrapper threads the faulting id and is classification-transparent.

    #[test]
    fn attribute_to_threads_the_faulting_record_id() {
        let e: EngineError = GenerateError::TruncatedReasoning("len".into()).into();
        let attributed = e.attribute_to("run-1-s0-seed5-a0-c2");
        assert_eq!(
            attributed.attributed_record(),
            Some("run-1-s0-seed5-a0-c2"),
            "the faulting sibling id is recoverable for the error-park"
        );
        // The wrapper is transparent to classification (a record-level content fault stays record-level).
        assert!(attributed.is_record_level());
    }

    #[test]
    fn attribute_to_is_idempotent_innermost_wins() {
        let e: EngineError = GenerateError::EmptyResponse("empty".into()).into();
        let once = e.attribute_to("rid-c2");
        // Re-wrapping while unwinding keeps the FIRST (innermost) attribution, closest to the failure.
        let twice = once.attribute_to("rid-c0");
        assert_eq!(twice.attributed_record(), Some("rid-c2"));
    }

    #[test]
    fn attributed_systemic_fault_is_still_infrastructure_fatal() {
        // Attribution must not launder a systemic fault into a record-level one.
        let e: EngineError = GenerateError::Provider(ProviderError::Status {
            status: 401,
            retryable: false,
            body: None,
        })
        .into();
        let attributed = e.attribute_to("rid-c1");
        assert!(!attributed.is_record_level());
    }

    #[test]
    fn unattributed_error_has_no_record() {
        let e: EngineError = GenerateError::EmptyResponse("empty".into()).into();
        assert_eq!(e.attributed_record(), None);
    }
}
