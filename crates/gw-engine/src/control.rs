//! Shared run-control policy passed through the executor's cooperative boundaries.

use gw_schema::BudgetBreach;
use tokio_util::sync::CancellationToken;

/// A new generation transition may be interrupted before dispatch without becoming a record fault.
pub(crate) enum GenerationOutcome<T> {
    /// The started generation completed and its record was persisted.
    Generated(T),
    /// Cancellation prevented dispatch; there is no new record or provider fault.
    Interrupted,
}

/// The cancellation token plus budget-breach policy for one engine run.
#[derive(Debug, Clone, Copy)]
pub struct RunControl<'a> {
    cancel: &'a CancellationToken,
    on_breach: BudgetBreach,
}

impl<'a> RunControl<'a> {
    /// Build run-control context from the shared cancellation token and breach policy.
    #[must_use]
    pub fn new(cancel: &'a CancellationToken, on_breach: BudgetBreach) -> Self {
        Self { cancel, on_breach }
    }

    /// Return the shared cancellation token.
    #[must_use]
    pub fn token(&self) -> &'a CancellationToken {
        self.cancel
    }

    /// Return the configured budget-breach behavior.
    #[must_use]
    pub fn on_breach(&self) -> BudgetBreach {
        self.on_breach
    }

    /// Return whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Request cancellation for every shard sharing this run token.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}
