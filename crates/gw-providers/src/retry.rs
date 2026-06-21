//! Hand-rolled exponential-backoff retry — no extra dependency.
//!
//! [`retry`] runs an async operation and retries it on retryable [`ProviderError`]s (HTTP 429,
//! HTTP 5xx, transport faults, stream resets — see [`ProviderError::is_retryable`]). Backoff is
//! exponential from `base` (doubling), capped at `max_delay`, with a hard `max_attempts`. When
//! the error carries a `Retry-After` (HTTP 429), that hint wins over the computed backoff.
//!
//! The sleep is injected (`sleeper`) so tests assert the *schedule* without real waiting; the
//! production [`retry`] entry point wires in [`tokio::time::sleep`].

use std::future::Future;
use std::time::Duration;

use crate::error::ProviderError;

/// Backoff policy for [`retry_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total attempts including the first try. `1` ⇒ no retries.
    pub max_attempts: u32,
    /// Base delay before the first retry; doubles each subsequent retry.
    pub base_delay: Duration,
    /// Upper bound on any single backoff delay.
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
        }
    }
}

impl RetryPolicy {
    /// The backoff delay before retry number `retry_index` (0-based: 0 is the first retry),
    /// honoring an optional server `Retry-After` hint (which overrides and is itself capped at
    /// `max_delay`). Exponential: `base * 2^retry_index`, saturating, capped at `max_delay`.
    #[must_use]
    pub fn delay_for(&self, retry_index: u32, retry_after: Option<Duration>) -> Duration {
        if let Some(hint) = retry_after {
            return hint.min(self.max_delay);
        }
        let factor = 1u64.checked_shl(retry_index).unwrap_or(u64::MAX);
        let scaled = self
            .base_delay
            .checked_mul(u32::try_from(factor).unwrap_or(u32::MAX))
            .unwrap_or(self.max_delay);
        scaled.min(self.max_delay)
    }
}

/// Run `op` with retries per `policy`, sleeping via `sleeper` between attempts.
///
/// `op` is a closure producing a fresh future per attempt (so the request is rebuilt each
/// try). A non-retryable error returns immediately; a retryable error is retried until
/// `max_attempts` is exhausted, after which the last error is returned.
///
/// # Errors
/// Returns the final [`ProviderError`] if every attempt fails (or the first non-retryable one).
pub async fn retry_with<T, Op, Fut, Sleep, SleepFut>(
    policy: RetryPolicy,
    mut sleeper: Sleep,
    mut op: Op,
) -> Result<T, ProviderError>
where
    Op: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ProviderError>>,
    Sleep: FnMut(Duration) -> SleepFut,
    SleepFut: Future<Output = ()>,
{
    let attempts = policy.max_attempts.max(1);
    let mut last_err: Option<ProviderError> = None;

    for attempt in 0..attempts {
        match op().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                let is_last = attempt + 1 >= attempts;
                if !err.is_retryable() || is_last {
                    return Err(err);
                }
                let delay = policy.delay_for(attempt, err.retry_after());
                last_err = Some(err);
                sleeper(delay).await;
            }
        }
    }
    // Unreachable in practice (the loop returns), but keep a defined result.
    Err(last_err.unwrap_or_else(|| ProviderError::Config("retry: no attempts run".into())))
}

/// Production entry point: [`retry_with`] wired to [`tokio::time::sleep`].
///
/// # Errors
/// Propagates the final [`ProviderError`] from [`retry_with`].
pub async fn retry<T, Op, Fut>(policy: RetryPolicy, op: Op) -> Result<T, ProviderError>
where
    Op: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ProviderError>>,
{
    retry_with(policy, |d| tokio::time::sleep(d), op).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::time::Duration;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn backoff_schedule_doubles_and_caps() {
        let p = RetryPolicy {
            max_attempts: 10,
            base_delay: ms(100),
            max_delay: ms(800),
        };
        assert_eq!(p.delay_for(0, None), ms(100));
        assert_eq!(p.delay_for(1, None), ms(200));
        assert_eq!(p.delay_for(2, None), ms(400));
        assert_eq!(p.delay_for(3, None), ms(800));
        // Capped from here on.
        assert_eq!(p.delay_for(4, None), ms(800));
        assert_eq!(p.delay_for(20, None), ms(800));
    }

    #[test]
    fn retry_after_hint_overrides_and_is_capped() {
        let p = RetryPolicy {
            max_attempts: 5,
            base_delay: ms(100),
            max_delay: ms(5000),
        };
        // Hint within cap is used verbatim, regardless of exponential schedule.
        assert_eq!(p.delay_for(0, Some(ms(2000))), ms(2000));
        // Hint above cap is clamped.
        assert_eq!(p.delay_for(3, Some(ms(99_999))), ms(5000));
    }

    #[tokio::test]
    async fn succeeds_first_try_no_sleep() {
        let slept: RefCell<Vec<Duration>> = RefCell::new(Vec::new());
        let calls = RefCell::new(0u32);
        let out: Result<u8, _> = retry_with(
            RetryPolicy::default(),
            |d| {
                slept.borrow_mut().push(d);
                async {}
            },
            || {
                *calls.borrow_mut() += 1;
                async { Ok(7u8) }
            },
        )
        .await;
        assert_eq!(out.unwrap(), 7);
        assert_eq!(*calls.borrow(), 1);
        assert!(slept.borrow().is_empty());
    }

    #[tokio::test]
    async fn retries_then_succeeds_recording_schedule() {
        let slept: RefCell<Vec<Duration>> = RefCell::new(Vec::new());
        let calls = RefCell::new(0u32);
        let policy = RetryPolicy {
            max_attempts: 5,
            base_delay: ms(100),
            max_delay: ms(10_000),
        };
        let out: Result<&str, _> = retry_with(
            policy,
            |d| {
                slept.borrow_mut().push(d);
                async {}
            },
            || {
                let n = {
                    let mut c = calls.borrow_mut();
                    *c += 1;
                    *c
                };
                async move {
                    if n < 3 {
                        Err(ProviderError::from_status(503, None))
                    } else {
                        Ok("done")
                    }
                }
            },
        )
        .await;
        assert_eq!(out.unwrap(), "done");
        assert_eq!(*calls.borrow(), 3);
        // Two retries → two sleeps with the exponential schedule.
        assert_eq!(*slept.borrow(), vec![ms(100), ms(200)]);
    }

    #[tokio::test]
    async fn non_retryable_returns_immediately() {
        let slept: RefCell<Vec<Duration>> = RefCell::new(Vec::new());
        let calls = RefCell::new(0u32);
        let out: Result<u8, _> = retry_with(
            RetryPolicy::default(),
            |d| {
                slept.borrow_mut().push(d);
                async {}
            },
            || {
                *calls.borrow_mut() += 1;
                async { Err::<u8, _>(ProviderError::from_status(400, None)) }
            },
        )
        .await;
        assert!(matches!(
            out,
            Err(ProviderError::Status { status: 400, .. })
        ));
        assert_eq!(*calls.borrow(), 1);
        assert!(slept.borrow().is_empty());
    }

    #[tokio::test]
    async fn exhausts_attempts_and_returns_last_error() {
        let calls = RefCell::new(0u32);
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: ms(10),
            max_delay: ms(100),
        };
        let out: Result<u8, _> = retry_with(
            policy,
            |_d| async {},
            || {
                *calls.borrow_mut() += 1;
                async { Err::<u8, _>(ProviderError::Transport("eof".into())) }
            },
        )
        .await;
        assert!(matches!(out, Err(ProviderError::Transport(_))));
        // Exactly max_attempts operation invocations.
        assert_eq!(*calls.borrow(), 3);
    }

    #[tokio::test]
    async fn rate_limited_uses_retry_after_in_schedule() {
        let slept: RefCell<Vec<Duration>> = RefCell::new(Vec::new());
        let calls = RefCell::new(0u32);
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: ms(50),
            max_delay: ms(10_000),
        };
        let _ = retry_with(
            policy,
            |d| {
                slept.borrow_mut().push(d);
                async {}
            },
            || {
                *calls.borrow_mut() += 1;
                async {
                    Err::<u8, _>(ProviderError::RateLimited {
                        retry_after: Some(ms(1234)),
                    })
                }
            },
        )
        .await;
        // Both retries honor the server Retry-After, not the exponential base.
        assert_eq!(*slept.borrow(), vec![ms(1234), ms(1234)]);
    }
}
