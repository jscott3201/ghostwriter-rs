//! [`RateLimiter`] — a thin `governor` GCRA wrapper keyed by requests-per-minute.
//!
//! One limiter guards one (provider/model) lane. It is built from a [`gw_schema::ProviderLimits`]
//! `rpm` (or a fallback default) and exposes a single async gate, [`RateLimiter::until_ready`],
//! that resolves once a cell is available — the call site `await`s it immediately before
//! dispatching an HTTP request. This is intentionally a thin, testable shim over governor's
//! direct (non-keyed) GCRA limiter; per-key fan-out is handled one-limiter-per-lane by the
//! caller (the client holds one limiter), keeping this type simple.

use std::num::NonZeroU32;

use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter as GovLimiter};

use gw_schema::ProviderLimits;

/// A GCRA rate limiter gating one provider/model lane at a fixed requests-per-minute budget.
///
/// The burst size equals the per-minute quota, so a fresh limiter admits a full minute's worth
/// of requests immediately, then refills at `rpm / 60` per second (GCRA's smooth replenish).
pub struct RateLimiter {
    inner: GovLimiter<NotKeyed, InMemoryState, DefaultClock>,
    rpm: u32,
}

impl RateLimiter {
    /// Build a limiter from a requests-per-minute budget. An `rpm` of `0` is clamped to `1`
    /// (governor quotas require a non-zero burst), so the limiter never deadlocks.
    #[must_use]
    pub fn per_minute(rpm: u32) -> Self {
        let burst = NonZeroU32::new(rpm.max(1)).expect("clamped to >= 1");
        let quota = Quota::per_minute(burst);
        Self {
            inner: GovLimiter::direct(quota),
            rpm: rpm.max(1),
        }
    }

    /// Build a limiter for a provider lane: use the lane's `rpm` from [`ProviderLimits`], or
    /// `default_rpm` when no per-provider override exists.
    #[must_use]
    pub fn from_limits(limits: Option<&ProviderLimits>, default_rpm: u32) -> Self {
        let rpm = limits.map_or(default_rpm, |l| l.rpm);
        Self::per_minute(rpm)
    }

    /// The effective requests-per-minute budget this limiter enforces (post-clamp).
    #[must_use]
    pub fn rpm(&self) -> u32 {
        self.rpm
    }

    /// Asynchronously wait until a request cell is available, then return. Resolving means the
    /// caller is cleared to dispatch exactly one request.
    pub async fn until_ready(&self) {
        self.inner.until_ready().await;
    }

    /// Non-blocking check: `true` if a cell is available right now (consuming it). Primarily
    /// for tests and fast-path probes.
    #[must_use]
    pub fn try_acquire(&self) -> bool {
        self.inner.check().is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn from_limits_prefers_provider_rpm() {
        let mut map = BTreeMap::new();
        map.insert(
            "parasail".to_string(),
            ProviderLimits {
                endpoint: None,
                rpm: 120,
                tpm: None,
                max_in_flight: None,
            },
        );
        let lim = RateLimiter::from_limits(map.get("parasail"), 60);
        assert_eq!(lim.rpm(), 120);
    }

    #[test]
    fn from_limits_falls_back_to_default() {
        let lim = RateLimiter::from_limits(None, 45);
        assert_eq!(lim.rpm(), 45);
    }

    #[test]
    fn zero_rpm_is_clamped_to_one() {
        let lim = RateLimiter::per_minute(0);
        assert_eq!(lim.rpm(), 1);
        // The first cell is available; the limiter does not deadlock.
        assert!(lim.try_acquire());
    }

    #[test]
    fn burst_then_throttle() {
        // rpm=2 → burst of 2 cells available immediately, third is denied.
        let lim = RateLimiter::per_minute(2);
        assert!(lim.try_acquire());
        assert!(lim.try_acquire());
        assert!(!lim.try_acquire());
    }

    #[tokio::test]
    async fn until_ready_resolves_within_budget() {
        let lim = RateLimiter::per_minute(60);
        // First call is immediate (burst available).
        lim.until_ready().await;
    }
}
