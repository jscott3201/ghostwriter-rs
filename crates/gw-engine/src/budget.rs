//! The cost meter + budget gate (ARCHITECTURE §3.5, D-CONFIG; INVARIANT 13).
//!
//! The budget cap (`cap_usd`) is the PRIMARY spend guard for a run. This module owns a thread-safe
//! cumulative cost meter: every teacher/judge spend is charged via [`BudgetMeter::charge`], and the
//! executor consults [`BudgetMeter::may_dispatch`] BEFORE pulling new work. Once the cap is reached
//! the gate closes — no NEW teacher work is dispatched — implementing the `Drain` breach (let
//! in-flight finish + persist, then halt). The `Abort` breach is the executor's to wire on top (it
//! cancels the per-job [`CancellationToken`](tokio_util::sync::CancellationToken)); the meter itself
//! only reports whether the cap has tripped.
//!
//! ## Why charge-after-spend, gate-before-dispatch
//!
//! OpenRouter returns the authoritative per-generation `cost.usd` on the response, so the true spend
//! is only known AFTER a call. The meter therefore CHARGES post-spend (so the running total is exact)
//! and GATES pre-dispatch (so once the total crosses the cap, the next record is never started). A
//! record already in flight when the cap trips is allowed to finish and persist (Drain) — its spend
//! is charged and may push the total past the cap, which is expected and bounded by `max_in_flight`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A shared, thread-safe cumulative USD cost meter with a hard cap.
///
/// Cheap to clone (`Arc`-backed), so every spawned shard worker shares ONE meter — the cap is
/// run-wide. The total is stored as `f64` bits in an [`AtomicU64`] and updated with a
/// compare-and-swap loop, so concurrent charges from many workers compose exactly without a lock.
#[derive(Debug, Clone)]
pub struct BudgetMeter {
    spent_bits: Arc<AtomicU64>,
    cap_usd: f64,
}

impl BudgetMeter {
    /// A meter with the given run-wide cap. A non-positive cap is treated as "unbounded" only when
    /// explicitly `f64::INFINITY`; a `cap_usd <= 0.0` cap means "spend nothing" and closes the gate
    /// immediately (a defensive read of a misconfigured zero cap).
    #[must_use]
    pub fn new(cap_usd: f64) -> Self {
        Self {
            spent_bits: Arc::new(AtomicU64::new(0.0f64.to_bits())),
            cap_usd,
        }
    }

    /// The configured run-wide cap in USD.
    #[must_use]
    pub fn cap(&self) -> f64 {
        self.cap_usd
    }

    /// The cumulative USD charged so far.
    #[must_use]
    pub fn spent(&self) -> f64 {
        f64::from_bits(self.spent_bits.load(Ordering::Acquire))
    }

    /// Charge `usd` against the meter and return the new cumulative total. A non-finite or negative
    /// `usd` is clamped to `0.0` (a missing/garbled cost must never CREDIT the meter). Lock-free CAS
    /// loop so concurrent charges from many shard workers compose exactly.
    pub fn charge(&self, usd: f64) -> f64 {
        let add = if usd.is_finite() && usd > 0.0 {
            usd
        } else {
            0.0
        };
        let mut current = self.spent_bits.load(Ordering::Acquire);
        loop {
            let next = f64::from_bits(current) + add;
            match self.spent_bits.compare_exchange_weak(
                current,
                next.to_bits(),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return next,
                Err(observed) => current = observed,
            }
        }
    }

    /// SET the cumulative spend to exactly `total` (clamped to `>= 0.0`), discarding the prior value.
    ///
    /// Used to REHYDRATE the meter at the start of a run from the spend already persisted on a prior
    /// (possibly crashed) launch (E3): an in-memory meter resets to 0 each process, so without this a
    /// restart would re-grant the FULL cap and a resumed run could spend up to `N × cap` across N
    /// launches. Setting (not adding) makes this idempotent across BOTH an in-process re-run (the same
    /// meter) and a cross-process restart (a fresh meter). Call ONCE, before dispatching work, while no
    /// charge can race it.
    pub fn reset_to(&self, total: f64) {
        let v = if total.is_finite() && total > 0.0 {
            total
        } else {
            0.0
        };
        self.spent_bits.store(v.to_bits(), Ordering::Release);
    }

    /// `true` when the cap has been REACHED (spent `>=` cap) — the gate is closed. The executor
    /// consults this before dispatching new work; a closed gate means "drain, do not start more".
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.spent() >= self.cap_usd
    }

    /// `true` when new teacher work MAY be dispatched (the cap has not been reached). The inverse of
    /// [`is_exhausted`](BudgetMeter::is_exhausted), named for the dispatch site so the guard reads
    /// affirmatively at the call.
    #[must_use]
    pub fn may_dispatch(&self) -> bool {
        !self.is_exhausted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_accumulates_and_gate_closes_at_cap() {
        let m = BudgetMeter::new(1.0);
        assert!(m.may_dispatch());
        assert_eq!(m.charge(0.4), 0.4);
        assert!(m.may_dispatch());
        assert_eq!(m.charge(0.6), 1.0);
        // Exactly at cap → exhausted (>=), gate closed.
        assert!(m.is_exhausted());
        assert!(!m.may_dispatch());
    }

    #[test]
    fn over_cap_stays_closed() {
        let m = BudgetMeter::new(0.5);
        m.charge(0.9);
        assert!(m.is_exhausted());
        assert!((m.spent() - 0.9).abs() < 1e-12);
    }

    #[test]
    fn negative_or_nan_cost_never_credits() {
        let m = BudgetMeter::new(10.0);
        m.charge(1.0);
        // A garbled/missing cost must not subtract from or corrupt the running total.
        assert_eq!(m.charge(-5.0), 1.0);
        assert_eq!(m.charge(f64::NAN), 1.0);
        assert_eq!(m.charge(f64::INFINITY), 1.0);
        assert!((m.spent() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_cap_closes_immediately() {
        let m = BudgetMeter::new(0.0);
        // A zero cap means spend nothing — the gate is closed from the start.
        assert!(m.is_exhausted());
        assert!(!m.may_dispatch());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_charges_compose_exactly() {
        let m = BudgetMeter::new(f64::INFINITY);
        let mut handles = Vec::new();
        for _ in 0..8 {
            let m = m.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..1000 {
                    m.charge(0.001);
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        // 8 * 1000 * 0.001 = 8.0, exactly (within fp tolerance).
        assert!((m.spent() - 8.0).abs() < 1e-6, "got {}", m.spent());
    }
}
