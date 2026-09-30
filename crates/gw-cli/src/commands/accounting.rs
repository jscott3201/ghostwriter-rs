//! Terminal reports always refresh SQLite after engine settlement; events are advisory.
use gw_schema::{AccountingPolicy, AccountingSnapshot, PolicyState, TokenEvidence};
use gw_storage::Store;

pub(crate) async fn terminal(
    store: &Store,
    run_id: &str,
    requested: &AccountingPolicy,
    report: Option<&gw_engine::RunReport>,
) {
    if let Some(report) = report {
        println!(
            "run {run_id} {} — admitted {}, exported {}, rejected {}, needs_review {}, revising {}, errored {}, pending items {}",
            if report.completed {
                "completed"
            } else {
                "halted"
            },
            report.admitted,
            report.exported,
            report.rejected,
            report.needs_review,
            report.revising,
            report.errored,
            report.pending_items
        );
        if let Some(reason) = &report.halted_reason {
            println!("  request admission: {reason}");
        }
    } else {
        println!("run {run_id} failed; accounting below is best effort");
    }
    println!("  requested policy: {}", policy(requested));
    match store.accounting_snapshot(run_id).await {
        Ok(mut snapshot) => {
            snapshot.configured = report
                .and_then(|r| r.accounting.as_ref())
                .and_then(|s| s.configured.clone());
            println!("{}", render(&snapshot));
        }
        Err(error) => eprintln!("  accounting unavailable: {error}"),
    }
}
fn policy(value: &AccountingPolicy) -> String {
    match value {
        AccountingPolicy::ObservationOnly => "observation_only".into(),
        AccountingPolicy::FiniteUsd { limit_usd } => {
            format!("finite_usd (dispatch threshold ${limit_usd:.4})")
        }
    }
}
fn authority(value: &Option<PolicyState>) -> String {
    value.as_ref().map_or_else(
        || "unknown".into(),
        |state| format!("{} epoch {}", policy(&state.policy), state.epoch),
    )
}
fn tokens(name: &str, evidence: &TokenEvidence) -> String {
    format!(
        "  {name} tokens: known {}; missing {}; invalid {}",
        evidence
            .known
            .map_or_else(|| "overflow".into(), |n| n.to_string()),
        evidence.missing_attempts,
        evidence.invalid_attempts
    )
}
pub(crate) fn render(s: &AccountingSnapshot) -> String {
    format!(
        "  configured: {}\n  effective: {}\n  accounting revision {}; history {:?}; unknown coverage lanes {}\n  physical attempts {}; known USD {}; unknown cost {}; invalid cost {}; conflicting {}; unresolved {}\n{}\n{}\n{}\n{}\n  client wall time: {} ms (not GPU time); token categories may overlap",
        authority(&s.configured),
        authority(&s.effective),
        s.revision,
        s.history,
        s.unknown_coverage_lanes,
        s.attempts,
        s.known_usd
            .map_or_else(|| "overflow".into(), |n| format!("${n:.4}")),
        s.unknown_cost_attempts,
        s.invalid_cost_attempts,
        s.conflicting_attempts,
        s.unresolved_attempts,
        tokens("prompt/input", &s.prompt_tokens),
        tokens("completion/output", &s.completion_tokens),
        tokens("reported total", &s.total_tokens),
        tokens("reasoning", &s.reasoning_tokens),
        s.elapsed_ms
            .map_or_else(|| "overflow".into(), |n| n.to_string())
    )
}
