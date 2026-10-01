//! Local supplied-file lexical planning; no configuration, database, or provider is opened.
use crate::{CommandOutcome, cli::ScreeningArgs};
use anyhow::Context;
use gw_schema::{
    FrozenScreeningPlan, LexicalScreeningStatus, ProtectedScreeningSet, ScreeningDeclaration,
    TrainingRecord,
};

/// Prepare or exactly verify a frozen plan from supplied files. Zero means complete lexical
/// coverage with no quarantined components; two means incomplete/quarantined; invalid input is one.
///
/// # Errors
/// Rejects invalid files/declarations, stale plans, or contradictory identities.
pub fn screen(args: ScreeningArgs) -> anyhow::Result<CommandOutcome> {
    let records: Vec<TrainingRecord> =
        serde_json::from_slice(&std::fs::read(&args.records).context("reading screening records")?)
            .context("parsing ordinary records")?;
    let declaration: ScreeningDeclaration = serde_json::from_slice(
        &std::fs::read(&args.declaration).context("reading screening declaration")?,
    )
    .context("parsing strict screening declaration")?;
    let protected: Vec<ProtectedScreeningSet> = serde_json::from_slice(
        &std::fs::read(&args.protected).context("reading local protected contents")?,
    )
    .context("parsing strict protected manifests")?;
    let read_plan = |path: &std::path::Path| -> anyhow::Result<FrozenScreeningPlan> {
        serde_json::from_slice(&std::fs::read(path).context("reading supplied frozen plan")?)
            .context("parsing supplied frozen plan")
    };
    let previous = args.previous.as_deref().map(read_plan).transpose()?;
    let check = args.check_plan.as_deref().map(read_plan).transpose()?;
    let prior = check
        .as_ref()
        .and_then(|plan| plan.previous.as_deref())
        .or(previous.as_ref());
    let plan = gw_eval::screening::prepare_screening(&records, &declaration, &protected, prior)?;
    if let Some(check) = check {
        anyhow::ensure!(
            serde_json::to_vec(&check)? == serde_json::to_vec(&plan)?,
            "stale or altered screening plan"
        );
    }
    println!("{}", serde_json::to_string_pretty(&plan)?);
    if plan.lexical_status == LexicalScreeningStatus::CompleteNoMatch
        && plan.groups.iter().all(|group| !group.quarantined)
    {
        Ok(CommandOutcome::Success)
    } else {
        Ok(CommandOutcome::GateRejected)
    }
}
