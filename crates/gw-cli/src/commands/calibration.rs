//! Offline evidence intake; this handler never constructs a provider or opens a database.
use crate::{CommandOutcome, cli::FitCalibrationArgs, config::Config};
use anyhow::Context;
use gw_schema::{CalibrationEvidence, CalibrationStatus, TrainingRecord};

/// Resolve the production panel, validate supplied evidence, and emit a sealed descriptive report.
/// ComputedUnqualified exits zero; incomplete evidence exits two; invalid evidence exits one.
///
/// # Errors
/// Propagates file/configuration/JSON errors or semantic invalidity after printing its typed report.
pub fn fit(args: FitCalibrationArgs) -> anyhow::Result<CommandOutcome> {
    std::fs::metadata(&args.config).context("reading production panel configuration")?;
    let config = Config::load(Some(&args.config))?;
    let area = config.area_config();
    let panel =
        gw_judge::ResolvedCalibrationPanel::new(&area.training_area, &area.judges, &area.rubric)?;
    let records: Vec<TrainingRecord> =
        serde_json::from_slice(&std::fs::read(&args.records).context("reading candidate records")?)
            .context("parsing candidate records")?;
    let evidence: CalibrationEvidence = serde_json::from_slice(
        &std::fs::read(&args.evidence).context("reading calibration evidence")?,
    )
    .context("parsing exact calibration evidence")?;
    let report = gw_judge::fit_calibration(&panel, &records, &evidence);
    println!("{}", serde_json::to_string_pretty(&report)?);
    match report.status {
        CalibrationStatus::ComputedUnqualified => Ok(CommandOutcome::Success),
        CalibrationStatus::IncompleteEvidence => Ok(CommandOutcome::GateRejected),
        CalibrationStatus::InvalidEvidence => {
            anyhow::bail!("invalid calibration evidence; see report reasons")
        }
    }
}
