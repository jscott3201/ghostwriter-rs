//! Strict applicability checks before any descriptive fitting. No evidence row is silently dropped.
use crate::calibration_contract::{identity, verdict};
use crate::{JUDGE_INTERPRETATION_VERSION, ResolvedCalibrationPanel};
use gw_schema::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(crate) struct Intake {
    pub invalid: Vec<String>,
    pub incomplete: Vec<String>,
}

impl Intake {
    fn invalid(&mut self, reason: impl Into<String>) {
        self.invalid.push(reason.into());
    }
    fn incomplete(&mut self, reason: impl Into<String>) {
        self.incomplete.push(reason.into());
    }
}

pub(crate) fn validate(
    panel: &ResolvedCalibrationPanel,
    records: &[TrainingRecord],
    evidence: &CalibrationEvidence,
) -> Intake {
    let mut state = Intake::default();
    if evidence.version != CALIBRATION_VERSION
        || evidence.group_map_version != 1
        || evidence.method != CALIBRATION_METHOD
        || evidence.numerical_recipe != CALIBRATION_NUMERICAL_RECIPE
    {
        state.invalid("unsupported evidence, group map, method or numerical recipe version");
    }
    if !same(&evidence.panel, panel.declaration()) || evidence.panel_identity != panel.identity() {
        state.invalid(
            "submitted stable panel does not match the independently resolved production panel",
        );
    }
    if !evidence.beta.is_finite() || evidence.beta < 0.0 {
        state.invalid("beta must be finite and nonnegative");
    }
    let CalibrationCorrelation::AssumedConstantRho { rho } = evidence.correlation;
    if let Err(error) = crate::validate_correlation_prior(panel.len(), rho) {
        state.invalid(error.to_string());
    }
    for (name, value) in [
        ("target name", &evidence.target.name),
        ("target version", &evidence.target.version),
        ("reference protocol", &evidence.target.protocol_revision),
        ("reference source", &evidence.provenance.source),
        (
            "reference artifact digest",
            &evidence.provenance.reference_digest,
        ),
        (
            "independence/blinding declaration",
            &evidence.provenance.independence_blinding,
        ),
    ] {
        if value.trim().is_empty() {
            state.invalid(format!("{name} must be explicit and nonempty"));
        }
    }
    let mut actual = BTreeMap::new();
    for record in records {
        let key = (&record.provenance.run_id, &record.record_id);
        if actual.insert(key, record).is_some() {
            state.invalid(format!(
                "duplicate loaded candidate {} in run {}",
                key.1, key.0
            ));
        }
    }
    let mut candidates = BTreeSet::new();
    let mut groups_by_prompt = BTreeMap::new();
    let mut partition_groups = [BTreeSet::new(), BTreeSet::new()];
    let mut partition_prompts = [BTreeSet::new(), BTreeSet::new()];
    for (partition, rows) in [&evidence.fit, &evidence.assessment]
        .into_iter()
        .enumerate()
    {
        let name = if partition == 0 { "fit" } else { "assessment" };
        if rows.is_empty() {
            state.incomplete(format!("{name} partition must be nonempty"));
        }
        for row in rows {
            let key = (&row.candidate.run_id, &row.candidate.record_id);
            let context = format!("{name} candidate {} in run {}", key.1, key.0);
            if !candidates.insert(key) {
                state.invalid(format!("{context}: duplicate candidate membership"));
            }
            if row.prompt_group.trim().is_empty() {
                state.invalid(format!("{context}: missing global prompt group"));
            }
            partition_groups[partition].insert(&row.prompt_group);
            partition_prompts[partition].insert(&row.candidate.prompt_hash);
            if let Some(group) =
                groups_by_prompt.insert(&row.candidate.prompt_hash, &row.prompt_group)
                && group != &row.prompt_group
            {
                state.invalid(format!(
                    "{context}: equal prompt hashes require the same global group across runs"
                ));
            }
            match &row.label {
                CalibrationLabel::Known { value }
                    if value.is_finite() && (0.0..=1.0).contains(value) => {}
                CalibrationLabel::Known { .. } => {
                    state.invalid(format!("{context}: label must be finite in [0,1]"))
                }
                CalibrationLabel::Unknown { .. } => {
                    state.incomplete(format!("{context}: unknown reference label"))
                }
            }
            let Some(record) = actual.get(&key) else {
                state.invalid(format!("{context}: declared candidate was not supplied"));
                continue;
            };
            match gw_storage::capture_candidate_binding(record) {
                Ok(binding) if binding == row.candidate => {}
                _ => state.invalid(format!(
                    "{context}: stale or contradictory candidate/run/area/prompt/content binding"
                )),
            }
            if row.observations.len() < panel.len() {
                state.incomplete(format!("{context}: incomplete panel coverage"));
            } else if row.observations.len() > panel.len() {
                state.invalid(format!("{context}: extra panel observations"));
            }
            for (column, observation) in row.observations.iter().enumerate() {
                if column >= panel.len() {
                    break;
                }
                validate_cell(
                    &mut state,
                    panel,
                    record,
                    row,
                    column,
                    observation,
                    &context,
                );
            }
        }
    }
    if !partition_groups[0].is_disjoint(&partition_groups[1]) {
        state.invalid("fit and assessment prompt groups overlap, including across run IDs");
    }
    if !partition_prompts[0].is_disjoint(&partition_prompts[1]) {
        state.invalid("fit and assessment prompt hashes overlap, including across run IDs");
    }
    state
}

fn same<T: serde::Serialize>(left: &T, right: &T) -> bool {
    // Exact numeric codecs convert float bits to strings; ordinary f64 PartialEq would hide -0.
    matches!((serde_json::to_vec(left), serde_json::to_vec(right)), (Ok(a), Ok(b)) if a == b)
}

fn validate_cell(
    state: &mut Intake,
    panel: &ResolvedCalibrationPanel,
    record: &TrainingRecord,
    row: &CalibrationRow,
    column: usize,
    observation: &CalibrationObservation,
    context: &str,
) {
    let context = format!("{context}, column {column}");
    if observation.candidate != row.candidate {
        state.invalid(format!(
            "{context}: observation belongs to another candidate"
        ));
    }
    if observation.column_identity != panel.declaration().judges[column].identity {
        state.invalid(format!(
            "{context}: observation is not aligned with its stable panel column"
        ));
    }
    match panel.request_contract(record, column) {
        Ok(expected) if same(&expected, &observation.request) => {}
        _ => state.invalid(format!(
            "{context}: actual request differs from the rebuilt production request"
        )),
    }
    if observation.interpretation_version != JUDGE_INTERPRETATION_VERSION {
        state.invalid(format!("{context}: unsupported interpretation version"));
    }
    match crate::calibration_observation_identity(observation) {
        Ok(expected) if expected == observation.identity => {}
        _ => state.invalid(format!(
            "{context}: observation identity mismatch or invalid numerical encoding"
        )),
    }
    match crate::calibration_payload_identity(&observation.raw) {
        Ok(expected) if expected == observation.payload_identity => {}
        _ => state.invalid(format!("{context}: payload identity mismatch")),
    }
    if observation.collection != CalibrationCollection::Observed {
        state.incomplete(format!(
            "{context}: supplied collection state is not observed"
        ));
    }
    let Some(raw) = &observation.raw else {
        if observation.score.is_some() || observation.verdict.is_some() {
            state.invalid(format!("{context}: interpreted claims lack raw.response"));
        } else {
            state.incomplete(format!("{context}: missing raw.response"));
        }
        return;
    };
    let parsed = match crate::interpret_judge_response(&raw.response) {
        Ok(parsed) => parsed,
        Err(_) => {
            state.invalid(format!("{context}: malformed raw.response"));
            return;
        }
    };
    if observation.score.is_none() || observation.verdict.is_none() {
        state.incomplete(format!("{context}: missing interpreted score or verdict"));
    }
    if let Some(score) = observation.score
        && (!score.is_finite()
            || !(0.0..=1.0).contains(&score)
            || score.to_bits() != parsed.score.to_bits())
    {
        state.invalid(format!(
            "{context}: claimed normalized score contradicts raw.response"
        ));
    }
    if let Some(claimed) = observation.verdict
        && claimed != verdict(parsed.verdict)
    {
        state.invalid(format!(
            "{context}: claimed verdict contradicts raw.response"
        ));
    }
    if parsed.verdict == crate::Verdict::Uncertain {
        state.incomplete(format!("{context}: uncertain interpreted verdict"));
    }
}

pub(crate) fn canonical_evidence(evidence: &CalibrationEvidence) -> CalibrationEvidence {
    let mut canonical = evidence.clone();
    for rows in [&mut canonical.fit, &mut canonical.assessment] {
        rows.sort_by(|a, b| (&a.prompt_group, &a.candidate).cmp(&(&b.prompt_group, &b.candidate)));
    }
    canonical
}

pub(crate) fn fit_identity(evidence: &CalibrationEvidence) -> crate::Result<String> {
    // Deliberate positive allowlist: no assessment inputs/results, whole-corpus provenance
    // digest, full-snapshot digest or rho. Fit labels and raw observations are bound directly.
    let mut fit_only =
        serde_json::to_value(evidence).map_err(crate::calibration_contract::serialization)?;
    let object = fit_only.as_object_mut().expect("evidence is a struct");
    // Use an allowlist, rather than accidentally including a future held-out field.
    object.retain(|key, _| {
        matches!(
            key.as_str(),
            "version"
                | "group_map_version"
                | "method"
                | "numerical_recipe"
                | "target"
                | "panel"
                | "panel_identity"
                | "beta"
                | "fit"
        )
    });
    identity("calibration-fit-v1", &fit_only)
}
