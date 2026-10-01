//! Resolution of supplied calibration declarations through the actual production request builder.
use crate::{
    EffectiveJudgeContract, JUDGE_CANDIDATE_RENDER_VERSION, JUDGE_INTERPRETATION_VERSION,
    JudgeError, PanelJudge, Result, Verdict, build_judge_request, interpret_judge_response,
    render_judge_candidate,
};
use gw_schema::{
    CALIBRATION_VERSION, CalibrationCollection, CalibrationJudgeContract, CalibrationObservation,
    CalibrationPanel, CalibrationRawResponse, CalibrationVerdict, TrainingRecord,
};
use serde::Serialize;

/// Stable, opaque live panel authority, constructed separately from supplied evidence.
///
/// Every column is resolved from an actual production empty-candidate request. Every observation
/// request is independently rebuilt from the complete candidate. No deserialization is provided.
#[derive(Debug, Clone)]
pub struct ResolvedCalibrationPanel {
    judges: Vec<PanelJudge>,
    rubric: String,
    declaration: CalibrationPanel,
    identity: String,
    // Retain live effective definitions, whose construction validates controls and distinctness.
    effective: Vec<EffectiveJudgeContract>,
}

impl ResolvedCalibrationPanel {
    /// Resolve ordered production declarations and reject duplicated effective definitions.
    ///
    /// # Errors
    /// Rejects empty panels/areas, invalid controls, duplicate effective judges, or serialization.
    pub fn new(training_area: &str, judges: &[PanelJudge], rubric: &str) -> Result<Self> {
        if training_area.trim().is_empty() || judges.is_empty() {
            return Err(JudgeError::Invariant(
                "calibration requires an area and nonempty panel".into(),
            ));
        }
        let effective = judges
            .iter()
            .map(|judge| {
                EffectiveJudgeContract::json_score(&build_judge_request(judge, rubric, ""))
            })
            .collect::<Result<Vec<_>>>()?;
        crate::effective_contract::validate_unique(&effective)?;
        let declaration = CalibrationPanel {
            version: CALIBRATION_VERSION,
            training_area: training_area.into(),
            render_contract: JUDGE_CANDIDATE_RENDER_VERSION.into(),
            judges: judges
                .iter()
                .map(|judge| contract(judge, rubric, ""))
                .collect::<Result<_>>()?,
        };
        let identity = identity("calibration-panel-v1", &declaration)?;
        Ok(Self {
            judges: judges.to_vec(),
            rubric: rubric.into(),
            declaration,
            identity,
            effective,
        })
    }

    /// Full ordered stable declarations for preparing a supplied evidence document.
    #[must_use]
    pub fn declaration(&self) -> &CalibrationPanel {
        &self.declaration
    }

    /// Content identity of the complete stable declaration.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Number of distinct effective columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.effective.len()
    }

    /// Whether this resolved panel has no columns (construction rejects that case).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effective.is_empty()
    }

    /// Rebuild the full candidate-specific production request for one ordered column.
    ///
    /// # Errors
    /// Rejects a wrong area, unknown column, invalid candidate rendering, or serialization failure.
    pub fn request_contract(
        &self,
        record: &TrainingRecord,
        column: usize,
    ) -> Result<CalibrationJudgeContract> {
        if record.training_area != self.declaration.training_area {
            return Err(JudgeError::Invariant(
                "candidate training area differs from resolved panel".into(),
            ));
        }
        let judge = self
            .judges
            .get(column)
            .ok_or_else(|| JudgeError::Invariant("unknown calibration column".into()))?;
        let candidate = render_judge_candidate(&record.messages).map_err(|error| {
            JudgeError::Invariant(format!("candidate rendering failed: {error}"))
        })?;
        contract(judge, &self.rubric, &candidate)
    }

    /// Bind already supplied response text. This makes no model call and grants no collection proof.
    /// Missing text is represented explicitly; supplied text uses the production interpreter.
    ///
    /// # Errors
    /// Rejects invalid request bindings, malformed supplied text, or serialization failure.
    pub fn observation(
        &self,
        record: &TrainingRecord,
        column: usize,
        collection: CalibrationCollection,
        response: Option<String>,
    ) -> Result<CalibrationObservation> {
        let request = self.request_contract(record, column)?;
        let interpreted = response
            .as_ref()
            .map(|text| {
                interpret_judge_response(text)
                    .map_err(|error| JudgeError::JudgeParse(error.to_string()))
            })
            .transpose()?;
        let raw = response.map(|response| CalibrationRawResponse { response });
        let mut observation = CalibrationObservation {
            candidate: gw_storage::capture_candidate_binding(record)?,
            column_identity: self.declaration.judges[column].identity.clone(),
            request,
            interpretation_version: JUDGE_INTERPRETATION_VERSION,
            collection,
            payload_identity: calibration_payload_identity(&raw)?,
            raw,
            score: interpreted.as_ref().map(|result| result.score),
            verdict: interpreted.as_ref().map(|result| verdict(result.verdict)),
            identity: String::new(),
        };
        observation.identity = calibration_observation_identity(&observation)?;
        Ok(observation)
    }
}

pub(crate) fn verdict(value: Verdict) -> CalibrationVerdict {
    match value {
        Verdict::Accept => CalibrationVerdict::Accept,
        Verdict::Revise => CalibrationVerdict::Revise,
        Verdict::Reject => CalibrationVerdict::Reject,
        Verdict::Uncertain => CalibrationVerdict::Uncertain,
    }
}

fn contract(judge: &PanelJudge, rubric: &str, candidate: &str) -> Result<CalibrationJudgeContract> {
    let projection =
        crate::request_identity::request_contract_for_candidate(judge, rubric, candidate)?;
    // This Value comes directly from the production resolver. Never deserialize this JSON later.
    let projection_json = serde_json::to_string(&projection).map_err(serialization)?;
    let mut contract = CalibrationJudgeContract {
        projection_json,
        identity: String::new(),
        temperature: judge.sampling.temperature,
        top_p: judge.sampling.top_p,
    };
    contract.identity = identity("calibration-request-v1", &contract)?;
    Ok(contract)
}

/// Recompute the digest of exact optional raw.response text without interpreting its contents.
///
/// # Errors
/// Returns a serialization error; does not perform I/O.
pub fn calibration_payload_identity(raw: &Option<CalibrationRawResponse>) -> Result<String> {
    identity("calibration-payload-v1", raw)
}

/// Recompute a supplied cell identity from all fields except its own identity.
/// This function grants no authority to the submitted request, payload, or score.
///
/// # Errors
/// Rejects nonfinite declared numerical fields or serialization errors.
pub fn calibration_observation_identity(observation: &CalibrationObservation) -> Result<String> {
    let mut value = observation.clone();
    value.identity.clear();
    identity("calibration-observation-v1", &value)
}

pub(crate) fn identity<T: Serialize>(domain: &str, value: &T) -> Result<String> {
    let value = serde_json::to_value(value).map_err(serialization)?;
    Ok(format!(
        "{domain}:{}",
        gw_storage::canonical_json_hash(&value)?
    ))
}

pub(crate) fn serialization(error: serde_json::Error) -> JudgeError {
    JudgeError::Invariant(format!(
        "could not encode exact calibration evidence: {error}"
    ))
}
