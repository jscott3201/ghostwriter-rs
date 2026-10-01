//! Synthetic software fixtures; the hand-assigned labels do not establish model quality.
#![allow(dead_code)]

use gw_eval::outcomes::*;
use gw_schema::{
    Content, Generation, Hashes, Judging, Lifecycle, Message, Provenance, Role, TeacherRef,
    TrainingRecord, Verification,
};

pub fn candidate(prompt: usize, index: u32, score: Option<f64>) -> TrainingRecord {
    let message = |role, text| Message {
        role,
        content: Content::Text(text),
        reasoning: None,
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    let mut record = TrainingRecord {
        record_id: format!("p{prompt}-c{index}"),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "synthetic-area".into(),
        tags: vec![],
        messages: vec![
            message(Role::User, format!("independent prompt {prompt}")),
            message(Role::Assistant, format!("candidate answer {index}")),
        ],
        tools: None,
        provenance: Provenance {
            run_id: "run-1".into(),
            parent_ids: vec![],
            teacher: TeacherRef {
                provider: "fixture".into(),
                slug: "synthetic".into(),
                served_by: None,
                model_card_revision: None,
            },
            user_synth_model: None,
            user_turn_kind: None,
            in_scope_safe: Some(true),
            judge_models: vec![],
            harness_version: "fixture".into(),
            git_commit: None,
        },
        generation: Generation {
            completion_index: Some(index),
            ..Default::default()
        },
        task_provenance: None,
        verification_contract: None,
        execution_evidence: None,
        verification: Verification {
            all_passed: true,
            ..Default::default()
        },
        judging: Judging {
            aggregate: score,
            ..Default::default()
        },
        reasoning_quality: None,
        lifecycle: Lifecycle::default(),
        hashes: Hashes::default(),
        cost: Default::default(),
    };
    let binding = gw_storage::capture_candidate_binding(&record).unwrap();
    record.hashes.prompt_hash = binding.prompt_hash;
    record.hashes.record_hash = binding.record_hash;
    record
}

pub fn evidence(records: &[TrainingRecord], values: &[f64]) -> OutcomeEvidence {
    assert_eq!(records.len(), values.len());
    let corpus: Vec<_> = records
        .iter()
        .map(|record| gw_storage::capture_candidate_binding(record).unwrap())
        .collect();
    let outcomes = corpus
        .iter()
        .zip(values)
        .map(|(candidate, &value)| CandidateOutcome {
            candidate: candidate.clone(),
            outcome: ReferenceOutcome::Known { value },
        })
        .collect();
    OutcomeEvidence {
        version: OUTCOME_EVIDENCE_VERSION,
        run_id: "run-1".into(),
        training_area: "synthetic-area".into(),
        metric: OutcomeMetric {
            name: "synthetic_reference_success".into(),
            version: "v1".into(),
            direction: MetricDirection::HigherIsBetter,
        },
        provenance: OutcomeProvenance {
            source: ReferenceSource::DeterministicTaskReference,
            protocol_revision: "hand-authored-fixture-v1".into(),
            reference_artifact_digest: ReferenceDigest {
                algorithm: DigestAlgorithm::Sha256,
                hex: "a".repeat(64),
            },
        },
        sampling_assumption: SamplingAssumption::IndependentPrompts,
        corpus,
        outcomes,
    }
}

pub fn corpus(prompts: usize, predictive: bool) -> (Vec<TrainingRecord>, OutcomeEvidence) {
    let mut records = Vec::with_capacity(prompts * 2);
    let mut values = Vec::with_capacity(prompts * 2);
    for prompt in 0..prompts {
        for index in 0..2 {
            let good = index == 0;
            records.push(candidate(
                prompt,
                index,
                Some(if good == predictive { 0.9 } else { 0.1 }),
            ));
            values.push(if good { 1.0 } else { 0.0 });
        }
    }
    let evidence = evidence(&records, &values);
    (records, evidence)
}
