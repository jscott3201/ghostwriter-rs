//! `gw-schema` — the canonical, byte-reproducible serde data contract for ghostwriter-rs.
//!
//! Every other crate depends on this one; it has **no internal dependencies** and performs
//! **no I/O**. Types are transcribed from the internal `DATA-SCHEMA.md` spec: `TrainingRecord`
//! and its envelope (`Message`, `Provenance`, `Generation`, `Verification`, `Judging`,
//! `ReasoningQuality`, `Lifecycle`, `Hashes`, `Cost`), the grading types (`Verdict`, `Decision`,
//! `Check`/`CheckKind`, consensus types), the rating types (`RatingRecord`), the export types
//! (`ExportManifest`, `PreferenceRecord`), and the configuration types (`Config`,
//! `ProviderLimits`, `EmbeddingConfig`, `SandboxConfig`, `ReasoningEffort`).
//!
//! ## Scope & invariants
//!
//! This crate is **types only** — pure structs/enums + `Default` impls. No business logic, no
//! algorithms (no `n_eff` computation, no hashing). It carries the load-bearing invariants of
//! the contract in its *shape*:
//!
//! - **(a)** reasoning is a first-class sibling of `content`, never inlined (`Message`,
//!   `PreferenceSide`).
//! - **(f)** admission is never a plain mean — the schema stores only the inputs/outputs of
//!   the harness-side consensus compute (`Judging`).
//! - **`reasoning_effort` is `xhigh`, never `max`** — `ReasoningEffort` has no `Max` variant.
//!
//! Timestamps are RFC3339 `String`s for v1 byte-reproducibility; `schema_version` /
//! `dataset_version` are `semver::Version`.
//!
//! Note: the 4-variant per-grade `gw-judge::Verdict`, the panel `gw-judge::Decision`, and the
//! `Grade`/consensus rail types live in `gw-judge` (they carry trait objects + raw provider
//! responses). `gw-schema` owns the *persisted* envelope `Verdict { Admit, Reject,
//! NeedsReview }` and the config-side `JudgeSampling` policy folded into the contract.

mod config;
mod cost;
mod decontam;
mod embedding;
mod export;
mod generation;
mod hashes;
mod judging;
mod lifecycle;
mod message;
mod preference;
mod provenance;
mod rating;
mod reasoning_quality;
mod record;
mod sandbox;
mod verification;
mod verification_contract;

// --- §1.11 the envelope ---
pub use record::TrainingRecord;

// --- §1.3 conversation ---
pub use message::{Content, ContentPart, FunctionCall, Message, ReasoningDetail, Role, ToolCall};

// --- §1.4 provenance ---
pub use provenance::{Provenance, TeacherRef};

// --- §1.5 generation ---
pub use generation::{Generation, ReasoningEffort};

// --- §1.6 verification (deterministic rail) ---
pub use verification::{Check, CheckKind, Verification};

// --- §1.7 judging (LLM panel rail) + folded judge sampling policy ---
pub use judging::{JudgeSampling, JudgeVote, Judging, Verdict};

// --- §1.13 reasoning quality (per-step CoT) ---
pub use reasoning_quality::{ReasoningQuality, StepAggregation, StepVerdict};

// --- §1.13b rating (Glicko-2 reputation) ---
pub use rating::{RatingRecord, RatingSubject};

// --- §1.8 lifecycle ---
pub use lifecycle::{Lifecycle, LifecycleState, StateTransition};

// --- §1.9 / §1.10 hashes + cost ---
pub use cost::Cost;
pub use hashes::Hashes;

// --- §3 / §4.3 export contracts ---
pub use export::{
    CorpusDiversityStats, CotPolicy, ExportManifest, ExportSchemaVersion, MultiTurnLoss, TrlFormat,
};

// --- §6.3 DPO preference export ---
pub use preference::{PreferenceRecord, PreferenceSide};

// --- §5.3 decontamination config ---
pub use decontam::{CANONICAL_PROTECTED_BENCHMARKS, DecontamConfig};

// --- USER-SYNTHESIS §8/§9 verification contract + user-turn QC verdict ---
pub use verification_contract::{Oracle, UserTurnVerdict, VerificationContract, VerificationKind};

// --- CONFIG global config + sub-configs ---
pub use config::{
    BudgetBreach, BudgetConfig, BudgetGranularity, Config, DataCollection, PromoteConfig,
    ProviderLimits, TeacherRouting,
};
pub use embedding::{
    DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_ENDPOINT, DEFAULT_EMBEDDING_MODEL, EmbeddingBackend,
    EmbeddingConfig, VectorIndex,
};
pub use sandbox::{CodeSandbox, SandboxConfig, SqlSandbox};

#[cfg(test)]
mod tests {
    use super::*;

    /// The serialized enum spellings must match the §1.12 JSON-Schema enum arrays exactly.
    #[test]
    fn enum_wire_spellings_match_schema() {
        let j = |v: &dyn JsonStr| v.json();
        assert_eq!(j(&Role::Assistant), "\"assistant\"");
        assert_eq!(j(&Role::Developer), "\"developer\"");
        assert_eq!(j(&ReasoningEffort::Xhigh), "\"xhigh\"");
        assert_eq!(j(&CheckKind::UnitTest), "\"unit_test\"");
        assert_eq!(j(&CheckKind::ReasoningPresent), "\"reasoning_present\"");
        assert_eq!(j(&CheckKind::Sandbox), "\"sandbox\"");
        assert_eq!(j(&CheckKind::Language), "\"language\"");
        assert_eq!(j(&Verdict::NeedsReview), "\"needs_review\"");
        assert_eq!(
            j(&LifecycleState::AssistantGenerated),
            "\"assistant_generated\""
        );
        assert_eq!(j(&StepAggregation::LateWeighted), "\"late_weighted\"");
        assert_eq!(j(&RatingSubject::Teacher), "\"teacher\"");
        assert_eq!(j(&MultiTurnLoss::AllAssistant), "\"all_assistant\"");
        assert_eq!(
            j(&VerificationKind::RefusalExpected),
            "\"refusal_expected\""
        );
    }

    trait JsonStr {
        fn json(&self) -> String;
    }
    impl<T: serde::Serialize> JsonStr for T {
        fn json(&self) -> String {
            serde_json::to_string(self).unwrap()
        }
    }

    /// `ReasoningDetail` is tagged on `type` with dotted variant names; `Content` is untagged.
    #[test]
    fn reasoning_detail_and_content_round_trip() {
        let d = ReasoningDetail::Text {
            text: "let me think".into(),
            signature: None,
            id: None,
            format: Some("anthropic-claude-v1".into()),
            index: 0,
        };
        let s = serde_json::to_string(&d).unwrap();
        assert!(s.contains("\"type\":\"reasoning.text\""));
        assert_eq!(d, serde_json::from_str(&s).unwrap());

        // untagged Content: a plain string stays a JSON string.
        let c = Content::Text("42".into());
        assert_eq!(serde_json::to_string(&c).unwrap(), "\"42\"");
        assert_eq!(c, serde_json::from_str::<Content>("\"42\"").unwrap());
    }

    /// Optional fields skip-serialize when None; defaults round-trip.
    #[test]
    fn pinned_defaults() {
        assert_eq!(JudgeSampling::default().temperature, 0.0);
        assert_eq!(MultiTurnLoss::default(), MultiTurnLoss::AllAssistant);
        assert_eq!(CotPolicy::default(), CotPolicy::Supervised);
        assert_eq!(Config::default().budget.cap_usd, 25.0);
        assert_eq!(DecontamConfig::default().decontam_ngram, [8, 13]);
        assert_eq!(EmbeddingConfig::default().dim, DEFAULT_EMBEDDING_DIM);

        // A minimal Message with only role+content emits no optional keys.
        let m = Message {
            role: Role::User,
            content: Content::Text("hi".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(
            serde_json::to_string(&m).unwrap(),
            "{\"role\":\"user\",\"content\":\"hi\"}"
        );
    }

    /// A full envelope round-trips byte-for-byte through serde_json.
    #[test]
    fn training_record_round_trip() {
        let rec = TrainingRecord {
            record_id: "01J8".into(),
            schema_version: semver::Version::new(1, 0, 0),
            dataset_version: Some(semver::Version::new(0, 3, 2)),
            training_area: "rust-async".into(),
            tags: vec!["tokio".into()],
            messages: vec![Message {
                role: Role::Assistant,
                content: Content::Text("96".into()),
                reasoning: Some("12*8=96".into()),
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            provenance: Provenance {
                run_id: "run-1".into(),
                parent_ids: vec![],
                teacher: TeacherRef {
                    provider: "openrouter".into(),
                    slug: "z-ai/glm-5.2".into(),
                    served_by: Some("Parasail".into()),
                    model_card_revision: None,
                },
                user_synth_model: None,
                user_turn_kind: Some("numeric_match".into()),
                in_scope_safe: Some(true),
                judge_models: vec![],
                harness_version: "0.1.0".into(),
                git_commit: None,
            },
            generation: Generation {
                reasoning_effort: Some(ReasoningEffort::Xhigh),
                ..Default::default()
            },
            verification_contract: None,
            verification: Verification::default(),
            judging: Judging::default(),
            reasoning_quality: None,
            lifecycle: Lifecycle::default(),
            hashes: Hashes::default(),
            cost: Cost::default(),
        };
        let s = serde_json::to_string(&rec).unwrap();
        let back: TrainingRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(rec, back);
    }
}
