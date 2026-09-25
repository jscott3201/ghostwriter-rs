//! User-turn synthesis + the pre-teacher-spend [`UserTurnVerdict`] QC gate (USER-SYNTHESIS §9).
//!
//! `gw-generate` synthesizes BOTH roles. This module produces a candidate USER turn and runs it
//! through the four-boolean QC gate BEFORE any teacher tokens are spent. The gate is the spend
//! guard that complements the budget cap: a candidate advances to `user_synthesized` (and on to
//! the teacher) ONLY iff all four bools are true.
//!
//! ## The gate (USER-SYNTHESIS §9) — enforced in code
//!
//! [`UserTurnVerdict`] (owned by `gw-schema`) carries `answerable`, `difficulty_targeted`,
//! `diverse`, `in_scope_safe`. `evaluate` computes the verdict; [`GatedUserTurn::passed`] is the
//! single predicate the orchestrator MUST consult — [`crate::synthesize_user_turn`] returns the
//! candidate gated, and [`crate::generate_assistant`] refuses to run on a failed candidate, so
//! there is NO path that spends teacher tokens on a turn that did not pass (the load-bearing
//! invariant). `in_scope_safe` is `true` for adversarial-by-construction prompts
//! ([`VerificationKind::RefusalExpected`]): refusal IS the wanted training signal, so the gate
//! must not scrub them.
//!
//! ## The diversity seam ([`Embedder`])
//!
//! `diverse` is an embedding cosine-dedup against already-admitted USER turns. `gw-generate` has
//! no embedder dependency (no heavy ML crate, no network in unit tests), so the embedder is an
//! injected trait with a deterministic default. [`NullEmbedder`] treats everything as novel (for
//! tests / a no-dedup run); a real run injects an OMLX-backed embedder from the engine.

use gw_schema::{
    Content, ContentPart, Message, Role, UserTurnVerdict, VerificationContract, VerificationKind,
};

use crate::error::{GenerateError, Result};

/// The default near-duplicate cosine-similarity threshold (CONFIG §8.1 /
/// USER-SYNTHESIS §6.3 `embedding_cosine_threshold`). A candidate whose cosine similarity to ANY
/// already-admitted USER turn is `>=` this is a near-repeat and fails `diverse`.
pub const DEFAULT_COSINE_THRESHOLD: f64 = 0.86;

/// An embedding backend for the `diverse` dedup check — the crate's only external-effect seam
/// besides the [`Provider`](gw_providers::Provider). Injected so unit tests stay hermetic and the
/// crate pulls no ML/vector dependency.
///
/// Implementations return a fixed-width embedding for a piece of text. The dedup math
/// ([`cosine`]) lives here, so an impl only has to produce vectors.
pub trait Embedder {
    /// Embed `text` into a dense vector.
    ///
    /// # Errors
    /// Returns a human-readable message (wrapped by the caller into
    /// [`GenerateError::Embed`]) if the backend fails.
    fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, String>;
}

/// A deterministic [`Embedder`] that declares everything novel: every candidate is `diverse`.
///
/// The hermetic default — it performs no embedding and never near-dups, so a run with no real
/// embedder configured still produces a valid (if un-deduped) verdict. Real diversity needs an
/// injected OMLX-backed embedder.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullEmbedder;

impl Embedder for NullEmbedder {
    fn embed(&self, _text: &str) -> std::result::Result<Vec<f32>, String> {
        // An empty vector → `cosine` against anything is 0.0 → never a near-dup.
        Ok(Vec::new())
    }
}

/// Cosine similarity of two equal-length vectors. Returns `0.0` when either is empty or has zero
/// norm (so the [`NullEmbedder`]'s empty vectors never trip the near-dup threshold) and when the
/// lengths differ (a dimension mismatch is treated as "not similar", never a panic).
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// The seed inputs that condition a synthesized USER turn (USER-SYNTHESIS §7). These flow into the
/// [`Generation`](gw_schema::Generation) reproducibility block (`persona`, `taxonomy_node`,
/// `prompt_template_id`) at assembly time.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UserSeed {
    /// The persona conditioning the turn (e.g. `"curious_user"`). `None` ⇒ unconditioned.
    pub persona: Option<String>,
    /// The node in the seed taxonomy / skill tree (e.g. `"sql.window_functions"`).
    pub taxonomy_node: Option<String>,
    /// The user-synth prompt-template id (e.g. a MAGPIE / SeedExpand template), for
    /// reproducibility. `None` ⇒ unrecorded.
    pub prompt_template_id: Option<String>,
    /// The requested difficulty band marker (e.g. `"hard"`), checked by `difficulty_targeted`.
    pub difficulty: Option<String>,
}

/// A candidate USER turn paired with the QC inputs needed to gate it (USER-SYNTHESIS §9). The
/// synthesizer fills these; `evaluate` turns them into a [`UserTurnVerdict`].
#[derive(Debug, Clone, PartialEq)]
pub struct UserTurnCandidate {
    /// The synthesized USER message (clean text; user turns carry no reasoning).
    pub message: Message,
    /// The seed inputs that conditioned it.
    pub seed: UserSeed,
    /// The verification contract classifying the turn (drives `in_scope_safe` for adversarial
    /// prompts and is mirrored to provenance downstream).
    pub contract: VerificationContract,
    /// Pre-judged: a competent teacher could answer it (set by the synthesizer's answerability
    /// check; not nonsense/contradictory).
    pub answerable: bool,
    /// Pre-judged: the tagged difficulty matches the requested band.
    pub difficulty_targeted: bool,
    /// Pre-judged: the turn is in-taxonomy + decontaminated + safety-classified. Adversarial
    /// prompts set this `true` (they are wanted).
    pub in_scope: bool,
}

impl UserTurnCandidate {
    /// The candidate's user-turn text, for embedding-dedup AND the control-token guard. For a
    /// multimodal turn the `ContentPart::Text` parts are concatenated (matching the gw-format
    /// flatten convention); non-text parts contribute nothing.
    fn text(&self) -> String {
        match &self.message.content {
            Content::Text(t) => t.clone(),
            Content::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
            // An explicitly absent user turn has no text to embed or control-token-scan.
            Content::Null => String::new(),
        }
    }
}

/// Every chat control token that may leak into a synthesized user turn — a transcription of
/// `gw-format`'s crate-private `CONTROL_TOKENS` (which is not part of its public surface, so it is
/// mirrored here). If a synthesized USER turn contains any of these RAW, an upstream
/// elicitation/template stage leaked channel markup; the gate must reject it (m8) rather than embed
/// it (which would defeat dedup) and later render it (which would double-frame the training target).
const CONTROL_TOKENS: &[&str] = &[
    // ChatML / Qwen / DeepSeek
    "<think>",
    "</think>",
    "<|im_start|>",
    "<|im_end|>",
    // Gemma-4 (asymmetric)
    "<|channel>",
    "<channel|>",
    "<|channel|>",
    "<|turn>",
    "<turn|>",
    "<|think|>",
    "<bos>",
    // Harmony
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|return|>",
];

/// The first control token contained in `text`, if any (the most descriptive marker first, mirroring
/// gw-format's ordering).
fn first_control_token(text: &str) -> Option<&'static str> {
    CONTROL_TOKENS
        .iter()
        .copied()
        .find(|tok| text.contains(tok))
}

/// A candidate that has been run through the QC gate. The orchestrator MUST consult [`passed`] —
/// it is the single gate that decides whether teacher tokens may be spent.
///
/// [`passed`]: GatedUserTurn::passed
#[derive(Debug, Clone, PartialEq)]
pub struct GatedUserTurn {
    /// The candidate (kept verbatim so the assembler can read its seed/contract/message).
    pub candidate: UserTurnCandidate,
    /// The computed four-boolean verdict.
    pub verdict: UserTurnVerdict,
}

impl GatedUserTurn {
    /// `true` iff ALL FOUR verdict booleans are true — the SINGLE condition under which a teacher
    /// may be called for this turn (USER-SYNTHESIS §9). This is the spend gate.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.verdict.answerable
            && self.verdict.difficulty_targeted
            && self.verdict.diverse
            && self.verdict.in_scope_safe
    }
}

/// Compute the [`UserTurnVerdict`] for a candidate, running the `diverse` embedding-dedup against
/// `prior_embeddings` (already-admitted USER-turn vectors) at `threshold`.
///
/// - First, FAIL LOUD if the user-turn text carries a raw control token (m8): a leak must never be
///   embedded (it would defeat dedup) or admitted (a later render would double-frame the target).
/// - `answerable` / `difficulty_targeted` come from the synthesizer's pre-judgement.
/// - `diverse` is `true` iff the candidate's max cosine similarity to any prior embedding is
///   `< threshold` (a `>=` match is a near-repeat → not diverse).
/// - `in_scope_safe` is `candidate.in_scope` OR'd with the adversarial-by-construction exemption:
///   a [`VerificationKind::RefusalExpected`] turn is ALWAYS `in_scope_safe` (refusal is the wanted
///   signal; the gate must not scrub it).
///
/// # Errors
/// - [`GenerateError::LeakedUserTurn`] if the user-turn text contains a chat control token.
/// - [`GenerateError::Embed`] if the embedder fails on the candidate text.
pub fn evaluate<E: Embedder + ?Sized>(
    candidate: &UserTurnCandidate,
    embedder: &E,
    prior_embeddings: &[Vec<f32>],
    threshold: f64,
) -> Result<UserTurnVerdict> {
    let text = candidate.text();
    if let Some(token) = first_control_token(&text) {
        return Err(GenerateError::LeakedUserTurn(token));
    }

    let embedding = embedder.embed(&text).map_err(GenerateError::Embed)?;

    let max_sim = prior_embeddings
        .iter()
        .map(|prior| cosine(&embedding, prior))
        .fold(0.0f64, f64::max);
    let diverse = max_sim < threshold;

    // Adversarial-by-construction prompts (RefusalExpected) are in_scope_safe regardless of the
    // synthesizer's generic in-scope judgement: refusal IS the oracle, and they are wanted/tagged.
    let in_scope_safe =
        candidate.in_scope || candidate.contract.kind == VerificationKind::RefusalExpected;

    Ok(UserTurnVerdict {
        answerable: candidate.answerable,
        difficulty_targeted: candidate.difficulty_targeted,
        diverse,
        in_scope_safe,
        notes: None,
    })
}

/// Gate a candidate at the default cosine threshold ([`DEFAULT_COSINE_THRESHOLD`]). Convenience
/// over `evaluate` that pairs the candidate with its verdict into a [`GatedUserTurn`].
///
/// # Errors
/// Returns [`GenerateError::Embed`] if the embedder fails.
pub fn gate<E: Embedder + ?Sized>(
    candidate: UserTurnCandidate,
    embedder: &E,
    prior_embeddings: &[Vec<f32>],
) -> Result<GatedUserTurn> {
    let verdict = evaluate(
        &candidate,
        embedder,
        prior_embeddings,
        DEFAULT_COSINE_THRESHOLD,
    )?;
    Ok(GatedUserTurn { candidate, verdict })
}

/// Build a clean USER [`Message`] from synthesized text. User turns carry no reasoning.
#[must_use]
pub fn user_message(text: impl Into<String>) -> Message {
    Message {
        role: Role::User,
        content: Content::Text(text.into()),
        reasoning: None,
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::Oracle;

    /// An [`Embedder`] that returns a fixed vector per exact text — lets a test stage a "prior"
    /// turn and a near-identical candidate to exercise the dedup branch deterministically.
    struct StubEmbedder;
    impl Embedder for StubEmbedder {
        fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, String> {
            match text {
                "dup" => Ok(vec![1.0, 0.0, 0.0]),
                "near" => Ok(vec![0.999, 0.044, 0.0]), // cosine ~0.999 vs "dup"
                "novel" => Ok(vec![0.0, 1.0, 0.0]),    // orthogonal to "dup"
                "boom" => Err("backend down".into()),
                _ => Ok(vec![0.0, 0.0, 1.0]),
            }
        }
    }

    fn candidate(text: &str, kind: VerificationKind, in_scope: bool) -> UserTurnCandidate {
        UserTurnCandidate {
            message: user_message(text),
            seed: UserSeed::default(),
            contract: VerificationContract {
                kind,
                oracle: Oracle::None,
                answer_marker: None,
            },
            answerable: true,
            difficulty_targeted: true,
            in_scope,
        }
    }

    #[test]
    fn cosine_handles_empty_and_mismatched_lengths() {
        assert_eq!(cosine(&[], &[1.0]), 0.0);
        assert_eq!(cosine(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-12);
    }

    #[test]
    fn novel_candidate_passes_all_four() {
        let c = candidate("novel", VerificationKind::NumericMatch, true);
        let prior = vec![vec![1.0f32, 0.0, 0.0]]; // the "dup" vector
        let gated = gate(c, &StubEmbedder, &prior).unwrap();
        assert!(gated.verdict.diverse);
        assert!(gated.passed());
    }

    #[test]
    fn near_repeat_fails_diverse_and_blocks_spend() {
        // The crux: a near-duplicate candidate must fail `diverse`, so `passed()` is false and the
        // orchestrator never spends teacher tokens on it.
        let c = candidate("near", VerificationKind::NumericMatch, true);
        let prior = vec![vec![1.0f32, 0.0, 0.0]]; // "dup"
        let gated = gate(c, &StubEmbedder, &prior).unwrap();
        assert!(!gated.verdict.diverse, "near-repeat must not be diverse");
        assert!(!gated.passed(), "a non-diverse turn must NOT pass the gate");
    }

    #[test]
    fn any_false_bool_blocks_the_gate() {
        // Each individual bool, when false, must sink the gate (teacher spend is all-or-nothing).
        let prior: Vec<Vec<f32>> = vec![];
        for flip in 0..3 {
            let mut c = candidate("novel", VerificationKind::NumericMatch, true);
            match flip {
                0 => c.answerable = false,
                1 => c.difficulty_targeted = false,
                _ => c.in_scope = false,
            }
            let gated = gate(c, &StubEmbedder, &prior).unwrap();
            assert!(!gated.passed(), "flip={flip} should block the gate");
        }
    }

    #[test]
    fn refusal_expected_is_in_scope_safe_even_when_in_scope_is_false() {
        // Adversarial-by-construction: in_scope=false at the candidate level, but RefusalExpected
        // forces in_scope_safe=true — the gate must NOT scrub the wanted refusal training signal.
        let c = candidate("seed-020", VerificationKind::RefusalExpected, false);
        let prior: Vec<Vec<f32>> = vec![];
        let gated = gate(c, &StubEmbedder, &prior).unwrap();
        assert!(gated.verdict.in_scope_safe);
        assert!(gated.passed());
    }

    #[test]
    fn embedder_failure_surfaces_as_embed_error() {
        let c = candidate("boom", VerificationKind::NumericMatch, true);
        let err = gate(c, &StubEmbedder, &[]).unwrap_err();
        assert!(matches!(err, GenerateError::Embed(_)));
    }

    #[test]
    fn null_embedder_declares_everything_diverse() {
        // With the hermetic default, even an identical-text prior never near-dups.
        let c = candidate("anything", VerificationKind::None, true);
        let prior = vec![vec![1.0f32, 2.0, 3.0]];
        let gated = gate(c, &NullEmbedder, &prior).unwrap();
        assert!(gated.verdict.diverse);
    }

    /// An [`Embedder`] producing unit 2-D vectors whose cosine vs the prior `[1,0]` equals a target,
    /// so a test can sit exactly on either side of the 0.86 boundary (M4).
    struct CosineEmbedder;
    impl Embedder for CosineEmbedder {
        fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, String> {
            // [cos, sin] is a unit vector at the named cosine vs the prior [1, 0].
            let cos: f32 = match text {
                "cos085" => 0.85,
                "cos086" => 0.86,
                "cos087" => 0.87,
                _ => 0.0,
            };
            Ok(vec![cos, (1.0 - cos * cos).sqrt()])
        }
    }

    #[test]
    fn threshold_constant_is_pinned() {
        // M4: pin the constant so a typo (0.86 -> 0.68/0.96) is caught by name.
        assert_eq!(DEFAULT_COSINE_THRESHOLD, 0.86);
    }

    #[test]
    fn diverse_boundary_is_strict_less_than_086() {
        // M4: cosine 0.85 < 0.86 => diverse; 0.87 >= 0.86 => not diverse; EXACTLY 0.86 => not
        // diverse (the gate is `max_sim < threshold`). Kills threshold-direction/typo mutants.
        let prior = vec![vec![1.0f32, 0.0]];

        let below = gate(
            candidate("cos085", VerificationKind::NumericMatch, true),
            &CosineEmbedder,
            &prior,
        )
        .unwrap();
        assert!(below.verdict.diverse, "cos 0.85 (< 0.86) must be diverse");

        let at = gate(
            candidate("cos086", VerificationKind::NumericMatch, true),
            &CosineEmbedder,
            &prior,
        )
        .unwrap();
        assert!(
            !at.verdict.diverse,
            "cos exactly 0.86 must NOT be diverse (>= threshold)"
        );

        let above = gate(
            candidate("cos087", VerificationKind::NumericMatch, true),
            &CosineEmbedder,
            &prior,
        )
        .unwrap();
        assert!(
            !above.verdict.diverse,
            "cos 0.87 (>= 0.86) must NOT be diverse"
        );
    }

    #[test]
    fn multimodal_user_turn_dedups_on_concatenated_text_parts() {
        // m5: a Parts user turn must embed its concatenated text (NOT ""), or dedup is defeated.
        let mut c = candidate("ignored", VerificationKind::NumericMatch, true);
        c.message.content = Content::Parts(vec![
            ContentPart::Text { text: "nov".into() },
            ContentPart::ImageUrl {
                image_url: "http://x/y.png".into(),
            },
            ContentPart::Text { text: "el".into() },
        ]);
        // text() concatenates to "novel"; StubEmbedder("novel") is orthogonal to the "dup" prior.
        let prior = vec![vec![1.0f32, 0.0, 0.0]];
        let gated = gate(c, &StubEmbedder, &prior).unwrap();
        assert!(gated.verdict.diverse);
    }

    #[test]
    fn control_token_in_user_turn_fails_the_gate() {
        // m8: a leaked control token must fail the gate loud (never embedded, never admitted).
        for laced in ["<|turn>user\nhi", "tell me<think> about", "x<|im_end|>"] {
            let c = candidate(laced, VerificationKind::NumericMatch, true);
            let err = gate(c, &StubEmbedder, &[]).unwrap_err();
            assert!(
                matches!(err, GenerateError::LeakedUserTurn(_)),
                "expected LeakedUserTurn for {laced:?}, got {err:?}"
            );
        }
        // A clean turn still passes (guard is not over-eager).
        let clean = candidate("novel", VerificationKind::NumericMatch, true);
        assert!(gate(clean, &StubEmbedder, &[]).is_ok());
    }
}
