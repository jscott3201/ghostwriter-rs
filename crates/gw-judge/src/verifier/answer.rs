//! Per-kind answer-correctness comparators for the Verifier rail (JUDGE-DESIGN §1.1, V1).
//!
//! Comparators report Match, NonMatch, or Undecided without applying admission policy. Numeric
//! extraction and tolerance come from the persisted contract. Order-insensitive set matching is
//! unchanged. SQL/schema
//! string inequality stays Undecided; the task's answer policy determines its admission consequence.

#[cfg(test)]
use gw_schema::NumericComparison;
use gw_schema::{VerificationContract, VerificationKind, VerificationOutcome};

/// The three-state outcome of a rule-based answer comparison. Distinct from a plain `bool` so the
/// `Undecided` case (a parse failure or unavailable oracle) remains distinct from `NonMatch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnswerComparison {
    /// The answer matches the oracle under the kind's comparator.
    Match,
    /// The answer is a clear, decidable non-match (the rule comparator succeeded and disagreed).
    NonMatch,
    /// The rule comparator could not decide: unavailable oracle or a parse failure on either side.
    Undecided,
}

/// Normalize an answer string for plain string comparison: trim + collapse internal whitespace +
/// lowercase. The conservative fallback comparator (used by `SchemaShape` / `SqlResultMatch`),
/// whose non-matches are routed to rescue, not hard-rejected.
fn normalize_answer(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

/// Split a set/ranking answer into normalized tokens on commas and whitespace, dropping empties.
fn set_tokens(s: &str) -> Vec<String> {
    s.split([',', ' ', '\t', '\n', ';'])
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

/// `VerificationKind::SetMatch`: tokenize both sides and compare as an ORDER-INSENSITIVE multiset
/// (sorted-token equality). So `c,b,a` matches `a,b,c`. An empty token list on either side (nothing
/// to compare) is `Undecided` → rescue.
fn compare_set(answer: &str, expected: &str) -> AnswerComparison {
    let mut a = set_tokens(answer);
    let mut e = set_tokens(expected);
    if a.is_empty() || e.is_empty() {
        return AnswerComparison::Undecided;
    }
    a.sort();
    e.sort();
    if a == e {
        AnswerComparison::Match
    } else {
        AnswerComparison::NonMatch
    }
}

/// The conservative string comparator for kinds without a cheap, clearly-specified rule
/// (`SchemaShape`, `SqlResultMatch`): exact normalized-string equality. A match is a `Match`, but a
/// NON-match is treated as `Undecided` (NOT a confident non-match) because plain string inequality
/// over a structured/SQL result is exactly the false-negative the judge rescue exists to catch.
/// A full structural `SchemaShape` comparator is a tracked follow-up.
fn compare_conservative(answer: &str, expected: &str) -> AnswerComparison {
    if normalize_answer(answer) == normalize_answer(expected) {
        AnswerComparison::Match
    } else {
        AnswerComparison::Undecided
    }
}

/// Compare the assistant `answer` against an oracle string using the per-kind comparator. `None`
/// expected ⇒ `Undecided` (no ground truth → judge rescue).
pub(super) fn compare_answer(
    contract: &VerificationContract,
    answer: &str,
    expected: Option<&str>,
) -> AnswerComparison {
    let Some(expected) = expected else {
        return AnswerComparison::Undecided;
    };
    match contract.kind {
        VerificationKind::NumericMatch => {
            contract
                .numeric
                .as_ref()
                .map_or(
                    AnswerComparison::Undecided,
                    |settings| match crate::evaluate_numeric_answer(
                        answer,
                        Some(expected),
                        settings,
                    ) {
                        VerificationOutcome::Pass => AnswerComparison::Match,
                        VerificationOutcome::Fail => AnswerComparison::NonMatch,
                        VerificationOutcome::Unknown => AnswerComparison::Undecided,
                    },
                )
        }
        VerificationKind::SetMatch => compare_set(answer, expected),
        // SchemaShape / SqlResultMatch have no cheap complete comparator yet → conservative, and a
        // non-match routes to rescue (Undecided), never a hard reject.
        VerificationKind::SchemaShape | VerificationKind::SqlResultMatch => {
            compare_conservative(answer, expected)
        }
        // RefusalExpected / None never reach this comparator (handled in run_verifier).
        VerificationKind::RefusalExpected | VerificationKind::None => AnswerComparison::Undecided,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_answer(
        kind: VerificationKind,
        answer: &str,
        expected: Option<&str>,
    ) -> AnswerComparison {
        let contract = VerificationContract {
            answer_policy: Some(gw_schema::VerificationPolicy::Advisory),
            execution_policy: Some(gw_schema::VerificationPolicy::Absent),
            required_tests: vec![],
            kind,
            oracle: gw_schema::Oracle::None,
            numeric: (kind == VerificationKind::NumericMatch).then(NumericComparison::default),
        };
        super::compare_answer(&contract, answer, expected)
    }

    #[test]
    fn numeric_accepts_only_decimal_formatting() {
        assert_eq!(
            compare_answer(VerificationKind::NumericMatch, "42.0", Some("42")),
            AnswerComparison::Match
        );
        assert_eq!(
            compare_answer(VerificationKind::NumericMatch, "1,000", Some("1000")),
            AnswerComparison::Undecided
        );
        assert_eq!(
            compare_answer(VerificationKind::NumericMatch, "$3.50", Some("3.5")),
            AnswerComparison::Undecided
        );
    }

    #[test]
    fn numeric_clear_mismatch_is_nonmatch() {
        assert_eq!(
            compare_answer(VerificationKind::NumericMatch, "41", Some("42")),
            AnswerComparison::NonMatch
        );
    }

    #[test]
    fn numeric_parse_failure_is_undecided() {
        assert_eq!(
            compare_answer(
                VerificationKind::NumericMatch,
                "about forty-two",
                Some("42")
            ),
            AnswerComparison::Undecided
        );
    }

    #[test]
    fn set_is_order_insensitive() {
        assert_eq!(
            compare_answer(VerificationKind::SetMatch, "c, b, a", Some("a, b, c")),
            AnswerComparison::Match
        );
        assert_eq!(
            compare_answer(VerificationKind::SetMatch, "a, b, z", Some("a, b, c")),
            AnswerComparison::NonMatch
        );
    }

    #[test]
    fn empty_set_is_undecided() {
        assert_eq!(
            compare_answer(VerificationKind::SetMatch, "", Some("a, b")),
            AnswerComparison::Undecided
        );
    }

    #[test]
    fn conservative_match_and_nonmatch_routes_to_rescue() {
        // SchemaShape / SqlResultMatch: an exact match is Match; any inequality is Undecided (rescue).
        assert_eq!(
            compare_answer(VerificationKind::SchemaShape, "{a,b,c}", Some("{a,b,c}")),
            AnswerComparison::Match
        );
        assert_eq!(
            compare_answer(VerificationKind::SchemaShape, "{a,b}", Some("{a,b,c}")),
            AnswerComparison::Undecided
        );
        assert_eq!(
            compare_answer(VerificationKind::SqlResultMatch, "7", Some("8")),
            AnswerComparison::Undecided
        );
    }

    #[test]
    fn no_expected_is_undecided() {
        assert_eq!(
            compare_answer(VerificationKind::NumericMatch, "42", None),
            AnswerComparison::Undecided
        );
    }
}
