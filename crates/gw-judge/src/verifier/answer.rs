//! Per-kind answer-correctness comparators for the Verifier rail (JUDGE-DESIGN §1.1, V1).
//!
//! THE LOAD-BEARING RULE (`rescue_negatives = true`): the deterministic verifier hard-rejects ONLY
//! what it is certain about. The reasoning-present Verify gate and decontam are authoritative; an
//! ANSWER-CORRECTNESS comparison — where a rule-based comparator can be wrong on
//! correct-but-differently-formatted data (`42.0` vs `42`, a reordered set) — returns a three-state
//! [`AnswerComparison`] so the rail's caller routes a residual non-match to `Uncertain` → judge
//! rescue rather than a silent hard reject (the hard reject is reserved for an area that explicitly
//! opts into rule-only-authoritative grading).

use gw_schema::VerificationKind;

/// Default relative tolerance for [`VerificationKind::NumericMatch`] (`1e-6`). A numeric answer
/// within this relative (or [`NUMERIC_ABS_TOL`] absolute) distance of the oracle counts as a match.
const NUMERIC_REL_TOL: f64 = 1e-6;
/// Default absolute tolerance for [`VerificationKind::NumericMatch`] (`1e-9`) — covers near-zero
/// expected values where a relative tolerance degenerates.
const NUMERIC_ABS_TOL: f64 = 1e-9;

/// The three-state outcome of a rule-based answer comparison. Distinct from a plain `bool` so the
/// CRITICAL `Undecided` case (a parse failure, or no oracle) routes to judge rescue rather than
/// being conflated with a hard `NonMatch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnswerComparison {
    /// The answer matches the oracle under the kind's comparator.
    Match,
    /// The answer is a clear, decidable non-match (the rule comparator succeeded and disagreed).
    NonMatch,
    /// The rule comparator could not decide (no oracle, or a parse failure on either side). Routes
    /// to `Uncertain` → judge rescue, NEVER a silent pass or a hard reject.
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

/// Parse a numeric answer, tolerating surrounding whitespace, a leading currency `$`, `%`, and
/// thousands separators (`1,000` → `1000`). Returns `None` if the cleaned token is not a finite
/// number — that drives the `Undecided` → rescue path rather than a false non-match.
fn parse_numeric(s: &str) -> Option<f64> {
    let cleaned: String = s
        .trim()
        .chars()
        .filter(|c| !matches!(c, ',' | '$' | '%' | '_' | ' '))
        .collect();
    if cleaned.is_empty() {
        return None;
    }
    cleaned.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// `VerificationKind::NumericMatch`: parse both sides and compare within a relative + absolute
/// tolerance. A parse failure on either side is `Undecided` (rescue), never a false non-match — so
/// `42.0` vs `42` and `1,000` vs `1000` are matches, while a genuinely different number is a clean
/// `NonMatch`.
fn compare_numeric(answer: &str, expected: &str) -> AnswerComparison {
    match (parse_numeric(answer), parse_numeric(expected)) {
        (Some(a), Some(e)) => {
            let diff = (a - e).abs();
            let tol = NUMERIC_ABS_TOL.max(NUMERIC_REL_TOL * e.abs());
            if diff <= tol {
                AnswerComparison::Match
            } else {
                AnswerComparison::NonMatch
            }
        }
        // A non-numeric answer/oracle is not a confident non-match — let the judge decide.
        _ => AnswerComparison::Undecided,
    }
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
    kind: VerificationKind,
    answer: &str,
    expected: Option<&str>,
) -> AnswerComparison {
    let Some(expected) = expected else {
        return AnswerComparison::Undecided;
    };
    match kind {
        VerificationKind::NumericMatch => compare_numeric(answer, expected),
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

    #[test]
    fn numeric_tolerates_formatting() {
        assert_eq!(
            compare_answer(VerificationKind::NumericMatch, "42.0", Some("42")),
            AnswerComparison::Match
        );
        assert_eq!(
            compare_answer(VerificationKind::NumericMatch, "1,000", Some("1000")),
            AnswerComparison::Match
        );
        assert_eq!(
            compare_answer(VerificationKind::NumericMatch, "$3.50", Some("3.5")),
            AnswerComparison::Match
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
