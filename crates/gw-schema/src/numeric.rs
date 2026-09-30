//! Explicit numeric answer semantics shared by task intake and verification. Decimal tokens are
//! rounded to IEEE-754 binary64; this is not arbitrary-precision decimal or integer equality.
//! No I/O.
use serde::{Deserialize, Serialize};

/// Where to read one numeric answer from assistant content. Reasoning is never searched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericExtraction {
    /// The entire content, except surrounding whitespace, must be one numeric token.
    WholeContent,
    /// Exactly one literal marker must begin the final nonempty line. The rest of that line must
    /// be one numeric token. Earlier content may contain prose but cannot contain the marker.
    FinalMarker {
        /// Nonempty, single-line, whitespace-trimmed marker such as `FINAL:`.
        marker: String,
    },
}

/// Finite nonnegative bounds. A match has distance at most `max(absolute, relative * abs(expected))`.
/// Distance is the absolute value of rounded binary64 subtraction; overflow exceeds every bound.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericTolerance {
    /// Absolute distance, including near zero.
    pub absolute: f64,
    /// Relative distance, scaled only by the expected value.
    pub relative: f64,
}

/// Persisted numeric extraction and comparison settings, with no hidden verifier tolerance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericComparison {
    /// The only accepted answer position.
    pub extraction: NumericExtraction,
    /// Explicit comparison bounds.
    pub tolerance: NumericTolerance,
}

impl Default for NumericComparison {
    /// Whole-content parsing with the historical relative `1e-6` and absolute `1e-9` bounds.
    /// These values are serialized into the contract before execution.
    fn default() -> Self {
        Self {
            extraction: NumericExtraction::WholeContent,
            tolerance: NumericTolerance {
                absolute: 1e-9,
                relative: 1e-6,
            },
        }
    }
}

impl NumericComparison {
    /// Validate extraction and tolerance declarations without resolving any oracle.
    ///
    /// # Errors
    /// Rejects blank/multiline/untrimmed markers and nonfinite or negative tolerances.
    pub fn validate(&self) -> Result<(), &'static str> {
        if let NumericExtraction::FinalMarker { marker } = &self.extraction
            && (marker.is_empty() || marker.trim() != marker || marker.contains(['\r', '\n']))
        {
            return Err("numeric final marker must be nonempty, trimmed, and single-line");
        }
        if !self.tolerance.absolute.is_finite()
            || !self.tolerance.relative.is_finite()
            || self.tolerance.absolute < 0.0
            || self.tolerance.relative < 0.0
        {
            return Err("numeric tolerances must be finite and nonnegative");
        }
        Ok(())
    }

    /// Resolve the finite comparison bound for one finite expected value.
    ///
    /// # Errors
    /// Rejects invalid settings/expected values or an overflowing relative bound.
    pub fn bound(&self, expected: f64) -> Result<f64, &'static str> {
        self.validate()?;
        if !expected.is_finite() {
            return Err("numeric expected value must be finite");
        }
        let magnitude = expected.abs();
        let relative = self.tolerance.relative;
        if relative > 1.0 && magnitude > f64::MAX / relative {
            return Err("numeric relative tolerance overflows for the expected value");
        }
        let scaled = relative * magnitude;
        if !scaled.is_finite() {
            return Err("numeric relative tolerance overflows for the expected value");
        }
        Ok(self.tolerance.absolute.max(scaled))
    }

    /// Extract the one declared token. Missing, repeated, or ambiguous markers yield `None`.
    #[must_use]
    pub fn extract<'a>(&self, content: &'a str) -> Option<&'a str> {
        match &self.extraction {
            NumericExtraction::WholeContent => Some(content.trim()),
            NumericExtraction::FinalMarker { marker } => {
                // Byte windows include overlapping literal occurrences, unlike match_indices.
                if self.validate().is_err()
                    || content
                        .as_bytes()
                        .windows(marker.len())
                        .filter(|window| *window == marker.as_bytes())
                        .take(2)
                        .count()
                        != 1
                {
                    return None;
                }
                let final_line = content
                    .lines()
                    .rev()
                    .find(|line| !line.trim().is_empty())?
                    .trim();
                final_line.strip_prefix(marker).map(str::trim)
            }
        }
    }
}

/// Parse one finite decimal/scientific token, allowing only surrounding whitespace.
///
/// Grammar: `[+-]?(digits(.digits*)?|.digits+)([eE][+-]?digits+)?`, with ASCII digits.
/// Currency, separators, percent, hexadecimal, NaN/infinity, and nonzero values that underflow to
/// zero are rejected. This function never searches prose for a number.
#[must_use]
pub fn parse_finite_decimal(text: &str) -> Option<f64> {
    let token = text.trim();
    let bytes = token.as_bytes();
    let mut i = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let mantissa_start = i;
    let mut digits = 0;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
        digits += 1;
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    let nonzero = bytes[mantissa_start..i]
        .iter()
        .any(|b| matches!(b, b'1'..=b'9'));
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let start = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return None;
        }
    }
    if i != bytes.len() {
        return None;
    }
    let value = token.parse::<f64>().ok()?;
    (value.is_finite() && (value != 0.0 || !nonzero)).then_some(value)
}
