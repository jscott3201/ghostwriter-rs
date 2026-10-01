//! Bounded, exact values exchanged with the isolated Python function wrapper.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Exact cross-language value contract. Integers are signed 64-bit; floats are unsupported.
/// Booleans are distinct from integers. Lists preserve order and length; object key order is
/// irrelevant. Strings retain exact Unicode codepoints without normalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum CodingValue {
    /// Python `None`.
    Null,
    /// Python `bool`, never an integer subclass match.
    Boolean(bool),
    /// Python `int` in the inclusive i64 range.
    Integer(i64),
    /// Exact UTF-8 string.
    String(String),
    /// Ordered Python list, not a tuple or iterable.
    Array(Vec<CodingValue>),
    /// Python dict with exact string keys.
    Object(BTreeMap<String, CodingValue>),
}
impl CodingValue {
    /// Parse one result, rejecting duplicate keys, floats, extra fields, and trailing output.
    ///
    /// # Errors
    /// Rejects malformed, unsupported, or oversized values.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > 32 * 1024 {
            return Err("coding result exceeds byte bound".into());
        }
        let raw = strict_coding_json(bytes)?;
        let value: Self = serde_json::from_value(raw.clone()).map_err(|error| error.to_string())?;
        if serde_json::to_value(&value).map_err(|error| error.to_string())? != raw {
            return Err("coding value has unsupported or omitted fields".into());
        }
        value.validate().map_err(str::to_owned)?;
        Ok(value)
    }

    /// Check depth, node count, and total text size before dispatch or comparison.
    ///
    /// # Errors
    /// Rejects depth over 16, more than 2048 nodes, or more than 16 KiB of UTF-8 text.
    pub fn validate(&self) -> Result<(), &'static str> {
        let mut nodes = 0;
        let mut bytes = 0;
        self.walk(0, &mut nodes, &mut bytes)
    }

    fn walk(&self, depth: usize, nodes: &mut usize, bytes: &mut usize) -> Result<(), &'static str> {
        *nodes += 1;
        if depth > 16 || *nodes > 2048 {
            return Err("coding value exceeds depth or node bound");
        }
        match self {
            Self::String(text) => *bytes += text.len(),
            Self::Array(values) => {
                for value in values {
                    value.walk(depth + 1, nodes, bytes)?;
                }
            }
            Self::Object(values) => {
                for (key, value) in values {
                    *bytes += key.len();
                    value.walk(depth + 1, nodes, bytes)?;
                }
            }
            Self::Null | Self::Boolean(_) | Self::Integer(_) => {}
        }
        if *bytes > 16 * 1024 {
            return Err("coding value exceeds text bound");
        }
        Ok(())
    }
}

/// Parse strict integer-only JSON without erasing duplicate object fields.
///
/// # Errors
/// Rejects malformed JSON, duplicate fields, floating-point numbers, and trailing data.
pub fn strict_coding_json(bytes: &[u8]) -> Result<serde_json::Value, String> {
    crate::prepared_sft_frame::strict_prepared_json(bytes)
        .map_err(|error| error.replace("prepared SFT", "coding"))
}

/// Domain-separated identity over exact captured bytes. This is integrity, not authentication.
#[must_use]
pub fn coding_digest(domain: &str, bytes: &[u8]) -> String {
    let mut hash = blake3::Hasher::new_derive_key(domain);
    hash.update(bytes);
    hash.finalize().to_hex().to_string()
}

pub(crate) fn coding_json_digest<T: Serialize>(domain: &str, value: &T) -> String {
    // Value sorts object keys; arrays and strings preserve their exact semantics.
    let canonical = serde_json::to_value(value).expect("coding contract is JSON-representable");
    coding_digest(domain, &serde_json::to_vec(&canonical).expect("JSON value"))
}

pub(crate) fn coding_hash_valid(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
