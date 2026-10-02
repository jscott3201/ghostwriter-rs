//! Protocol-local strict JSON. Existing integer-only and prepared-data decoders stay unchanged.
use super::{REPOSITORY_EPISODE_MAX_BYTES, RepositoryEpisodeError, Result};
use serde::de::{DeserializeSeed, Error, MapAccess, SeqAccess, Visitor};
use serde_json::{Number, Value};
use std::collections::VecDeque;

/// Parse bounded finite JSON, rejecting duplicate keys and unrepresentable numbers.
///
/// Integers must fit i64/u64. Fractional/exponent numbers use Rust's correctly rounded f64 parser;
/// nonzero underflow and nonfinite values are rejected. Bare `-0` is canonically floating `-0.0`,
/// as are `-0.0` and `-0e0`; positive integer `0` remains distinct in serialized identities.
/// Numeric source spelling otherwise has no identity. Preserve source spelling in `raw_arguments`
/// when it matters. Strings are never inspected as JSON or altered.
///
/// # Errors
/// Returns fixed diagnostics without including supplied text or field names.
pub fn strict_repository_json(bytes: &[u8]) -> Result<Value> {
    if bytes.len() > REPOSITORY_EPISODE_MAX_BYTES {
        return Err(RepositoryEpisodeError(
            "repository JSON exceeds 32 MiB bound",
        ));
    }
    let (structure, mut numbers) = numeric_structure(bytes)?;
    let mut decoder = serde_json::Deserializer::from_slice(&structure);
    let value = JsonSeed(&mut numbers)
        .deserialize(&mut decoder)
        .map_err(|_| RepositoryEpisodeError("invalid or duplicate repository JSON"))?;
    decoder
        .end()
        .map_err(|_| RepositoryEpisodeError("trailing repository JSON data"))?;
    if !numbers.is_empty() {
        return Err(RepositoryEpisodeError("invalid repository number sequence"));
    }
    Ok(value)
}

// Validate the JSON number grammar and interpret each value exactly once. Replace those tokens
// with integer zero plus padding for structural decoding, so serde never imposes a second numeric
// acceptance boundary. Strings and all other bytes stay exact; numeric visits consume the saved
// values in input order. Padding preserves delimiters and offsets without expanding the input.
fn numeric_structure(bytes: &[u8]) -> Result<(Vec<u8>, VecDeque<Number>)> {
    let mut structure = bytes.to_vec();
    let mut values = VecDeque::new();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                index += 1;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => index += 2,
                        b'"' => {
                            index += 1;
                            break;
                        }
                        _ => index += 1,
                    }
                }
            }
            b'-' | b'0'..=b'9' => {
                let start = index;
                while index < bytes.len()
                    && matches!(bytes[index], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    index += 1;
                }
                let token = std::str::from_utf8(&bytes[start..index])
                    .map_err(|_| RepositoryEpisodeError("invalid repository number"))?;
                number_grammar(token.as_bytes())?;
                values.push_back(number(token)?);
                structure[start] = b'0';
                structure[start + 1..index].fill(b' ');
            }
            _ => index += 1,
        }
    }
    Ok((structure, values))
}
fn number_grammar(token: &[u8]) -> Result<()> {
    let invalid = || RepositoryEpisodeError("invalid repository number grammar");
    let mut index = usize::from(token.first() == Some(&b'-'));
    match token.get(index) {
        Some(b'0') => index += 1,
        Some(b'1'..=b'9') => {
            index += 1;
            while token.get(index).is_some_and(u8::is_ascii_digit) {
                index += 1;
            }
        }
        _ => return Err(invalid()),
    }
    if token.get(index) == Some(&b'.') {
        index += 1;
        let first = index;
        while token.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if first == index {
            return Err(invalid());
        }
    }
    if matches!(token.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(token.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let first = index;
        while token.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if first == index {
            return Err(invalid());
        }
    }
    if index != token.len() {
        return Err(invalid());
    }
    Ok(())
}
fn number(token: &str) -> Result<Number> {
    let invalid = || RepositoryEpisodeError("unsupported repository number");
    if token == "-0" {
        return Number::from_f64(-0.0).ok_or_else(invalid);
    }
    if !token.contains(['.', 'e', 'E']) {
        return if token.starts_with('-') {
            token
                .parse::<i64>()
                .map(Number::from)
                .map_err(|_| invalid())
        } else {
            token
                .parse::<u64>()
                .map(Number::from)
                .map_err(|_| invalid())
        };
    }
    let value = token.parse::<f64>().map_err(|_| invalid())?;
    let significand = token.split(['e', 'E']).next().ok_or_else(invalid)?;
    if !value.is_finite() || (value == 0.0 && significand.bytes().any(|b| matches!(b, b'1'..=b'9')))
    {
        return Err(invalid());
    }
    Number::from_f64(value).ok_or_else(invalid)
}
struct JsonSeed<'a>(&'a mut VecDeque<Number>);
impl<'de> DeserializeSeed<'de> for JsonSeed<'_> {
    type Value = Value;
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Value, D::Error> {
        d.deserialize_any(self)
    }
}
impl JsonSeed<'_> {
    fn numeric<E: Error>(&mut self) -> std::result::Result<Value, E> {
        self.0
            .pop_front()
            .map(Value::Number)
            .ok_or_else(|| E::custom("missing number"))
    }
}
impl<'de> Visitor<'de> for JsonSeed<'_> {
    type Value = Value;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("finite JSON with unique object keys")
    }
    fn visit_bool<E: Error>(self, v: bool) -> std::result::Result<Value, E> {
        Ok(v.into())
    }
    fn visit_i64<E: Error>(mut self, _: i64) -> std::result::Result<Value, E> {
        self.numeric()
    }
    fn visit_u64<E: Error>(mut self, _: u64) -> std::result::Result<Value, E> {
        self.numeric()
    }
    fn visit_f64<E: Error>(mut self, _: f64) -> std::result::Result<Value, E> {
        self.numeric()
    }
    fn visit_str<E: Error>(self, v: &str) -> std::result::Result<Value, E> {
        Ok(v.into())
    }
    fn visit_string<E: Error>(self, v: String) -> std::result::Result<Value, E> {
        Ok(v.into())
    }
    fn visit_unit<E: Error>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element_seed(JsonSeed(self.0))? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Value, A::Error> {
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom("duplicate JSON key"));
            }
            values.insert(key, map.next_value_seed(JsonSeed(self.0))?);
        }
        Ok(Value::Object(values))
    }
}
