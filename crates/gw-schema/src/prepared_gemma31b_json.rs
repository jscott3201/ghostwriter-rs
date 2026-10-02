//! Preserve duplicate-key and numeric evidence before parsing serial source JSON into maps.
use serde::de::{Error, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = Unique;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("finite JSON with unique keys")
            }
            fn visit_bool<E: Error>(self, v: bool) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_i64<E: Error>(self, v: i64) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_u64<E: Error>(self, v: u64) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_f64<E: Error>(self, v: f64) -> Result<Unique, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| Unique(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite serial number"))
            }
            fn visit_str<E: Error>(self, v: &str) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_string<E: Error>(self, v: String) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_unit<E: Error>(self) -> Result<Unique, E> {
                Ok(Unique(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Unique, A::Error> {
                let mut values = Vec::new();
                while let Some(Unique(v)) = seq.next_element()? {
                    values.push(v);
                }
                Ok(Unique(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Unique, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some((key, Unique(value))) = map.next_entry::<String, Unique>()? {
                    if values.insert(key, value).is_some() {
                        return Err(A::Error::custom("duplicate serial JSON key"));
                    }
                }
                Ok(Unique(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(JsonVisitor)
    }
}
pub(super) fn parse(raw: &str) -> Result<Value, &'static str> {
    serde_json::from_str::<Unique>(raw)
        .map(|v| v.0)
        .map_err(|_| "invalid/duplicate serial source JSON")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_numeric_and_duplicate_cases_match_literal_contract() {
        let cases: Value = serde_json::from_str(include_str!(
            "../../../adapters/trl/tests/gemma31b/argument_cases.json"
        ))
        .unwrap();
        for case in cases.as_array().unwrap() {
            let result = parse(case["raw"].as_str().unwrap())
                .and_then(|v| crate::prepared_gemma31b_render::argument(&v, false));
            if let Some(expected) = case["rendered"].as_str() {
                assert_eq!(result.unwrap(), expected, "{}", case["name"]);
            } else {
                assert!(result.is_err(), "{}", case["name"]);
            }
        }
    }
}
