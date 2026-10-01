//! Private finite binary64 primitive shared by versioned evidence codecs.
use serde_json::Value;
const TAG: &str = "binary64";

pub(super) fn encode(value: &mut Value) -> Result<(), String> {
    let number = value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or("preference numeric evidence must be finite")?;
    *value = serde_json::json!({TAG: format!("{:016x}", number.to_bits())});
    Ok(())
}

pub(super) fn decode(value: &mut Value) -> Result<(), String> {
    let object = value
        .as_object()
        .filter(|object| object.len() == 1)
        .ok_or("preference numbers require a binary64 bit object")?;
    let text = object
        .get(TAG)
        .and_then(Value::as_str)
        .ok_or("preference binary64 bits must be exactly 16 lowercase hexadecimal digits")?;
    let number = serde_json::Number::from_f64(parse_bits(text)?)
        .ok_or("preference numeric evidence must be finite")?;
    *value = Value::Number(number);
    Ok(())
}

/// Shared bit syntax and finite-number checks. Preference visitors retain their existing Value
/// boundary; new calibration fields use the strict streaming map visitor below.
fn parse_bits(text: &str) -> Result<f64, String> {
    if text.len() != 16
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(
            "preference binary64 bits must be exactly 16 lowercase hexadecimal digits".into(),
        );
    }
    let bits = u64::from_str_radix(text, 16).map_err(|error| error.to_string())?;
    let number = f64::from_bits(bits);
    if !number.is_finite() {
        return Err("preference numeric evidence must be finite".into());
    }
    Ok(number)
}

/// Decode directly from MapAccess so duplicate keys cannot collapse into a JSON Value first.
struct FiniteBinary64(f64);

impl<'de> serde::Deserialize<'de> for FiniteBinary64 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = FiniteBinary64;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object containing exactly one binary64 string")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                use serde::de::Error;
                let Some(key) = map.next_key::<String>()? else {
                    return Err(M::Error::missing_field(TAG));
                };
                if key != TAG {
                    return Err(M::Error::unknown_field(&key, &[TAG]));
                }
                let text = map.next_value::<String>()?;
                if let Some(key) = map.next_key::<String>()? {
                    return Err(if key == TAG {
                        M::Error::duplicate_field(TAG)
                    } else {
                        M::Error::unknown_field(&key, &[TAG])
                    });
                }
                parse_bits(&text)
                    .map(FiniteBinary64)
                    .map_err(M::Error::custom)
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

pub(super) mod scalar {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(number: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(number).map_err(serde::ser::Error::custom)?;
        super::encode(&mut value).map_err(serde::ser::Error::custom)?;
        value.serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        super::FiniteBinary64::deserialize(deserializer).map(|value| value.0)
    }
}

pub(super) mod optional {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        number: &Option<f64>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(number).map_err(serde::ser::Error::custom)?;
        if number.is_some() {
            super::encode(&mut value).map_err(serde::ser::Error::custom)?;
        }
        value.serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<f64>, D::Error> {
        Option::<super::FiniteBinary64>::deserialize(deserializer)
            .map(|value| value.map(|value| value.0))
    }
}

pub(super) mod vector {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(numbers: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        let mut values = numbers
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::ser::Error::custom)?;
        for value in &mut values {
            super::encode(value).map_err(serde::ser::Error::custom)?;
        }
        values.serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
        Vec::<super::FiniteBinary64>::deserialize(deserializer)
            .map(|values| values.into_iter().map(|value| value.0).collect())
    }
}
