//! Exact floating-point wire representation scoped to preference version 1. Ordinary record JSON
//! remains unchanged. The existing JSON number parser can shift some finite values by one ULP;
//! bit strings preserve evidence identity, including signed zero, without an equality tolerance.
//!
//! Only declared numeric positions are visited. Arbitrary dimension keys, explanations and raw
//! response strings are never interpreted as tags. Optional numbers are omitted when absent.
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::DeserializeOwned};
use serde_json::Value;

type Transform = dyn FnMut(&mut Value) -> Result<(), String>;
type Visit = fn(&mut Value, &mut Transform) -> Result<(), String>;

use crate::finite_numbers::{decode, encode};

fn serialize<T: Serialize, S: Serializer>(
    value: &T,
    serializer: S,
    visit: Visit,
) -> Result<S::Ok, S::Error> {
    let mut value = serde_json::to_value(value).map_err(serde::ser::Error::custom)?;
    visit(&mut value, &mut encode).map_err(serde::ser::Error::custom)?;
    value.serialize(serializer)
}

fn deserialize<'de, T: DeserializeOwned, D: Deserializer<'de>>(
    deserializer: D,
    visit: Visit,
) -> Result<T, D::Error> {
    let mut value = Value::deserialize(deserializer)?;
    visit(&mut value, &mut decode).map_err(serde::de::Error::custom)?;
    // from_value consumes the already exact Number; it does not parse a decimal token again.
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

fn fields(value: &mut Value, names: &[&str], transform: &mut Transform) -> Result<(), String> {
    for name in names {
        if let Some(number) = value.get_mut(*name) {
            transform(number)?;
        }
    }
    Ok(())
}

fn judging(value: &mut Value, transform: &mut Transform) -> Result<(), String> {
    fields(
        value,
        &["aggregate", "agreement", "n_eff", "threshold_at_decision"],
        transform,
    )?;
    if let Some(panel) = value.get_mut("panel").and_then(Value::as_array_mut) {
        for vote in panel {
            fields(vote, &["score", "temperature", "top_p"], transform)?;
            if let Some(dimensions) = vote.get_mut("dimensions").and_then(Value::as_object_mut) {
                for score in dimensions.values_mut() {
                    transform(score)?;
                }
            }
        }
    }
    Ok(())
}

fn verification(value: &mut Value, transform: &mut Transform) -> Result<(), String> {
    if let Some(checks) = value.get_mut("checks").and_then(Value::as_array_mut) {
        for check in checks {
            fields(check, &["score"], transform)?;
        }
    }
    Ok(())
}

fn contract(value: &mut Value, transform: &mut Transform) -> Result<(), String> {
    if let Some(tolerance) = value
        .get_mut("numeric")
        .and_then(|numeric| numeric.get_mut("tolerance"))
    {
        fields(tolerance, &["absolute", "relative"], transform)?;
    }
    Ok(())
}

fn quality(value: &mut Value, transform: &mut Transform) -> Result<(), String> {
    fields(value, &["reasoning_score", "fsf"], transform)?;
    if let Some(steps) = value.get_mut("steps").and_then(Value::as_array_mut) {
        for step in steps {
            fields(step, &["score"], transform)?;
        }
    }
    Ok(())
}

fn scalar(value: &mut Value, transform: &mut Transform) -> Result<(), String> {
    transform(value)
}

macro_rules! adapter {
    ($name:ident, $ty:ty, $visit:ident) => {
        pub(super) mod $name {
            /// Emit exact finite bit strings at this material snapshot's numeric positions.
            pub fn serialize<S: serde::Serializer>(
                value: &$ty,
                serializer: S,
            ) -> Result<S::Ok, S::Error> {
                super::serialize(value, serializer, super::$visit)
            }
            /// Reconstruct exact finite values without decimal parsing or interpreting prose.
            pub fn deserialize<'de, D: serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<$ty, D::Error> {
                super::deserialize(deserializer, super::$visit)
            }
        }
    };
}
adapter!(margin, f64, scalar);
adapter!(judge, crate::Judging, judging);
adapter!(verifier, crate::Verification, verification);
adapter!(task, Option<crate::VerificationContract>, contract);
adapter!(reasoning, Option<crate::ReasoningQuality>, quality);
