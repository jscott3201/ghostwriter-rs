//! Field-independent accounting extraction before teacher, judge, vector or delta parsing.
use gw_schema::{AttemptMetadata, ReportedCost};
use serde::Deserialize;
use serde_json::value::RawValue;

#[derive(Deserialize)]
struct Envelope {
    #[serde(default)]
    id: Option<Box<RawValue>>,
    #[serde(default)]
    model: Option<Box<RawValue>>,
    #[serde(default)]
    provider: Option<Box<RawValue>>,
    #[serde(default)]
    usage: Option<Box<RawValue>>,
}
#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: Option<Box<RawValue>>,
    #[serde(default)]
    completion_tokens: Option<Box<RawValue>>,
    #[serde(default)]
    total_tokens: Option<Box<RawValue>>,
    #[serde(default)]
    completion_tokens_details: Option<Box<RawValue>>,
    #[serde(default)]
    cost: Option<Box<RawValue>>,
}
#[derive(Deserialize)]
struct Details {
    #[serde(default)]
    reasoning_tokens: Option<Box<RawValue>>,
}

pub(crate) fn extract_json(bytes: &[u8]) -> Option<AttemptMetadata> {
    // RawValue isolates each field, including out-of-range JSON numbers: one bad cost or
    // content field must not erase valid token evidence elsewhere in the same envelope.
    let envelope: Envelope = serde_json::from_slice(bytes).ok()?;
    let mut metadata = AttemptMetadata::default();
    for (name, raw, slot) in [
        ("id", envelope.id, &mut metadata.response_id),
        ("model", envelope.model, &mut metadata.model),
        ("provider", envelope.provider, &mut metadata.provider),
    ] {
        if let Some(raw) = raw {
            match serde_json::from_str::<String>(raw.get()) {
                Ok(text) if !text.is_empty() => *slot = Some(text),
                _ => metadata.invalid_fields.push(name.into()),
            }
        }
    }
    let Some(raw_usage) = envelope.usage else {
        return Some(metadata);
    };
    let Ok(usage) = serde_json::from_str::<Usage>(raw_usage.get()) else {
        metadata.invalid_fields.push("usage".into());
        return Some(metadata);
    };
    for (name, raw, slot) in [
        (
            "prompt_tokens",
            usage.prompt_tokens,
            &mut metadata.prompt_tokens,
        ),
        (
            "completion_tokens",
            usage.completion_tokens,
            &mut metadata.completion_tokens,
        ),
        (
            "total_tokens",
            usage.total_tokens,
            &mut metadata.total_tokens,
        ),
    ] {
        if let Some(raw) = raw {
            *slot = exact_count(raw.get());
            if slot.is_none() {
                metadata.invalid_fields.push(name.into());
            }
        }
    }
    if let Some(raw) = usage.completion_tokens_details {
        match serde_json::from_str::<Details>(raw.get()) {
            Ok(details) => {
                if let Some(raw) = details.reasoning_tokens {
                    metadata.reasoning_tokens = exact_count(raw.get());
                    if metadata.reasoning_tokens.is_none() {
                        metadata.invalid_fields.push("reasoning_tokens".into());
                    }
                }
            }
            Err(_) => metadata
                .invalid_fields
                .push("completion_tokens_details".into()),
        }
    }
    if let Some(raw) = usage.cost {
        let number = serde_json::from_str::<f64>(raw.get()).ok().or_else(|| {
            serde_json::from_str::<String>(raw.get())
                .ok()
                .and_then(|text| text.parse::<f64>().ok())
        });
        metadata.cost_usd = match number {
            Some(cost) if cost.is_finite() && cost >= 0.0 => ReportedCost::Known(cost),
            _ => {
                metadata.invalid_fields.push("cost".into());
                ReportedCost::Invalid
            }
        };
    }
    Some(metadata)
}
fn exact_count(raw: &str) -> Option<u64> {
    serde_json::from_str::<u64>(raw).ok().or_else(|| {
        serde_json::from_str::<String>(raw)
            .ok()
            .and_then(|text| text.parse::<u64>().ok())
    })
}

/// Update only supplied fields so ordinary content chunks cannot cause writes by omitting usage.
pub(crate) fn merge_present(current: &mut AttemptMetadata, patch: &AttemptMetadata) {
    if let Some(value) = patch.prompt_tokens {
        current.prompt_tokens = Some(value);
    }
    if let Some(value) = patch.completion_tokens {
        current.completion_tokens = Some(value);
    }
    if let Some(value) = patch.total_tokens {
        current.total_tokens = Some(value);
    }
    if let Some(value) = patch.reasoning_tokens {
        current.reasoning_tokens = Some(value);
    }
    if let Some(value) = &patch.response_id {
        current.response_id = Some(value.clone());
    }
    if let Some(value) = &patch.model {
        current.model = Some(value.clone());
    }
    if let Some(value) = &patch.provider {
        current.provider = Some(value.clone());
    }
    if patch.cost_usd != ReportedCost::Missing {
        current.cost_usd = patch.cost_usd.clone();
    }
    for field in &patch.invalid_fields {
        if !current.invalid_fields.contains(field) {
            current.invalid_fields.push(field.clone());
        }
    }
}
