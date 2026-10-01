use super::{ArtifactAssessmentError, SuppliedModelPolicy};
use gw_schema::DigestAlgorithm;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(super) fn bytes_digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

pub(super) fn policy_bytes_match(policy: &SuppliedModelPolicy) -> bool {
    let expected = &policy.declaration.document.content;
    let observed = match expected.algorithm {
        DigestAlgorithm::Blake3 => bytes_digest(&policy.bytes),
        DigestAlgorithm::Sha256 => format!("{:x}", Sha256::digest(&policy.bytes)),
    };
    expected.hex == observed
}

pub(super) fn text_valid(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

pub(super) fn hash<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<String, ArtifactAssessmentError> {
    let mut value = serde_json::to_value(value).map_err(|_| ArtifactAssessmentError)?;
    sort_keys(&mut value);
    let bytes = serde_json::to_vec(&value).map_err(|_| ArtifactAssessmentError)?;
    let mut hasher = blake3::Hasher::new_derive_key(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn sort_keys(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // sort_keys is required even if another workspace consumer enables preserve_order.
            map.sort_keys();
            for value in map.values_mut() {
                sort_keys(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                sort_keys(value);
            }
        }
        _ => {}
    }
}
