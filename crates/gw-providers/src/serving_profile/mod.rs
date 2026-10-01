//! Offline serving contracts and supplied-snapshot consistency checks.
//!
//! Profiles declare wire behavior; preparation does not construct clients, resolve credentials,
//! inspect a deployment, or grant execution/cache authority. Supplied loader/gateway documents
//! are unauthenticated claims. A real loader must measure the actual loaded generation, retain
//! that same generation atomically through inference, and authenticate attempt-bound evidence.
//! None of those lifecycle guarantees follows from these pure consistency checks.
//!
//! An owned-Modal profile stores two secret references and uses ObservationOnly defaults:
//!
//! ```
//! use gw_providers::serving_profile::{
//!     ProfileBehavior, SecretReference, ServingDialect, ServingProfile,
//! };
//! use gw_schema::{AccountingPolicy, ModelOperation};
//! let profile = ServingProfile::modal(
//!     "https://fixture.modal.run/v1",
//!     SecretReference::new("MODAL_PROXY_ID")?,
//!     SecretReference::new("MODAL_PROXY_SECRET")?,
//!     ProfileBehavior {
//!         version: 1,
//!         dialect: ServingDialect::VllmV1,
//!         operations: vec![ModelOperation::ChatCompletion],
//!         capabilities: vec![],
//!         reasoning_efforts: vec![],
//!     },
//! )?;
//! assert_eq!(profile.operations.accounting, AccountingPolicy::ObservationOnly);
//! # Ok::<(), gw_providers::serving_profile::ProfileError>(())
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Non-secret failures at the offline serving boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ProfileError {
    /// Unsupported document version, field, or combination.
    #[error("invalid or unsupported serving profile")]
    InvalidProfile,
    /// Supplied execution declarations are incomplete or inconsistent with the profile.
    #[error("incomplete or mismatched execution semantics")]
    InvalidSemantics,
    /// Unsupported or malformed canonical request data.
    #[error("invalid or unsupported request data")]
    InvalidRequest,
    /// A requested required control is not supported by the explicit declaration.
    #[error("required request control is unsupported: {0:?}")]
    UnsupportedRequiredControl(ProfileControl),
    /// A response could not be safely interpreted by the supported decoder.
    #[error("invalid or unsupported response data")]
    InvalidResponse,
    /// Both aliases were supplied with different reasoning text.
    #[error("conflicting reasoning aliases")]
    ConflictingReasoning,
    /// Canonical representation could not be encoded.
    #[error("serving contract could not be encoded")]
    Encoding,
}
type Result<T> = std::result::Result<T, ProfileError>;

// Every decoded profile document revalidates, including when nested. Public fields remain
// convenient for configuration; pure operations validate them again before using them.
macro_rules! document {
    ($(#[$meta:meta])* pub struct $name:ident { $( $(#[$field_meta:meta])* pub $field:ident: $ty:ty ),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Serialize)]
        pub struct $name { $( $(#[$field_meta])* pub $field: $ty ),* }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Wire { $( $field: $ty ),* }
                let wire = Wire::deserialize(deserializer)
                    .map_err(|_| serde::de::Error::custom("invalid serving contract"))?;
                let value = Self { $( $field: wire.$field ),* };
                value.validate().map_err(serde::de::Error::custom)?;
                Ok(value)
            }
        }
        impl $name {
            /// Strictly decode supplied JSON without I/O or rejected input in errors.
            pub fn from_json(input: &str) -> Result<Self> {
                serde_json::from_str(input).map_err(|_| ProfileError::InvalidProfile)
            }
        }
    };
}

fn text_valid(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}
fn canonical_bytes(value: &impl Serialize) -> Result<Vec<u8>> {
    fn sort(value: &mut Value) {
        match value {
            Value::Object(map) => {
                map.sort_keys();
                for child in map.values_mut() {
                    sort(child);
                }
            }
            Value::Array(values) => values.iter_mut().for_each(sort),
            _ => {}
        }
    }
    let mut value = serde_json::to_value(value).map_err(|_| ProfileError::Encoding)?;
    sort(&mut value);
    serde_json::to_vec(&value).map_err(|_| ProfileError::Encoding)
}
fn body_digest(bytes: &[u8]) -> gw_schema::ContentDigest {
    gw_schema::ContentDigest {
        algorithm: gw_schema::DigestAlgorithm::Blake3,
        hex: blake3::hash(bytes).to_hex().to_string(),
    }
}

mod authentication;
mod consistency;
mod controls;
mod gateway;
mod profile;
mod request;
mod response;
mod validity;
pub use authentication::*;
pub use gateway::*;
pub use profile::*;
pub use request::*;
pub use response::*;
