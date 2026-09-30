//! Pure model declarations and supplied evidence, independent of run-manifest versions.
//!
//! Validation establishes shape and stable digests only. It performs no I/O, verifies no
//! signatures or loaded bytes, and grants no policy, cache, training, or deployment authority.
//! Deserializing a top-level document validates its structure and emits non-secret errors.
//! Rust callers can edit public fields; identity methods always revalidate before hashing.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, fmt};

/// A structural error containing no caller-supplied text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelIdentityError(&'static str);
impl fmt::Display for ModelIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for ModelIdentityError {}
type Result<T> = std::result::Result<T, ModelIdentityError>;

// Keep validated public documents and their strict wire representations in sync.
macro_rules! document {
    ($(#[$meta:meta])* pub struct $name:ident { $( $(#[$field_meta:meta])* pub $field:ident: $ty:ty ),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
        pub struct $name { $( $(#[$field_meta])* pub $field: $ty ),* }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Wire { $( $field: $ty ),* }
                let wire = Wire::deserialize(deserializer)
                    .map_err(|_| serde::de::Error::custom("invalid model identity document"))?;
                let result = Self { $( $field: wire.$field ),* };
                result.validate().map_err(serde::de::Error::custom)?;
                Ok(result)
            }
        }
        impl $name {
            /// Decode and structurally validate JSON without including input in errors.
            pub fn from_json(input: &str) -> Result<Self> {
                serde_json::from_str(input).map_err(|_| ModelIdentityError("invalid model identity document"))
            }
        }
    };
}

/// An explicit unknown or a supplied declaration. Neither variant establishes verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "status",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Declaration<T> {
    /// No declaration is available; never inferred from an alias or endpoint.
    Unknown,
    /// A caller supplied this value, pending any required independent qualification.
    Declared(T),
}
impl<T> Declaration<T> {
    fn check(&self, check: impl FnOnce(&T) -> Result<()>) -> Result<()> {
        match self {
            Self::Unknown => Ok(()),
            Self::Declared(value) => check(value),
        }
    }
}

/// Algorithm used by a supplied file or evidence digest. Both use 32-byte digests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DigestAlgorithm {
    /// BLAKE3, without a keyed or derive-key context.
    Blake3,
    /// SHA-256.
    Sha256,
}
document! {
    /// Declared digest of exact bytes; validating it does not read or hash those bytes.
    pub struct ContentDigest {
        /// Algorithm used by the producer.
        pub algorithm: DigestAlgorithm,
        /// Exactly 64 lowercase hexadecimal characters.
        pub hex: String,
    }
}
impl ContentDigest {
    /// Check the digest encoding.
    pub fn validate(&self) -> Result<()> {
        digest(&self.hex)
    }
}
macro_rules! identity {
    ($name:ident, $domain:literal, $docs:literal) => {
        document! {
            #[doc = $docs]
            pub struct $name {
                /// Digest contract version; only version 1 is supported.
                pub version: u32,
                /// Lowercase BLAKE3 derive-key digest in this identity's distinct domain.
                pub digest: String,
            }
        }
        impl $name {
            /// Check the version and digest encoding, without resolving the referenced document.
            pub fn validate(&self) -> Result<()> {
                version(self.version)?;
                digest(&self.digest)
            }
            fn of<T: Serialize>(value: &T) -> Result<Self> {
                Ok(Self {
                    version: 1,
                    digest: hash($domain, value)?,
                })
            }
        }
    };
}
identity!(
    ArtifactIdentity,
    "ghostwriter.model-artifact.v1",
    "Identity of a pinned artifact declaration; no proof of acquired or loaded files."
);
identity!(
    PolicyDocumentIdentity,
    "ghostwriter.model-policy-document.v1",
    "Identity of one pinned policy document and its declared role/use; no policy approval."
);
identity!(
    DeploymentEvidenceIdentity,
    "ghostwriter.deployment-evidence.v1",
    "Binding of supplied deployment evidence, including its endpoint and incarnation."
);
identity!(
    SemanticExecutionIdentity,
    "ghostwriter.model-execution-semantics.v1",
    "Identity of declared execution semantics; unknowns never imply safe cache reuse."
);
identity!(
    ObservedExecutionIdentity,
    "ghostwriter.observed-model-execution.v1",
    "Binding of a model observation to its durable physical attempt reference."
);

/// A credential-free locator in a deliberately narrow, non-normalizing grammar.
///
/// HTTP(S) locators accept a DNS/IPv4 or bracketed IPv6 host, optional numeric port,
/// and an ASCII path. URNs accept a nonempty namespace and colon-separated identifier.
/// Userinfo, query, fragment, percent escapes, backslashes, whitespace, and dot path
/// segments are rejected. Authentication belongs outside persisted identity documents.
/// A safe locator is not proof that the resource exists or is immutable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ModelReference(String);
impl ModelReference {
    /// Validate a locator without retaining rejected input in an error.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        reference(&value)?;
        Ok(Self(value))
    }
    /// Return the unchanged locator; spelling contributes to evidence identities.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
    fn endpoint(&self) -> Result<()> {
        if self.0.starts_with("http://") || self.0.starts_with("https://") {
            Ok(())
        } else {
            Err(ModelIdentityError("endpoint must use HTTP or HTTPS"))
        }
    }
}
impl<'de> Deserialize<'de> for ModelReference {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = String::deserialize(deserializer)
            .map_err(|_| serde::de::Error::custom("invalid model reference"))?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

document! {
    /// A declared immutable source revision. No network access or pin verification occurs.
    pub struct PinnedModelSource {
        /// Credential-free source locator; its spelling contributes to identity.
        pub reference: ModelReference,
        /// Producer-declared immutable revision, never a request to resolve a moving branch.
        pub revision: String,
    }
}
impl PinnedModelSource {
    /// Check the explicit source revision.
    pub fn validate(&self) -> Result<()> {
        nonempty(&self.revision)
    }
}

fn version(value: u32) -> Result<()> {
    if value == 1 {
        Ok(())
    } else {
        Err(ModelIdentityError("unsupported model identity version"))
    }
}
fn nonempty(value: &str) -> Result<()> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        Err(ModelIdentityError("missing or invalid model declaration"))
    } else {
        Ok(())
    }
}
fn digest(value: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(ModelIdentityError("invalid model digest"))
    }
}
fn semantic(value: &crate::SemanticDeclaration) -> Result<()> {
    nonempty(&value.implementation)?;
    nonempty(&value.revision)?;
    if value.is_valid() {
        Ok(())
    } else {
        Err(ModelIdentityError("invalid semantic declaration"))
    }
}
fn file_path(value: &str) -> Result<()> {
    let valid = !value.is_empty()
        && value.split('/').all(|part| {
            let stem = part
                .split('.')
                .next()
                .unwrap_or_default()
                .to_ascii_uppercase();
            !part.is_empty()
                && part != "."
                && part != ".."
                && !part.ends_with('.')
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                && !matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
                && !(stem.len() == 4
                    && (stem.starts_with("COM") || stem.starts_with("LPT"))
                    && matches!(stem.as_bytes()[3], b'1'..=b'9'))
        });
    if valid {
        Ok(())
    } else {
        Err(ModelIdentityError(
            "invalid relative artifact file identifier",
        ))
    }
}
fn reference(value: &str) -> Result<()> {
    let bad = || ModelIdentityError("invalid credential-free model reference");
    if !value.is_ascii()
        || value
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b.is_ascii_control() || b"@?#%\\".contains(&b))
    {
        return Err(bad());
    }
    if let Some(urn) = value.strip_prefix("urn:") {
        if urn.split(':').count() < 2
            || !urn.split(':').all(|part| {
                !part.is_empty()
                    && part
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            })
        {
            return Err(bad());
        }
        return Ok(());
    }
    let rest = value
        .strip_prefix("https://")
        .or_else(|| value.strip_prefix("http://"))
        .ok_or_else(bad)?;
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = if authority.starts_with('[') {
        let (host, tail) = authority.split_once(']').ok_or_else(bad)?;
        host[1..].parse::<std::net::Ipv6Addr>().map_err(|_| bad())?;
        (
            host,
            if tail.is_empty() {
                None
            } else {
                Some(tail.strip_prefix(':').ok_or_else(bad)?)
            },
        )
    } else {
        let (host, port) = authority
            .split_once(':')
            .map_or((authority, None), |(h, p)| (h, Some(p)));
        if host.is_empty()
            || !host.split('.').all(|label| {
                !label.is_empty()
                    && !label.starts_with('-')
                    && !label.ends_with('-')
                    && label
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            })
        {
            return Err(bad());
        }
        (host, port)
    };
    if host.is_empty()
        || port.is_some_and(|p| {
            p.is_empty()
                || !p.bytes().all(|b| b.is_ascii_digit())
                || p.parse::<u16>().map_or(true, |p| p == 0)
        })
    {
        return Err(bad());
    }
    if !path
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"/._-~".contains(&b))
        || path.split('/').any(|part| part == "." || part == "..")
    {
        return Err(bad());
    }
    Ok(())
}
fn canonical(value: &Value, output: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Object(map) => {
            output.push(b'{');
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)
                    .map_err(|_| ModelIdentityError("invalid canonical identity"))?;
                output.push(b':');
                canonical(&map[*key], output)?;
            }
            output.push(b'}');
        }
        Value::Array(values) => {
            output.push(b'[');
            for (i, value) in values.iter().enumerate() {
                if i > 0 {
                    output.push(b',');
                }
                canonical(value, output)?;
            }
            output.push(b']');
        }
        _ => serde_json::to_writer(output, value)
            .map_err(|_| ModelIdentityError("invalid canonical identity"))?,
    }
    Ok(())
}
fn hash<T: Serialize>(domain: &str, value: &T) -> Result<String> {
    let value = serde_json::to_value(value)
        .map_err(|_| ModelIdentityError("invalid canonical identity"))?;
    let mut bytes = Vec::new();
    canonical(&value, &mut bytes)?;
    let mut hasher = blake3::Hasher::new_derive_key(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}
fn artifact_set(values: &[ArtifactIdentity]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        value.validate()?;
        if !seen.insert(&value.digest) {
            return Err(ModelIdentityError("duplicate artifact identity"));
        }
    }
    Ok(())
}
fn sort_artifacts(values: &mut Declaration<Vec<ArtifactIdentity>>) {
    if let Declaration::Declared(values) = values {
        values.sort_by(|a, b| a.digest.cmp(&b.digest));
    }
}

mod artifact;
mod execution;
mod policy;
pub use artifact::*;
pub use execution::*;
pub use policy::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identical_payloads_have_distinct_typed_digest_domains() {
        let value = serde_json::json!({"version":1,"same":"payload"});
        let digests = [
            ArtifactIdentity::of(&value).unwrap().digest,
            PolicyDocumentIdentity::of(&value).unwrap().digest,
            DeploymentEvidenceIdentity::of(&value).unwrap().digest,
            SemanticExecutionIdentity::of(&value).unwrap().digest,
            ObservedExecutionIdentity::of(&value).unwrap().digest,
        ];
        assert_eq!(digests.iter().collect::<BTreeSet<_>>().len(), 5);
    }
}
