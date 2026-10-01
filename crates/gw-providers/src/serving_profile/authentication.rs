use super::*;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};

/// Validated reference name in a caller-owned secret store. Never a stored credential value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct SecretReference(String);
impl SecretReference {
    /// Validate a nonempty ASCII reference name without reading any secret source.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty()
            || !value
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-./:".contains(&c))
        {
            return Err(ProfileError::InvalidProfile);
        }
        Ok(Self(value))
    }
    /// Reference name to pass to an explicitly injected resolver.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl<'de> Deserialize<'de> for SecretReference {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = String::deserialize(deserializer)
            .map_err(|_| serde::de::Error::custom("invalid secret reference"))?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Explicit authentication policy; neither an SDK credential lookup nor a no-auth fallback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProfileAuthentication {
    /// Explicit unauthenticated configuration. Does not prove endpoint permission.
    None {},
    /// One caller-resolved secret used as Authorization: Bearer.
    Bearer {
        /// Reference to the bearer value.
        secret: SecretReference,
    },
    /// Modal's token-ID/token-secret pair, always emitted as two separately sensitive headers.
    ModalProxy {
        /// Reference resolved into Modal-Key.
        token_id: SecretReference,
        /// Different reference resolved into Modal-Secret.
        token_secret: SecretReference,
    },
}
impl ProfileAuthentication {
    /// Require separate references for the Modal pair. No secret source is accessed.
    pub fn validate(&self) -> Result<()> {
        if let Self::ModalProxy {
            token_id,
            token_secret,
        } = self
            && token_id == token_secret
        {
            return Err(ProfileError::InvalidProfile);
        }
        Ok(())
    }
}

/// Caller-owned secret value with redacted Debug and no serialization or display implementation.
pub struct SecretValue(String);
impl SecretValue {
    /// Wrap a resolver's value. Header-domain validation happens in resolve_authentication.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}
impl std::fmt::Debug for SecretValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretValue([REDACTED])")
    }
}

/// Static resolution failures never include credential values or resolver error text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AuthenticationError {
    /// The injected resolver failed.
    #[error("credential resolver failed")]
    Resolver,
    /// No value exists for a required reference.
    #[error("required credential is missing")]
    Missing,
    /// Empty, whitespace-bearing or invalid HTTP header credential.
    #[error("credential is not a valid nonempty header token")]
    InvalidValue,
    /// Authentication references violate the explicit configuration contract.
    #[error("invalid authentication configuration")]
    InvalidConfiguration,
}

/// Resolved headers with sensitive values, deliberately separate from persisted profile data.
/// This is credential material, not an execution or deployment qualification capability.
pub struct ResolvedAuthentication(HeaderMap);
impl std::fmt::Debug for ResolvedAuthentication {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ResolvedAuthentication([REDACTED])")
    }
}
impl ResolvedAuthentication {
    /// Borrow sensitive headers for an explicitly owned transport boundary. No client is built.
    #[must_use]
    pub fn headers(&self) -> &HeaderMap {
        &self.0
    }
}

/// Resolve only through the injected callback, producing exact bearer or Modal proxy headers.
/// The callback owns any credential source; this module never discovers environment/SDK values.
///
/// # Errors
/// Missing, empty or invalid values fail the whole operation. There is no no-auth fallback.
pub fn resolve_authentication(
    policy: &ProfileAuthentication,
    mut resolve: impl FnMut(
        &SecretReference,
    ) -> std::result::Result<Option<SecretValue>, AuthenticationError>,
) -> std::result::Result<ResolvedAuthentication, AuthenticationError> {
    policy
        .validate()
        .map_err(|_| AuthenticationError::InvalidConfiguration)?;
    let mut header = |reference: &SecretReference, bearer: bool| {
        let value = resolve(reference)
            .map_err(|_| AuthenticationError::Resolver)?
            .ok_or(AuthenticationError::Missing)?;
        if value.0.is_empty() || !value.0.bytes().all(|c| c.is_ascii_graphic()) {
            return Err(AuthenticationError::InvalidValue);
        }
        let value = if bearer {
            format!("Bearer {}", value.0)
        } else {
            value.0
        };
        let mut value =
            HeaderValue::from_str(&value).map_err(|_| AuthenticationError::InvalidValue)?;
        value.set_sensitive(true);
        Ok(value)
    };
    let mut headers = HeaderMap::new();
    match policy {
        ProfileAuthentication::None {} => {}
        ProfileAuthentication::Bearer { secret } => {
            headers.insert(AUTHORIZATION, header(secret, true)?);
        }
        ProfileAuthentication::ModalProxy {
            token_id,
            token_secret,
        } => {
            headers.insert("Modal-Key", header(token_id, false)?);
            headers.insert("Modal-Secret", header(token_secret, false)?);
        }
    }
    Ok(ResolvedAuthentication(headers))
}
