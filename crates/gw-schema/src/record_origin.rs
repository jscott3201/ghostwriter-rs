//! Truthful record origins and the explicit historical generated-v1 envelope codec.
use crate::{Generation, Provenance};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The generation facts that actually exist for a model-generated record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedOrigin {
    /// Model, harness, and parent lineage.
    pub provenance: Provenance,
    /// Actual generation settings.
    pub generation: Generation,
}

/// Declared authorship category. This does not authenticate an external person or service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceActorKind {
    /// The operator declares that a person performed the work.
    Human,
    /// The operator declares that an agent performed the work.
    Agent,
}

/// Public authorship categories; private names and review notes are deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceAuthorship {
    /// Declared reference author category.
    pub author: ReferenceActorKind,
    /// Declared reviewer category, without a claim of independent external authentication.
    pub reviewer: ReferenceActorKind,
}

/// Versioned redacted binding of a reference to its registration and observed import batch.
/// These serialized declarations confer no registration or fresh execution authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedReferenceOrigin {
    /// Exactly one for this representation.
    pub version: u32,
    /// Exact captured catalogue population identity.
    pub catalogue_id: String,
    /// Application-owned local operator registration identity.
    pub registration_id: String,
    /// Complete atomic import identity, also the reference run ledger key.
    pub batch_id: String,
    /// Ordered catalogue member binding.
    pub member_id: String,
    /// Exact complete UTF-8 reference module identity.
    pub reference_code_id: String,
    /// Complete redacted execution suite identity.
    pub suite_id: String,
    /// Native verified result identity for this member's fresh import execution.
    pub native_result_id: String,
    /// Declared author/reviewer categories, without private review material.
    pub authorship: ReferenceAuthorship,
    /// Whole related component, retained for split and screening reconciliation.
    pub component: crate::NamespacedTaskId,
    /// The reviewed permitted use for this record.
    pub permitted_use: crate::TaskPermittedUse,
}
impl ReviewedReferenceOrigin {
    /// Check the strict version and digest shape. Registration and eligibility require storage.
    ///
    /// # Errors
    /// Rejects unsupported versions or malformed bindings.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.component.namespace.trim().is_empty()
            || self.component.id.trim().is_empty()
            || self.version != 1
            || [
                &self.catalogue_id,
                &self.registration_id,
                &self.batch_id,
                &self.member_id,
                &self.reference_code_id,
                &self.suite_id,
                &self.native_result_id,
            ]
            .into_iter()
            .any(|id| !crate::coding_value::coding_hash_valid(id))
        {
            return Err("unsupported or malformed reviewed-reference origin");
        }
        Ok(())
    }
}

/// Actual origin of a record. References never contain fictitious teacher or generation facts.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordOrigin {
    /// Model-generated provenance and generation settings.
    Generated(Box<GeneratedOrigin>),
    /// Locally accepted reference with observed native execution bindings.
    ReviewedReference(Box<ReviewedReferenceOrigin>),
}
impl RecordOrigin {
    /// The ledger partition shared by both record origins.
    #[must_use]
    pub fn run_id(&self) -> &str {
        match self {
            Self::Generated(value) => &value.provenance.run_id,
            Self::ReviewedReference(value) => &value.batch_id,
        }
    }
    /// Generation facts, present only when generation actually occurred.
    #[must_use]
    pub fn generated(&self) -> Option<&GeneratedOrigin> {
        match self {
            Self::Generated(value) => Some(value),
            Self::ReviewedReference(_) => None,
        }
    }
    /// Mutable generation facts for the generation pipeline and its fixtures.
    pub fn generated_mut(&mut self) -> Option<&mut GeneratedOrigin> {
        match self {
            Self::Generated(value) => Some(value),
            Self::ReviewedReference(_) => None,
        }
    }
    /// Redacted public origin, bound by the version-four export row identity.
    #[must_use]
    pub fn projection(&self) -> ExportRecordOrigin {
        match self {
            Self::Generated(_) => ExportRecordOrigin::Generated { version: 1 },
            Self::ReviewedReference(value) => ExportRecordOrigin::ReviewedReference(value.clone()),
        }
    }
}

// Explicit generated-v1 codec: flatten exactly the old provenance/generation fields, in order.
// A reference uses a distinct strict origin field. Mixed old/new fields cannot be decoded.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceWire {
    origin: Box<ReviewedReferenceOrigin>,
}
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum OriginWire {
    Generated(Box<GeneratedOrigin>),
    Reference(ReferenceWire),
}
impl Serialize for RecordOrigin {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Generated(value) => value.serialize(serializer),
            Self::ReviewedReference(origin) => ReferenceWire {
                origin: origin.clone(),
            }
            .serialize(serializer),
        }
    }
}
impl<'de> Deserialize<'de> for RecordOrigin {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match OriginWire::deserialize(deserializer)? {
            OriginWire::Generated(value) => Ok(Self::Generated(value)),
            OriginWire::Reference(value) => {
                value.origin.validate().map_err(serde::de::Error::custom)?;
                Ok(Self::ReviewedReference(value.origin))
            }
        }
    }
}

/// Strict public origin projection; generated rows remain explicit without inventing reference facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExportRecordOrigin {
    /// Generated-v1 record; generation metadata remains in the authoritative record envelope.
    Generated {
        /// Exactly one for this representation.
        version: u32,
    },
    /// Only stable reference bindings and declared actor categories are exported.
    ReviewedReference(Box<ReviewedReferenceOrigin>),
}
impl ExportRecordOrigin {
    /// Validate the versioned public representation, without asserting local registration.
    ///
    /// # Errors
    /// Rejects unsupported versions and malformed reference bindings.
    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::Generated { version: 1 } => Ok(()),
            Self::Generated { .. } => Err("unsupported generated-origin projection version"),
            Self::ReviewedReference(value) => value.validate(),
        }
    }
}
