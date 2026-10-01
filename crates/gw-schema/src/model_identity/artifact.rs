use super::*;
use crate::SemanticDeclaration;

/// Purpose of an inventory file, included in the artifact identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFilePurpose {
    /// Model parameter bytes, including shards.
    Weights,
    /// Tokenizer vocabulary or behavior definition.
    Tokenizer,
    /// Prompt/chat serialization template.
    ChatTemplate,
    /// Adapter parameter bytes.
    Adapter,
    /// Checkpoint state.
    Checkpoint,
    /// Quantization metadata.
    Quantization,
    /// Other behavior-affecting configuration.
    Configuration,
}
document! {
    /// One relative file in a declared pinned inventory.
    pub struct ModelArtifactFile {
        /// Portable relative identifier; no traversal, drive prefix, or ambiguous casing duplicates.
        pub path: String,
        /// Intended role of these bytes.
        pub purpose: ModelFilePurpose,
        /// Declared digest of the file's exact bytes.
        pub content: ContentDigest,
    }
}
impl ModelArtifactFile {
    /// Check the relative identifier and content digest.
    pub fn validate(&self) -> Result<()> {
        file_path(&self.path)?;
        self.content.validate()
    }
}
document! {
    /// A pinned tokenizer or template file, independent of its enclosing artifact's identity.
    /// This avoids a circular self-reference when the component is in the artifact inventory.
    pub struct ModelComponentReference {
        /// Source and immutable revision of the component.
        pub source: PinnedModelSource,
        /// Relative file identifier, purpose, and exact byte digest.
        pub file: ModelArtifactFile,
    }
}
impl ModelComponentReference {
    /// Validate the source and component file.
    pub fn validate(&self) -> Result<()> {
        self.source.validate()?;
        self.file.validate()
    }
}

/// Explicit lineage claims, requiring later resolution and compatibility qualification.
/// Optional links distinguish unknown, declared absence, and a supplied exact identity.
/// Declared absence is a caller's claim and requires the same independent review as presence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelArtifactLineage {
    /// A declared base artifact.
    Base {},
    /// A derivative with a primary base and any additional parent artifacts.
    Derived {
        /// Primary base identity.
        base: ArtifactIdentity,
        /// Additional parent set; a declared empty set means no additional parents.
        additional_parents: Declaration<Vec<ArtifactIdentity>>,
        /// Declared transformation implementation and revision.
        transformation: SemanticDeclaration,
    },
    /// Quantized weights derived from a pinned base.
    Quantized {
        /// Source of the unquantized weights.
        base: ArtifactIdentity,
        /// Quantizer revision and semantic settings.
        quantization: SemanticDeclaration,
    },
    /// Adapter weights for a declared base and optional prior checkpoint.
    Adapter {
        /// Base model expected by this adapter.
        base: ArtifactIdentity,
        /// Prior checkpoint: unknown, declared absent, or a supplied exact identity.
        parent_checkpoint: Declaration<Option<ArtifactIdentity>>,
        /// Adapter implementation, revision, and settings.
        configuration: SemanticDeclaration,
    },
    /// Training checkpoint with explicit base, parent, and adapter claims.
    Checkpoint {
        /// Base model from which this checkpoint descends.
        base: ArtifactIdentity,
        /// Prior checkpoint: unknown, declared absent, or a supplied exact identity.
        parent: Declaration<Option<ArtifactIdentity>>,
        /// Incorporated adapter: unknown, declared absent, or a supplied exact identity.
        adapter: Declaration<Option<ArtifactIdentity>>,
    },
}
impl ModelArtifactLineage {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Base {} => Ok(()),
            Self::Derived {
                base,
                additional_parents,
                transformation,
            } => {
                base.validate()?;
                additional_parents.check(|parents| {
                    artifact_set(parents)?;
                    if parents.contains(base) {
                        return Err(ModelIdentityError("duplicate base artifact"));
                    }
                    Ok(())
                })?;
                semantic(transformation)
            }
            Self::Quantized { base, quantization } => {
                base.validate()?;
                semantic(quantization)
            }
            Self::Adapter {
                base,
                parent_checkpoint,
                configuration,
            } => {
                base.validate()?;
                parent_checkpoint.check(optional_identity)?;
                semantic(configuration)
            }
            Self::Checkpoint {
                base,
                parent,
                adapter,
            } => {
                base.validate()?;
                parent.check(optional_identity)?;
                adapter.check(optional_identity)
            }
        }
    }
}

fn optional_identity(value: &Option<ArtifactIdentity>) -> Result<()> {
    value.as_ref().map_or(Ok(()), ArtifactIdentity::validate)
}

document! {
    /// Version 1 pinned artifact declaration; source and inventory are claims, not loaded-byte proof.
    pub struct PinnedModelArtifact {
        /// Independent artifact document version; only 1 is supported.
        pub version: u32,
        /// Human display label, excluded from artifact identity.
        pub label: String,
        /// Declared source and immutable revision; both contribute to identity.
        pub source: PinnedModelSource,
        /// Nonempty file set; identity sorts by exact path, rejecting case-insensitive duplicates.
        pub files: Vec<ModelArtifactFile>,
        /// Base, derivation, quantization, adapter, or checkpoint claims.
        pub lineage: ModelArtifactLineage,
        /// Pinned tokenizer file or an explicit unknown.
        pub tokenizer: Declaration<ModelComponentReference>,
        /// Pinned chat-template file or an explicit unknown.
        pub chat_template: Declaration<ModelComponentReference>,
    }
}
impl PinnedModelArtifact {
    /// Validate inventory uniqueness and all supplied component and lineage declarations.
    pub fn validate(&self) -> Result<()> {
        version(self.version)?;
        nonempty(&self.label)?;
        self.source.validate()?;
        if self.files.is_empty() {
            return Err(ModelIdentityError("empty model artifact inventory"));
        }
        let mut paths = BTreeSet::new();
        for file in &self.files {
            file.validate()?;
            if !paths.insert(file.path.to_ascii_lowercase()) {
                return Err(ModelIdentityError("duplicate artifact file identifier"));
            }
        }
        self.lineage.validate()?;
        self.component(&self.tokenizer, ModelFilePurpose::Tokenizer)?;
        self.component(&self.chat_template, ModelFilePurpose::ChatTemplate)
    }
    fn component(
        &self,
        component: &Declaration<ModelComponentReference>,
        purpose: ModelFilePurpose,
    ) -> Result<()> {
        component.check(|component| {
            component.validate()?;
            if component.file.purpose != purpose {
                return Err(ModelIdentityError("incorrect model component purpose"));
            }
            if component.source == self.source && !self.files.contains(&component.file) {
                return Err(ModelIdentityError(
                    "component does not match artifact inventory",
                ));
            }
            Ok(())
        })
    }
    /// Hash canonical JSON in the artifact domain, excluding only the display label.
    /// File and additional-parent order is ignored; locators, revisions, paths, purposes,
    /// algorithms, byte digests, components, and lineage contribute. Arrays in semantic
    /// configuration retain their order; JSON object keys are recursively sorted.
    pub fn identity(&self) -> Result<ArtifactIdentity> {
        self.validate()?;
        let mut value = self.clone();
        value.files.sort_by(|a, b| a.path.cmp(&b.path));
        if let ModelArtifactLineage::Derived {
            additional_parents, ..
        } = &mut value.lineage
        {
            sort_artifacts(additional_parents);
        }
        let mut value = serde_json::to_value(value)
            .map_err(|_| ModelIdentityError("invalid artifact identity"))?;
        value
            .as_object_mut()
            .ok_or(ModelIdentityError("invalid artifact identity"))?
            .remove("label");
        ArtifactIdentity::of(&value)
    }
}
