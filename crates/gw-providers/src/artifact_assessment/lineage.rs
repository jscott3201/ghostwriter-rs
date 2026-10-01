//! Resolution of exact declared lineage links in the supplied offline artifact bundle.
use super::{evidence::deny, *};
use gw_schema::{Declaration, ModelArtifactLineage};
use std::collections::BTreeMap;

pub(super) fn parents(
    identity: &ArtifactIdentity,
    artifact: &PinnedModelArtifact,
    artifacts: &BTreeMap<String, &PinnedModelArtifact>,
    denials: &mut Vec<ArtifactDenial>,
) -> Vec<ArtifactIdentity> {
    let mut parents = Vec::new();
    match &artifact.lineage {
        ModelArtifactLineage::Base {} => {}
        ModelArtifactLineage::Derived {
            base,
            additional_parents,
            ..
        } => {
            parents.push(base.clone());
            match additional_parents {
                Declaration::Unknown => deny(
                    denials,
                    Some(identity),
                    ArtifactDenialReason::UnknownLineage,
                ),
                Declaration::Declared(additional) => parents.extend(additional.iter().cloned()),
            }
        }
        ModelArtifactLineage::Quantized { base, .. } => parents.push(base.clone()),
        ModelArtifactLineage::Adapter {
            base,
            parent_checkpoint,
            ..
        } => {
            parents.push(base.clone());
            optional(parent_checkpoint, identity, &mut parents, denials);
            check_link(parent_checkpoint, base, false, identity, artifacts, denials);
        }
        ModelArtifactLineage::Checkpoint {
            base,
            parent,
            adapter,
        } => {
            parents.push(base.clone());
            optional(parent, identity, &mut parents, denials);
            optional(adapter, identity, &mut parents, denials);
            check_link(parent, base, false, identity, artifacts, denials);
            check_link(adapter, base, true, identity, artifacts, denials);
        }
    }
    parents
}

fn optional(
    claim: &Declaration<Option<ArtifactIdentity>>,
    subject: &ArtifactIdentity,
    parents: &mut Vec<ArtifactIdentity>,
    denials: &mut Vec<ArtifactDenial>,
) {
    match claim {
        Declaration::Unknown => deny(denials, Some(subject), ArtifactDenialReason::UnknownLineage),
        Declaration::Declared(Some(identity)) => parents.push(identity.clone()),
        Declaration::Declared(None) => {}
    }
}

fn check_link(
    claim: &Declaration<Option<ArtifactIdentity>>,
    base: &ArtifactIdentity,
    adapter: bool,
    subject: &ArtifactIdentity,
    artifacts: &BTreeMap<String, &PinnedModelArtifact>,
    denials: &mut Vec<ArtifactDenial>,
) {
    let Declaration::Declared(Some(parent)) = claim else {
        return;
    };
    let Some(parent) = artifacts.get(&parent.digest) else {
        return;
    };
    let linked_base = match (&parent.lineage, adapter) {
        (ModelArtifactLineage::Adapter { base, .. }, true)
        | (ModelArtifactLineage::Checkpoint { base, .. }, false) => Some(base),
        _ => None,
    };
    if linked_base != Some(base) {
        deny(
            denials,
            Some(subject),
            ArtifactDenialReason::ConflictingLineage,
        );
    }
}
