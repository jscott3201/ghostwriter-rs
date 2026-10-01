use super::*;
use ConsistencyReason as R;
use ConsistencyState::{Mismatch, Unknown};
use gw_schema::{
    ArtifactIdentity, Declaration, ModelArtifactLineage, ModelExecutionSemantics, ModelOperation,
    PinnedModelArtifact,
};
use std::collections::{BTreeMap, BTreeSet};

fn compare<T: PartialEq>(
    a: &Declaration<T>,
    b: &Declaration<T>,
    axis: &mut ConsistencyAxis,
    mismatch: R,
) {
    match (a, b) {
        (Declaration::Declared(a), Declaration::Declared(b)) if a != b => {
            axis.note(Mismatch, mismatch)
        }
        (Declaration::Unknown, _) | (_, Declaration::Unknown) => {
            axis.note(Unknown, R::UnknownSemantics)
        }
        _ => {}
    }
}
fn normalized_set(
    value: &Declaration<Vec<ArtifactIdentity>>,
) -> Declaration<Vec<ArtifactIdentity>> {
    let mut value = value.clone();
    if let Declaration::Declared(values) = &mut value {
        values.sort_by(|a, b| a.digest.cmp(&b.digest));
    }
    value
}

pub(super) fn semantics(
    requested: &ModelExecutionSemantics,
    effective: Option<&ModelExecutionSemantics>,
    artifacts: &[PinnedModelArtifact],
    axis: &mut ConsistencyAxis,
) {
    declaration(requested, axis);
    let mut indexed = BTreeMap::new();
    for artifact in artifacts {
        match artifact.identity() {
            Ok(identity) => {
                if indexed.insert(identity.digest, artifact).is_some() {
                    axis.note(Mismatch, R::DuplicateArtifact);
                }
            }
            Err(_) => axis.note(Mismatch, R::InvalidArtifact),
        }
    }
    resolve(requested, &indexed, axis);
    let Some(effective) = effective else {
        axis.note(Unknown, R::UnknownSemantics);
        return;
    };
    declaration(effective, axis);
    resolve(effective, &indexed, axis);
    if requested.version != effective.version
        || requested.alias != effective.alias
        || requested.operation != effective.operation
        || requested.adapter_behavior != effective.adapter_behavior
    {
        axis.note(Mismatch, R::SemanticMismatch);
    }
    compare(
        &requested.serving_profile,
        &effective.serving_profile,
        axis,
        R::SemanticMismatch,
    );
    compare(
        &requested.artifact,
        &effective.artifact,
        axis,
        R::SemanticMismatch,
    );
    compare(
        &normalized_set(&requested.additional_artifacts),
        &normalized_set(&effective.additional_artifacts),
        axis,
        R::SemanticMismatch,
    );
    compare(
        &requested.tokenizer,
        &effective.tokenizer,
        axis,
        R::SemanticMismatch,
    );
    compare(
        &requested.chat_template,
        &effective.chat_template,
        axis,
        R::SemanticMismatch,
    );
    compare(
        &requested.runtime,
        &effective.runtime,
        axis,
        R::SemanticMismatch,
    );
    compare(
        &requested.parser,
        &effective.parser,
        axis,
        R::SemanticMismatch,
    );
    compare(
        &requested.configuration,
        &effective.configuration,
        axis,
        R::SemanticMismatch,
    );
}
fn declaration(value: &ModelExecutionSemantics, axis: &mut ConsistencyAxis) {
    if value.validate().is_err() {
        axis.note(Mismatch, R::SemanticMismatch);
    }
    if !super::request::semantics_complete(value) {
        axis.note(Unknown, R::UnknownSemantics);
    }
    if value.operation == ModelOperation::ChatCompletion
        && value.chat_template == Declaration::Declared(None)
    {
        axis.note(Mismatch, R::ComponentMismatch);
    }
    if let Declaration::Declared(profile) = &value.serving_profile {
        let behavior = serde_json::from_value::<ProfileBehavior>(profile.configuration.clone());
        let matches = profile.implementation == "gw-providers/offline-serving-profile"
            && profile.revision == "1"
            && behavior.is_ok_and(|behavior| {
                behavior.declaration().as_ref() == Ok(profile)
                    && behavior.adapter_behavior(value.operation).as_ref()
                        == Ok(&value.adapter_behavior)
            });
        if !matches {
            axis.note(Mismatch, R::UnsupportedProfileBehavior);
        }
    }
}

fn required_artifacts(value: &ModelExecutionSemantics) -> Option<Vec<ArtifactIdentity>> {
    let (Declaration::Declared(primary), Declaration::Declared(additional)) =
        (&value.artifact, &value.additional_artifacts)
    else {
        return None;
    };
    let mut result = vec![primary.clone()];
    result.extend(additional.iter().cloned());
    result.sort_by(|a, b| a.digest.cmp(&b.digest));
    Some(result)
}
pub(super) fn loaded_artifacts(
    value: &ModelExecutionSemantics,
    loaded: &Declaration<Vec<ArtifactIdentity>>,
    axis: &mut ConsistencyAxis,
) {
    match (required_artifacts(value), normalized_set(loaded)) {
        (Some(expected), Declaration::Declared(actual)) if expected != actual => {
            axis.note(Mismatch, R::LoadedArtifactsMismatch)
        }
        (None, _) | (_, Declaration::Unknown) => axis.note(Unknown, R::UnknownLoadedArtifacts),
        _ => {}
    }
}

fn resolve(
    value: &ModelExecutionSemantics,
    artifacts: &BTreeMap<String, &PinnedModelArtifact>,
    axis: &mut ConsistencyAxis,
) {
    let mut pending = Vec::new();
    match &value.artifact {
        Declaration::Declared(primary) => pending.push(primary.clone()),
        Declaration::Unknown => axis.note(Unknown, R::UnknownSemantics),
    }
    match &value.additional_artifacts {
        Declaration::Declared(additional) => pending.extend(additional.iter().cloned()),
        Declaration::Unknown => axis.note(Unknown, R::UnknownSemantics),
    }
    if let Declaration::Declared(primary) = &value.artifact
        && let Some(primary) = artifacts.get(&primary.digest)
    {
        compare(
            &value.tokenizer,
            &primary.tokenizer,
            axis,
            R::ComponentMismatch,
        );
        // Text completion/embedding may intentionally not use a template even when one is
        // distributed in the artifact. A declared present effective template must match its pin.
        if value.chat_template != Declaration::Declared(None)
            || value.operation == ModelOperation::ChatCompletion
        {
            compare(
                &value.chat_template,
                &primary.chat_template,
                axis,
                R::ComponentMismatch,
            );
        }
    }
    let mut seen = BTreeSet::new();
    while let Some(identity) = pending.pop() {
        if !seen.insert(identity.digest.clone()) {
            continue;
        }
        let Some(artifact) = artifacts.get(&identity.digest) else {
            axis.note(Unknown, R::MissingArtifact);
            continue;
        };
        match &artifact.lineage {
            ModelArtifactLineage::Base {} => {}
            ModelArtifactLineage::Derived {
                base,
                additional_parents,
                ..
            } => {
                pending.push(base.clone());
                match additional_parents {
                    Declaration::Unknown => axis.note(Unknown, R::UnknownArtifactLineage),
                    Declaration::Declared(values) => pending.extend(values.iter().cloned()),
                }
            }
            ModelArtifactLineage::Quantized { base, .. } => pending.push(base.clone()),
            ModelArtifactLineage::Adapter {
                base,
                parent_checkpoint,
                ..
            } => {
                pending.push(base.clone());
                optional_parent(
                    parent_checkpoint,
                    base,
                    false,
                    artifacts,
                    &mut pending,
                    axis,
                );
            }
            ModelArtifactLineage::Checkpoint {
                base,
                parent,
                adapter,
            } => {
                pending.push(base.clone());
                optional_parent(parent, base, false, artifacts, &mut pending, axis);
                optional_parent(adapter, base, true, artifacts, &mut pending, axis);
            }
        }
    }
}
fn optional_parent(
    value: &Declaration<Option<ArtifactIdentity>>,
    base: &ArtifactIdentity,
    adapter: bool,
    artifacts: &BTreeMap<String, &PinnedModelArtifact>,
    pending: &mut Vec<ArtifactIdentity>,
    axis: &mut ConsistencyAxis,
) {
    match value {
        Declaration::Unknown => axis.note(Unknown, R::UnknownArtifactLineage),
        Declaration::Declared(None) => {}
        Declaration::Declared(Some(parent)) => {
            pending.push(parent.clone());
            if let Some(parent) = artifacts.get(&parent.digest) {
                let linked = match (&parent.lineage, adapter) {
                    (ModelArtifactLineage::Adapter { base, .. }, true)
                    | (ModelArtifactLineage::Checkpoint { base, .. }, false) => Some(base),
                    _ => None,
                };
                if linked != Some(base) {
                    axis.note(Mismatch, R::ConflictingLineage);
                }
            }
        }
    }
}
