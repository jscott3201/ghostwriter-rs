//! Redacted complete held-out population identities; private oracle values have no fields here.
use crate::coding_value::{coding_hash_valid, coding_json_digest};
use crate::{CodingSuiteBinding, NamespacedTaskId, TaskProvenance, TaskSplitRole};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// One public held-out member, tied to its position in the complete registered catalogue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingPopulationMember {
    /// Original 112-member catalogue ordinal, preserving accepted order.
    pub ordinal: u32,
    /// Exact accepted reference member identity; its reference answer is never copied.
    pub member_id: String,
    /// Redacted source, task, family, split and review declarations.
    pub provenance: TaskProvenance,
    /// Whole accepted related component.
    pub component: NamespacedTaskId,
    /// Redacted complete execution coverage.
    pub suite: CodingSuiteBinding,
    /// Public problem, callable signature and deliberately visible examples only.
    pub prompt: String,
}
impl CodingPopulationMember {
    /// Reconstruct the redacted authoritative execution contract, without any oracle values.
    #[must_use]
    pub fn contract(&self) -> crate::VerificationContract {
        crate::VerificationContract {
            answer_policy: Some(crate::VerificationPolicy::Absent),
            execution_policy: Some(crate::VerificationPolicy::Authoritative),
            required_tests: self.suite.case_ids.clone(),
            kind: crate::VerificationKind::None,
            oracle: crate::Oracle::CodingSuite {
                suite: self.suite.clone(),
            },
            numeric: None,
        }
    }
}

/// Compact bindings of an accepted Train record captured from the current committed import.
/// The content digest covers the byte-exact stable public export, without repeating its metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingTrainingBinding {
    /// Accepted catalogue member identity.
    pub member_id: String,
    /// Stable reference record identity, independent of lifecycle/export transitions.
    pub record_id: String,
    /// Native-derived digest of record identity, canonical messages, task and accepted origin.
    pub content_id: String,
    /// Accepted semantic task identity, without repeating large source citations.
    pub task_identity: crate::SemanticTaskIdentity,
    /// Whole accepted task family.
    pub group: NamespacedTaskId,
    /// Whole accepted related component.
    pub component: NamespacedTaskId,
    /// Accepted redacted complete execution suite identity.
    pub suite_id: String,
}
impl CodingTrainingBinding {
    /// Digest the exact stable canonical export strings. This alone grants no current eligibility.
    #[must_use]
    pub fn compute_content_id(
        record_id: &str,
        messages_json: &str,
        task_json: &str,
        origin_json: &str,
    ) -> String {
        coding_json_digest(
            "ghostwriter.coding-training-content.v1",
            &(record_id, messages_json, task_json, origin_json),
        )
    }
    fn validate(&self, population: &CodingPopulation) -> Result<(), &'static str> {
        if [&self.content_id, &self.suite_id, &self.task_identity.digest]
            .into_iter()
            .any(|id| !coding_hash_valid(id))
            || self.task_identity.version != 2
            || [
                &self.group.namespace,
                &self.group.id,
                &self.component.namespace,
                &self.component.id,
            ]
            .into_iter()
            .any(|value| value.trim().is_empty())
            || self.record_id
                != coding_json_digest(
                    "ghostwriter.reference-record.v1",
                    &(&population.batch_id, &self.member_id),
                )
            || population.members.iter().any(|m| {
                m.component == self.component
                    || m.provenance.group == self.group
                    || m.provenance.identity == self.task_identity
            })
        {
            return Err("Train content contradicts accepted population bindings");
        }
        Ok(())
    }
}

/// Complete selected held-out split captured from one committed operator registration.
/// Deserialization validates declarations and never authenticates current database membership.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingPopulation {
    /// Exactly one.
    pub version: u32,
    /// Domain-separated identity of all other fields.
    pub population_id: String,
    /// Entire accepted 112-member capture identity.
    pub catalogue_id: String,
    /// Explicit local operator registration identity.
    pub registration_id: String,
    /// Complete committed reference-import identity.
    pub batch_id: String,
    /// Validation or Test; Train is never a comparison population.
    pub split: TaskSplitRole,
    /// Complete native-derived Train content bindings for every reference-trained candidate source.
    pub training_members: Vec<CodingTrainingBinding>,
    /// All 16 Validation or all 32 Test members, in their accepted order.
    pub members: Vec<CodingPopulationMember>,
}
impl CodingPopulation {
    /// Recompute integrity over every ordered public binding, without granting registration.
    #[must_use]
    pub fn computed_id(&self) -> String {
        let mut copy = self.clone();
        copy.population_id.clear();
        coding_json_digest("ghostwriter.coding-population.v1", &copy)
    }
    /// Validate exact held-out coverage and family/identity consistency.
    ///
    /// # Errors
    /// Rejects training, incomplete or reordered members, duplicate identities and mixed splits.
    pub fn validate(&self) -> Result<(), &'static str> {
        let count = match self.split {
            TaskSplitRole::Validation => 16,
            TaskSplitRole::Test => 32,
            TaskSplitRole::Train => return Err("comparison population must be held out"),
        };
        if self.version != 1
            || self.members.len() != count
            || [
                &self.catalogue_id,
                &self.registration_id,
                &self.batch_id,
                &self.population_id,
            ]
            .into_iter()
            .any(|id| !coding_hash_valid(id))
            || self.population_id != self.computed_id()
        {
            return Err("invalid complete coding population identity or count");
        }
        let mut ids = HashSet::new();
        if self.training_members.len() != 64 {
            return Err("coding population requires all accepted Train content bindings");
        }
        for member in &self.training_members {
            member.validate(self)?;
            if !coding_hash_valid(&member.member_id) || !ids.insert(&member.member_id) {
                return Err("coding population requires distinct accepted Train identities");
            }
        }
        let mut tasks = HashSet::new();
        let mut families = HashMap::new();
        let mut previous = None;
        for member in &self.members {
            member.provenance.validate_declarations()?;
            member.suite.validate()?;
            member.suite.validate_split(self.split)?;
            if member.provenance.identity
                != crate::coding_task::coding_semantic_identity(
                    &member.provenance.source,
                    &member.prompt,
                    &member.contract(),
                )?
            {
                return Err("coding population prompt differs from its semantic task identity");
            }
            if member.ordinal >= 112
                || previous.is_some_and(|n| member.ordinal <= n)
                || !coding_hash_valid(&member.member_id)
                || !ids.insert(&member.member_id)
                || !tasks.insert(&member.provenance.identity.digest)
                || member.provenance.split.role != self.split
                || member.component.namespace.trim().is_empty()
                || member.component.id.trim().is_empty()
                || member.prompt.is_empty()
                || member.prompt.len() > 65536
            {
                return Err("invalid ordered held-out member or split binding");
            }
            previous = Some(member.ordinal);
            *families.entry(&member.provenance.group).or_insert(0) += 1;
        }
        if families.len() != count / 4 || families.values().any(|n| *n != 4) {
            return Err("coding population requires complete four-member families");
        }
        Ok(())
    }
}
