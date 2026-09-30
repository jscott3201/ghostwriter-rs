//! Capture, validate, identify and execute one immutable set of source outputs.
use crate::{AreaConfig, EngineError, Result, SeedItem, SeedSource};
use gw_schema::{
    ClientSemantics, InputPlanIdentity, RUN_MANIFEST_VERSION, RunManifest, SemanticDeclaration,
    UnattestedDeployment,
};
use std::collections::HashSet;

/// An owned seed plan captured exactly once per effective shard, including empty shards.
/// Private vectors prevent mutation after validation and hashing.
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedSeedPlan {
    pub(crate) shards: Vec<Vec<SeedItem>>,
    identity: InputPlanIdentity,
}
impl CapturedSeedPlan {
    /// Materialize all source outputs once and validate record/cursor identities before registration.
    /// No provider/client operation is involved; input media URLs are retained without fetching.
    ///
    /// # Errors
    /// Rejects an unrepresentable shard index, invalid offsets, or duplicate seeds within a shard.
    pub fn capture(source: &(impl SeedSource + ?Sized)) -> Result<Self> {
        let count = source.shard_count().max(1);
        i64::try_from(count).map_err(|_| {
            EngineError::Invariant(
                "shard count cannot be represented by the runtime shard index".into(),
            )
        })?;
        let mut shards = Vec::new();
        let mut tasks = gw_schema::TaskDeclarations::default();
        shards
            .try_reserve_exact(count)
            .map_err(|_| EngineError::Invariant("captured shard plan is too large".into()))?;
        for shard in 0..count {
            let items = source.items_for_shard(shard as i64);
            let mut seeds = HashSet::new();
            for (ordinal, item) in items.iter().enumerate() {
                item.candidate.validate_contract()?;
                if let Some(task) = &item.candidate.task_provenance {
                    tasks
                        .insert(task)
                        .map_err(|reason| EngineError::Invariant(reason.into()))?;
                }
                if item.offset != ordinal as u64 || item.offset.checked_add(1).is_none() {
                    return Err(EngineError::Invariant(format!(
                        "seed plan shard {shard}: each offset must equal its vector ordinal and be advanceable"
                    )));
                }
                if !seeds.insert(item.seed) {
                    return Err(EngineError::Invariant(format!(
                        "seed plan shard {shard}: duplicate seed aliases a record identity"
                    )));
                }
            }
            shards.push(items);
        }
        let content_hash = gw_storage::canonical_json_hash(
            &serde_json::json!({"encoding": "seed-plan-v3", "shards": shards}),
        )?;
        let identity = InputPlanIdentity {
            content_hash,
            shard_items: shards.iter().map(|items| items.len() as u64).collect(),
        };
        Ok(Self { shards, identity })
    }
    /// The full canonical input identity, including every ordered shard and candidate field.
    #[must_use]
    pub fn identity(&self) -> &InputPlanIdentity {
        &self.identity
    }
    /// Read the exact captured inputs that execution will consume.
    #[must_use]
    pub fn shards(&self) -> &[Vec<SeedItem>] {
        &self.shards
    }
}

/// A pure prepared run: owned input plan plus effective semantic contract.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRun {
    pub(crate) plan: CapturedSeedPlan,
    manifest: RunManifest,
}
impl PreparedRun {
    /// Prepare effective generation/admission meaning without opening credentials or running clients.
    ///
    /// # Errors
    /// Rejects invalid execution settings, source identities, or missing declarations.
    pub fn capture(
        source: &(impl SeedSource + ?Sized),
        area: &AreaConfig,
        clients: ClientSemantics,
    ) -> Result<Self> {
        Self::from_plan(CapturedSeedPlan::capture(source)?, area, clients)
    }
    /// Prepare from an already-captured plan without querying the source again.
    ///
    /// # Errors
    /// Rejects invalid effective settings or incomplete client declarations.
    pub fn from_plan(
        plan: CapturedSeedPlan,
        area: &AreaConfig,
        clients: ClientSemantics,
    ) -> Result<Self> {
        let manifest = RunManifest {
            version: RUN_MANIFEST_VERSION,
            input_plan: plan.identity.clone(),
            execution: SemanticDeclaration::new(
                "gw-engine/generation-admission",
                "3",
                crate::behavior::contract(area)?,
            ),
            clients,
            unattested_deployment: UnattestedDeployment::default(),
        };
        manifest
            .validate()
            .map_err(|reason| EngineError::Invariant(reason.into()))?;
        Ok(Self { plan, manifest })
    }
    /// Immutable manifest to compare before constructing credential-bearing clients.
    #[must_use]
    pub fn manifest(&self) -> &RunManifest {
        &self.manifest
    }
    /// Read the exact captured source values.
    #[must_use]
    pub fn plan(&self) -> &CapturedSeedPlan {
        &self.plan
    }
    pub(crate) fn validate_actual(
        &self,
        area: &AreaConfig,
        clients: ClientSemantics,
    ) -> Result<()> {
        let mut actual = self.manifest.clone();
        actual.execution.configuration = crate::behavior::contract(area)?;
        actual.clients = clients;
        actual
            .validate()
            .map_err(|reason| EngineError::Invariant(reason.into()))?;
        if actual != self.manifest {
            return Err(EngineError::Invariant("prepared run differs from the actual injected clients or effective settings; prepare a new run with the actual objects".into()));
        }
        Ok(())
    }
}
