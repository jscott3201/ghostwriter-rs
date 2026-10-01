//! Text-free publication evidence over a fully captured screening population.
use crate::{FrozenScreeningPlan, ScreeningRecordId};
use serde::{Deserialize, Serialize};

/// Authoritative validation procedure required before the receipt was prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreeningValidation {
    /// The trusted application reran the full version-two planner on captured inputs.
    PlannerRerunV2,
}
/// Publication-side membership proof, separate from the supplied-file planner report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenedPopulationCheck {
    /// Complete membership and raw-input bindings were rechecked inside the write transaction.
    TransactionChecked,
}
/// Consumer example layout bound to the screening supervision policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenedExampleLayout {
    /// One complete prefix per supported assistant target.
    AssistantPrefixV1,
    /// Complete conversation through its terminal assistant target.
    FullConversationFinalV1,
}
/// Selected row's connected component, distinct from its declared task group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenedExportMember {
    /// Exact selected output coordinate.
    pub record: ScreeningRecordId,
    /// Frozen connected component identity.
    pub component_id: String,
}
/// Source-level qualification; semantic and effective-template separation remain unknown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenedExportQualification {
    /// Supported witness version (two), requiring exported-semantics bindings.
    pub version: u32,
    /// Initial complete planner rerun authority.
    pub validation: ScreeningValidation,
    /// Membership was checked against the publication database transaction.
    pub population_check: ScreenedPopulationCheck,
    /// Identity of declared runs and every captured raw-input binding.
    pub population_id: String,
    /// Pinned expansion layout.
    pub layout: ScreenedExampleLayout,
    /// Complete text-free frozen source report, including exclusions and protected identities.
    pub plan: FrozenScreeningPlan,
    /// Exactly the emitted rows and their components, ordered by record identity.
    pub members: Vec<ScreenedExportMember>,
}
