//! `SandboxConfig` — tool-exec isolation (CONFIG §6.2, REMEDIATION ITEM 3).

use serde::{Deserialize, Serialize};

/// Tool-execution isolation policy. Default = read-only ephemeral SQL copy + locked
/// subprocess; no network; control tools stubbed/refuse.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub sql_mode: SqlSandbox,
    pub code_mode: CodeSandbox,
    /// `RLIMIT_CPU`. default 10.
    pub cpu_secs: u32,
    /// `RLIMIT_AS`. default 512.
    pub mem_mb: u32,
    /// tokio timeout kill. default 30.
    pub wallclock_secs: u32,
    /// default false (HARD: never true for untrusted code).
    pub allow_network: bool,
    /// default false (control tools STUBBED/refuse in v1).
    pub control_tools_live: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            sql_mode: SqlSandbox::ReadOnlyEphemeralCopy,
            code_mode: CodeSandbox::LockedSubprocess,
            cpu_secs: 10,
            mem_mb: 512,
            wallclock_secs: 30,
            allow_network: false,
            control_tools_live: false,
        }
    }
}

/// SQL sandbox mode (only variant in v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SqlSandbox {
    #[default]
    ReadOnlyEphemeralCopy,
}

/// Code sandbox mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeSandbox {
    #[default]
    LockedSubprocess,
    Container,
    MicroVm,
}
