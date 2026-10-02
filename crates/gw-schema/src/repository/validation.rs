use super::*;
use crate::{Content, Role};
use std::collections::{BTreeMap, BTreeSet};
use unicode_normalization::UnicodeNormalization;

pub(super) fn err(message: &'static str) -> RepositoryEpisodeError {
    RepositoryEpisodeError(message)
}
pub(super) fn nonblank(value: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(err("missing repository declaration"))
    } else {
        Ok(())
    }
}
fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn revision(value: &RepositoryRevision) -> Result<()> {
    let length = match value.algorithm {
        RepositoryRevisionAlgorithm::GitSha1 => 40,
        RepositoryRevisionAlgorithm::GitSha256 => 64,
    };
    if hex(&value.hex, length) {
        Ok(())
    } else {
        Err(err("invalid immutable repository revision"))
    }
}
pub(super) fn task(value: &RepositoryTask, pending: &mut Vec<String>) -> Result<()> {
    for text in [
        &value.task_id,
        &value.source.namespace,
        &value.source.item,
        &value.source.citation,
        &value.problem,
        &value.group.namespace,
        &value.group.id,
        &value.split.manifest.namespace,
        &value.split.manifest.id,
        &value.split.revision,
        &value.rights.reviewer,
    ] {
        nonblank(text)?;
    }
    if value.rights.evidence.is_empty() || value.rights.permitted_uses.is_empty() {
        return Err(err("missing rights declaration"));
    }
    for text in &value.rights.evidence {
        nonblank(text)?;
    }
    for (index, usage) in value.rights.permitted_uses.iter().enumerate() {
        if value.rights.permitted_uses[..index].contains(usage) {
            return Err(err("duplicate rights use"));
        }
    }
    if value.source.revision.trim().is_empty() {
        pending.push("source_revision_missing".into());
    }
    for (name, revision_value) in [
        ("upstream_base_missing", &value.upstream_base),
        ("actor_baseline_missing", &value.actor_baseline),
    ] {
        if let Some(value) = revision_value {
            revision(value)?;
        } else {
            pending.push(name.into());
        }
    }
    if let Some(environment) = &value.environment {
        environment
            .digest
            .validate()
            .map_err(|_| err("invalid environment digest"))?;
        nonblank(&environment.platform)?;
        nonblank(&environment.recipe)?;
    } else {
        pending.push("environment_missing".into());
    }
    if let Some(contract) = &value.private_contract {
        contract
            .digest
            .validate()
            .map_err(|_| err("invalid private contract digest"))?;
        let mut seen = BTreeSet::new();
        for id in &contract.required_test_ids {
            nonblank(id)?;
            if !seen.insert(id) {
                return Err(err("duplicate required test ID"));
            }
        }
        if contract.required_test_ids.is_empty() {
            pending.push("required_test_ids_missing".into());
        }
    } else {
        pending.push("private_contract_missing".into());
    }
    Ok(())
}
fn path(value: &str) -> Result<String> {
    if value.is_empty()
        || !value.nfc().eq(value.chars())
        || value
            .chars()
            .any(|c| c.is_control() || "\\:*?\"<>|".contains(c))
    {
        return Err(err("invalid or ambiguous repository path"));
    }
    for component in value.split('/') {
        let folded = component.to_uppercase();
        let stem = folded.split('.').next().unwrap_or_default();
        if component.is_empty()
            || matches!(component, "." | "..")
            || component.ends_with(['.', ' '])
            || folded == ".GIT"
            || matches!(stem, "CON" | "PRN" | "AUX" | "NUL")
            || (stem.len() == 4
                && (stem.starts_with("COM") || stem.starts_with("LPT"))
                && matches!(stem.as_bytes()[3], b'1'..=b'9'))
        {
            return Err(err("invalid or ambiguous repository path"));
        }
    }
    Ok(value.to_lowercase().to_uppercase().nfc().collect())
}
fn state(value: &RepositoryFileState) -> Result<()> {
    if matches!(value, RepositoryFileState::Text { text, .. } if text.contains('\0')) {
        return Err(err("binary content requires unsupported declaration"));
    }
    Ok(())
}
fn exists(value: &RepositoryFileState) -> bool {
    matches!(value, RepositoryFileState::Text { .. })
}
fn delta(value: &RepositoryDelta, pending: &mut Vec<String>) -> Result<()> {
    if value.enumeration == RepositoryEnumeration::Incomplete {
        pending.push("delta_enumeration_incomplete".into());
    }
    if !value.unsupported.is_empty() {
        pending.push("unsupported_delta".into());
    }
    for (index, kind) in value.unsupported.iter().enumerate() {
        if value.unsupported[..index].contains(kind) {
            return Err(err("duplicate unsupported category"));
        }
    }
    let mut paths = BTreeMap::new();
    for change in &value.changes {
        let key = path(&change.path)?;
        state(&change.before)?;
        state(&change.after)?;
        if change.before == change.after {
            return Err(err("file delta contains unchanged state"));
        }
        if paths.insert(key, change).is_some() {
            return Err(err("duplicate or ambiguous changed path"));
        }
    }
    // Reject impossible file/directory overlaps in either snapshot; deletion followed by adding a
    // directory's child is valid when the old ancestor is absent in the resulting snapshot.
    for (key, change) in &paths {
        for (index, _) in key.match_indices('/') {
            if let Some(parent) = paths.get(&key[..index])
                && ((exists(&parent.before) && exists(&change.before))
                    || (exists(&parent.after) && exists(&change.after)))
            {
                return Err(err("overlapping regular file paths"));
            }
        }
    }
    Ok(())
}
fn trajectory(value: &RepositoryCandidate, pending: &mut Vec<String>) -> Result<()> {
    nonblank(&value.attempt)?;
    if !value.trajectory_complete {
        pending.push("trajectory_incomplete".into());
    }
    if value.messages.is_empty() {
        pending.push("messages_missing".into());
    }
    let mut definitions = BTreeSet::new();
    if let Some(tools) = &value.tools {
        for tool in tools {
            if tool.get("type").and_then(Value::as_str) != Some("function") {
                pending.push("unsupported_tool_definition".into());
                continue;
            }
            let Some(name) = tool
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
            else {
                return Err(err("invalid function definition"));
            };
            nonblank(name)?;
            if !definitions.insert(name) {
                return Err(err("duplicate function definition"));
            }
        }
    } else {
        pending.push("tool_definitions_missing".into());
    }
    let mut calls = BTreeMap::new();
    let mut replies = BTreeSet::new();
    for message in &value.messages {
        if matches!(message.content, Content::Parts(_)) {
            pending.push("unsupported_message_content".into());
        }
        if let Some(list) = &message.tool_calls {
            if message.role != Role::Assistant {
                return Err(err("calls require assistant role"));
            }
            for call in list {
                let id = call
                    .id
                    .as_deref()
                    .ok_or_else(|| err("call requires explicit ID"))?;
                nonblank(id)?;
                nonblank(&call.function.name)?;
                if !call.function.arguments.is_object() {
                    return Err(err("call arguments must be an object"));
                }
                if value.tools.is_some() && !definitions.contains(call.function.name.as_str()) {
                    return Err(err("call has no declared function"));
                }
                if calls.insert(id, call.function.name.as_str()).is_some() {
                    return Err(err("duplicate call ID"));
                }
            }
        }
        if message.role == Role::Tool {
            let id = message
                .tool_call_id
                .as_deref()
                .ok_or_else(|| err("result requires explicit call ID"))?;
            let called = calls
                .get(id)
                .ok_or_else(|| err("result precedes or lacks its call"))?;
            if !replies.insert(id) || message.name.as_deref().is_some_and(|name| name != *called) {
                return Err(err("duplicate or contradictory tool result"));
            }
        } else if message.tool_call_id.is_some() || message.name.is_some() {
            return Err(err("result identity requires tool role"));
        }
    }
    if calls.len() != replies.len() {
        if value.trajectory_complete {
            return Err(err("complete trajectory has unanswered calls"));
        }
        pending.push("unanswered_calls".into());
    }
    delta(&value.delta, pending)
}
impl RepositoryEpisodeRequest {
    /// Validate received shape and return deterministic pending reasons. This performs no I/O.
    ///
    /// # Errors
    /// Rejects invalid declarations, paths, call links, and nonfinite or negative usage costs.
    pub fn pending(&self) -> Result<Vec<String>> {
        if self.version != 1 {
            return Err(err("unsupported repository request version"));
        }
        let mut pending = Vec::new();
        task(&self.task, &mut pending)?;
        trajectory(&self.candidate, &mut pending)?;
        if !self.generation.settings.is_object() {
            return Err(err("generation settings must be an object"));
        }
        if self
            .generation
            .usage
            .cost_usd
            .is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(err("invalid declared usage cost"));
        }
        if let Some(report) = &self.report {
            nonblank(&report.publisher)?;
        }
        pending.sort();
        pending.dedup();
        Ok(pending)
    }
}

// Canonical Message and ExecutionEvidence retain legacy permissive serde implementations. Check
// only their protected structure here, without changing those historical input contracts. Tool
// definitions, arguments, settings, model claims and original reports are intentional opaque JSON.
fn keys(value: &Value, allowed: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| err("invalid protected repository object"))?;
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(err("unknown protected repository field"));
    }
    Ok(())
}
pub(super) fn protected(request: &Value) -> Result<()> {
    if let Some(messages) = request
        .pointer("/candidate/messages")
        .and_then(Value::as_array)
    {
        for message in messages {
            keys(
                message,
                &[
                    "role",
                    "content",
                    "reasoning",
                    "reasoning_details",
                    "tool_calls",
                    "tool_call_id",
                    "name",
                ],
            )?;
            if let Some(details) = message.get("reasoning_details").and_then(Value::as_array) {
                for detail in details {
                    let allowed: &[&str] = match detail.get("type").and_then(Value::as_str) {
                        Some("reasoning.text") => {
                            &["type", "text", "signature", "id", "format", "index"]
                        }
                        Some("reasoning.summary") => &["type", "summary", "id", "format", "index"],
                        Some("reasoning.encrypted") => &["type", "data", "id", "format", "index"],
                        _ => return Err(err("unsupported reasoning detail")),
                    };
                    keys(detail, allowed)?;
                }
            }
            if let Some(parts) = message.get("content").and_then(Value::as_array) {
                for part in parts {
                    let allowed: &[&str] = match part.get("type").and_then(Value::as_str) {
                        Some("text") => &["type", "text"],
                        Some("image_url") => &["type", "image_url"],
                        Some("input_audio") => &["type", "audio_url", "format"],
                        _ => return Err(err("unsupported content part")),
                    };
                    keys(part, allowed)?;
                }
            }
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    keys(call, &["id", "function"])?;
                    keys(&call["function"], &["name", "arguments", "raw_arguments"])?;
                }
            }
        }
    }
    if let Some(report) = request
        .pointer("/report/execution")
        .filter(|v| !v.is_null())
    {
        keys(
            report,
            &[
                "outcome",
                "required_tests",
                "cases",
                "exit_code",
                "errors",
                "source_ref",
                "binding",
            ],
        )?;
        keys(&report["binding"], &["task", "attempt", "patch_hash"])?;
        if let Some(cases) = report.get("cases").and_then(Value::as_array) {
            for case in cases {
                keys(case, &["node", "status"])?;
            }
        }
    }
    Ok(())
}
