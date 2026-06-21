//! Content-addressed hashing helpers + the "never re-spend" call cache (DATA-SCHEMA §5.2, §4.1).
//!
//! Two concerns live here:
//!
//! 1. **Canonical content hashing** ([`record_hash`], [`prompt_hash`], [`completion_hash`]):
//!    BLAKE3 over a *canonicalized* projection of a [`TrainingRecord`]. Canonical = serde_json
//!    with sorted keys (serde_json's `Value` -> `BTreeMap` ordering). Every hash is built from a
//!    **positive allowlist** of CONTENT fields only — never a denylist over the full envelope —
//!    so two byte-identical regenerations hash equal regardless of provenance, run, cost,
//!    lifecycle state, judging, or any other non-content field. `completion_hash` further
//!    excludes `reasoning` so the same final answer reached via different CoT collapses.
//!
//! 2. **The call cache**: [`Store::cache_get`](crate::Store::cache_get) /
//!    [`cache_put`](crate::Store::cache_put), keyed by `(content_hash, kind, model, rubric_id)`,
//!    so teacher/verify/judge reruns and crash-restarts never re-spend tokens.

use blake3::Hasher;
use gw_schema::{Content, Message, ReasoningDetail, TrainingRecord};
use serde_json::Value;

use crate::error::Result;
use crate::store::{Store, now_rfc3339};

/// Lower-hex BLAKE3 of `bytes`.
fn blake3_hex(bytes: &[u8]) -> String {
    let mut h = Hasher::new();
    h.update(bytes);
    h.finalize().to_hex().to_string()
}

/// Serialize `value` to canonical bytes: serde_json sorts object keys lexicographically when a
/// `serde_json::Value` is re-serialized, so round-tripping through `Value` yields a stable,
/// key-sorted byte string independent of struct field order.
fn canonical_bytes(value: &Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

/// BLAKE3 of the record's CONTENT, via a positive allowlist (the exact-dedup key,
/// [`gw_schema::Hashes::record_hash`]).
///
/// The hash is built ONLY from content-bearing fields — `schema_version`, `training_area`,
/// sorted `tags`, the per-turn `(role, content, reasoning, reasoning_details, tool_calls, name)`,
/// and `tools` — and EXCLUDES every non-content field: `record_id`, `provenance`, `generation`,
/// `verification`, `judging`, `reasoning_quality`, `lifecycle` (including the mutating `state`),
/// `hashes`, `cost`, and `dataset_version`. So two byte-identical regenerations hash equal
/// regardless of run/teacher/served-by/cost/judge votes, and a record's hash does not change as
/// its lifecycle advances. Both the flat `reasoning` text AND the `reasoning_details` content
/// payloads ARE content (different CoT → different hash); only the volatile per-detail
/// `id` / `index` / `signature` / `format` are excluded.
///
/// # Errors
/// Returns [`StorageError::Serde`](crate::StorageError::Serde) if the record fails to serialize.
pub fn record_hash(rec: &TrainingRecord) -> Result<String> {
    let mut tags = rec.tags.clone();
    tags.sort();
    let messages: Vec<Value> = rec
        .messages
        .iter()
        .map(message_content_value)
        .collect::<Result<_>>()?;
    let mut projection = serde_json::Map::new();
    projection.insert(
        "schema_version".into(),
        serde_json::to_value(&rec.schema_version)?,
    );
    projection.insert(
        "training_area".into(),
        Value::String(rec.training_area.clone()),
    );
    projection.insert("tags".into(), serde_json::to_value(tags)?);
    projection.insert("messages".into(), Value::Array(messages));
    if let Some(tools) = &rec.tools {
        projection.insert("tools".into(), serde_json::to_value(tools)?);
    }
    Ok(blake3_hex(&canonical_bytes(&Value::Object(projection))?))
}

/// The content projection of one message used by [`record_hash`]: role + clean content + flat
/// reasoning text + a CONTENT-ONLY projection of `reasoning_details` + `tool_calls` + `name`.
///
/// `reasoning_details` is verbatim CoT content (the Verify-gate substrate) — a record can carry
/// its whole chain-of-thought in `reasoning_details[].text` with flat `reasoning = None`, so it
/// MUST contribute to the hash, but only its content payload (`text` / `summary` / `data`) plus
/// the variant discriminant — the volatile `id` / `index` / `signature` / `format` are excluded.
/// `name` (speaker / tool name) is content too: distinct speakers / tool names must not collide.
///
/// # Errors
/// Returns [`StorageError::Serde`](crate::StorageError::Serde) if a content part fails to
/// serialize.
fn message_content_value(m: &Message) -> Result<Value> {
    let reasoning_details = m.reasoning_details.as_ref().map(|details| {
        details
            .iter()
            .map(reasoning_detail_value)
            .collect::<Vec<_>>()
    });
    Ok(serde_json::json!({
        "role": m.role,
        "content": content_value(&m.content)?,
        "reasoning": m.reasoning,
        "reasoning_details": reasoning_details,
        "tool_calls": m.tool_calls,
        "name": m.name,
    }))
}

/// Project one [`gw_schema::ReasoningDetail`] to its content payload + variant discriminant,
/// EXCLUDING the volatile `id` / `index` / `signature` / `format` fields.
fn reasoning_detail_value(d: &ReasoningDetail) -> Value {
    match d {
        ReasoningDetail::Text { text, .. } => {
            serde_json::json!({ "type": "reasoning.text", "text": text })
        }
        ReasoningDetail::Summary { summary, .. } => {
            serde_json::json!({ "type": "reasoning.summary", "summary": summary })
        }
        ReasoningDetail::Encrypted { data, .. } => {
            serde_json::json!({ "type": "reasoning.encrypted", "data": data })
        }
    }
}

/// BLAKE3 of the canonicalized prompt (every non-assistant message's role + clean `content`).
///
/// This is the DPO pairing key / sibling-group id ([`gw_schema::Hashes::prompt_hash`]); it
/// ignores assistant turns, reasoning, and tool calls entirely.
///
/// # Errors
/// Returns [`StorageError::Serde`](crate::StorageError::Serde) on a serialization failure.
pub fn prompt_hash(messages: &[Message]) -> Result<String> {
    let projected: Vec<Value> = messages
        .iter()
        .filter(|m| m.role != gw_schema::Role::Assistant)
        .map(|m| {
            Ok(serde_json::json!({
                "role": m.role,
                "content": content_value(&m.content)?,
            }))
        })
        .collect::<Result<_>>()?;
    Ok(blake3_hex(&canonical_bytes(&Value::Array(projected))?))
}

/// BLAKE3 of the ordered prompt list that defines a run's seed partition manifest.
///
/// Each string is one post-filter user prompt in file/source order. Array order is deliberately
/// preserved by `serde_json`, so reordering prompts changes the hash while object-key
/// canonicalization remains available if a richer prompt source later stores structured entries.
///
/// # Errors
/// Returns [`StorageError::Serde`](crate::StorageError::Serde) on a serialization failure.
pub fn prompts_hash(prompts: &[String]) -> Result<String> {
    let projected = prompts
        .iter()
        .map(|prompt| Value::String(prompt.clone()))
        .collect();
    Ok(blake3_hex(&canonical_bytes(&Value::Array(projected))?))
}

/// BLAKE3 of the assistant `content` only ([`gw_schema::Hashes::completion_hash`]).
///
/// Deliberately EXCLUDES `reasoning`, so the same final answer reached via different CoT
/// collapses to one completion hash (cross-CoT answer dedup).
///
/// # Errors
/// Returns [`StorageError::Serde`](crate::StorageError::Serde) on a serialization failure.
pub fn completion_hash(messages: &[Message]) -> Result<String> {
    let answers: Vec<Value> = messages
        .iter()
        .filter(|m| m.role == gw_schema::Role::Assistant)
        .map(|m| content_value(&m.content))
        .collect::<Result<_>>()?;
    Ok(blake3_hex(&canonical_bytes(&Value::Array(answers))?))
}

/// Project a [`Content`] to a canonical JSON value for hashing. Plain text becomes a JSON
/// string; multimodal `Parts` route through `serde_json::to_value` so they pick up the same
/// sorted-key canonicalization as the rest of the projection (image/audio refs still contribute).
///
/// # Errors
/// Returns [`StorageError::Serde`](crate::StorageError::Serde) if a multimodal part fails to
/// serialize — propagated rather than masked, so distinct payloads can never collapse to `null`.
fn content_value(c: &Content) -> Result<Value> {
    match c {
        Content::Text(t) => Ok(Value::String(t.clone())),
        Content::Parts(parts) => Ok(serde_json::to_value(parts)?),
    }
}

impl Store {
    /// Look up a cached teacher/verify/judge result by its content-addressed key.
    ///
    /// `rubric_id` is normalized to `""` when `None` so the composite key is well-defined for
    /// teacher/verify calls that have no rubric. Returns `Ok(None)` on a cache miss.
    ///
    /// # Errors
    /// Returns [`StorageError`](crate::StorageError) on a SQL fault or if the stored value fails
    /// to parse as JSON.
    pub async fn cache_get(
        &self,
        content_hash: &str,
        kind: &str,
        model: &str,
        rubric_id: Option<&str>,
    ) -> Result<Option<Value>> {
        let row: Option<(String,)> = sqlx::query_as(
            "SELECT value_json FROM cache \
             WHERE content_hash = ?1 AND kind = ?2 AND model = ?3 AND rubric_id = ?4",
        )
        .bind(content_hash)
        .bind(kind)
        .bind(model)
        .bind(rubric_id.unwrap_or(""))
        .fetch_optional(self.pool())
        .await?;
        match row {
            Some((json,)) => Ok(Some(serde_json::from_str(&json)?)),
            None => Ok(None),
        }
    }

    /// Insert (or replace) a cached result for `(content_hash, kind, model, rubric_id)`.
    ///
    /// Idempotent: re-caching the same key overwrites the prior value (a deterministic recompute
    /// yields the same bytes, so this is a no-op in practice). `rubric_id` normalizes `None` to
    /// `""` to match [`cache_get`](Self::cache_get).
    ///
    /// # Errors
    /// Returns [`StorageError`](crate::StorageError) on a SQL fault or if `value` fails to
    /// serialize.
    pub async fn cache_put(
        &self,
        content_hash: &str,
        kind: &str,
        model: &str,
        rubric_id: Option<&str>,
        value: &Value,
    ) -> Result<()> {
        let json = serde_json::to_string(value)?;
        sqlx::query(
            "INSERT INTO cache (content_hash, kind, model, rubric_id, value_json, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(content_hash, kind, model, rubric_id) \
             DO UPDATE SET value_json = excluded.value_json, created_at = excluded.created_at",
        )
        .bind(content_hash)
        .bind(kind)
        .bind(model)
        .bind(rubric_id.unwrap_or(""))
        .bind(json)
        .bind(now_rfc3339())
        .execute(self.pool())
        .await?;
        Ok(())
    }
}
