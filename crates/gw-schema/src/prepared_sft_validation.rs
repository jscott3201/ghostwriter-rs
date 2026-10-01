//! Structural invariants for complete prepared inputs, without implementing tokenizer semantics.
use crate::{CotPolicy, MultiTurnLoss, PreparedSftExample, PreparedSftPayload, PreparedSftRecipe};
use serde_json::Value;
use std::collections::{BTreeSet, HashSet};

fn hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn keys(value: &Value, expected: &[&str]) -> Result<(), &'static str> {
    let Some(map) = value.as_object() else {
        return Err("prepared SFT metadata must be an object");
    };
    if map.len() != expected.len() || expected.iter().any(|key| !map.contains_key(*key)) {
        return Err("unexpected prepared SFT metadata fields");
    }
    Ok(())
}
fn text(value: &Value) -> Result<&str, &'static str> {
    value
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or("prepared SFT metadata requires nonempty text")
}

impl PreparedSftPayload {
    /// Parse strict raw JSON before any map conversion, then validate all structural invariants.
    ///
    /// # Errors
    /// Rejects malformed UTF-8/JSON, duplicate fields, floats, unsupported versions, or invalid data.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let value = crate::prepared_sft_frame::strict_prepared_json(bytes)?;
        let payload: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        payload.validate().map_err(str::to_owned)?;
        Ok(payload)
    }

    /// Validate the complete ordered payload before any examples are imported.
    ///
    /// # Errors
    /// Rejects inconsistent recipe, identity, shape, ownership, accounting, or qualification claims.
    pub fn validate(&self) -> Result<(), &'static str> {
        let manifest = &self.manifest;
        if self.version != 1
            || manifest.build_manifest_version != 1
            || !hex(&self.source.artifact_id)
            || !hex(&self.source.snapshot_blake3)
            || self.source.byte_length == 0
            || !hex(&manifest.recipe_id)
        {
            return Err("unsupported prepared SFT version or source/recipe identity");
        }
        let vocab = manifest.recipe.validate()?;
        validate_limits(
            &manifest.qualification_limits,
            manifest.recipe.screening.is_some(),
        )?;
        let mut ids = HashSet::new();
        let mut targets = HashSet::new();
        let mut records = HashSet::new();
        let mut supervised = 0;
        let mut context = 0;
        let mut shifted = 0;
        let mut answers = 0;
        for example in &self.examples {
            if !hex(&example.example_id)
                || !ids.insert(&example.example_id)
                || !targets.insert((&example.source.record_id, example.target_index))
                || example.source.artifact_id != self.source.artifact_id
            {
                return Err("duplicate or mismatched prepared SFT example identity");
            }
            example.validate(vocab, &manifest.recipe)?;
            records.insert(&example.source.record_id);
            supervised += example.supervised_tokens;
            context += example.context_tokens;
            shifted += example.effective_shifted_supervised_tokens;
            answers += example.shifted_answer_token_indices.len() as u64;
        }
        let mut rejections = HashSet::new();
        for item in &manifest.rejections {
            if item.record_id.trim().is_empty()
                || item.reason.trim().is_empty()
                || !rejections.insert((&item.record_id, item.target_index))
                || item
                    .target_index
                    .is_some_and(|index| targets.contains(&(&item.record_id, index)))
            {
                return Err("duplicate, contradictory, or malformed prepared SFT rejection");
            }
        }
        let rejected_records = manifest
            .rejections
            .iter()
            .filter(|r| r.target_index.is_none())
            .count() as u64;
        let rejected_targets = manifest.rejections.len() as u64 - rejected_records;
        if manifest.expanded_example_count != self.examples.len() as u64
            || manifest.example_ids.len() != self.examples.len()
            || manifest
                .example_ids
                .iter()
                .zip(&self.examples)
                .any(|(id, example)| id != &example.example_id)
            || manifest.accepted_source_record_count != records.len() as u64
            || manifest.source_record_count < records.len() as u64
            || manifest.source_records_with_no_examples
                != manifest.source_record_count - records.len() as u64
            || manifest.rejected_item_count != manifest.rejections.len() as u64
            || manifest.rejected_record_count != rejected_records
            || manifest.rejected_target_count != rejected_targets
            || manifest.candidate_target_count != self.examples.len() as u64 + rejected_targets
            || manifest.supervised_token_count != supervised
            || manifest.context_token_count != context
            || manifest.effective_shifted_supervised_token_count != shifted
            || manifest.effective_shifted_answer_token_count != answers
        {
            return Err("prepared SFT counts or ordered example references are contradictory");
        }
        Ok(())
    }
}

impl PreparedSftRecipe {
    /// Validate declared recipe shape and return its vocabulary bound. Actual recipe identity and
    /// official rendering are checked by the installed Python consumer against pinned local data.
    ///
    /// # Errors
    /// Rejects unsupported policies, malformed token metadata, or contradictory target declarations.
    pub fn validate(&self) -> Result<u32, &'static str> {
        let layout = match self.multi_turn_loss {
            MultiTurnLoss::AllAssistant => "assistant_prefix_v1",
            MultiTurnLoss::FinalTurnOnly => "full_conversation_final_v1",
        };
        if self.version != 1
            || self.adapter_version.trim().is_empty()
            || !hex(&self.adapter_source_sha256)
            || self.dependencies.is_empty()
            || self
                .dependencies
                .iter()
                .any(|(k, v)| k.is_empty() || v.is_empty())
            || self.layout != layout
            || self.max_length == 0
            || self.offset_unit != "python_unicode_codepoint"
            || self.labels != "unshifted_causal_lm"
            || self.add_special_tokens
            || self.truncation
        {
            return Err("unsupported prepared SFT recipe");
        }
        keys(
            &self.runtime,
            &["python", "implementation", "system", "machine"],
        )?;
        for value in self.runtime.as_object().ok_or("invalid runtime")?.values() {
            text(value)?;
        }
        keys(&self.tokenizer_target, &["repository", "revision"])?;
        keys(
            &self.tokenizer,
            &[
                "repository",
                "revision",
                "files",
                "license",
                "declared_parent",
                "parent_revision",
                "chat_template_sha256",
            ],
        )?;
        for field in ["repository", "revision"] {
            if text(&self.tokenizer[field])? != text(&self.tokenizer_target[field])? {
                return Err("tokenizer target mismatch");
            }
        }
        for field in ["license", "declared_parent"] {
            text(&self.tokenizer[field])?;
        }
        if !self.tokenizer["parent_revision"].is_null() {
            return Err("unsupported tokenizer parent revision claim");
        }
        if !hex(text(&self.tokenizer["chat_template_sha256"])?) {
            return Err("invalid template identity");
        }
        let files = self.tokenizer["files"]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or("missing tokenizer files")?;
        let mut names = HashSet::new();
        for file in files {
            keys(file, &["name", "bytes", "sha256", "repository_oid"])?;
            if !names.insert(text(&file["name"])?)
                || !hex(text(&file["sha256"])?)
                || file["bytes"].as_u64().is_none_or(|n| n == 0)
            {
                return Err("invalid tokenizer file declaration");
            }
            text(&file["repository_oid"])?;
        }
        keys(
            &self.tokenizer_policy,
            &[
                "policy_version",
                "backend_sha256",
                "wrapper",
                "vocab_size",
                "added_tokens",
            ],
        )?;
        if self.tokenizer_policy["policy_version"].as_u64() != Some(1)
            || !hex(text(&self.tokenizer_policy["backend_sha256"])?)
        {
            return Err("invalid tokenizer policy identity");
        }
        let vocab = self.tokenizer_policy["vocab_size"]
            .as_u64()
            .filter(|n| *n > 0 && *n <= u64::from(u32::MAX))
            .ok_or("invalid vocabulary bound")? as u32;
        let wrapper = &self.tokenizer_policy["wrapper"];
        keys(
            wrapper,
            &[
                "special_tokens_map",
                "split_special_tokens",
                "padding_side",
                "truncation_side",
                "model_input_names",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
            ],
        )?;
        if wrapper["split_special_tokens"] != false
            || wrapper["padding_side"] != "right"
            || wrapper["truncation_side"] != "right"
            || wrapper["model_input_names"] != serde_json::json!(["input_ids", "attention_mask"])
        {
            return Err("unsupported tokenizer wrapper");
        }
        for field in ["eos_token_id", "pad_token_id"] {
            if wrapper[field]
                .as_u64()
                .is_none_or(|id| id >= u64::from(vocab))
            {
                return Err("invalid tokenizer special ID");
            }
        }
        if !wrapper["bos_token_id"].is_null()
            && wrapper["bos_token_id"]
                .as_u64()
                .is_none_or(|id| id >= u64::from(vocab))
        {
            return Err("invalid tokenizer beginning ID");
        }
        keys(
            &wrapper["special_tokens_map"],
            &["eos_token", "pad_token", "additional_special_tokens"],
        )?;
        text(&wrapper["special_tokens_map"]["eos_token"])?;
        text(&wrapper["special_tokens_map"]["pad_token"])?;
        let additional = wrapper["special_tokens_map"]["additional_special_tokens"]
            .as_array()
            .ok_or("invalid special-token literals")?;
        for literal in additional {
            text(literal)?;
        }
        let tokens = self.tokenizer_policy["added_tokens"]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or("missing added-token declarations")?;
        let mut ids = HashSet::new();
        let mut literals = HashSet::new();
        for token in tokens {
            keys(
                token,
                &[
                    "id",
                    "content",
                    "single_word",
                    "lstrip",
                    "rstrip",
                    "normalized",
                    "special",
                ],
            )?;
            let id = token["id"]
                .as_u64()
                .filter(|id| *id < u64::from(vocab))
                .ok_or("invalid added-token ID")?;
            if !ids.insert(id) || !literals.insert(text(&token["content"])?) {
                return Err("duplicate added-token declaration");
            }
            for flag in ["single_word", "lstrip", "rstrip", "normalized", "special"] {
                if !token[flag].is_boolean() {
                    return Err("invalid added-token flag");
                }
            }
        }
        Ok(vocab)
    }
}

fn validate_limits(value: &Value, screened: bool) -> Result<(), &'static str> {
    let mut expected = vec![
        "lifecycle_eligibility",
        "rights_and_execution_lineage",
        "student_parent_revision",
        "grouped_split_qualification",
        "contamination_screening",
        "heldout_training_benefit",
        "semantic_screening",
        "effective_prompt_separation",
        "student_weights",
        "execution_lineage",
        "decision_lineage",
    ];
    if screened {
        expected.extend(["screening_population", "screening_lexical_scope"]);
    }
    keys(value, &expected)?;
    for (field, expected) in [
        (
            "lifecycle_eligibility",
            "not_reconstructed_by_artifact_verification",
        ),
        ("rights_and_execution_lineage", "unknown"),
        ("heldout_training_benefit", "unknown"),
        ("semantic_screening", "not_run"),
        ("effective_prompt_separation", "unknown"),
        ("student_weights", "unbound"),
        ("execution_lineage", "unbound"),
        ("decision_lineage", "unbound"),
    ] {
        if value[field] != expected {
            return Err("unsupported prepared SFT qualification claim");
        }
    }
    if !value["student_parent_revision"].is_null() {
        return Err("student lineage must remain unbound");
    }
    if screened {
        if value["grouped_split_qualification"] != "declared_train_components_source_screened"
            || !matches!(
                value["contamination_screening"].as_str(),
                Some("complete_no_match" | "match_quarantined")
            )
            || value["screening_population"] != "transaction_checked"
            || value["screening_lexical_scope"] != "canonical_source_and_pinned_export_policy"
        {
            return Err("unsupported prepared SFT source-screening claim");
        }
    } else if value["grouped_split_qualification"] != "unknown"
        || value["contamination_screening"] != "unknown"
    {
        return Err("unscreened source cannot claim qualified grouping or contamination screening");
    }
    Ok(())
}

impl PreparedSftExample {
    fn validate(&self, vocab: u32, recipe: &PreparedSftRecipe) -> Result<(), &'static str> {
        let length = self.input_ids.len();
        let characters: Vec<_> = self.rendered.chars().collect();
        if length == 0
            || length as u64 > recipe.max_length
            || self.rendered.is_empty()
            || self.attention_mask.len() != length
            || self.labels.len() != length
            || self.offset_mapping.len() != length
            || self.ownership_offsets.len() != length
            || self.token_kinds.len() != length
            || self.input_ids.iter().any(|id| *id >= vocab)
            || self.attention_mask.iter().any(|mask| *mask != 1)
            || self.source.record_id.trim().is_empty()
            || !hex(&self.source.group_id)
            || !hex(&self.source.record_hash)
            || !hex(&self.source.prompt_hash)
        {
            return Err("invalid prepared SFT token shape or source identity");
        }
        let mut end = 0;
        let mut message = 0;
        for span in &self.spans {
            let expected = match span.kind.as_str() {
                "answer" | "end" => span.message_index == self.target_index,
                "reasoning" | "reasoning_wrapper" => {
                    span.message_index == self.target_index
                        && recipe.cot_policy == CotPolicy::Supervised
                }
                "header" | "separator" | "context" => false,
                _ => return Err("unsupported prepared SFT ownership kind"),
            };
            if span.start != end
                || span.end <= span.start
                || span.end > characters.len() as u64
                || span.message_index < message
                || span.message_index > message + 1
                || span.message_index > self.target_index
                || span.supervised != expected
            {
                return Err("inconsistent prepared SFT ownership span");
            }
            message = span.message_index;
            end = span.end;
        }
        if self.spans.first().is_none_or(|s| s.message_index != 0)
            || message != self.target_index
            || end != characters.len() as u64
        {
            return Err("incomplete prepared SFT ownership ledger");
        }
        let mut answer_indices = Vec::new();
        let mut previous = 0;
        for index in 0..length {
            let [raw_start, raw_end] = self.offset_mapping[index];
            let [start, end] = self.ownership_offsets[index];
            if raw_start < previous
                || raw_start >= raw_end
                || start > raw_start
                || end < raw_end
                || start >= end
                || end > characters.len() as u64
            {
                return Err("invalid prepared SFT token ownership offsets");
            }
            previous = raw_start;
            let owners: Vec<_> = self
                .spans
                .iter()
                .filter(|span| span.start < end && span.end > start)
                .collect();
            let Some(owner) = owners.first() else {
                return Err("unowned prepared SFT token");
            };
            if owners
                .iter()
                .any(|other| other.supervised != owner.supervised)
            {
                return Err("mixed prepared SFT loss ownership");
            }
            let kinds: Vec<_> = owners
                .iter()
                .map(|span| span.kind.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let label = if owner.supervised {
                i64::from(self.input_ids[index])
            } else {
                -100
            };
            if self.token_kinds[index] != kinds || self.labels[index] != label {
                return Err("prepared SFT labels or ownership kinds disagree");
            }
            if index > 0
                && owner.supervised
                && kinds == ["answer"]
                && characters[start as usize..end as usize]
                    .iter()
                    .any(|c| !python_whitespace(*c))
            {
                answer_indices.push(index as u64);
            }
        }
        let supervised = self.labels.iter().filter(|label| **label != -100).count() as u64;
        if answer_indices.is_empty()
            || answer_indices != self.shifted_answer_token_indices
            || self.supervised_tokens != supervised
            || self.context_tokens != length as u64 - supervised
            || self.effective_shifted_supervised_tokens
                != self.labels.iter().skip(1).filter(|l| **l != -100).count() as u64
        {
            return Err("contradictory prepared SFT supervised/context/answer counts");
        }
        Ok(())
    }
}

// CPython 3.12 str.isspace: Unicode White_Space plus the four information separators.
fn python_whitespace(c: char) -> bool {
    matches!(c, '\u{0009}'..='\u{000d}' | '\u{001c}'..='\u{0020}' | '\u{0085}' | '\u{00a0}' |
        '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}
