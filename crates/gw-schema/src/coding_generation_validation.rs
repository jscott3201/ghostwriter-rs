//! Strict complete generated-pair declarations; no saved field grants fresh model authority.
use crate::coding_value::{coding_hash_valid as hash, coding_json_digest};
use crate::*;
use serde_json::Value;

fn policy(text: &str) -> Value {
    serde_json::from_str(text).expect("embedded qualified generation policy")
}
impl CodingComparisonModels {
    /// Stable side-specific model identity; the adapter is never treated as a full model.
    #[must_use]
    pub fn model_id(&self, side: CodingModelSide) -> String {
        coding_json_digest(
            "ghostwriter.coding-comparison-model.v1",
            &serde_json::json!({
                "base":self.base_model.tensor_content_id,
                "adapter":if side == CodingModelSide::Candidate {Some(&self.final_adapter.tensor_content_id)} else {None}
            }),
        )
    }
    /// Validate declared model, completion and complete file identities without granting eligibility.
    ///
    /// # Errors
    /// Rejects malformed identities, unsupported source kinds, invented parent revisions or files.
    pub fn validate(&self) -> Result<(), &'static str> {
        if [
            &self.completion_id,
            &self.prepared_build_id,
            &self.producer_recipe_id,
            &self.training_source_sha256,
            &self.preparation_source_sha256,
        ]
        .into_iter()
        .any(|id| !hash(id))
            || ![GEMMA_RELEASE, GEMMA_FIXTURE].contains(&self.source_authorization.as_str())
            || self.publisher_model != "google/gemma-4-E2B-it"
            || self.publisher_revision != "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
            || self.declared_parent != "google/gemma-4-E2B"
            || self.parent_revision.is_some()
        {
            return Err("unsupported coding model binding or invented upstream lineage");
        }
        for model in [&self.base_model, &self.final_adapter] {
            if !hash(&model.tensor_content_id)
                || !(1..=6_000_000_000).contains(&model.parameter_count)
                || !(1..=4096).contains(&model.tensor_count)
            {
                return Err("invalid coding model tensor inventory");
            }
        }
        if self.checkpoint_files.len() != LORA_FILES.len() {
            return Err("incomplete coding checkpoint inventory");
        }
        for (file, path) in self.checkpoint_files.iter().zip(LORA_FILES) {
            if file.path != path
                || file.byte_length == 0
                || file.byte_length > MAX_LORA_BYTES
                || !hash(&file.blake3)
            {
                return Err("invalid coding checkpoint file binding");
            }
        }
        Ok(())
    }
}
impl GemmaCodingGenerationRecipe {
    /// Exact common protocol identity, separately versioned from preparation and LoRA.
    #[must_use]
    pub fn recipe_id(&self) -> String {
        coding_json_digest("ghostwriter.gemma-coding-generation.v1", self)
    }
    /// Validate the complete qualified CPU protocol and actual effective config declaration.
    ///
    /// # Errors
    /// Rejects changed policy, unsupported generation controls or invalid runtime declarations.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != 1
            || !hash(&self.source_sha256)
            || self.profile != "gemma4_e2b_text_v1"
            || !(1..=512).contains(&self.max_new_tokens)
            || !(1..=2048).contains(&self.max_prompt_tokens)
            || self.system_prompt.len() > 8192
            || !self.add_generation_prompt
            || self.enable_thinking
            || self.preserve_thinking
            || self.device != "cpu"
            || self.dtype != "float32"
            || self.attention != "eager"
            || self.threads != 1
            || self.processes != 1
            || !self.deterministic_algorithms
            || self.seed != 0
            || self.padding
            || self.truncation
            || !self.fresh_cache
            || self.cache != "dynamic"
            || self.compile
        {
            return Err("unsupported Gemma coding generation recipe");
        }
        if self.tokenizer
            != policy(include_str!(
                "../../../adapters/trl/src/ghostwriter_trl/profiles/gemma_manifest.json"
            ))
            || self.tokenizer_policy
                != policy(include_str!(
                    "../../../adapters/trl/src/ghostwriter_trl/profiles/gemma_policy.json"
                ))
            || serde_json::to_value(&self.dependencies).map_err(|_| "invalid dependencies")?
                != policy(include_str!(
                    "../../../adapters/trl/src/ghostwriter_trl/lora/dependencies.json"
                ))
        {
            return Err("generation dependency or official tokenizer policy changed");
        }
        if self.runtime.len() != 4
            || ["python", "implementation", "system", "machine"]
                .iter()
                .any(|key| {
                    self.runtime.get(*key).is_none_or(|s| {
                        s.is_empty() || s.len() > 128 || s.chars().any(char::is_control)
                    })
                })
        {
            return Err("invalid generation runtime identity");
        }
        let mut expected = policy(include_str!(
            "../../../adapters/trl/src/ghostwriter_trl/comparison/generation_config.json"
        ));
        expected["max_new_tokens"] = self.max_new_tokens.into();
        if self.effective_config_json
            != serde_json::to_string(&expected).map_err(|_| "invalid config policy")?
        {
            return Err("effective generation config differs from the complete pinned policy");
        }
        if self
            .control_literals()
            .any(|s| self.system_prompt.contains(s))
        {
            return Err("generation system prompt contains reserved controls");
        }
        Ok(())
    }
    fn control_literals(&self) -> impl Iterator<Item = &str> {
        self.tokenizer_policy["added_tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry["content"].as_str())
    }
    /// Expected official one-task rendered text. Only the template's own whitespace rule applies.
    #[must_use]
    pub fn rendered_prompt(&self, public_prompt: &str) -> String {
        let strip = |text: &str| {
            text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
                .to_owned()
        };
        let mut rendered = String::from("<bos>");
        if !self.system_prompt.is_empty() {
            rendered.push_str(&format!(
                "<|turn>system\n{}<turn|>\n",
                strip(&self.system_prompt)
            ));
        }
        rendered.push_str(&format!(
            "<|turn>user\n{}<turn|>\n<|turn>model\n",
            strip(public_prompt)
        ));
        rendered
    }
}
impl CapturedCodingGeneration {
    /// Verify lossless token boundaries and deterministic format classification.
    /// Exact tokenizer decoding is separately replayed by the Python consumer.
    ///
    /// # Errors
    /// Rejects changed prompt, omitted IDs, invented terminals, hidden controls or invalid bounds.
    pub fn validate(
        &self,
        recipe: &GemmaCodingGenerationRecipe,
        public_prompt: &str,
    ) -> Result<(), &'static str> {
        let p = &self.prompt;
        let o = &self.output;
        let max = recipe.max_new_tokens;
        let vocab = recipe.tokenizer_policy["vocab_size"]
            .as_u64()
            .ok_or("missing vocabulary")?;
        p.validate(recipe, public_prompt)?;
        if self.effective_max_length != p.input_ids.len() + max
            || self.cache_type != "transformers.cache_utils.DynamicCache"
            || o.sequence_ids.len() > 4096
            || o.original_suffix_text.len() > 131072
            || o.body_text.len() > 131072
            || o.sequence_ids.iter().any(|id| u64::from(*id) >= vocab)
        {
            return Err("invalid exact coding generation runtime or token population");
        }
        use CodingGeneratedRepresentation::{Module, Unknown, Unsupported};
        use CodingGenerationTermination::*;
        if !o.sequence_ids.starts_with(&p.input_ids) {
            if o.termination != PrefixMismatch
                || o.representation != Unknown
                || !o.suffix_ids.is_empty()
                || !o.body_ids.is_empty()
                || !o.original_suffix_text.is_empty()
                || !o.body_text.is_empty()
                || o.terminal_id.is_some()
                || o.at_token_bound
            {
                return Err("prefix mismatch must retain only the unmodified returned sequence");
            }
            return Ok(());
        }
        let suffix = &o.sequence_ids[p.input_ids.len()..];
        let final_id = suffix
            .last()
            .copied()
            .filter(|id| [1, 106, 50].contains(id));
        let body = if matches!(final_id, Some(1 | 106)) {
            &suffix[..suffix.len() - 1]
        } else {
            suffix
        };
        let termination = if suffix.len() > max {
            OverBound
        } else {
            match final_id {
                Some(1) => Eos,
                Some(106) => TurnEnd,
                Some(50) => ToolHandoff,
                _ if suffix.len() == max => LengthLimit,
                _ => ShortReturn,
            }
        };
        let unsupported = final_id == Some(50)
            || body.is_empty()
            || o.body_text.is_empty()
            || o.body_text.len() > 65536
            || body.iter().any(|id| {
                recipe.tokenizer_policy["added_tokens"]
                    .as_array()
                    .expect("validated policy")
                    .iter()
                    .any(|entry| entry["id"].as_u64() == Some(u64::from(*id)))
            })
            || recipe.control_literals().any(|s| o.body_text.contains(s));
        let representation = if matches!(termination, ShortReturn | OverBound) {
            Unknown
        } else if unsupported {
            Unsupported
        } else {
            Module
        };
        if o.suffix_ids != suffix
            || o.body_ids != body
            || o.terminal_id != final_id
            || o.at_token_bound != (suffix.len() == max)
            || o.termination != termination
            || o.representation != representation
        {
            return Err("coding suffix, terminal or representation was altered");
        }
        Ok(())
    }
}
impl CodingGenerationPrompt {
    fn validate(
        &self,
        recipe: &GemmaCodingGenerationRecipe,
        public_prompt: &str,
    ) -> Result<(), &'static str> {
        let p = self;
        let vocab = recipe.tokenizer_policy["vocab_size"]
            .as_u64()
            .ok_or("missing vocabulary")?;
        if p.input_ids.is_empty()
            || p.input_ids.len() > recipe.max_prompt_tokens
            || p.input_ids.first() != Some(&2)
            || p.input_ids.iter().filter(|i| **i == 2).count() != 1
            || p.attention_mask != vec![1; p.input_ids.len()]
            || p.rendered != recipe.rendered_prompt(public_prompt)
            || recipe.control_literals().any(|s| public_prompt.contains(s))
            || p.input_ids.iter().any(|id| u64::from(*id) >= vocab)
        {
            return Err("invalid exact coding generation prompt, runtime or token population");
        }
        Ok(())
    }
}
impl CodingPairRequest {
    /// Stable complete source/generation request identity, including both ordered sides.
    #[must_use]
    pub fn request_id(&self) -> String {
        coding_json_digest("ghostwriter.coding-pair-request.v1", self)
    }
    /// Require both exact sides of every captured member before any coding execution starts.
    ///
    /// # Errors
    /// Rejects missing, duplicate, reordered, swapped, foreign or contradictory generated answers.
    pub fn validate(&self, population: &CodingPopulation) -> Result<(), &'static str> {
        population.validate()?;
        self.models.validate()?;
        self.recipe.validate()?;
        let s = &self.separation;
        if s.semantic_screening != "not_run"
            || !["not_run", "declared_source_screened"].contains(&s.source_screening.as_str())
        {
            return Err("unsupported coding comparison separation claim");
        }
        let unrendered: Vec<_> = population
            .members
            .iter()
            .filter(|m| s.unrendered_member_ids.contains(&m.member_id))
            .map(|m| m.member_id.clone())
            .collect();
        if unrendered != s.unrendered_member_ids
            || s.effective_prompt_separation
                != if unrendered.is_empty() {
                    "generation_recipe_prompt_ids_disjoint"
                } else {
                    "incomplete_unrenderable_heldout"
                }
        {
            return Err("inconsistent unrendered held-out separation scope");
        }
        if s.training_population == "registered_reference_train" {
            if s.training_member_ids
                != population
                    .training_members
                    .iter()
                    .map(|m| m.member_id.clone())
                    .collect::<Vec<_>>()
            {
                return Err("prepared training membership differs from accepted references");
            }
        } else if s.training_population != "owned_software_fixture"
            || self.models.source_authorization != GEMMA_FIXTURE
            || !s.training_member_ids.is_empty()
        {
            return Err("real comparison requires the complete registered Train population");
        }
        if self.version != 1
            || self.population_id != population.population_id
            || self.rows.len() != population.members.len() * 2
        {
            return Err("generated pair does not cover the complete captured population");
        }
        for (member, pair) in population.members.iter().zip(self.rows.as_chunks::<2>().0) {
            for (row, side) in pair
                .iter()
                .zip([CodingModelSide::Base, CodingModelSide::Candidate])
            {
                if row.side != side
                    || row.member_id != member.member_id
                    || row.model_id != self.models.model_id(side)
                {
                    return Err("generated pair side, member or model binding was swapped");
                }
                if unrendered.contains(&member.member_id)
                    != (row.failure.as_deref() == Some("prompt_rejected"))
                {
                    return Err("prompt rejection differs from unrendered separation scope");
                }
                match (&row.generation, &row.failure) {
                    (Some(generation), None) if row.failed_prompt.is_none() => {
                        generation.validate(&self.recipe, &member.prompt)?
                    }
                    (None, Some(reason))
                        if ["generation_error", "prompt_rejected", "cancelled"]
                            .contains(&reason.as_str()) =>
                    {
                        if let Some(prompt) = &row.failed_prompt {
                            prompt.validate(&self.recipe, &member.prompt)?;
                        }
                        if (reason == "generation_error" && row.failed_prompt.is_none())
                            || (reason == "prompt_rejected" && row.failed_prompt.is_some())
                        {
                            return Err("generation failure lost its actual captured input");
                        }
                    }
                    _ => {
                        return Err(
                            "generated answer must retain either one observation or an explicit failure",
                        );
                    }
                }
            }
            if let (Some(a), Some(b)) = (&pair[0].generation, &pair[1].generation)
                && (a.prompt != b.prompt
                    || a.effective_max_length != b.effective_max_length
                    || a.cache_type != b.cache_type)
            {
                return Err("paired models did not consume identical complete prompt inputs");
            }
        }
        Ok(())
    }
}
