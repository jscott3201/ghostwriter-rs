//! Explicit text profiles and immutable policy commitments. This module performs no I/O and
//! does not execute tokenization; the installed Python consumer separately replays the renderer.
use crate::{PreparedSftExample, PreparedSftRecipe};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Supported, qualified text-only preparation profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreparedSftProfileName {
    /// The pinned Qwen3-0.6B tokenizer and existing qualified dependency stack.
    #[serde(rename = "qwen3_text_v1")]
    Qwen3TextV1,
    /// The pinned Gemma4 E2B processor's text subset and separately qualified stack.
    #[serde(rename = "gemma4_e2b_text_v1")]
    Gemma4E2bTextV1,
}

/// Complete supported official renderer controls; tool and multimodal controls are unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftRenderControls {
    /// Qwen requires true; Gemma supports an explicit true/false thinking preamble.
    pub enable_thinking: bool,
    /// Always false for complete training conversations.
    pub add_generation_prompt: bool,
    /// Gemma explicitly disables historical tool-thinking preservation; absent for Qwen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preserve_thinking: Option<bool>,
}

/// Exact profile selection and controls bound into the version-two preparation recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftProfile {
    /// Closed set of supported model-specific text contracts.
    pub name: PreparedSftProfileName,
    /// Exact renderer controls; saved declarations do not grant execution authority.
    pub controls: PreparedSftRenderControls,
}

impl PreparedSftProfile {
    pub(crate) fn validate_recipe(&self, recipe: &PreparedSftRecipe) -> Result<u32, &'static str> {
        use PreparedSftProfileName::{Gemma4E2bTextV1, Qwen3TextV1};
        if self.controls.add_generation_prompt
            || match self.name {
                Qwen3TextV1 => {
                    !self.controls.enable_thinking || self.controls.preserve_thinking.is_some()
                }
                Gemma4E2bTextV1 => self.controls.preserve_thinking != Some(false),
            }
        {
            return Err("unsupported prepared text-profile rendering controls");
        }
        // Ordinary BLAKE3 over compact UTF-8 JSON with recursively sorted keys. These pins
        // commit every release file, template, wrapper/BOS/EOS/pad setting, backend, added
        // token, and dependency in the independently qualified packaged profile data.
        // Keep the schema independent of Python files or build-time adapter availability.
        let (pins, vocabulary) = match self.name {
            Qwen3TextV1 => (
                [
                    "943d1250f5aac099a750f727e03bf69ac8317fc07a54433ea87aaf270826ea19",
                    "92d8b5779f4c26f6f91a152d0f356070207ff3ab19fb06039a8cc16446994025",
                    "428f189b94afe5048d4cf4c660bfe4b3baaf87fc75a39352dd7c5ddb88e1ef4c",
                ],
                151_669,
            ),
            Gemma4E2bTextV1 => (
                [
                    "bdf713b7d07d72f53807041f493837c2ab12d8e019fb7b4832f3c6e81b21dff3",
                    "d97d6a08801124a98a11bd211d23fc966a7ded707496830954fefecac8df5628",
                    "08c6a7480c62167482471321ab8b421838c56de65063246c12b7912ab0424d58",
                ],
                262_144,
            ),
        };
        let dependencies = serde_json::to_value(&recipe.dependencies)
            .map_err(|_| "invalid prepared text-profile dependencies")?;
        for (value, expected) in [&recipe.tokenizer, &recipe.tokenizer_policy, &dependencies]
            .into_iter()
            .zip(pins)
        {
            let bytes = serde_json::to_vec(&sorted(value))
                .map_err(|_| "invalid prepared text-profile policy")?;
            if blake3::hash(&bytes).to_hex().as_str() != expected {
                return Err("prepared text-profile policy differs from immutable commitments");
            }
        }
        if recipe.tokenizer_target
            != serde_json::json!({
                "repository": recipe.tokenizer["repository"],
                "revision": recipe.tokenizer["revision"],
            })
        {
            return Err("prepared text-profile tokenizer target mismatch");
        }
        Ok(vocabulary)
    }

    pub(crate) fn validate_example(
        &self,
        example: &PreparedSftExample,
    ) -> Result<(), &'static str> {
        if self.name != PreparedSftProfileName::Gemma4E2bTextV1 {
            return Ok(());
        }
        let has_thinking = example
            .rendered
            .starts_with("<bos><|turn>system\n<|think|>\n");
        if !example.rendered.starts_with("<bos><|turn>")
            || !example.rendered.ends_with("<turn|>\n")
            || example.input_ids.first() != Some(&2)
            || !example.input_ids.ends_with(&[106, 107])
            || example.offset_mapping.first() != Some(&[0, 5])
            || example.labels.first() != Some(&-100)
            || !example.labels.ends_with(&[106, -100])
            || example.rendered.matches("<bos>").count() != 1
            || has_thinking != self.controls.enable_thinking
            || example.rendered.matches("<|think|>").count()
                != usize::from(self.controls.enable_thinking)
        {
            return Err("Gemma prepared BOS, turn ending, or thinking controls disagree");
        }
        Ok(())
    }
}

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let entries: std::collections::BTreeMap<_, _> = map
                .iter()
                .map(|(key, value)| (key.clone(), sorted(value)))
                .collect();
            Value::Object(entries.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}
