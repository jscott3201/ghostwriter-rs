//! Exact approved Gemma layouts, inert configurations and narrow local adapter recipe.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Exact text embedding name in the official conditional-generation architecture.
pub const GEMMA_EMBEDDING: &str = "model.language_model.embed_tokens.weight";
/// Logical tied output alias, optionally omitted from safe serialized weights.
pub const GEMMA_HEAD: &str = "lm_head.weight";
/// Four persistent scalar bounds of the official modality clippable linear.
pub const GEMMA_CLIPS: [&str; 4] = ["input_min", "input_max", "output_min", "output_max"];
/// Application-owned exact release identity.
pub const GEMMA_RELEASE: &str = "gemma4_e2b_3e22461_student_lora_v1";
/// Explicit seeded reduced random software fixture, never pretrained execution evidence.
pub const GEMMA_FIXTURE: &str = "owned_gemma4_text_fixture_v1";
/// Exact approved single safe weights file size.
pub const GEMMA_RELEASE_BYTES: u64 = 10_246_621_918;
/// SHA256 of the approved single safe weights file.
pub const GEMMA_RELEASE_SHA256: &str =
    "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550";

/// Only two exact reviewed configuration documents are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaBaseKind {
    /// Full selected publisher release, including frozen vision/audio towers.
    Release,
    /// Owned small text-only random CPU fixture with the full vocabulary.
    Fixture,
}
impl GemmaBaseKind {
    /// Match exact inert bytes, rejecting alternate or duplicate fields and loader options.
    ///
    /// # Errors
    /// Rejects every configuration except the reviewed release or canonical owned fixture.
    pub fn from_config(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes == include_bytes!("gemma/release_config.json") {
            Ok(Self::Release)
        } else if bytes.trim_ascii() == include_bytes!("gemma/fixture_config.json").trim_ascii() {
            Ok(Self::Fixture)
        } else {
            Err("unsupported exact Gemma base configuration")
        }
    }
    /// Scoped declared source class, never historical execution authority.
    #[must_use]
    pub fn authorization(self) -> &'static str {
        if self == Self::Release {
            GEMMA_RELEASE
        } else {
            GEMMA_FIXTURE
        }
    }
    /// Complete reviewed logical shapes, including buffers and the tied head alias.
    #[must_use]
    pub fn base_shapes(self) -> BTreeMap<String, Vec<u64>> {
        let release = self == Self::Release;
        let (h, p, count, shared, heads, local, global, intermediate) = if release {
            (1536, 256, 35, 20, 8, 256, 512, 6144)
        } else {
            (16, 4, 4, 2, 2, 8, 16, 32)
        };
        let root = "model.language_model.";
        let mut result = BTreeMap::from([
            (GEMMA_EMBEDDING.into(), vec![262_144, h]),
            (GEMMA_HEAD.into(), vec![262_144, h]),
            (format!("{root}norm.weight"), vec![h]),
            (
                format!("{root}embed_tokens_per_layer.weight"),
                vec![262_144, count * p],
            ),
            (
                format!("{root}per_layer_model_projection.weight"),
                vec![count * p, h],
            ),
            (format!("{root}per_layer_projection_norm.weight"), vec![p]),
        ]);
        for index in 0..count {
            let full = if release {
                index % 5 == 4
            } else {
                index % 2 == 1
            };
            let d = if full { global } else { local };
            let q = heads * d;
            let i = intermediate * if index >= count - shared { 2 } else { 1 };
            let prefix = format!("{root}layers.{index}.");
            for (name, shape) in [
                ("layer_scalar", vec![1]),
                ("mlp.gate_proj.weight", vec![i, h]),
                ("mlp.up_proj.weight", vec![i, h]),
                ("mlp.down_proj.weight", vec![h, i]),
                ("per_layer_input_gate.weight", vec![p, h]),
                ("per_layer_projection.weight", vec![h, p]),
                ("self_attn.q_norm.weight", vec![d]),
                ("self_attn.q_proj.weight", vec![q, h]),
                ("self_attn.o_proj.weight", vec![h, q]),
            ] {
                result.insert(format!("{prefix}{name}"), shape);
            }
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "post_feedforward_layernorm",
                "post_per_layer_input_norm",
                "pre_feedforward_layernorm",
            ] {
                result.insert(format!("{prefix}{name}.weight"), vec![h]);
            }
            if index < count - shared {
                for (name, shape) in [
                    ("k_norm", vec![d]),
                    ("k_proj", vec![d, h]),
                    ("v_proj", vec![d, h]),
                ] {
                    result.insert(format!("{prefix}self_attn.{name}.weight"), shape);
                }
            }
        }
        if release {
            modalities(&mut result);
        }
        result
    }
    /// Distinct parameters exclude tied aliases and all persistent buffers.
    #[must_use]
    pub fn parameter_count(self) -> u64 {
        self.base_shapes()
            .iter()
            .filter(|(name, _)| {
                name.as_str() != GEMMA_HEAD
                    && !name.ends_with(".layer_scalar")
                    && !GEMMA_CLIPS.contains(&name.rsplit('.').next().unwrap_or(""))
            })
            .map(|(_, shape)| shape.iter().product::<u64>())
            .sum()
    }
    /// Exact q/v text projection target set; full release has 35 q plus 15 v projections.
    #[must_use]
    pub fn targets(self) -> Vec<GemmaLoraTarget> {
        self.base_shapes()
            .into_iter()
            .filter(|(name, _)| {
                name.starts_with("model.language_model.layers.")
                    && (name.ends_with("self_attn.q_proj.weight")
                        || name.ends_with("self_attn.v_proj.weight"))
            })
            .map(|(name, shape)| GemmaLoraTarget {
                path: name.trim_end_matches(".weight").into(),
                input_features: shape[1],
                output_features: shape[0],
            })
            .collect()
    }
    /// Safe PEFT saved parameter names, with no embedding or base population.
    #[must_use]
    pub fn adapter_shapes(self) -> BTreeMap<String, Vec<u64>> {
        let mut result = BTreeMap::new();
        for target in self.targets() {
            let prefix = format!("base_model.model.{}.lora_", target.path);
            result.insert(format!("{prefix}A.weight"), vec![8, target.input_features]);
            result.insert(format!("{prefix}B.weight"), vec![target.output_features, 8]);
        }
        result
    }
    /// Complete strict inert adapter configuration, consumed only as data.
    #[must_use]
    pub fn adapter_config(self) -> serde_json::Value {
        serde_json::json!({"version":1,"kind":"gemma_text_qv_lora","rank":8,"alpha":8,
            "dropout":0,"bias":"none","task":"causal_lm","targets":self.targets(),
            "modules_to_save":[],"quantization":false,"dora":false,"rslora":false,"save_embedding_layers":false})
    }
}

/// One measured eligible text-attention linear projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GemmaLoraTarget {
    /// Exact official projection module path.
    pub path: String,
    /// Input width from its actual base weight.
    pub input_features: u64,
    /// Output width from its actual base weight.
    pub output_features: u64,
}

fn linear(result: &mut BTreeMap<String, Vec<u64>>, prefix: &str, output: u64, input: u64) {
    result.insert(format!("{prefix}.linear.weight"), vec![output, input]);
    for name in GEMMA_CLIPS {
        result.insert(format!("{prefix}.{name}"), vec![]);
    }
}
fn modalities(r: &mut BTreeMap<String, Vec<u64>>) {
    let root = "model.vision_tower.";
    r.insert(
        format!("{root}patch_embedder.input_proj.weight"),
        vec![768, 768],
    );
    r.insert(
        format!("{root}patch_embedder.position_embedding_table"),
        vec![2, 10240, 768],
    );
    r.insert(
        "model.embed_vision.embedding_projection.weight".into(),
        vec![1536, 768],
    );
    for index in 0..16 {
        let p = format!("{root}encoder.layers.{index}.");
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            r.insert(format!("{p}{name}.weight"), vec![768]);
        }
        for name in ["q_norm", "k_norm"] {
            r.insert(format!("{p}self_attn.{name}.weight"), vec![64]);
        }
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            linear(r, &format!("{p}self_attn.{name}"), 768, 768);
        }
        for name in ["gate_proj", "up_proj"] {
            linear(r, &format!("{p}mlp.{name}"), 3072, 768);
        }
        linear(r, &format!("{p}mlp.down_proj"), 768, 3072);
    }
    let root = "model.audio_tower.";
    r.insert(format!("{root}output_proj.bias"), vec![1536]);
    r.insert(format!("{root}output_proj.weight"), vec![1536, 1024]);
    r.insert(
        format!("{root}subsample_conv_projection.input_proj_linear.weight"),
        vec![1024, 1024],
    );
    r.insert(
        "model.embed_audio.embedding_projection.weight".into(),
        vec![1536, 1536],
    );
    for (index, output, input) in [(0, 128, 1), (1, 32, 128)] {
        let p = format!("{root}subsample_conv_projection.layer{index}.");
        r.insert(format!("{p}conv.weight"), vec![output, input, 3, 3]);
        r.insert(format!("{p}norm.weight"), vec![output]);
    }
    for index in 0..12 {
        let p = format!("{root}layers.{index}.");
        for ffw in ["feed_forward1", "feed_forward2"] {
            linear(r, &format!("{p}{ffw}.ffw_layer_1"), 4096, 1024);
            linear(r, &format!("{p}{ffw}.ffw_layer_2"), 1024, 4096);
            for name in ["pre_layer_norm", "post_layer_norm"] {
                r.insert(format!("{p}{ffw}.{name}.weight"), vec![1024]);
            }
        }
        for name in ["conv_norm", "pre_layer_norm"] {
            r.insert(format!("{p}lconv1d.{name}.weight"), vec![1024]);
        }
        r.insert(
            format!("{p}lconv1d.depthwise_conv1d.weight"),
            vec![1024, 1, 5],
        );
        linear(r, &format!("{p}lconv1d.linear_start"), 2048, 1024);
        linear(r, &format!("{p}lconv1d.linear_end"), 1024, 1024);
        for name in ["norm_out", "norm_post_attn", "norm_pre_attn"] {
            r.insert(format!("{p}{name}.weight"), vec![1024]);
        }
        for name in ["q_proj", "k_proj", "v_proj", "post"] {
            linear(r, &format!("{p}self_attn.{name}"), 1024, 1024);
        }
        r.insert(
            format!("{p}self_attn.relative_k_proj.weight"),
            vec![1024, 1024],
        );
        r.insert(format!("{p}self_attn.per_dim_scale"), vec![128]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reviewed_layouts_preserve_sharing_buffers_and_distinct_head_widths() {
        let release = GemmaBaseKind::Release;
        assert_eq!(release.base_shapes().len(), 1952);
        assert_eq!(release.parameter_count(), 5_104_297_504);
        assert_eq!(release.targets().len(), 50);
        assert_eq!(
            release
                .targets()
                .iter()
                .filter(|t| t.path.ends_with("q_proj"))
                .count(),
            35
        );
        let shapes = release.base_shapes();
        assert_eq!(
            shapes["model.language_model.layers.0.self_attn.q_proj.weight"],
            [2048, 1536]
        );
        assert_eq!(
            shapes["model.language_model.layers.4.self_attn.q_proj.weight"],
            [4096, 1536]
        );
        assert!(!shapes.contains_key("model.language_model.layers.15.self_attn.v_proj.weight"));
        assert_eq!(
            shapes["model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min"],
            Vec::<u64>::new()
        );
        assert_eq!(GemmaBaseKind::Fixture.parameter_count(), 8_402_844);
        assert_eq!(GemmaBaseKind::Fixture.targets().len(), 6);
        assert_eq!(
            GemmaBaseKind::Fixture
                .adapter_shapes()
                .values()
                .map(|s| s.iter().product::<u64>())
                .sum::<u64>(),
            1728
        );
    }
    #[test]
    fn configuration_authority_is_exact_and_never_generic_loader_data() {
        assert_eq!(
            GemmaBaseKind::from_config(include_bytes!("gemma/release_config.json")),
            Ok(GemmaBaseKind::Release)
        );
        assert_eq!(
            GemmaBaseKind::from_config(include_bytes!("gemma/fixture_config.json")),
            Ok(GemmaBaseKind::Fixture)
        );
        assert!(GemmaBaseKind::from_config(br#"{"model_type":"gemma4","auto_map":{}}"#).is_err());
        let mut value: serde_json::Value =
            serde_json::from_slice(include_bytes!("gemma/fixture_config.json")).unwrap();
        value["text_config"]["hidden_size"] = 32.into();
        assert!(GemmaBaseKind::from_config(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
