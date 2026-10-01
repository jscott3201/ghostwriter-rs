//! Complete checkpoint inspection: actual captured bytes and consistent declarations, no execution.
use crate::{Result, artifact::integrity, checkpoint_tensors, verify_prepared_sft_snapshot};
use gw_schema::{
    CHECKPOINT_HASH_DOMAIN, CHECKPOINT_MAGIC, CheckpointModelConfig, CheckpointModelSummary,
    MAX_CHECKPOINT_MANIFEST_BYTES, TrainingCheckpointManifest, TrainingMicrobatch,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;

/// Receipt from native captured-byte inspection, separate from Python replay and training evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointReport {
    /// Receipt contract version.
    pub report_version: u32,
    /// Complete independently measured completion identity.
    pub completion_id: String,
    /// Original immutable input build identity.
    pub prepared_build_id: String,
    /// Complete captured stream length.
    pub byte_length: u64,
    /// Measured initial model content.
    pub initial_model: CheckpointModelSummary,
    /// Measured final model content.
    pub checkpoint_model: CheckpointModelSummary,
    /// Byte, structure, tensor and accounting validation result.
    pub structural_validation: String,
    /// Saved declarations cannot authenticate historical training.
    pub historical_training: String,
    /// Native inspection does not import or run the Python model.
    pub model_reload: String,
    /// Native structure verification does not execute the official tokenizer.
    pub tokenizer_replay: String,
    /// Original manifest declarations, visibly distinct from measured fields.
    pub declarations: TrainingCheckpointManifest,
}

struct Hashed<'a, R> {
    reader: &'a mut R,
    body: &'a mut blake3::Hasher,
    file: blake3::Hasher,
    sha: Sha256,
    remaining: u64,
}
impl<R: Read> Read for Hashed<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let count = self.remaining.min(buf.len() as u64) as usize;
        if count == 0 {
            return Ok(0);
        }
        let n = self.reader.read(&mut buf[..count])?;
        self.body.update(&buf[..n]);
        self.file.update(&buf[..n]);
        self.sha.update(&buf[..n]);
        self.remaining -= n as u64;
        Ok(n)
    }
}

/// Stream and verify a complete bounded checkpoint without extracting files or loading a model.
///
/// # Errors
/// Rejects corrupt framing, unsafe/incomplete weights, mismatched source, or contradictory recipe
/// and batch accounting. Success authenticates captured bytes, never historical model execution.
pub fn verify_training_checkpoint(mut reader: impl Read) -> Result<TrainingCheckpointReport> {
    let mut header = [0; 48];
    reader.read_exact(&mut header)?;
    if &header[..8] != CHECKPOINT_MAGIC {
        return Err(integrity("unsupported checkpoint magic"));
    }
    let manifest_length = u64::from_be_bytes(
        header[40..48]
            .try_into()
            .map_err(|_| integrity("invalid frame"))?,
    );
    if manifest_length == 0 || manifest_length > MAX_CHECKPOINT_MANIFEST_BYTES as u64 {
        return Err(integrity("checkpoint manifest length exceeds bound"));
    }
    let mut raw = vec![0; manifest_length as usize];
    reader.read_exact(&mut raw)?;
    let manifest = TrainingCheckpointManifest::from_json(&raw).map_err(integrity)?;
    let mut body = blake3::Hasher::new_derive_key(CHECKPOINT_HASH_DOMAIN);
    body.update(&header[40..]);
    body.update(&raw);
    let mut configs = Vec::new();
    let mut models = Vec::new();
    let mut prepared = None;
    for file in &manifest.files {
        let mut stream = Hashed {
            reader: &mut reader,
            body: &mut body,
            file: blake3::Hasher::new(),
            sha: Sha256::new(),
            remaining: file.byte_length,
        };
        if file.path.ends_with("config.json") {
            let mut bytes = Vec::new();
            stream.read_to_end(&mut bytes)?;
            configs.push(CheckpointModelConfig::from_json(&bytes).map_err(integrity)?);
        } else if file.path.ends_with(".safetensors") {
            let config = configs
                .last()
                .ok_or_else(|| integrity("checkpoint config missing"))?;
            models.push(checkpoint_tensors::measure(
                &mut stream,
                file.byte_length,
                config,
                models.is_empty(),
            )?);
        } else {
            let mut bytes = Vec::new();
            stream.read_to_end(&mut bytes)?;
            prepared = Some(verify_prepared_sft_snapshot(bytes)?);
        }
        if stream.remaining != 0 || stream.file.finalize().to_hex().as_str() != file.blake3 {
            return Err(integrity(
                "captured checkpoint file length or digest mismatch",
            ));
        }
        if manifest.source_authorization == "qwen3_0_6b_c1899de_student_training_v1" {
            let pin = match file.path.as_str() {
                "initial/config.json" => {
                    Some("660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd")
                }
                "initial/model.safetensors" => {
                    Some("f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b")
                }
                _ => None,
            };
            if pin.is_some_and(|pin| format!("{:x}", stream.sha.finalize()) != pin) {
                return Err(integrity(
                    "initial model differs from the approved exact release",
                ));
            }
        }
    }
    let digest = body.finalize();
    if digest.as_bytes() != &header[8..40] {
        return Err(integrity("checkpoint completion digest mismatch"));
    }
    if reader.read(&mut [0])? != 0 {
        return Err(integrity("trailing checkpoint bytes"));
    }
    if configs.len() != 2
        || !configs[0].same_architecture(&configs[1])
        || models.len() != 2
        || models[0] != manifest.checkpoint_model
        || models[1] != manifest.initial_model
    {
        return Err(integrity(
            "measured checkpoint model/config differs from declarations",
        ));
    }
    let prepared = prepared.ok_or_else(|| integrity("missing prepared input"))?;
    if prepared.report().build_id != manifest.prepared_build_id {
        return Err(integrity("checkpoint prepared input identity mismatch"));
    }
    validate_inputs(&manifest, prepared.payload(), &configs[0])?;
    let byte_length =
        48 + manifest_length + manifest.files.iter().map(|f| f.byte_length).sum::<u64>();
    Ok(TrainingCheckpointReport {
        report_version: 1,
        completion_id: digest.to_hex().to_string(),
        prepared_build_id: manifest.prepared_build_id.clone(),
        byte_length,
        checkpoint_model: models[0].clone(),
        initial_model: models[1].clone(),
        structural_validation: "passed".into(),
        historical_training: "declared".into(),
        model_reload: "not_run".into(),
        tokenizer_replay: "not_run".into(),
        declarations: manifest,
    })
}

fn validate_inputs(
    manifest: &TrainingCheckpointManifest,
    prepared: &gw_schema::PreparedSftPayload,
    config: &CheckpointModelConfig,
) -> Result<()> {
    let recipe = &manifest.recipe;
    let source = &prepared.manifest.recipe;
    let examples = &prepared.examples;
    if recipe.preparation_source_sha256 != source.adapter_source_sha256
        || recipe.dependencies != source.dependencies
        || examples.is_empty()
        || config.vocab_size
            < source.tokenizer_policy["vocab_size"]
                .as_u64()
                .unwrap_or(u64::MAX)
        || config.eos_token_id
            != source.tokenizer_policy["wrapper"]["eos_token_id"]
                .as_u64()
                .unwrap_or(u64::MAX)
        || config.pad_token_id.is_some_and(|id| {
            Some(id) != source.tokenizer_policy["wrapper"]["pad_token_id"].as_u64()
        })
        || examples.iter().any(|e| {
            e.input_ids.len() > recipe.max_sequence_length as usize
                || e.input_ids.len() as u64 > config.max_position_embeddings
                || e.input_ids
                    .iter()
                    .any(|id| u64::from(*id) >= config.vocab_size)
        })
    {
        return Err(integrity(
            "checkpoint recipe, tokenizer or complete examples mismatch",
        ));
    }
    let mut observed = manifest.observations.microbatches.iter();
    let batches: Vec<_> = examples.chunks(recipe.batch_size as usize).collect();
    let mut update = 0;
    while update < recipe.max_steps {
        for group in batches.chunks(recipe.accumulation as usize) {
            update += 1;
            for batch in group {
                let expected = TrainingMicrobatch {
                    update,
                    example_ids: batch.iter().map(|e| e.example_id.clone()).collect(),
                    input_tokens: batch.iter().map(|e| e.input_ids.len() as u64).sum(),
                    shifted_supervised_tokens: batch
                        .iter()
                        .map(|e| {
                            e.labels
                                .iter()
                                .skip(1)
                                .filter(|label| **label != -100)
                                .count() as u64
                        })
                        .sum(),
                };
                if observed.next() != Some(&expected) {
                    return Err(integrity(
                        "declared microbatches differ from the complete sequential sampler",
                    ));
                }
            }
            if update == recipe.max_steps {
                break;
            }
        }
    }
    if observed.next().is_some() {
        return Err(integrity("excess checkpoint microbatches"));
    }
    Ok(())
}
