//! Complete native Gemma LoRA inspection. Historical execution remains declared.
use crate::{Result, artifact::integrity, lora_tensors, verify_prepared_sft_snapshot};
use gw_schema::{
    CheckpointModelSummary, GEMMA_RELEASE_BYTES, GEMMA_RELEASE_SHA256, GemmaBaseKind,
    LORA_HASH_DOMAIN, LORA_MAGIC, LoraCheckpointManifest, LoraTrainable,
    MAX_CHECKPOINT_MANIFEST_BYTES,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;

/// Measured captured-byte receipt, never an observed training or candidate capability.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraCheckpointReport {
    /// Native report version.
    pub report_version: u32,
    /// Independently measured complete artifact identity.
    pub completion_id: String,
    /// Original prepared input identity.
    pub prepared_build_id: String,
    /// Exact complete captured bytes.
    pub byte_length: u64,
    /// Measured complete base, including buffers and tied aliases.
    pub base_model: CheckpointModelSummary,
    /// Measured initial adapter state.
    pub initial_adapter: CheckpointModelSummary,
    /// Measured final adapter state.
    pub final_adapter: CheckpointModelSummary,
    /// Passed only after byte, tensor, recipe and accounting checks.
    pub structural_validation: String,
    /// Always declared: saved observations do not establish actual historical execution.
    pub historical_training: String,
    /// Native inspection does not run a model.
    pub model_reload: String,
    /// Native inspection does not execute a tokenizer.
    pub tokenizer_replay: String,
    /// Original strict saved declarations, distinct from measured receipt fields.
    pub declarations: LoraCheckpointManifest,
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

/// Stream the separate complete LoRA frame without extracting paths or allocating full weights.
///
/// # Errors
/// Rejects unsafe/corrupt tensor populations, altered configurations, forged accounting or bindings.
pub fn verify_lora_checkpoint(mut reader: impl Read) -> Result<LoraCheckpointReport> {
    let mut header = [0; 48];
    reader.read_exact(&mut header)?;
    if &header[..8] != LORA_MAGIC {
        return Err(integrity("unsupported LoRA magic"));
    }
    let size = u64::from_be_bytes(
        header[40..]
            .try_into()
            .map_err(|_| integrity("invalid LoRA frame"))?,
    );
    if size == 0 || size > MAX_CHECKPOINT_MANIFEST_BYTES as u64 {
        return Err(integrity("LoRA manifest exceeds bound"));
    }
    let mut raw = vec![0; size as usize];
    reader.read_exact(&mut raw)?;
    let manifest = LoraCheckpointManifest::from_json(&raw).map_err(integrity)?;
    let mut body = blake3::Hasher::new_derive_key(LORA_HASH_DOMAIN);
    body.update(&header[40..]);
    body.update(&raw);
    let mut kind = None;
    let mut models = Vec::new();
    let mut contents = Vec::new();
    let mut prepared = None;
    for file in &manifest.files {
        let mut stream = Hashed {
            reader: &mut reader,
            body: &mut body,
            file: blake3::Hasher::new(),
            sha: Sha256::new(),
            remaining: file.byte_length,
        };
        if file.path == "base/config.json" {
            let mut bytes = Vec::new();
            stream.read_to_end(&mut bytes)?;
            let k = GemmaBaseKind::from_config(&bytes).map_err(integrity)?;
            if k.authorization() != manifest.source_authorization {
                return Err(integrity(
                    "Gemma source authorization differs from configuration",
                ));
            }
            kind = Some(k);
        } else if file.path.ends_with("config.json") {
            let mut bytes = Vec::new();
            stream.read_to_end(&mut bytes)?;
            let config = gw_schema::strict_coding_json(&bytes).map_err(integrity)?;
            if config
                != kind
                    .ok_or_else(|| integrity("missing base configuration"))?
                    .adapter_config()
            {
                return Err(integrity("unsupported Gemma adapter configuration"));
            }
        } else if file.path.ends_with(".safetensors") {
            let k = kind.ok_or_else(|| integrity("missing base configuration"))?;
            let (summary, content) =
                lora_tensors::measure(&mut stream, file.byte_length, k, models.is_empty())?;
            models.push(summary);
            contents.push(content);
        } else {
            let mut bytes = Vec::new();
            stream.read_to_end(&mut bytes)?;
            prepared = Some(verify_prepared_sft_snapshot(bytes)?);
        }
        if stream.remaining != 0 || stream.file.finalize().to_hex().as_str() != file.blake3 {
            return Err(integrity("LoRA file length or digest mismatch"));
        }
        if kind == Some(GemmaBaseKind::Release)
            && file.path == "base/model.safetensors"
            && (file.byte_length != GEMMA_RELEASE_BYTES
                || format!("{:x}", stream.sha.finalize()) != GEMMA_RELEASE_SHA256)
        {
            return Err(integrity(
                "Gemma release weights differ from approved bytes",
            ));
        }
    }
    let digest = body.finalize();
    if digest.as_bytes() != &header[8..40] || reader.read(&mut [0])? != 0 {
        return Err(integrity(
            "LoRA completion digest or trailing bytes mismatch",
        ));
    }
    if models.len() != 3
        || models[0] != manifest.base_model
        || models[1] != manifest.final_adapter
        || models[2] != manifest.initial_adapter
    {
        return Err(integrity("measured LoRA tensors differ from declarations"));
    }
    if contents[1]
        .iter()
        .any(|(name, value)| contents[2].get(name) == Some(value))
    {
        return Err(integrity("an adapter tensor did not change"));
    }
    let prepared = prepared.ok_or_else(|| integrity("missing LoRA prepared input"))?;
    if prepared.report().build_id != manifest.prepared_build_id {
        return Err(integrity("LoRA prepared binding mismatch"));
    }
    validate_inputs(
        &manifest,
        prepared.payload(),
        kind.ok_or_else(|| integrity("missing base configuration"))?,
    )?;
    Ok(LoraCheckpointReport {
        report_version: 1,
        completion_id: digest.to_hex().to_string(),
        prepared_build_id: manifest.prepared_build_id.clone(),
        byte_length: 48 + size + manifest.files.iter().map(|f| f.byte_length).sum::<u64>(),
        base_model: models[0].clone(),
        final_adapter: models[1].clone(),
        initial_adapter: models[2].clone(),
        structural_validation: "passed".into(),
        historical_training: "declared".into(),
        model_reload: "not_run".into(),
        tokenizer_replay: "not_run".into(),
        declarations: manifest,
    })
}

fn validate_inputs(
    m: &LoraCheckpointManifest,
    p: &gw_schema::PreparedSftPayload,
    k: GemmaBaseKind,
) -> Result<()> {
    let r = &m.recipe;
    let source = &p.manifest.recipe;
    let pins: std::collections::BTreeMap<String, String> =
        serde_json::from_str(include_str!("lora_dependencies.json"))?;
    if r.preparation_source_sha256 != source.adapter_source_sha256
        || r.dependencies != pins
        || source
            .dependencies
            .iter()
            .any(|(key, value)| r.dependencies.get(key) != Some(value))
        || source.preparation_profile.as_ref().is_none_or(|profile| {
            profile.name != gw_schema::PreparedSftProfileName::Gemma4E2bTextV1
        })
        || p.examples.is_empty()
        || p.examples.iter().any(|e| {
            e.input_ids.len() > r.max_sequence_length as usize
                || e.input_ids.iter().any(|id| *id >= 262_144)
        })
    {
        return Err(integrity(
            "Gemma LoRA preparation, source or dependency mismatch",
        ));
    }
    let expected: Vec<_> = k
        .adapter_shapes()
        .into_iter()
        .map(|(name, shape)| LoraTrainable {
            name: name.replace(".weight", ".default.weight"),
            parameters: shape.iter().product(),
            shape,
            dtype: "float32".into(),
            device: "cpu".into(),
        })
        .collect();
    if m.observations.trainables != expected {
        return Err(integrity(
            "LoRA trainable population differs from exact q/v recipe",
        ));
    }
    let mut observed = m.observations.microbatches.iter();
    let batches: Vec<_> = p.examples.chunks(r.batch_size as usize).collect();
    let mut update = 0;
    while update < r.max_steps {
        for group in batches.chunks(r.accumulation as usize) {
            update += 1;
            for batch in group {
                let next = observed
                    .next()
                    .ok_or_else(|| integrity("missing LoRA microbatch"))?;
                let ids: Vec<_> = batch.iter().map(|e| e.example_id.clone()).collect();
                let inputs: u64 = batch.iter().map(|e| e.input_ids.len() as u64).sum();
                let labels: u64 = batch
                    .iter()
                    .map(|e| e.labels.iter().skip(1).filter(|x| **x != -100).count() as u64)
                    .sum();
                if next.update != update
                    || next.example_ids != ids
                    || next.input_tokens != inputs
                    || next.shifted_supervised_tokens != labels
                {
                    return Err(integrity(
                        "LoRA microbatch order, inputs or shifted labels mismatch",
                    ));
                }
            }
            if update == r.max_steps {
                break;
            }
        }
    }
    if observed.next().is_some() {
        return Err(integrity("excess LoRA microbatches"));
    }
    Ok(())
}
