//! Framing and strict integer-only JSON for self-contained prepared SFT input builds.
use serde::de::{Error, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// Version-one framing magic. The following fields are `digest[32]`, two big-endian u64 lengths,
/// exact UTF-8 JSON payload bytes, and original Parquet bytes, with no trailing data.
pub const PREPARED_SFT_MAGIC: &[u8; 8] = b"GWSFT001";
/// Derive-key domain for the complete length-framed payload and captured source.
pub const PREPARED_SFT_HASH_DOMAIN: &str = "ghostwriter.prepared-sft-input.v1";
/// Maximum complete captured build size, including its embedded source (256 MiB).
pub const MAX_PREPARED_SFT_BYTES: usize = 256 * 1024 * 1024;
const HEADER: usize = 56;

/// Borrowed, integrity-checked framing. Payload semantics and source verification are separate.
pub struct PreparedSftFrame<'a> {
    /// Complete immutable input identity, independent of later trainer/optimization evidence.
    pub build_id: String,
    /// Exact payload bytes committed by that identity.
    pub payload: &'a [u8],
    /// Exact original source bytes, never a path or an asserted verification report.
    pub source: &'a [u8],
}

/// Encode framing only. Consumers must still validate the payload and embedded source.
///
/// # Errors
/// Rejects overflow or a build exceeding the input limit.
pub fn encode_prepared_sft_frame(payload: &[u8], source: &[u8]) -> Result<Vec<u8>, &'static str> {
    let length = HEADER
        .checked_add(payload.len())
        .and_then(|n| n.checked_add(source.len()))
        .filter(|n| *n <= MAX_PREPARED_SFT_BYTES)
        .ok_or("prepared SFT build exceeds size limit")?;
    let mut bytes = Vec::with_capacity(length);
    bytes.extend_from_slice(PREPARED_SFT_MAGIC);
    bytes.extend_from_slice(&[0; 32]);
    bytes.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    bytes.extend_from_slice(&(source.len() as u64).to_be_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(source);
    let mut hash = blake3::Hasher::new_derive_key(PREPARED_SFT_HASH_DOMAIN);
    hash.update(&bytes[40..]);
    bytes[8..40].copy_from_slice(hash.finalize().as_bytes());
    Ok(bytes)
}

/// Verify exact framing, lengths, absence of trailing bytes, and the complete declared digest.
///
/// # Errors
/// Rejects unsupported versions, malformed lengths, truncated/extra bytes, or digest mismatch.
pub fn decode_prepared_sft_frame(bytes: &[u8]) -> Result<PreparedSftFrame<'_>, &'static str> {
    if bytes.len() < HEADER
        || bytes.len() > MAX_PREPARED_SFT_BYTES
        || &bytes[..8] != PREPARED_SFT_MAGIC
    {
        return Err("unsupported or malformed prepared SFT framing");
    }
    let length = |range: std::ops::Range<usize>| -> Result<usize, &'static str> {
        let encoded: [u8; 8] = bytes[range]
            .try_into()
            .map_err(|_| "invalid prepared SFT length")?;
        usize::try_from(u64::from_be_bytes(encoded)).map_err(|_| "invalid prepared SFT length")
    };
    let payload_len = length(40..48)?;
    let source_len = length(48..56)?;
    let boundary = HEADER
        .checked_add(payload_len)
        .ok_or("prepared SFT length overflow")?;
    if boundary.checked_add(source_len) != Some(bytes.len()) {
        return Err("prepared SFT length mismatch or trailing bytes");
    }
    let mut hash = blake3::Hasher::new_derive_key(PREPARED_SFT_HASH_DOMAIN);
    hash.update(&bytes[40..]);
    let digest = hash.finalize();
    if bytes[8..40] != *digest.as_bytes() {
        return Err("prepared SFT complete payload/source digest mismatch");
    }
    Ok(PreparedSftFrame {
        build_id: digest.to_hex().to_string(),
        payload: &bytes[HEADER..boundary],
        source: &bytes[boundary..],
    })
}

// Parse every object before conversion to Value, retaining duplicate-field evidence even inside
// opaque tokenizer recipe metadata. This wire has only integers; decimal task/oracle data remains
// in original source strings/Parquet rather than undergoing another floating-point conversion.
struct StrictValue(Value);
impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StrictVisitor;
        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = StrictValue;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("strict JSON with unique object keys and integer numbers")
            }
            fn visit_bool<E: Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(StrictValue(value.into()))
            }
            fn visit_i64<E: Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(StrictValue(value.into()))
            }
            fn visit_u64<E: Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(StrictValue(value.into()))
            }
            fn visit_f64<E: Error>(self, _: f64) -> Result<Self::Value, E> {
                Err(E::custom("prepared SFT JSON numbers must be integers"))
            }
            fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(StrictValue(value.into()))
            }
            fn visit_string<E: Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(StrictValue(value.into()))
            }
            fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(StrictValue(value)) = seq.next_element()? {
                    values.push(value);
                }
                Ok(StrictValue(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(A::Error::custom("duplicate prepared SFT JSON field"));
                    }
                    let StrictValue(value) = map.next_value()?;
                    values.insert(key, value);
                }
                Ok(StrictValue(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(StrictVisitor)
    }
}

pub(crate) fn strict_prepared_json(bytes: &[u8]) -> Result<Value, String> {
    serde_json::from_slice::<StrictValue>(bytes)
        .map(|value| value.0)
        .map_err(|error| error.to_string())
}
