use super::*;
use serde_json::json;

fn adapter(mutation: &str) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (index, (name, shape)) in GemmaBaseKind::Fixture
        .adapter_shapes()
        .into_iter()
        .enumerate()
    {
        let count = shape.iter().product::<u64>() as usize;
        let start = data.len();
        for element in 0..count {
            let bits = if index == 0 && element == 0 {
                match mutation {
                    "nan" => 0x7fc0_0000,
                    "inf" => 0x7f80_0000,
                    "negative_inf" => 0xff80_0000,
                    _ => 0x3f80_0000,
                }
            } else {
                0x3f80_0000_u32
            };
            data.extend_from_slice(&bits.to_le_bytes());
        }
        header.insert(
            name,
            json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
        );
    }
    let first = header.keys().next().unwrap().clone();
    match mutation {
        "missing" => {
            header.remove(&first);
        }
        "unknown" => {
            let value = header.remove(&first).unwrap();
            header.insert("unexpected.input_min".into(), value);
        }
        "shape" => {
            header.get_mut(&first).unwrap()["shape"] = json!([1]);
        }
        "gap" => {
            header.get_mut(&first).unwrap()["data_offsets"] = json!([4, 516]);
        }
        "metadata" => {
            header.insert(
                "__metadata__".into(),
                json!({"base_model_name_or_path":"untrusted"}),
            );
        }
        _ => {}
    }
    let raw = serde_json::to_vec(&header).unwrap();
    let mut output = (raw.len() as u64).to_le_bytes().to_vec();
    output.extend(raw);
    output.extend(data);
    output
}
#[test]
fn measured_adapter_is_complete_and_every_value_is_finite() {
    let bytes = adapter("");
    let (summary, _) = measure(
        &mut bytes.as_slice(),
        bytes.len() as u64,
        GemmaBaseKind::Fixture,
        false,
    )
    .unwrap();
    assert_eq!(summary.parameter_count, 1728);
    assert_eq!(summary.tensor_count, 12);
    for corruption in [
        "nan",
        "inf",
        "negative_inf",
        "missing",
        "unknown",
        "shape",
        "gap",
        "metadata",
    ] {
        let bytes = adapter(corruption);
        assert!(
            measure(
                &mut bytes.as_slice(),
                bytes.len() as u64,
                GemmaBaseKind::Fixture,
                false
            )
            .is_err(),
            "{corruption}"
        );
    }
}
#[test]
fn tensor_framing_rejects_truncation_and_excess_allocation_claims() {
    let bytes = adapter("");
    for removed in [1, 4, 100] {
        let short = &bytes[..bytes.len() - removed];
        assert!(
            measure(
                &mut &short[..],
                bytes.len() as u64,
                GemmaBaseKind::Fixture,
                false
            )
            .is_err()
        );
    }
    let raw = u64::MAX.to_le_bytes();
    assert!(measure(&mut raw.as_slice(), 100, GemmaBaseKind::Fixture, false).is_err());
}

#[test]
fn cuda_bf16_rounding_preserves_signed_zero_subnormals_and_even_ties() {
    for (input, expected) in [
        (0x0000_0000, 0x0000),
        (0x8000_0000, 0x8000),
        (0x0000_0001, 0x0000),
        (0x8000_0001, 0x8000),
        (0x0000_8000, 0x0000),
        (0x0000_8001, 0x0001),
        (0x0001_8000, 0x0002),
        (0x8001_8000, 0x8002),
        (0x3f80_7fff, 0x3f80),
        (0x3f80_8000, 0x3f80),
        (0x3f80_8001, 0x3f81),
        (0x3f81_8000, 0x3f82),
        (0xbf80_8000, 0xbf80),
        (0xbf81_8000, 0xbf82),
        (0x7f7f_0000, 0x7f7f),
        (0xff7f_0000, 0xff7f),
    ] {
        assert_eq!(bf16_bits(input).unwrap(), expected, "{input:08x}");
    }
}

#[test]
fn cuda_bf16_rejects_nonfinite_source_and_finite_conversion_overflow() {
    for bits in [
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7f80_0001,
        0x7f7f_8000,
        0xff7f_8000,
        0x7f7f_ffff,
        0xff7f_ffff,
    ] {
        assert!(bf16_bits(bits).is_err(), "{bits:08x}");
    }
}
