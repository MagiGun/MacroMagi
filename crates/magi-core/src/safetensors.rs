//! Safetensors: the weights file the trainer writes and the native engine
//! reads, and half-precision conversions.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::error::{Error, Result};

/// A tensor read from a safetensors file: its shape and values.
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Reads the float tensors of a safetensors file (F32, F16 or BF16).
pub fn read_safetensors(bytes: &[u8]) -> Result<BTreeMap<String, Tensor>> {
    if bytes.len() < 8 {
        return Err(Error::format("the weights file is too short"));
    }
    let size = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let header: Value =
        serde_json::from_slice(bytes.get(8..8 + size).ok_or_else(|| Error::format("the weights header is cut off"))?)?;
    let body = &bytes[8 + size..];
    let mut out = BTreeMap::new();
    for (name, info) in header.as_object().into_iter().flatten() {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info["dtype"].as_str().unwrap_or_default();
        let shape: Vec<usize> =
            info["shape"].as_array().into_iter().flatten().filter_map(Value::as_u64).map(|d| d as usize).collect();
        let offsets: Vec<usize> = info["data_offsets"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .map(|d| d as usize)
            .collect();
        let raw =
            body.get(offsets[0]..offsets[1]).ok_or_else(|| Error::format(format!("the tensor {name} is cut off")))?;
        let data: Vec<f32> = match dtype {
            "F32" => raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            "F16" => raw.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect(),
            "BF16" => {
                raw.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)).collect()
            }
            _ => continue,
        };
        out.insert(name.clone(), Tensor { shape, data });
    }
    Ok(out)
}

/// Writes tensors as a safetensors file (F32).
pub fn write_safetensors(tensors: &[(String, &Tensor)], metadata: Option<Value>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if let Some(meta) = metadata {
        header.insert("__metadata__".into(), meta);
    }
    let mut offset = 0usize;
    for (name, tensor) in tensors {
        let size = tensor.data.len() * 4;
        header.insert(
            name.clone(),
            json!({ "dtype": "F32", "shape": tensor.shape, "data_offsets": [offset, offset + size] }),
        );
        offset += size;
    }
    let mut blob = serde_json::to_vec(&Value::Object(header)).unwrap_or_default();
    while !blob.len().is_multiple_of(8) {
        blob.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + blob.len() + offset);
    out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    out.extend_from_slice(&blob);
    for (_, tensor) in tensors {
        for value in &tensor.data {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normalize it.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// f32 to IEEE half precision, rounding to nearest even.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mut mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        mant |= 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = mant + half - 1 + ((mant >> shift) & 1);
        return sign | (rounded >> shift) as u16;
    }
    let rounded = mant + 0xfff + ((mant >> 13) & 1);
    if rounded & 0x80_0000 != 0 {
        let e = e + 1;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        return sign | ((e as u16) << 10);
    }
    sign | ((e as u16) << 10) | (rounded >> 13) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_round_trips() {
        for v in [0.0f32, 1.0, -2.5, 0.000_06, 65504.0, 3.25, -0.001] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
    }

    #[test]
    fn safetensors_round_trip() {
        let tensor = Tensor { shape: vec![2, 2], data: vec![1.0, -2.0, 0.5, 4.0] };
        let bytes = write_safetensors(&[("w".into(), &tensor)], Some(json!({ "a": "b" })));
        let back = read_safetensors(&bytes).unwrap();
        assert_eq!(back["w"].shape, [2, 2]);
        assert_eq!(back["w"].data, tensor.data);
    }
}
