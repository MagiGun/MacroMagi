//! MacroMagi's own engine for its text models made of Transformer Layers
//! (the Llama, Qwen3, Gemma 3, GLM and Magi layouts).
//!
//! It runs the weights straight from the trainer's safetensors file and keeps
//! a key/value cache: every token it reads leaves its keys and values behind,
//! so each new token is one small step instead of the whole context again (the
//! ONNX path's way). Between requests the cache is kept, and a request that
//! starts the same way – the system text and tools – only reads what is new.
//!
//! Long contexts:
//! - positions past the trained context are stretched with YaRN or NTK
//!   ([`RopeScaling`]), dynamically: within the trained context they are
//!   exactly as the model learned them, and the cache keeps keys unturned so
//!   the stretch can grow with the text;
//! - the cache is kept in blocks of 256 tokens in three tiers
//!   ([`MemorySpec`]): *hot* (the first block, with its attention sinks, and
//!   the latest ones, at full precision), *warm* (older blocks in system
//!   memory, as f16 or 8-bit), and *cold* (blocks written to a file on disk
//!   when the warm tier is full, read back when needed);
//! - with `read_blocks`, each token reads only the first block, the latest
//!   ones and those whose mean key matches its query best;
//! - with `persist_prefix`, the memory of the system text is saved to disk
//!   under the SHA-256 of the model and its tokens, and loaded instead of
//!   being read again.
//!
//! The math is the trainer's (`runtime/python/magi_train/blocks.py`):
//! RMSNorm, rotary positions in the halves layout, grouped-query attention,
//! SwiGLU or GeGLU, optional QK-norm, biases, partial rotation, sliding
//! windows, sandwich norms, tied and scaled embeddings.

use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use magi_core::export::{Tensor, f16_to_f32, f32_to_f16, read_safetensors};
use magi_core::magic::{Architecture, MemorySpec, RopeScaling};
use rayon::prelude::*;
use serde_json::Value;

use crate::{Generation, MlError};

/// Tokens per cache block.
pub const BLOCK: usize = 256;
const EPS: f32 = 1e-5;

struct Layer {
    input_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    attn_out_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    mlp_out_norm: Option<Vec<f32>>,
    hidden: usize,
    window: usize,
    turned: usize,
    geglu: bool,
}

/// What the cache holds and has done, for the Memory & Cache block.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KvTelemetry {
    pub tokens: usize,
    pub hot_bytes: u64,
    pub hot_capacity: u64,
    pub warm_bytes: u64,
    pub warm_capacity: u64,
    pub cold_bytes: u64,
    pub cold_capacity: u64,
    /// Tokens of requests found in the cache (kept, or loaded from disk), and all of them.
    pub reused: u64,
    pub read: u64,
    /// The last request's prefix came from a file on disk.
    pub from_disk: bool,
}

impl KvTelemetry {
    pub fn hit_rate(&self) -> f64 {
        if self.read == 0 { 0.0 } else { self.reused as f64 / self.read as f64 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Precision {
    F32,
    F16,
    Q8,
}

enum Store {
    Hot(Vec<f32>),
    Warm(Vec<u8>),
    Cold { offset: u64, len: usize },
}

/// One block of the cache: per layer, the keys then the values of up to
/// `BLOCK` positions, each `kv_dim` numbers.
struct Block {
    store: Store,
    len: usize,
    /// Per layer, the mean key: what the block is about, to choose blocks by.
    landmark: Vec<f32>,
}

struct Cache {
    layers: usize,
    kv_dim: usize,
    blocks: Vec<Block>,
    precision: Precision,
    hot_blocks: usize,
    warm_budget: u64,
    disk_budget: u64,
    disk_dir: PathBuf,
    file: Option<(std::fs::File, PathBuf, u64)>,
    /// Recently read cold blocks, decoded: (block index, data).
    recent: VecDeque<(usize, Vec<f32>)>,
}

impl Cache {
    fn block_floats(&self) -> usize {
        self.layers * 2 * BLOCK * self.kv_dim
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.blocks.iter().map(|b| b.len).sum()
    }

    fn bytes(&self) -> (u64, u64, u64) {
        let (mut hot, mut warm, mut cold) = (0, 0, 0);
        for block in &self.blocks {
            match &block.store {
                Store::Hot(data) => hot += data.len() as u64 * 4,
                Store::Warm(data) => warm += data.len() as u64,
                Store::Cold { len, .. } => cold += *len as u64,
            }
        }
        (hot, warm, cold)
    }

    fn clear(&mut self) {
        self.blocks.clear();
        self.recent.clear();
        if let Some((_, path, _)) = self.file.take() {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Forgets everything from `position` on (to a block's start when the
    /// block is not hot).
    fn truncate(&mut self, position: usize) -> usize {
        let mut kept = 0;
        let mut keep_blocks = 0;
        for block in &mut self.blocks {
            if kept + block.len <= position {
                kept += block.len;
                keep_blocks += 1;
                continue;
            }
            if let Store::Hot(_) = block.store
                && position > kept
            {
                block.len = position - kept;
                kept = position;
                keep_blocks += 1;
            }
            break;
        }
        self.blocks.truncate(keep_blocks);
        self.recent.retain(|(index, _)| *index < keep_blocks);
        kept
    }

    fn offset(&self, layer: usize, value: bool, slot: usize) -> usize {
        ((layer * 2 + usize::from(value)) * BLOCK + slot) * self.kv_dim
    }

    /// Appends one position's keys and values (per layer, `kv_dim` each).
    fn push(&mut self, keys: &[Vec<f32>], values: &[Vec<f32>]) -> Result<(), MlError> {
        if self.blocks.last().is_none_or(|b| b.len == BLOCK) {
            self.demote()?;
            let floats = self.block_floats();
            self.blocks.push(Block { store: Store::Hot(vec![0.0; floats]), len: 0, landmark: Vec::new() });
        }
        let kv_dim = self.kv_dim;
        let (offsets, slot): (Vec<(usize, usize)>, usize) = {
            let slot = self.blocks.last().expect("a block").len;
            ((0..self.layers).map(|l| (self.offset(l, false, slot), self.offset(l, true, slot))).collect(), slot)
        };
        let block = self.blocks.last_mut().expect("a block");
        let Store::Hot(data) = &mut block.store else { return Err(MlError::new("the newest cache block is not hot")) };
        for (layer, (k_at, v_at)) in offsets.into_iter().enumerate() {
            data[k_at..k_at + kv_dim].copy_from_slice(&keys[layer]);
            data[v_at..v_at + kv_dim].copy_from_slice(&values[layer]);
        }
        block.len = slot + 1;
        Ok(())
    }

    /// Moves full blocks past the hot window to the warm tier, and warm ones
    /// past the warm budget to the cold one.
    fn demote(&mut self) -> Result<(), MlError> {
        let count = self.blocks.len();
        let (kv_dim, layers, precision) = (self.kv_dim, self.layers, self.precision);
        for index in 1..count.saturating_sub(self.hot_blocks.saturating_sub(1)) {
            let block = &mut self.blocks[index];
            if let Store::Hot(data) = &block.store {
                block.landmark = landmark(data, layers, kv_dim, block.len);
                block.store = Store::Warm(pack(data, precision, kv_dim));
            }
        }
        let (_, mut warm, _) = self.bytes();
        let mut index = 1;
        while warm > self.warm_budget && index < self.blocks.len() {
            if let Store::Warm(bytes) = &self.blocks[index].store {
                let bytes = bytes.clone();
                let len = bytes.len();
                let offset = self.write_cold(&bytes)?;
                self.blocks[index].store = Store::Cold { offset, len };
                warm -= len as u64;
            }
            index += 1;
        }
        Ok(())
    }

    fn write_cold(&mut self, bytes: &[u8]) -> Result<u64, MlError> {
        if self.file.is_none() {
            std::fs::create_dir_all(&self.disk_dir)
                .map_err(|e| MlError::new(format!("the cache folder could not be made: {e}")))?;
            let path = self.disk_dir.join(format!("kv-{}-{}.bin", std::process::id(), unique()));
            let file = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&path)
                .map_err(|e| MlError::new(format!("the cache file could not be made: {e}")))?;
            self.file = Some((file, path, 0));
        }
        let (file, _, end) = self.file.as_mut().expect("opened above");
        if *end + bytes.len() as u64 > self.disk_budget {
            return Err(MlError::new(
                "the text is longer than the memory allows: raise the disk tier in the Memory & Cache block",
            ));
        }
        let offset = *end;
        file.seek(SeekFrom::Start(offset)).map_err(|e| MlError::new(e.to_string()))?;
        file.write_all(bytes).map_err(|e| MlError::new(e.to_string()))?;
        *end += bytes.len() as u64;
        Ok(offset)
    }

    /// One block's keys and values for every layer, as numbers.
    fn floats(&mut self, index: usize) -> Result<std::borrow::Cow<'_, [f32]>, MlError> {
        let floats = self.block_floats();
        let (kv_dim, precision) = (self.kv_dim, self.precision);
        match &self.blocks[index].store {
            Store::Hot(data) => Ok(std::borrow::Cow::Borrowed(&data[..])),
            Store::Warm(bytes) => Ok(std::borrow::Cow::Owned(unpack(bytes, precision, kv_dim, floats))),
            Store::Cold { offset, len } => {
                if let Some(position) = self.recent.iter().position(|(i, _)| *i == index) {
                    let found = self.recent.remove(position).expect("found");
                    self.recent.push_front(found);
                    return Ok(std::borrow::Cow::Borrowed(&self.recent[0].1));
                }
                let (offset, len) = (*offset, *len);
                let (file, ..) = self.file.as_mut().ok_or_else(|| MlError::new("the cache file is gone"))?;
                let mut bytes = vec![0u8; len];
                file.seek(SeekFrom::Start(offset)).map_err(|e| MlError::new(e.to_string()))?;
                file.read_exact(&mut bytes).map_err(|e| MlError::new(e.to_string()))?;
                self.recent.push_front((index, unpack(&bytes, precision, kv_dim, floats)));
                self.recent.truncate(8);
                Ok(std::borrow::Cow::Borrowed(&self.recent[0].1))
            }
        }
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        self.clear();
    }
}

fn unique() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Per layer, the mean of a block's keys.
fn landmark(data: &[f32], layers: usize, kv_dim: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0.0; layers * kv_dim];
    for layer in 0..layers {
        let base = layer * 2 * BLOCK * kv_dim;
        for slot in 0..len {
            let key = &data[base + slot * kv_dim..base + (slot + 1) * kv_dim];
            for (o, k) in out[layer * kv_dim..(layer + 1) * kv_dim].iter_mut().zip(key) {
                *o += k / len.max(1) as f32;
            }
        }
    }
    out
}

fn pack(data: &[f32], precision: Precision, kv_dim: usize) -> Vec<u8> {
    match precision {
        Precision::F32 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Precision::F16 => data.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect(),
        Precision::Q8 => {
            let mut out = Vec::with_capacity(data.len() + data.len() / kv_dim * 4);
            for vector in data.chunks(kv_dim) {
                let amax = vector.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale = amax / 127.0;
                let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                out.extend_from_slice(&scale.to_le_bytes());
                out.extend(vector.iter().map(|v| (v * inverse).round() as i8 as u8));
            }
            out
        }
    }
}

fn unpack(bytes: &[u8], precision: Precision, kv_dim: usize, floats: usize) -> Vec<f32> {
    match precision {
        Precision::F32 => bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        Precision::F16 => bytes.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect(),
        Precision::Q8 => {
            let mut out = Vec::with_capacity(floats);
            for chunk in bytes.chunks(4 + kv_dim) {
                let scale = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.extend(chunk[4..].iter().map(|q| f32::from(*q as i8) * scale));
            }
            out
        }
    }
}

/// The rotary frequencies of `turned` channels, and YaRN's attention factor.
fn frequencies(turned: usize, theta: f64, scaling: Option<&RopeScaling>) -> (Vec<f32>, f32) {
    let half = turned / 2;
    let mut base = theta;
    if let Some(s) = scaling.filter(|s| s.kind == "ntk" && s.factor > 1.0) {
        base *= s.factor.powf(turned as f64 / (turned as f64 - 2.0).max(1.0));
    }
    let plain: Vec<f64> = (0..half).map(|i| 1.0 / base.powf(2.0 * i as f64 / turned as f64)).collect();
    let Some(s) = scaling.filter(|s| s.kind == "yarn" && s.factor > 1.0) else {
        return (plain.iter().map(|f| *f as f32).collect(), 1.0);
    };
    // YaRN: channels that turn many times within the trained context keep their
    // frequency, those that turn less than once are stretched, those between blend.
    let dim = turned as f64;
    let correction = |rotations: f64| {
        dim * (f64::from(s.original.max(1)) / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
    };
    let low = correction(32.0).floor().max(0.0);
    let high = correction(1.0).ceil().min(dim - 1.0);
    let freqs = plain
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let ramp = ((i as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
            let keep = 1.0 - ramp;
            (f / s.factor * (1.0 - keep) + f * keep) as f32
        })
        .collect();
    (freqs, s.attention_factor() as f32)
}

fn rms_norm(x: &mut [f32], weight: &[f32]) {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean + EPS).sqrt();
    for (v, w) in x.iter_mut().zip(weight) {
        *v *= scale * w;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            lanes[i] += x[i] * y[i];
        }
    }
    lanes.iter().sum::<f32>() + a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum::<f32>()
}

/// `rows` inputs of `inputs` numbers times a `[outputs, inputs]` weight: `[rows, outputs]`.
fn linear(x: &[f32], rows: usize, weight: &[f32], outputs: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let inputs = weight.len() / outputs;
    let mut out = vec![0.0; rows * outputs];
    let work = rows * outputs * inputs;
    let row = |(r, out_row): (usize, &mut [f32])| {
        let x_row = &x[r * inputs..(r + 1) * inputs];
        for (o, y) in out_row.iter_mut().enumerate() {
            *y = dot(x_row, &weight[o * inputs..(o + 1) * inputs]) + bias.map_or(0.0, |b| b[o]);
        }
    };
    if work > 1 << 20 && rows > 1 {
        out.par_chunks_mut(outputs).enumerate().for_each(row);
    } else if work > 1 << 20 {
        // One row: split its outputs.
        out.par_chunks_mut(64).enumerate().for_each(|(chunk, ys)| {
            for (i, y) in ys.iter_mut().enumerate() {
                let o = chunk * 64 + i;
                *y = dot(x, &weight[o * inputs..(o + 1) * inputs]) + bias.map_or(0.0, |b| b[o]);
            }
        });
    } else {
        out.chunks_mut(outputs).enumerate().for_each(row);
    }
    out
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SHA-256, for naming saved prefix memories.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for chunk in message.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let choose = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(choose).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let majority = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(majority);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// A text model of Transformer Layers, run with a key/value cache.
pub struct NativeModel {
    width: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    vocab: usize,
    embed: Vec<f32>,
    head: Option<Vec<f32>>,
    norm: Vec<f32>,
    embed_scale: f32,
    layers: Vec<Layer>,
    /// Per distinct rotated width, its rotary table at the present stretch.
    rope: Vec<Table>,
    theta: f64,
    /// The Context Scaler's scaling, applied only past the trained context.
    scaling: Option<RopeScaling>,
    cache: Cache,
    /// The tokens whose memory is in the cache, in order.
    tokens: Vec<u32>,
    /// The longest text it reads (the target of its context scaling).
    pub context: usize,
    read_blocks: usize,
    persist: bool,
    fingerprint: [u8; 32],
    stats: KvTelemetry,
}

impl NativeModel {
    /// Loads a model from its architecture and safetensors weights.
    pub fn load(a: &Architecture, weights: &[u8], memory: &MemorySpec, data_dir: &Path) -> Result<Self, MlError> {
        if let Some(why) = a.native_problem() {
            return Err(MlError::new(format!("the built-in engine runs Transformer Layers only; {why}")));
        }
        let mut tensors = read_safetensors(weights).map_err(|e| MlError::new(e.to_string()))?;
        let mut take = |name: &str| -> Result<Tensor, MlError> {
            tensors.remove(name).ok_or_else(|| MlError::new(format!("the weights have no {name}")))
        };
        let (width, heads) = (a.width as usize, a.heads as usize);
        let kv_heads = (a.kv_heads as usize).max(1);
        let dim = width / heads.max(1);
        let embed = take("embed_tokens.weight")?;
        let vocab = embed.shape[0];
        let norm = take("norm.weight")?.data;
        let head = if a.tied { None } else { Some(take("lm_head.weight")?.data) };
        let mut layers = Vec::new();
        for (i, spec) in a.layers.iter().enumerate() {
            let p = |name: &str| format!("layers.{i}.{name}");
            let flag = |path: &str| spec.pointer(path).and_then(Value::as_bool).unwrap_or(false);
            let rotary = spec.pointer("/attention/rotary").and_then(Value::as_f64).unwrap_or(1.0);
            let turned = ((dim as f64 * rotary) as usize / 2 * 2).clamp(2, dim);
            let gate = take(&p("mlp.gate_proj.weight"))?;
            let bias = flag("/attention/bias");
            let qk = flag("/attention/qk_norm");
            let sandwich = spec.pointer("/norm").and_then(Value::as_str) == Some("sandwich");
            layers.push(Layer {
                input_norm: take(&p("input_layernorm.weight"))?.data,
                q: take(&p("self_attn.q_proj.weight"))?.data,
                k: take(&p("self_attn.k_proj.weight"))?.data,
                v: take(&p("self_attn.v_proj.weight"))?.data,
                o: take(&p("self_attn.o_proj.weight"))?.data,
                q_bias: if bias { Some(take(&p("self_attn.q_proj.bias"))?.data) } else { None },
                k_bias: if bias { Some(take(&p("self_attn.k_proj.bias"))?.data) } else { None },
                v_bias: if bias { Some(take(&p("self_attn.v_proj.bias"))?.data) } else { None },
                q_norm: if qk { Some(take(&p("self_attn.q_norm.weight"))?.data) } else { None },
                k_norm: if qk { Some(take(&p("self_attn.k_norm.weight"))?.data) } else { None },
                attn_out_norm: if sandwich { Some(take(&p("attn_out_norm.weight"))?.data) } else { None },
                ffn_norm: take(&p("post_attention_layernorm.weight"))?.data,
                hidden: gate.shape[0],
                gate: gate.data,
                up: take(&p("mlp.up_proj.weight"))?.data,
                down: take(&p("mlp.down_proj.weight"))?.data,
                mlp_out_norm: if sandwich { Some(take(&p("mlp_out_norm.weight"))?.data) } else { None },
                window: spec.pointer("/attention/window").and_then(Value::as_u64).unwrap_or(0) as usize,
                turned,
                geglu: spec.pointer("/ffn/activation").and_then(Value::as_str) == Some("geglu"),
            });
        }
        let mut rope: Vec<Table> = Vec::new();
        for layer in &layers {
            if !rope.iter().any(|t| t.turned == layer.turned) {
                rope.push(Table::new(layer.turned, a.rope_theta));
            }
        }
        let precision = match memory.precision.as_str() {
            "f32" => Precision::F32,
            "q8_0" | "q4_0" => Precision::Q8,
            _ => Precision::F16,
        };
        let kv_dim = kv_heads * dim;
        let block_bytes = (layers.len() * 2 * BLOCK * kv_dim * 4) as u64;
        let disk_dir = if memory.disk_dir.trim().is_empty() {
            data_dir.join("kv-cache")
        } else {
            PathBuf::from(memory.disk_dir.trim())
        };
        let mut fingerprint_data = weights.len().to_le_bytes().to_vec();
        fingerprint_data.extend(weights.iter().step_by(4099).copied());
        fingerprint_data.extend_from_slice(serde_json::to_string(a).unwrap_or_default().as_bytes());
        let context = a.rope_scaling.as_ref().map_or(a.context, |s| s.target.max(a.context)) as usize;
        Ok(Self {
            width,
            heads,
            kv_heads,
            dim,
            vocab,
            embed: embed.data,
            head,
            norm,
            embed_scale: if a.embed_scale { (width as f32).sqrt() } else { 1.0 },
            layers,
            rope,
            theta: a.rope_theta,
            scaling: a.rope_scaling.clone().filter(|s| s.factor > 1.0 && s.kind != "none"),
            cache: Cache {
                layers: a.layers.len(),
                kv_dim,
                blocks: Vec::new(),
                precision,
                hot_blocks: ((u64::from(memory.hot_mb) << 20) / block_bytes.max(1)).max(2) as usize,
                warm_budget: u64::from(memory.warm_mb) << 20,
                disk_budget: u64::from(memory.disk_gb) << 30,
                disk_dir,
                file: None,
                recent: VecDeque::new(),
            },
            tokens: Vec::new(),
            context,
            read_blocks: memory.read_blocks as usize,
            persist: memory.persist_prefix,
            fingerprint: sha256(&fingerprint_data),
            stats: KvTelemetry::default(),
        })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn telemetry(&self) -> KvTelemetry {
        let (hot, warm, cold) = self.cache.bytes();
        let block_bytes = (self.cache.block_floats() * 4) as u64;
        KvTelemetry {
            tokens: self.tokens.len(),
            hot_bytes: hot,
            hot_capacity: self.cache.hot_blocks as u64 * block_bytes,
            warm_bytes: warm,
            warm_capacity: self.cache.warm_budget,
            cold_bytes: cold,
            cold_capacity: self.cache.disk_budget,
            ..self.stats.clone()
        }
    }

    /// Forgets the text it has read.
    pub fn reset(&mut self) {
        self.cache.clear();
        self.tokens.clear();
    }

    /// The scores for the token after `tokens`, reading only what the cache does not hold.
    pub fn scores(&mut self, tokens: &[u32], prefix: usize) -> Result<Vec<f32>, MlError> {
        if tokens.is_empty() {
            return Err(MlError::new("there is nothing to continue"));
        }
        if tokens.len() > self.context {
            return Err(MlError::new(format!(
                "the text is {} tokens, longer than the model's context window ({}): give it a Context Scaler",
                tokens.len(),
                self.context
            )));
        }
        let same = self.tokens.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        // The last token is read again, for its scores, unless it is new anyway.
        let keep = self.cache.truncate(same.min(tokens.len() - 1));
        self.tokens.truncate(keep);
        self.stats.read += tokens.len() as u64;
        self.stats.reused += keep as u64;
        self.stats.from_disk = false;
        if self.persist && prefix > keep && prefix < tokens.len() && self.load_prefix(&tokens[..prefix])? {
            self.stats.reused += (prefix - keep) as u64;
            self.stats.from_disk = true;
        }
        let start = self.tokens.len();
        let mut last = Vec::new();
        for chunk_start in (start..tokens.len()).step_by(64) {
            let chunk = &tokens[chunk_start..(chunk_start + 64).min(tokens.len())];
            last = self.forward(chunk)?;
            if self.persist && prefix > 0 && self.tokens.len() >= prefix && chunk_start < prefix {
                self.save_prefix(&tokens[..prefix])?;
            }
        }
        Ok(last)
    }

    /// Writes as `how` says, after `prompt`; `prefix` tokens of it (the system
    /// text) may be kept on disk.
    pub fn generate(&mut self, prompt: &[u32], how: &Generation, prefix: usize) -> Result<Vec<u32>, MlError> {
        let mut random = crate::Random::new(how.seed);
        let mut scores = self.scores(prompt, prefix)?;
        let mut answer = Vec::new();
        for _ in 0..how.max_tokens {
            let next = crate::pick(&scores, how.temperature, how.top_k, random.next());
            if how.stop.contains(&next) {
                break;
            }
            answer.push(next);
            if self.tokens.len() >= self.context {
                break;
            }
            scores = self.forward(&[next])?;
        }
        Ok(answer)
    }

    fn prefix_path(&self, tokens: &[u32]) -> PathBuf {
        let mut data = self.fingerprint.to_vec();
        data.extend(tokens.iter().flat_map(|t| t.to_le_bytes()));
        let hash: String = sha256(&data).iter().map(|b| format!("{b:02x}")).collect();
        self.cache.disk_dir.join(format!("prefix-{hash}.kv"))
    }

    /// Loads the memory of `tokens` saved before; false when there is none.
    fn load_prefix(&mut self, tokens: &[u32]) -> Result<bool, MlError> {
        let Ok(bytes) = std::fs::read(self.prefix_path(tokens)) else { return Ok(false) };
        let (layers, kv_dim) = (self.cache.layers, self.cache.kv_dim);
        let per_token = layers * 2 * kv_dim * 2;
        if bytes.len() != tokens.len() * per_token {
            return Ok(false);
        }
        self.reset();
        for (t, token) in tokens.iter().enumerate() {
            let at = |layer: usize, value: usize| ((t * layers + layer) * 2 + value) * kv_dim * 2;
            let read = |from: usize| -> Vec<f32> {
                bytes[from..from + kv_dim * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
                    .collect()
            };
            let keys: Vec<Vec<f32>> = (0..layers).map(|l| read(at(l, 0))).collect();
            let values: Vec<Vec<f32>> = (0..layers).map(|l| read(at(l, 1))).collect();
            self.cache.push(&keys, &values)?;
            self.tokens.push(*token);
        }
        Ok(true)
    }

    fn save_prefix(&mut self, tokens: &[u32]) -> Result<(), MlError> {
        let path = self.prefix_path(tokens);
        if path.is_file() || self.tokens.len() < tokens.len() {
            return Ok(());
        }
        let (layers, kv_dim) = (self.cache.layers, self.cache.kv_dim);
        let mut out = Vec::with_capacity(tokens.len() * layers * 2 * kv_dim * 2);
        let mut position = 0;
        for index in 0..self.cache.blocks.len() {
            let len = self.cache.blocks[index].len;
            let offsets: Vec<usize> = (0..len)
                .flat_map(|slot| (0..layers).flat_map(move |l| [(l, false, slot), (l, true, slot)]))
                .map(|(l, v, slot)| ((l * 2 + usize::from(v)) * BLOCK + slot) * kv_dim)
                .collect();
            let data = self.cache.floats(index)?;
            for at in offsets {
                if position >= tokens.len() {
                    break;
                }
                out.extend(data[at..at + kv_dim].iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()));
            }
            position += len;
            if position >= tokens.len() {
                break;
            }
        }
        out.truncate(tokens.len() * layers * 2 * kv_dim * 2);
        let _ = std::fs::create_dir_all(&self.cache.disk_dir);
        std::fs::write(&path, out).map_err(|e| MlError::new(format!("the prefix memory could not be saved: {e}")))
    }

    /// How far positions are stretched for a text of `total` tokens: not at
    /// all within the trained context (exactly as it learned), then by the
    /// next power of two that fits, up to the Context Scaler's factor –
    /// dynamic scaling, so short texts are never changed.
    fn stretch(&self, total: usize) -> f64 {
        match &self.scaling {
            Some(s) if s.kind != "none" && total > s.original as usize => {
                let needed = total as f64 / f64::from(s.original.max(1));
                needed.log2().ceil().exp2().min(s.factor.max(1.0))
            }
            _ => 1.0,
        }
    }

    /// Makes the rotary tables cover `total` positions at stretch `factor`.
    fn prepare_rope(&mut self, total: usize, factor: f64) {
        for table in &mut self.rope {
            if table.factor != factor {
                let scaling =
                    self.scaling.as_ref().filter(|_| factor > 1.0).map(|s| RopeScaling { factor, ..s.clone() });
                let (freqs, gain) = frequencies(table.turned, self.theta, scaling.as_ref());
                *table = Table { turned: table.turned, factor, freqs, gain, cos: Vec::new(), sin: Vec::new() };
            }
            let half = table.turned / 2;
            for position in table.cos.len() / half.max(1)..total {
                for f in &table.freqs {
                    let (sin, cos) = (position as f32 * f).sin_cos();
                    table.cos.push(cos * table.gain);
                    table.sin.push(sin * table.gain);
                }
            }
        }
    }

    /// Reads `tokens` after what the cache holds; the scores after the last one.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, MlError> {
        let n = tokens.len();
        let start = self.tokens.len();
        let (width, dim, heads, kv_heads) = (self.width, self.dim, self.heads, self.kv_heads);
        let kv_dim = kv_heads * dim;
        let factor = self.stretch(start + n);
        self.prepare_rope(start + n, factor);
        let mut x = Vec::with_capacity(n * width);
        for token in tokens {
            let t = (*token as usize).min(self.vocab - 1);
            x.extend(self.embed[t * width..(t + 1) * width].iter().map(|v| v * self.embed_scale));
        }
        // Layer by layer for all the new tokens; their keys (not yet turned:
        // turned when read, by the scaling of the moment) and values go into
        // the cache once every layer has made them.
        let mut layer_keys: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());
        let mut layer_values: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());
        for li in 0..self.layers.len() {
            let layer = &self.layers[li];
            let mut h = x.clone();
            for row in h.chunks_mut(width) {
                rms_norm(row, &layer.input_norm);
            }
            let mut q = linear(&h, n, &layer.q, heads * dim, layer.q_bias.as_deref());
            let mut k = linear(&h, n, &layer.k, kv_dim, layer.k_bias.as_deref());
            let v = linear(&h, n, &layer.v, kv_dim, layer.v_bias.as_deref());
            if let Some(w) = &layer.q_norm {
                q.chunks_mut(dim).for_each(|head| rms_norm(head, w));
            }
            if let Some(w) = &layer.k_norm {
                k.chunks_mut(dim).for_each(|head| rms_norm(head, w));
            }
            // What the query is about, before it is turned: blocks are chosen by it.
            let plain_q = q.clone();
            let table = self.rope.iter().find(|t| t.turned == layer.turned).expect("made at load");
            for t in 0..n {
                for head in q[t * heads * dim..(t + 1) * heads * dim].chunks_mut(dim) {
                    table.turn(head, start + t);
                }
            }
            let attended = attend(
                &mut self.cache,
                table,
                &Attention { layer: li, heads, kv_heads, dim, window: layer.window, read_blocks: self.read_blocks },
                (&q, &plain_q, &k, &v),
                n,
                start,
            )?;
            let mut out = linear(&attended, n, &layer.o, width, None);
            if let Some(w) = &layer.attn_out_norm {
                for row in out.chunks_mut(width) {
                    rms_norm(row, w);
                }
            }
            for (a, b) in x.iter_mut().zip(&out) {
                *a += b;
            }
            let mut h = x.clone();
            for row in h.chunks_mut(width) {
                rms_norm(row, &layer.ffn_norm);
            }
            let gate = linear(&h, n, &layer.gate, layer.hidden, None);
            let up = linear(&h, n, &layer.up, layer.hidden, None);
            let mixed: Vec<f32> =
                gate.iter().zip(&up).map(|(g, u)| if layer.geglu { gelu_tanh(*g) * u } else { silu(*g) * u }).collect();
            let mut down = linear(&mixed, n, &layer.down, width, None);
            if let Some(w) = &layer.mlp_out_norm {
                for row in down.chunks_mut(width) {
                    rms_norm(row, w);
                }
            }
            for (a, b) in x.iter_mut().zip(&down) {
                *a += b;
            }
            layer_keys.push(k);
            layer_values.push(v);
        }
        for t in 0..n {
            let keys: Vec<Vec<f32>> = layer_keys.iter().map(|k| k[t * kv_dim..(t + 1) * kv_dim].to_vec()).collect();
            let values: Vec<Vec<f32>> = layer_values.iter().map(|v| v[t * kv_dim..(t + 1) * kv_dim].to_vec()).collect();
            self.cache.push(&keys, &values)?;
            self.tokens.push(tokens[t]);
        }
        let mut last = x[(n - 1) * width..n * width].to_vec();
        rms_norm(&mut last, &self.norm);
        let table = self.head.as_deref().unwrap_or(&self.embed);
        Ok(linear(&last, 1, table, self.vocab, None))
    }
}

/// Rotary positions for one rotated width, at one stretch: the frequencies
/// and, per position, their cosines and sines (times YaRN's attention factor).
struct Table {
    turned: usize,
    factor: f64,
    freqs: Vec<f32>,
    gain: f32,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Table {
    fn new(turned: usize, theta: f64) -> Self {
        let (freqs, gain) = frequencies(turned, theta, None);
        Self { turned, factor: 1.0, freqs, gain, cos: Vec::new(), sin: Vec::new() }
    }

    /// Turns one head (the halves layout) to its position.
    fn turn(&self, x: &mut [f32], position: usize) {
        let half = self.turned / 2;
        let (cos, sin) =
            (&self.cos[position * half..(position + 1) * half], &self.sin[position * half..(position + 1) * half]);
        for i in 0..half {
            let (a, b) = (x[i], x[i + half]);
            x[i] = a * cos[i] - b * sin[i];
            x[i + half] = b * cos[i] + a * sin[i];
        }
    }
}

/// The shape of one layer's attention.
struct Attention {
    layer: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    window: usize,
    read_blocks: usize,
}

/// Attention of the new tokens over the cache and each other: `[n, heads·dim]`.
/// `q` is turned, `plain_q` is not; `k` (not turned) and `v` are the new tokens'.
fn attend(
    cache: &mut Cache,
    table: &Table,
    shape: &Attention,
    (q, plain_q, k, v): (&[f32], &[f32], &[f32], &[f32]),
    n: usize,
    start: usize,
) -> Result<Vec<f32>, MlError> {
    let Attention { layer: li, heads, kv_heads, dim, window, read_blocks } = *shape;
    let kv_dim = kv_heads * dim;
    let group = heads / kv_heads;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0f32; n * heads * dim];
    // Scores and running sums per (token, head), block by block (online softmax).
    let mut best = vec![f32::NEG_INFINITY; n * heads];
    let mut total = vec![0.0f32; n * heads];
    let mut add = |t: usize, head: usize, key: &[f32], value: &[f32], out: &mut [f32]| {
        let query = &q[(t * heads + head) * dim..(t * heads + head + 1) * dim];
        let s = dot(query, key) * scale;
        let i = t * heads + head;
        let o = &mut out[i * dim..(i + 1) * dim];
        if s > best[i] {
            let shrink = (best[i] - s).exp();
            total[i] *= shrink;
            for x in o.iter_mut() {
                *x *= shrink;
            }
            best[i] = s;
        }
        let w = (s - best[i]).exp();
        total[i] += w;
        for (x, y) in o.iter_mut().zip(value) {
            *x += w * y;
        }
    };
    // The cached blocks: all of them, or the first, the latest and the most relevant.
    let count = cache.blocks.len();
    let chosen: Vec<usize> = if read_blocks == 0 || count <= read_blocks {
        (0..count).collect()
    } else {
        let recent = 2.min(read_blocks);
        let mut picked: Vec<usize> = vec![0];
        picked.extend(count.saturating_sub(recent)..count);
        let mut scored: Vec<(f32, usize)> = (1..count.saturating_sub(recent))
            .filter(|i| !cache.blocks[*i].landmark.is_empty())
            .map(|i| {
                let mark = &cache.blocks[i].landmark[li * kv_dim..(li + 1) * kv_dim];
                let mut score = 0.0;
                for t in 0..n {
                    for head in 0..heads {
                        let g = head / group;
                        score += dot(
                            &plain_q[(t * heads + head) * dim..(t * heads + head + 1) * dim],
                            &mark[g * dim..(g + 1) * dim],
                        );
                    }
                }
                (score, i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        picked.extend(scored.iter().take(read_blocks.saturating_sub(picked.len())).map(|(_, i)| *i));
        picked.sort_unstable();
        picked.dedup();
        picked
    };
    let mut position = 0usize;
    let mut starts = Vec::with_capacity(count);
    for block in &cache.blocks {
        starts.push(position);
        position += block.len;
    }
    let mut turned = vec![0.0f32; BLOCK * kv_dim];
    for index in chosen {
        let (first, len) = (starts[index], cache.blocks[index].len);
        let data = cache.floats(index)?;
        let base_k = (li * 2) * BLOCK * kv_dim;
        let base_v = (li * 2 + 1) * BLOCK * kv_dim;
        // This block's keys, turned to their positions.
        turned[..len * kv_dim].copy_from_slice(&data[base_k..base_k + len * kv_dim]);
        for slot in 0..len {
            for head in turned[slot * kv_dim..(slot + 1) * kv_dim].chunks_mut(dim) {
                table.turn(head, first + slot);
            }
        }
        for slot in 0..len {
            let j = first + slot;
            for t in 0..n {
                if window > 0 && start + t - j >= window {
                    continue;
                }
                for head in 0..heads {
                    let g = head / group;
                    let key = &turned[slot * kv_dim + g * dim..slot * kv_dim + (g + 1) * dim];
                    let value = &data[base_v + slot * kv_dim + g * dim..base_v + slot * kv_dim + (g + 1) * dim];
                    add(t, head, key, value, &mut out);
                }
            }
        }
    }
    // The new tokens themselves, causally.
    let mut new_keys = k.to_vec();
    for t in 0..n {
        for head in new_keys[t * kv_dim..(t + 1) * kv_dim].chunks_mut(dim) {
            table.turn(head, start + t);
        }
    }
    for t in 0..n {
        for u in 0..=t {
            if window > 0 && t - u >= window {
                continue;
            }
            for head in 0..heads {
                let g = head / group;
                let key = &new_keys[(u * kv_heads + g) * dim..(u * kv_heads + g + 1) * dim];
                let value = &v[(u * kv_heads + g) * dim..(u * kv_heads + g + 1) * dim];
                add(t, head, key, value, &mut out);
            }
        }
    }
    for (i, sum) in total.iter().enumerate() {
        let o = &mut out[i * dim..(i + 1) * dim];
        for x in o.iter_mut() {
            *x /= sum.max(1e-30);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With MAGI_TEST_MAGIC_DIR (an unpacked .magic file of a trained text
    /// model): the native engine against the ONNX model, and how fast each is.
    #[test]
    #[ignore = "needs a trained model"]
    fn the_native_engine_matches_the_onnx_model() {
        let dir = PathBuf::from(std::env::var("MAGI_TEST_MAGIC_DIR").expect("MAGI_TEST_MAGIC_DIR"));
        let model: magi_core::magic::MagicModel =
            serde_json::from_slice(&std::fs::read(dir.join("magic.json")).unwrap()).unwrap();
        let spec = model.text.clone().unwrap();
        let a = model.architecture.clone().unwrap();
        let weights = std::fs::read(dir.join("model").join("weights.safetensors")).unwrap();
        let onnx = std::fs::read(dir.join("model").join("model.onnx")).unwrap();
        let prompt = spec.encode(&spec.prompt("What is 487 times 26?"));
        let memory = MemorySpec { persist_prefix: false, ..MemorySpec::default() };
        let mut native = NativeModel::load(&a, &weights, &memory, &std::env::temp_dir()).unwrap();
        let started = std::time::Instant::now();
        let ours = native.scores(&prompt, 0).unwrap();
        println!("native prefill of {} tokens: {:?}", prompt.len(), started.elapsed());
        let mut reference = crate::TextModel::load(&onnx, a.context as usize).unwrap();
        let theirs = reference.next_scores(&prompt).unwrap();
        let diff = ours.iter().zip(&theirs).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        println!("largest difference in the scores: {diff}");
        assert!(diff < 1e-2, "{diff}");
        let how = Generation { max_tokens: 40, temperature: 0.0, top_k: 0, stop: vec![], seed: 1 };
        let started = std::time::Instant::now();
        let fast = native.generate(&prompt, &how, 0).unwrap();
        println!("native: 40 tokens in {:?}: {:?}", started.elapsed(), spec.decode(&fast));
        let how = Generation { max_tokens: 8, ..how };
        let started = std::time::Instant::now();
        let slow = reference.generate(&prompt, &how).unwrap();
        println!("ONNX: 8 tokens in {:?}", started.elapsed());
        assert_eq!(&fast[..8], &slow[..], "the same tokens");
        // Asked again, the prompt is in the cache already.
        let started = std::time::Instant::now();
        let again = native.scores(&prompt, 0).unwrap();
        println!("asked again: {:?}, {:?}", started.elapsed(), native.telemetry());
        assert!((again[0] - ours[0]).abs() < 1e-4);
        // Past the trained context, positions stretch (with a Context Scaler) and it still answers.
        if a.rope_scaling.is_some() {
            let long: Vec<u32> = spec.encode(&spec.prompt(&"the cat sat on the mat. ".repeat(130)));
            assert!(long.len() > a.context as usize, "{} tokens", long.len());
            let started = std::time::Instant::now();
            let scores = native.scores(&long, 0).unwrap();
            println!("{} tokens (stretched ×{}) in {:?}", long.len(), native.stretch(long.len()), started.elapsed());
            assert!(scores.iter().all(|s| s.is_finite()) && native.stretch(long.len()) > 1.0);
        }
    }

    #[test]
    fn sha256_matches_the_standard() {
        let hex = |d: [u8; 32]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn yarn_keeps_fast_channels_and_stretches_slow_ones() {
        let scaling =
            RopeScaling { kind: "yarn".into(), factor: 32.0, original: 2048, target: 65536, static_in_exports: false };
        let (plain, one) = frequencies(64, 10000.0, None);
        let (yarn, factor) = frequencies(64, 10000.0, Some(&scaling));
        assert_eq!(one, 1.0);
        assert!((factor - (0.1 * 32f32.ln() + 1.0)).abs() < 1e-5);
        assert!((yarn[0] - plain[0]).abs() < 1e-6, "the fastest channel keeps its frequency");
        assert!((yarn[31] - plain[31] / 32.0).abs() < 1e-9, "the slowest is stretched");
    }

    #[test]
    fn cache_blocks_move_between_tiers_and_come_back_the_same() {
        let dir = std::env::temp_dir().join(format!("magi-kv-test-{}", std::process::id()));
        let mut cache = Cache {
            layers: 1,
            kv_dim: 4,
            blocks: Vec::new(),
            precision: Precision::F16,
            hot_blocks: 2,
            warm_budget: 0,
            disk_budget: 1 << 30,
            disk_dir: dir.clone(),
            file: None,
            recent: VecDeque::new(),
        };
        for p in 0..(BLOCK * 4 + 3) {
            let x = p as f32 / 8.0;
            cache.push(&[vec![x, -x, 1.0, 0.5]], &[vec![x; 4]]).unwrap();
        }
        assert_eq!(cache.len(), BLOCK * 4 + 3);
        let (hot, _, cold) = cache.bytes();
        assert!(hot > 0 && cold > 0, "the first and latest are hot, the middle went to disk");
        // A block from disk reads back as written (to f16 precision).
        let data = cache.floats(2).unwrap().into_owned();
        let p = BLOCK * 2 + 5;
        assert!((data[5 * 4] - p as f32 / 8.0).abs() < 0.1);
        assert_eq!(cache.truncate(BLOCK * 4 + 1), BLOCK * 4 + 1);
        cache.clear();
        let _ = std::fs::remove_dir_all(dir);
    }
}
