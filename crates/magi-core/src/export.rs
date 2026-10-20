//! Exporting text models trained in MacroMagi to the formats other tools
//! read: GGUF (llama.cpp, LM Studio, Ollama), LM Studio's `model.yaml`, and
//! a Hugging Face folder (`transformers`). Only models of Transformer Layers
//! in a layout llama.cpp knows – Llama, Qwen3 or Gemma 3 – can be exported;
//! [`Architecture::gguf_arch`] says which.
//!
//! The weights come from the trainer's safetensors file. Both formats keep
//! MacroMagi's byte-level tokens: every byte is one token (written in GPT-2's
//! byte-to-character form, with no merges), then the special tags.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::magic::{Architecture, MagicModel, TextSpec};

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

/// GPT-2's byte-to-character table: printable bytes stand for themselves, the
/// rest for characters from U+0100 on, so every byte is a visible character.
pub fn byte_chars() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).expect("a valid char")
        } else {
            extra += 1;
            char::from_u32(255 + extra).expect("a valid char")
        };
    }
    table
}

/// The tokens' texts, in id order.
fn vocabulary(text: &TextSpec) -> Vec<String> {
    let chars = byte_chars();
    let mut tokens: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    tokens.extend(text.specials.iter().cloned());
    tokens
}

/// A string as a single-quoted Jinja literal.
fn jinja_str(text: &str) -> String {
    let mut out = String::from("'");
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// The model's chat template, in the Jinja form chat programs read. It writes
/// what [`TextSpec::prompt_with`] and the trainer write: the system text with
/// the reasoning effort (`reasoning_effort`) and the tools offered, tool
/// results in their own turns, and an empty thinking span when thinking is
/// off (`enable_thinking`). Earlier answers lose their reasoning.
///
/// It is laid out the way common tool-calling templates are (tools listed one
/// by one, calls and tool turns written directly), which is how LM Studio
/// tells that a model was trained to call tools.
pub fn chat_template(text: &TextSpec) -> String {
    let s = jinja_str;
    let (system, user, assistant, end) = (&text.system_tag, &text.user_tag, &text.assistant_tag, &text.end_tag);
    let (think_open, think_close) = (&text.think_tags.0, &text.think_tags.1);
    let with_tools = !text.tool_tag.is_empty();
    let mut t = String::new();
    // A message's text, whether it is a string or a list of parts.
    let content = |var: &str| {
        format!(
            "{{%- if {var}.content is string -%}}{{%- set ns.content = {var}.content -%}}\
             {{%- else -%}}{{%- set ns.content = '' -%}}\
             {{%- for part in {var}.content or [] -%}}{{%- if part.text is defined -%}}{{%- set ns.content = ns.content + part.text -%}}{{%- endif -%}}{{%- endfor -%}}\
             {{%- endif -%}}"
        )
    };
    t.push_str("{%- set ns = namespace(content='', last_user=-1) -%}");
    t.push_str("{%- set system_text = '' -%}");
    t.push_str("{%- if messages and messages[0].role == 'system' -%}");
    t.push_str(&content("messages[0]"));
    t.push_str("{%- set system_text = ns.content | trim -%}{%- endif -%}");
    if !text.system.trim().is_empty() {
        t.push_str(&format!(
            "{{%- if not system_text -%}}{{%- set system_text = {} -%}}{{%- endif -%}}",
            s(text.system.trim())
        ));
    }
    if let Some(effort) = &text.effort {
        let levels: Vec<String> = effort.levels.iter().map(|l| s(l)).collect();
        t.push_str(&format!(
            "{{%- if reasoning_effort is defined and reasoning_effort in [{levels}] -%}}\
             {{%- set effort = reasoning_effort -%}}{{%- else -%}}{{%- set effort = {default} -%}}{{%- endif -%}}",
            levels = levels.join(", "),
            default = s(&effort.default),
        ));
    }
    // The system turn: the system text, the effort and the tools, separated by blank lines.
    let mut any = vec!["system_text"];
    if text.effort.is_some() {
        any.push("true");
    }
    if with_tools {
        any.push("tools");
    }
    t.push_str(&format!("{{%- if {} -%}}{{{{- {} }}}}", any.join(" or "), s(system)));
    t.push_str("{%- if system_text -%}{{- system_text }}{%- endif -%}");
    if text.effort.is_some() {
        t.push_str("{%- if system_text -%}{{- '\n\n' }}{%- endif -%}{{- 'Reasoning effort: ' + effort }}");
    }
    if with_tools {
        let before = if text.effort.is_some() { "true" } else { "system_text" };
        t.push_str(&format!(
            "{{%- if tools -%}}\
             {{%- if {before} -%}}{{{{- '\n\n' }}}}{{%- endif -%}}\
             {{{{- 'You can use these tools:' }}}}\
             {{%- for tool in tools -%}}\
             {{%- if tool.function is defined -%}}{{%- set tool = tool.function -%}}{{%- endif -%}}\
             {{{{- '\n- ' + tool.name + ': ' + (tool.description | default('') | trim) + ' Parameters: ' }}}}\
             {{{{- tool.parameters | tojson }}}}\
             {{%- endfor -%}}\
             {{{{- {how} }}}}\
             {{%- endif -%}}",
            how = s(&format!(
                "\nTo use one, write {}{{\"name\": \"<tool name>\", \"arguments\": {{...}}}}{}. Its result comes back in the next turn.",
                text.tool_tags.0, text.tool_tags.1
            )),
        ));
    }
    t.push_str(&format!("{{{{- {} }}}}{{%- endif -%}}", s(end)));
    // Reasoning sent back is shown only since the last request, as the model's own tool loop.
    t.push_str("{%- for message in messages -%}{%- if message.role == 'user' -%}{%- set ns.last_user = loop.index0 -%}{%- endif -%}{%- endfor -%}");
    t.push_str("{%- for message in messages -%}");
    t.push_str(&content("message"));
    t.push_str(&format!(
        "{{%- if message.role == 'user' -%}}{{{{- {user} + ns.content + {end} }}}}\
         {{%- elif message.role == 'system' -%}}{{%- if not loop.first -%}}{{{{- {system} + ns.content + {end} }}}}{{%- endif -%}}\
         {{%- elif message.role == 'assistant' -%}}\
         {{%- if {close} in ns.content -%}}{{%- set ns.content = ns.content.split({close})[-1] | trim -%}}{{%- endif -%}}\
         {{{{- {assistant} }}}}\
         {{%- if message.reasoning_content is string and message.reasoning_content and loop.index0 > ns.last_user -%}}\
         {{{{- {open} + (message.reasoning_content | trim) + {close} }}}}\
         {{%- endif -%}}\
         {{{{- ns.content }}}}\
         {{%- if message.tool_calls -%}}\
         {{%- for tool_call in message.tool_calls -%}}\
         {{%- if tool_call.function is defined -%}}{{%- set tool_call = tool_call.function -%}}{{%- endif -%}}\
         {{{{- {open_call} + tool_call.name + '\", \"arguments\": ' }}}}\
         {{%- if tool_call.arguments is string -%}}{{{{- tool_call.arguments }}}}{{%- else -%}}{{{{- tool_call.arguments | tojson }}}}{{%- endif -%}}\
         {{{{- '}}' + {close_call} }}}}\
         {{%- endfor -%}}\
         {{%- endif -%}}\
         {{{{- {end} }}}}",
        user = s(user),
        system = s(system),
        close = s(think_close),
        open = s(think_open),
        assistant = s(assistant),
        open_call = s(&format!("{}{{\"name\": \"", text.tool_tags.0)),
        close_call = s(&text.tool_tags.1),
        end = s(end),
    ));
    let tool_turn = if with_tools { s(&text.tool_tag) } else { format!("{} + 'Tool result: '", s(user)) };
    t.push_str(&format!(
        "{{%- elif message.role == 'tool' -%}}{{{{- {tool_turn} + ns.content + {end} }}}}\
         {{%- endif -%}}{{%- endfor -%}}",
        end = s(end)
    ));
    t.push_str(&format!("{{%- if add_generation_prompt -%}}{{{{- {} }}}}", s(assistant)));
    let empty = s(&format!("{think_open}{think_close}"));
    match &text.thinking {
        Some(thinking) if thinking.on_by_default => t.push_str(&format!(
            "{{%- if enable_thinking is defined and enable_thinking is false -%}}{{{{- {empty} }}}}{{%- endif -%}}"
        )),
        Some(_) => t.push_str(&format!(
            "{{%- if not (enable_thinking is defined and enable_thinking) -%}}{{{{- {empty} }}}}{{%- endif -%}}"
        )),
        None => {}
    }
    t.push_str("{%- endif -%}");
    t
}

/// How a GGUF file keeps its matrices: half precision, or quantized in
/// blocks of 32 values (8 or 4 bits each, and one scale). Norms stay F32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quant {
    #[default]
    F16,
    Q8_0,
    Q4_0,
}

impl Quant {
    pub const ALL: [Quant; 3] = [Quant::F16, Quant::Q8_0, Quant::Q4_0];

    /// The name GGUF tools use, which also ends the file's name.
    pub fn name(self) -> &'static str {
        match self {
            Quant::F16 => "F16",
            Quant::Q8_0 => "Q8_0",
            Quant::Q4_0 => "Q4_0",
        }
    }

    pub fn parse(name: &str) -> Option<Quant> {
        Quant::ALL.into_iter().find(|q| q.name().eq_ignore_ascii_case(name.trim()))
    }

    pub fn about(self) -> &'static str {
        match self {
            Quant::F16 => "F16 – exact, the largest",
            Quant::Q8_0 => "Q8_0 – half the size, nearly exact",
            Quant::Q4_0 => "Q4_0 – a quarter of the size, loses some quality",
        }
    }

    fn ggml_type(self) -> u32 {
        match self {
            Quant::F16 => 1,
            Quant::Q8_0 => 8,
            Quant::Q4_0 => 2,
        }
    }

    /// `general.file_type` (llama.cpp's `llama_ftype`).
    fn file_type(self) -> u32 {
        match self {
            Quant::F16 => 1,
            Quant::Q8_0 => 7,
            Quant::Q4_0 => 2,
        }
    }

    /// The bytes of a matrix whose rows are `values`, row after row.
    fn encode(self, values: &[f32]) -> Vec<u8> {
        match self {
            Quant::F16 => values.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect(),
            Quant::Q8_0 => {
                let mut out = Vec::with_capacity(values.len() / 32 * 34);
                for block in values.chunks(32) {
                    let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let d = amax / 127.0;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
                    out.extend(block.iter().map(|v| (v * id).round() as i8 as u8));
                }
                out
            }
            Quant::Q4_0 => {
                let mut out = Vec::with_capacity(values.len() / 32 * 18);
                for block in values.chunks(32) {
                    // The value furthest from zero, with its sign, maps to -8.
                    let max = block.iter().fold(0.0f32, |m, v| if v.abs() > m.abs() { *v } else { m });
                    let d = max / -8.0;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
                    let q = |v: f32| ((v * id + 8.5) as i8).clamp(0, 15) as u8;
                    for j in 0..16 {
                        out.push(q(block[j]) | (q(block[j + 16]) << 4));
                    }
                }
                out
            }
        }
    }
}

/// A parameter count the way model names give it: 950K, 3.4M, 1.2B.
pub fn size_label(parameters: u64) -> String {
    let n = parameters as f64;
    let (value, unit) = if n >= 1e9 {
        (n / 1e9, "B")
    } else if n >= 1e6 {
        (n / 1e6, "M")
    } else if n >= 1e3 {
        (n / 1e3, "K")
    } else {
        return parameters.to_string();
    };
    if value >= 100.0 { format!("{value:.0}{unit}") } else { format!("{:.1}{unit}", value).replace(".0", "") }
}

/// The exported layout of a model: which GGUF architecture, and its tensors.
struct Layout {
    arch: &'static str,
    layers: usize,
    hidden: usize,
    vocab: usize,
    qk_norm: bool,
    sandwich: bool,
    tied: bool,
}

impl Layout {
    fn new(a: &Architecture, tensors: &BTreeMap<String, Tensor>, text: &TextSpec) -> Result<Self> {
        let arch = a.gguf_arch().ok_or_else(|| {
            Error::format(
                "only models made of Transformer Layers in the Llama, Qwen3 or Gemma 3 layout (no other attention, \
                 experts or hyper-connections) can be exported to GGUF or Hugging Face",
            )
        })?;
        let get = |name: &str| tensors.get(name).ok_or_else(|| Error::format(format!("the weights have no {name}")));
        let vocab = get("embed_tokens.weight")?.shape[0];
        if vocab != text.vocab_size() {
            return Err(Error::format("the weights do not match the model's tokens; train it again"));
        }
        let layers = a.layers.len();
        let hidden = get("layers.0.mlp.gate_proj.weight")?.shape[0];
        for i in 0..layers {
            if get(&format!("layers.{i}.mlp.gate_proj.weight"))?.shape[0] != hidden {
                return Err(Error::format("every layer must have the same feed-forward width to export"));
            }
        }
        let tied = !tensors.contains_key("lm_head.weight");
        if arch == "gemma3" && !tied {
            return Err(Error::format("Gemma 3 models share their input and output tables (Tie input and output)"));
        }
        Ok(Self {
            arch,
            layers,
            hidden,
            vocab,
            qk_norm: tensors.contains_key("layers.0.self_attn.q_norm.weight"),
            sandwich: tensors.contains_key("layers.0.attn_out_norm.weight"),
            tied,
        })
    }

    /// (our name, Hugging Face name, GGUF name) for every tensor.
    fn names(&self) -> Vec<(String, String, String)> {
        let mut names = vec![
            ("embed_tokens.weight".into(), "model.embed_tokens.weight".into(), "token_embd.weight".into()),
            ("norm.weight".into(), "model.norm.weight".into(), "output_norm.weight".into()),
        ];
        if !self.tied {
            names.push(("lm_head.weight".into(), "lm_head.weight".into(), "output.weight".into()));
        }
        for i in 0..self.layers {
            let mut parts = vec![
                ("input_layernorm.weight", "input_layernorm.weight", "attn_norm.weight"),
                ("self_attn.q_proj.weight", "self_attn.q_proj.weight", "attn_q.weight"),
                ("self_attn.k_proj.weight", "self_attn.k_proj.weight", "attn_k.weight"),
                ("self_attn.v_proj.weight", "self_attn.v_proj.weight", "attn_v.weight"),
                ("self_attn.o_proj.weight", "self_attn.o_proj.weight", "attn_output.weight"),
                ("post_attention_layernorm.weight", "post_attention_layernorm.weight", "ffn_norm.weight"),
                ("mlp.gate_proj.weight", "mlp.gate_proj.weight", "ffn_gate.weight"),
                ("mlp.up_proj.weight", "mlp.up_proj.weight", "ffn_up.weight"),
                ("mlp.down_proj.weight", "mlp.down_proj.weight", "ffn_down.weight"),
            ];
            if self.qk_norm {
                parts.push(("self_attn.q_norm.weight", "self_attn.q_norm.weight", "attn_q_norm.weight"));
                parts.push(("self_attn.k_norm.weight", "self_attn.k_norm.weight", "attn_k_norm.weight"));
            }
            if self.sandwich {
                parts.push(("attn_out_norm.weight", "post_attention_layernorm.weight", "post_attention_norm.weight"));
                parts.push(("mlp_out_norm.weight", "post_feedforward_layernorm.weight", "post_ffw_norm.weight"));
            }
            for (ours, hf, gguf) in parts {
                names.push((format!("layers.{i}.{ours}"), format!("model.layers.{i}.{hf}"), format!("blk.{i}.{gguf}")));
            }
        }
        names
    }
}

/// Reorders the rows of a query or key weight from the halves layout of
/// rotary positions (Hugging Face's) to the interleaved one (llama.cpp's).
fn interleave_rope(tensor: &Tensor, heads: usize) -> Tensor {
    let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
    let dim = rows / heads;
    let half = dim / 2;
    let mut data = vec![0.0; tensor.data.len()];
    for h in 0..heads {
        for j in 0..half {
            for t in 0..2 {
                let from = h * dim + t * half + j;
                let to = h * dim + 2 * j + t;
                data[to * cols..(to + 1) * cols].copy_from_slice(&tensor.data[from * cols..(from + 1) * cols]);
            }
        }
    }
    Tensor { shape: tensor.shape.clone(), data }
}

// ---- GGUF -------------------------------------------------------------------------

enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strings(Vec<String>),
    I32s(Vec<i32>),
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_meta(out: &mut Vec<u8>, key: &str, value: &Meta) {
    put_str(out, key);
    match value {
        Meta::U32(v) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        Meta::F32(v) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        Meta::Bool(v) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(u8::from(*v));
        }
        Meta::Str(v) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            put_str(out, v);
        }
        Meta::Strings(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                put_str(out, item);
            }
        }
        Meta::I32s(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                out.extend_from_slice(&item.to_le_bytes());
            }
        }
    }
}

const ALIGN: usize = 32;

/// The RMSNorm epsilon the trainer uses.
const NORM_EPS: f32 = 1e-5;

/// What an export says about a model: its GGUF architecture and size.
#[derive(Clone, Debug, PartialEq)]
pub struct Exported {
    pub arch: &'static str,
    pub parameters: u64,
}

/// What a trained text model exports as, without writing it.
pub fn describe(model: &MagicModel, weights: &[u8]) -> Result<Exported> {
    let a = model.architecture.as_ref().ok_or_else(|| Error::format("the model has not been trained in MacroMagi"))?;
    let text = model.text.as_ref().ok_or_else(|| Error::format("only text models can be exported to GGUF"))?;
    let tensors = read_safetensors(weights)?;
    let layout = Layout::new(a, &tensors, text)?;
    let parameters = layout.names().iter().map(|(ours, ..)| tensors[ours].data.len() as u64).sum();
    Ok(Exported { arch: layout.arch, parameters })
}

/// The longest text a model runs at: its Context Scaler's target, or what it
/// was trained at. Written as the GGUF's context length so llama.cpp allows it.
fn context_of(a: &Architecture) -> u32 {
    a.rope_scaling.as_ref().map_or(a.context, |s| s.target.max(a.context))
}

/// The rotary base a model runs with: raised for NTK scaling.
fn rope_base(a: &Architecture) -> f64 {
    match a.rope_scaling.as_ref().filter(|s| s.kind == "ntk" && s.factor > 1.0 && s.static_in_exports) {
        Some(s) => {
            let dim = f64::from(a.width / a.heads.max(1));
            a.rope_theta * s.factor.powf(dim / (dim - 2.0).max(1.0))
        }
        None => a.rope_theta,
    }
}

/// A GGUF file of a trained text model, in half precision.
pub fn gguf(model: &MagicModel, weights: &[u8]) -> Result<Vec<u8>> {
    gguf_with(model, weights, Quant::F16)
}

/// A GGUF file of a trained text model (llama, qwen3 or gemma3 architecture;
/// matrices as `quant` says, norms in F32), with its tokens, chat template
/// and the name, size and quantization LM Studio shows.
pub fn gguf_with(model: &MagicModel, weights: &[u8], quant: Quant) -> Result<Vec<u8>> {
    let a = model.architecture.as_ref().ok_or_else(|| Error::format("the model has not been trained in MacroMagi"))?;
    let text = model.text.as_ref().ok_or_else(|| Error::format("only text models can be exported to GGUF"))?;
    let tensors = read_safetensors(weights)?;
    let layout = Layout::new(a, &tensors, text)?;
    let arch = layout.arch;
    let end = text.special_id(&text.end_tag).unwrap_or(0);
    let dim = (a.width / a.heads) as usize;
    let tokens = vocabulary(text);
    // Turn tags are control tokens chat programs hide; thinking and tool tags are text they show.
    let types: Vec<i32> = (0..tokens.len())
        .map(|i| match i {
            0..256 => 1,
            _ if text.is_text_tag(&tokens[i]) => 4,
            _ => 3,
        })
        .collect();
    let parameters: u64 = layout.names().iter().map(|(ours, ..)| tensors[ours].data.len() as u64).sum();
    let name = if model.name.trim().is_empty() { "MacroMagi model".to_owned() } else { model.name.clone() };
    let key = |k: &str| format!("{arch}.{k}");
    let mut meta: Vec<(String, Meta)> = vec![
        ("general.architecture".into(), Meta::Str(arch.into())),
        ("general.type".into(), Meta::Str("model".into())),
        ("general.name".into(), Meta::Str(name.clone())),
        ("general.basename".into(), Meta::Str(name)),
        ("general.size_label".into(), Meta::Str(size_label(parameters))),
        ("general.description".into(), Meta::Str(model.description.clone())),
        ("general.author".into(), Meta::Str(model.author.clone())),
        ("general.license".into(), Meta::Str(model.license.clone())),
        ("general.file_type".into(), Meta::U32(quant.file_type())),
        ("general.quantization_version".into(), Meta::U32(2)),
        ("general.alignment".into(), Meta::U32(ALIGN as u32)),
        (key("context_length"), Meta::U32(context_of(a))),
        (key("embedding_length"), Meta::U32(a.width)),
        (key("block_count"), Meta::U32(layout.layers as u32)),
        (key("feed_forward_length"), Meta::U32(layout.hidden as u32)),
        (key("attention.head_count"), Meta::U32(a.heads)),
        (key("attention.head_count_kv"), Meta::U32(a.kv_heads)),
        (key("rope.freq_base"), Meta::F32(rope_base(a) as f32)),
        (key("attention.layer_norm_rms_epsilon"), Meta::F32(NORM_EPS)),
        (key("vocab_size"), Meta::U32(layout.vocab as u32)),
    ];
    if arch == "llama" {
        meta.push((key("rope.dimension_count"), Meta::U32(dim as u32)));
    } else {
        meta.push((key("attention.key_length"), Meta::U32(dim as u32)));
        meta.push((key("attention.value_length"), Meta::U32(dim as u32)));
    }
    if let Some(scaling) = a.rope_scaling.as_ref().filter(|s| s.kind == "yarn" && s.factor > 1.0 && s.static_in_exports)
    {
        // llama.cpp and LM Studio read these and stretch positions the same way.
        meta.push((key("rope.scaling.type"), Meta::Str("yarn".into())));
        meta.push((key("rope.scaling.factor"), Meta::F32(scaling.factor as f32)));
        meta.push((key("rope.scaling.original_context_length"), Meta::U32(scaling.original)));
    }
    if arch == "gemma3" {
        // No window: one as long as the context, so every layer sees everything.
        let window = a.local_window().unwrap_or(u64::from(a.context)) as u32;
        meta.push((key("attention.sliding_window"), Meta::U32(window)));
        meta.push((key("rope.freq_base_swa"), Meta::F32(a.rope_theta as f32)));
    }
    meta.extend([
        ("tokenizer.ggml.model".into(), Meta::Str("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Meta::Str("default".into())),
        ("tokenizer.ggml.tokens".into(), Meta::Strings(tokens)),
        ("tokenizer.ggml.token_type".into(), Meta::I32s(types)),
        ("tokenizer.ggml.merges".into(), Meta::Strings(Vec::new())),
        ("tokenizer.ggml.bos_token_id".into(), Meta::U32(end)),
        ("tokenizer.ggml.eos_token_id".into(), Meta::U32(end)),
        ("tokenizer.ggml.eot_token_id".into(), Meta::U32(end)),
        ("tokenizer.ggml.padding_token_id".into(), Meta::U32(end)),
        ("tokenizer.ggml.add_bos_token".into(), Meta::Bool(false)),
        ("tokenizer.ggml.add_eos_token".into(), Meta::Bool(false)),
        ("tokenizer.chat_template".into(), Meta::Str(chat_template(text))),
        // So MacroMagi knows the model again when it is imported: its tags, tools and shape.
        ("macromagi.text".into(), Meta::Str(serde_json::to_string(text)?)),
        ("macromagi.architecture".into(), Meta::Str(serde_json::to_string(a)?)),
    ]);

    // The tensors, in GGUF's order of dimensions (innermost first).
    let mut entries: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    for (ours, _, name) in layout.names() {
        let tensor = &tensors[&ours];
        // llama.cpp turns llama's queries and keys in interleaved pairs; Qwen3 and Gemma 3 in halves, as trained.
        let tensor = if arch == "llama" && ours.ends_with("q_proj.weight") {
            interleave_rope(tensor, a.heads as usize)
        } else if arch == "llama" && ours.ends_with("k_proj.weight") {
            interleave_rope(tensor, a.kv_heads as usize)
        } else {
            Tensor { shape: tensor.shape.clone(), data: tensor.data.clone() }
        };
        let dims: Vec<u64> = tensor.shape.iter().rev().map(|d| *d as u64).collect();
        let (ty, data) = if tensor.shape.len() >= 2 {
            // Quantized blocks run along a row: rows must hold whole blocks.
            let q = if tensor.shape[tensor.shape.len() - 1].is_multiple_of(32) { quant } else { Quant::F16 };
            (q.ggml_type(), q.encode(&tensor.data))
        } else {
            (0u32, tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect())
        };
        entries.push((name, dims, ty, data));
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    out.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (key, value) in &meta {
        put_meta(&mut out, key, value);
    }
    let mut offset = 0u64;
    for (name, dims, ty, data) in &entries {
        put_str(&mut out, name);
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += data.len() as u64;
        offset = offset.div_ceil(ALIGN as u64) * ALIGN as u64;
    }
    while !out.len().is_multiple_of(ALIGN) {
        out.push(0);
    }
    for (_, _, _, data) in &entries {
        out.extend_from_slice(data);
        while !out.len().is_multiple_of(ALIGN) {
            out.push(0);
        }
    }
    Ok(out)
}

// ---- LM Studio ---------------------------------------------------------------------

/// A name as LM Studio's keys write it: lowercase, words joined by dashes.
pub fn lmstudio_key(name: &str) -> String {
    let mut key = String::new();
    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() || c == '.' {
            key.push(c.to_ascii_lowercase());
        } else if !key.ends_with('-') && !key.is_empty() {
            key.push('-');
        }
    }
    let key = key.trim_end_matches('-').to_owned();
    if key.is_empty() { "model".into() } else { key }
}

fn yaml_str(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_default()
}

/// LM Studio's description of a model (`model.yaml`, see modelyaml.org): its
/// size and context, whether it reasons and calls tools, the sampling it
/// wants, how to find its thinking, and the Enable Thinking and Reasoning
/// Effort controls its chat template reads. `owner`/`repo` name the GGUF
/// folder in LM Studio's models folder.
pub fn lmstudio_model_yaml(
    model: &MagicModel,
    exported: &Exported,
    owner: &str,
    repo: &str,
    gguf_bytes: u64,
) -> Result<String> {
    let a = model.architecture.as_ref().ok_or_else(|| Error::format("the model has not been trained in MacroMagi"))?;
    let text = model.text.as_ref().ok_or_else(|| Error::format("only text models go to LM Studio"))?;
    let name = lmstudio_key(&model.name);
    let mut y = String::new();
    y.push_str("# model.yaml is an open standard for defining cross-platform, composable AI models\n");
    y.push_str("# Learn more at https://modelyaml.org\n");
    y.push_str(&format!("# Written by MacroMagi.\nmodel: {}/{name}\n", lmstudio_key(owner)));
    y.push_str(&format!(
        "base:\n  - key: {}/{}\n    sources:\n      - type: huggingface\n        user: {owner}\n        repo: {repo}\n",
        lmstudio_key(owner),
        lmstudio_key(repo)
    ));
    let reasons = text.thinking.is_some() || text.effort.is_some();
    y.push_str(&format!(
        "metadataOverrides:\n  domain: llm\n  architectures:\n    - {arch}\n  compatibilityTypes:\n    - gguf\n  \
         paramsStrings:\n    - {size}\n  minMemoryUsageBytes: {memory}\n  contextLengths:\n    - {context}\n  \
         vision: false\n  reasoning: {reasons}\n  trainedForToolUse: {tools}\n",
        arch = exported.arch,
        size = size_label(exported.parameters),
        memory = gguf_bytes + gguf_bytes / 4 + 64 * 1024 * 1024,
        context = context_of(a),
        tools = !text.tool_tag.is_empty(),
    ));
    y.push_str(&format!(
        "config:\n  operation:\n    fields:\n      - key: llm.prediction.temperature\n        value: {temperature}\n      \
         - key: llm.prediction.topKSampling\n        value: {top_k}\n      - key: llm.prediction.reasoning.parsing\n        \
         value:\n          enabled: true\n          startString: {open}\n          endString: {close}\n",
        temperature = text.temperature,
        top_k = text.top_k.max(1),
        open = yaml_str(&text.think_tags.0),
        close = yaml_str(&text.think_tags.1),
    ));
    if reasons {
        y.push_str("customFields:\n");
    }
    if let Some(effort) = &text.effort {
        y.push_str(&format!(
            "  - key: reasoningEffort\n    displayName: Reasoning Effort\n    description: Controls how much reasoning the \
             model should perform.\n    type: select\n    defaultValue: {}\n    options:\n",
            yaml_str(&effort.default)
        ));
        for level in &effort.levels {
            let mut label: Vec<char> = level.chars().collect();
            if let Some(first) = label.first_mut() {
                *first = first.to_ascii_uppercase();
            }
            y.push_str(&format!(
                "      - value: {}\n        label: {}\n",
                yaml_str(level),
                yaml_str(&label.into_iter().collect::<String>())
            ));
        }
        y.push_str("    effects:\n      - type: setJinjaVariable\n        variable: reasoning_effort\n");
    }
    if let Some(thinking) = &text.thinking {
        y.push_str(&format!(
            "  - key: enableThinking\n    displayName: Enable Thinking\n    description: Controls whether the model will \
             think before replying\n    type: boolean\n    defaultValue: {}\n    effects:\n      - type: setJinjaVariable\n        \
             variable: enable_thinking\n",
            thinking.on_by_default
        ));
    }
    Ok(y)
}

/// The `manifest.json` beside an LM Studio `model.yaml`.
pub fn lmstudio_manifest(model: &MagicModel, owner: &str, repo: &str) -> String {
    let manifest = json!({
        "type": "model",
        "owner": lmstudio_key(owner),
        "name": lmstudio_key(&model.name),
        "dependencies": [{
            "type": "model",
            "purpose": "baseModel",
            "modelKeys": [format!("{}/{}", lmstudio_key(owner), lmstudio_key(repo))],
            "sources": [{ "type": "huggingface", "user": owner, "repo": repo }],
        }],
        "revision": 1,
    });
    serde_json::to_string_pretty(&manifest).unwrap_or_default()
}

// Reads GGUF values from bytes.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let slice = self.bytes.get(self.at..self.at + n).ok_or_else(|| Error::format("the GGUF file is cut off"))?;
        self.at += n;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    /// A value of a type; arrays are counted, not kept (they can hold a whole vocabulary).
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => json!(self.take(1)?[0]),
            1 => json!(self.take(1)?[0] as i8),
            7 => json!(self.take(1)?[0] != 0),
            2 => json!(u16::from_le_bytes(self.take(2)?.try_into().expect("2 bytes"))),
            3 => json!(i16::from_le_bytes(self.take(2)?.try_into().expect("2 bytes"))),
            4 => json!(self.u32()?),
            5 => json!(self.u32()? as i32),
            6 => json!(f32::from_bits(self.u32()?)),
            8 => json!(self.string()?),
            10 => json!(self.u64()?),
            11 => json!(self.u64()? as i64),
            12 => json!(f64::from_bits(self.u64()?)),
            9 => {
                let inner = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.value(inner)?;
                }
                json!({ "array": n })
            }
            _ => return Err(Error::format("the GGUF file has a value of an unknown type")),
        })
    }
}

/// What a GGUF file says about itself: its metadata (arrays counted, not
/// kept) and how many tensors it has.
pub fn read_gguf_metadata(bytes: &[u8]) -> Result<(BTreeMap<String, Value>, u64)> {
    let mut c = Cursor { bytes, at: 0 };
    if c.take(4)? != b"GGUF" {
        return Err(Error::format("this is not a GGUF file"));
    }
    if c.u32()? < 2 {
        return Err(Error::format("this GGUF file is too old"));
    }
    let tensors = c.u64()?;
    let count = c.u64()?;
    let mut out = BTreeMap::new();
    for _ in 0..count {
        let key = c.string()?;
        let ty = c.u32()?;
        let value = c.value(ty)?;
        out.insert(key, value);
    }
    Ok((out, tensors))
}

// ---- Hugging Face -----------------------------------------------------------------

/// Writes a Hugging Face folder of a trained text model: `config.json`,
/// `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
/// `generation_config.json`, `chat_template.jinja` and a `README.md`.
pub fn huggingface(model: &MagicModel, weights: &[u8], folder: &Path) -> Result<()> {
    let a = model.architecture.as_ref().ok_or_else(|| Error::format("the model has not been trained in MacroMagi"))?;
    let text = model.text.as_ref().ok_or_else(|| Error::format("only text models can be exported"))?;
    let tensors = read_safetensors(weights)?;
    let layout = Layout::new(a, &tensors, text)?;
    let (class, model_type) = match layout.arch {
        "llama" => ("LlamaForCausalLM", "llama"),
        "qwen3" => ("Qwen3ForCausalLM", "qwen3"),
        _ => return Err(Error::format("Hugging Face folders are written for Llama and Qwen3 layouts; use GGUF")),
    };
    std::fs::create_dir_all(folder)?;
    let end = text.special_id(&text.end_tag).unwrap_or(0);
    let config = json!({
        "architectures": [class],
        "model_type": model_type,
        "hidden_size": a.width,
        "intermediate_size": layout.hidden,
        "num_hidden_layers": layout.layers,
        "num_attention_heads": a.heads,
        "num_key_value_heads": a.kv_heads,
        "head_dim": a.width / a.heads,
        "max_position_embeddings": context_of(a),
        "rms_norm_eps": NORM_EPS,
        "rope_theta": rope_base(a),
        "rope_scaling": a.rope_scaling.as_ref().filter(|s| s.kind == "yarn" && s.factor > 1.0 && s.static_in_exports).map(|s| json!({
            "rope_type": "yarn", "factor": s.factor, "original_max_position_embeddings": s.original,
        })),
        "vocab_size": layout.vocab,
        "hidden_act": "silu",
        "tie_word_embeddings": layout.tied,
        "attention_bias": false,
        "mlp_bias": false,
        "bos_token_id": end,
        "eos_token_id": end,
        "pad_token_id": end,
        "torch_dtype": "float32",
    });
    let renamed: Vec<(String, &Tensor)> =
        layout.names().into_iter().map(|(ours, hf, _)| (hf, &tensors[&ours])).collect();
    std::fs::write(folder.join("model.safetensors"), write_safetensors(&renamed, Some(json!({ "format": "pt" }))))?;
    std::fs::write(folder.join("config.json"), serde_json::to_vec_pretty(&config)?)?;

    let chars = byte_chars();
    let vocab: serde_json::Map<String, Value> =
        chars.iter().enumerate().map(|(i, c)| (c.to_string(), json!(i))).collect();
    let added: Vec<Value> = text
        .specials
        .iter()
        .enumerate()
        .map(|(i, tag)| json!({ "id": 256 + i, "content": tag, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }))
        .collect();
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false },
        "post_processor": null,
        "decoder": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false },
        "model": { "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                   "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                   "vocab": vocab, "merges": [] },
    });
    std::fs::write(folder.join("tokenizer.json"), serde_json::to_vec(&tokenizer)?)?;
    let template = chat_template(text);
    let tokenizer_config = json!({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": a.context,
        "eos_token": text.end_tag,
        "pad_token": text.end_tag,
        "bos_token": null,
        "clean_up_tokenization_spaces": false,
        "chat_template": template,
    });
    std::fs::write(folder.join("tokenizer_config.json"), serde_json::to_vec_pretty(&tokenizer_config)?)?;
    std::fs::write(folder.join("chat_template.jinja"), &template)?;
    let generation = json!({ "eos_token_id": end, "pad_token_id": end, "do_sample": text.temperature > 0.0,
                             "temperature": text.temperature, "top_k": text.top_k, "max_new_tokens": text.max_tokens });
    std::fs::write(folder.join("generation_config.json"), serde_json::to_vec_pretty(&generation)?)?;
    let mut readme = std::fs::File::create(folder.join("README.md"))?;
    writeln!(
        readme,
        "---\nlibrary_name: transformers\npipeline_tag: text-generation\n{licence}---\n\n# {name}\n\n{description}\n\n\
         A {class} model trained in [MacroMagi](https://github.com/MagiGun/MacroMagi): {layers} layers, \
         {width} wide, {context} tokens of context. Its tokens are bytes (no merges), so any text can be read.\n\n\
         Tool calls are written as `{open}{{\"name\": …, \"arguments\": {{…}}}}{close}`.\n",
        licence = if model.license.trim().is_empty() {
            String::new()
        } else {
            format!("license: {}\n", model.license.trim().to_lowercase())
        },
        name = if model.name.is_empty() { "MacroMagi model" } else { &model.name },
        description = model.description,
        class = class,
        layers = layout.layers,
        width = a.width,
        context = a.context,
        open = text.tool_tags.0,
        close = text.tool_tags.1,
    )?;
    Ok(())
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
    fn every_byte_has_its_own_character() {
        let chars = byte_chars();
        let distinct: std::collections::BTreeSet<char> = chars.iter().copied().collect();
        assert_eq!(distinct.len(), 256);
        assert_eq!(chars[b'A' as usize], 'A');
        assert_eq!(chars[b' ' as usize], 'Ġ');
        assert_eq!(chars[b'\n' as usize], 'Ċ');
    }

    #[test]
    fn rope_rows_are_interleaved_per_head() {
        // Two heads of dimension 4, one column: rows 0..8.
        let t = Tensor { shape: vec![8, 1], data: (0..8).map(|i| i as f32).collect() };
        let out = interleave_rope(&t, 2);
        assert_eq!(out.data, [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    /// With MAGI_EXPORT_DIR set: writes a training plan for the example text
    /// model there (step 1), or exports what the trainer made of it (step 2).
    #[test]
    #[ignore = "a manual check with the trainer, llama.cpp and transformers"]
    fn export_round_trip_with_other_tools() {
        use crate::graph::Graph;
        use crate::nodes::studio;
        let dir = std::path::PathBuf::from(std::env::var("MAGI_EXPORT_DIR").expect("MAGI_EXPORT_DIR"));
        let weights = dir.join("out").join("weights.safetensors");
        if !weights.exists() {
            // The example model, as New text model makes it; MAGI_EXPORT_COMMON picks another architecture.
            let mut text = TextSpec {
                temperature: 0.3,
                max_tokens: 200,
                system: studio::EXAMPLE_SYSTEM.into(),
                tools: studio::example_tools(),
                thinking: Some(crate::magic::ThinkingSpec::default()),
                effort: Some(crate::magic::EffortSpec::default()),
                ..TextSpec::default()
            };
            text.ensure_specials();
            let model = MagicModel {
                kind: crate::magic::ModelKind::Text,
                text: Some(text),
                name: "Magi".into(),
                ..MagicModel::default()
            };
            let files: Vec<(String, bool)> = studio::example_datasets().into_iter().map(|(f, _, n)| (f, n)).collect();
            let mut graph = Graph::from(studio::default_canvas(&model, false, false, &files));
            if let Ok(name) = std::env::var("MAGI_EXPORT_COMMON") {
                let common = studio::commons().into_iter().find(|c| c.name == name).expect("a common architecture");
                studio::apply_common(&mut graph, &common, [0.0, 0.0]);
            }
            if let Ok(steps) = std::env::var("MAGI_EXPORT_STEPS") {
                let steps: u32 = steps.parse().unwrap();
                let ids: Vec<_> =
                    graph.nodes().filter(|n| studio::kinds::STAGES.contains(&n.kind.as_str())).map(|n| n.id).collect();
                for id in ids {
                    graph.node_mut(id).unwrap().set_param("steps", steps);
                }
            }
            let mut model = model;
            studio::derive(&mut graph, &mut model, false);
            for (file, content, _) in studio::example_datasets() {
                std::fs::write(dir.join(file.rsplit('/').next().unwrap()), content).unwrap();
            }
            let pipeline = studio::pipeline(&graph).unwrap();
            let d = dir.clone();
            let dataset = move |f: &str| Some(d.join(f.rsplit('/').next().unwrap()).to_string_lossy().into_owned());
            let plan = studio::plan(
                &pipeline,
                &graph,
                &model,
                &studio::PlanFiles {
                    dataset: &dataset,
                    weights: None,
                    pictures: None,
                    out_dir: dir.join("out").to_string_lossy().into_owned(),
                    devices: vec![0],
                },
            )
            .unwrap();
            model.architecture = Some(pipeline.architecture);
            std::fs::write(dir.join("plan.json"), serde_json::to_vec_pretty(&plan).unwrap()).unwrap();
            std::fs::write(dir.join("model.json"), serde_json::to_vec_pretty(&model).unwrap()).unwrap();
        } else {
            let model: MagicModel = serde_json::from_slice(&std::fs::read(dir.join("model.json")).unwrap()).unwrap();
            let bytes = std::fs::read(weights).unwrap();
            std::fs::write(dir.join("model.gguf"), gguf(&model, &bytes).unwrap()).unwrap();
            if model.architecture.as_ref().unwrap().gguf_arch() != Some("gemma3") {
                huggingface(&model, &bytes, &dir.join("hf")).unwrap();
            }
            let (meta, _) = read_gguf_metadata(&std::fs::read(dir.join("model.gguf")).unwrap()).unwrap();
            assert_eq!(meta["general.architecture"], model.architecture.as_ref().unwrap().gguf_arch().unwrap());
            for quant in [Quant::Q8_0, Quant::Q4_0] {
                let name = format!("model-{}.gguf", quant.name());
                std::fs::write(dir.join(name), gguf_with(&model, &bytes, quant).unwrap()).unwrap();
            }
            std::fs::write(dir.join("chat_template.jinja"), chat_template(model.text.as_ref().unwrap())).unwrap();
        }
    }

    #[test]
    fn the_chat_template_writes_the_model_tags() {
        let text = TextSpec {
            thinking: Some(crate::magic::ThinkingSpec::default()),
            effort: Some(crate::magic::EffortSpec::default()),
            system: "Be brief.".into(),
            ..TextSpec::default()
        };
        let template = chat_template(&text);
        assert!(template.contains("'<|user|>' + ns.content + '<|end|>'"));
        assert!(template.contains("{%- for tool in tools -%}"));
        assert!(template.contains("reasoning_effort in ['none', 'low', 'medium', 'high']"));
        assert!(template.contains("enable_thinking is defined and enable_thinking is false"));
        assert!(template.contains("'<|tool|>' + ns.content"));
        assert!(template.ends_with("{%- endif -%}"));
        assert_eq!(template.matches("{%").count(), template.matches("%}").count());
        // To render it elsewhere (a Jinja check against the trainer's prompts).
        if let Ok(dir) = std::env::var("MAGI_TEST_TEMPLATE_DIR") {
            let dir = std::path::Path::new(&dir);
            let mut magi = TextSpec {
                system: "You are Magi.".into(),
                tools: crate::nodes::studio::example_tools(),
                thinking: Some(crate::magic::ThinkingSpec::default()),
                effort: Some(crate::nodes::studio::magi_effort()),
                ..TextSpec::default()
            };
            magi.ensure_specials();
            let plain = TextSpec { tool_tag: String::new(), ..TextSpec::default() };
            for (name, spec) in [("magi", &magi), ("tagged", &text), ("plain", &plain)] {
                std::fs::write(dir.join(format!("{name}.jinja")), chat_template(spec)).unwrap();
                std::fs::write(dir.join(format!("{name}.json")), serde_json::to_vec(spec).unwrap()).unwrap();
            }
        }
    }

    #[test]
    fn quantized_blocks_keep_their_values() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) / 8.0).collect();
        let q8 = Quant::Q8_0.encode(&values);
        assert_eq!(q8.len(), 2 * 34);
        let d = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));
        for (j, v) in values[..32].iter().enumerate() {
            assert!((f32::from(q8[2 + j] as i8) * d - v).abs() <= d);
        }
        let q4 = Quant::Q4_0.encode(&values);
        assert_eq!(q4.len(), 2 * 18);
        let d = f16_to_f32(u16::from_le_bytes([q4[0], q4[1]]));
        let first = f32::from(q4[2] & 15) - 8.0;
        assert!((first * d - values[0]).abs() <= d.abs());
        assert_eq!(size_label(3_412_000), "3.4M");
        assert_eq!(size_label(950_000), "950K");
        assert_eq!(size_label(2_000_000_000), "2B");
        assert_eq!(Quant::parse("q8_0"), Some(Quant::Q8_0));
    }

    /// Weights of a one-layer model of width 64, named as the trainer names them.
    fn tiny_weights(text: &TextSpec, qk_norm: bool, sandwich: bool, tied: bool) -> Vec<u8> {
        let tensor = |shape: Vec<usize>| {
            let n = shape.iter().product::<usize>();
            Tensor { shape, data: (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 400.0).collect() }
        };
        let (w, kv, hidden, vocab) = (64, 32, 128, text.vocab_size());
        let mut named = vec![
            ("embed_tokens.weight".to_owned(), tensor(vec![vocab, w])),
            ("norm.weight".to_owned(), tensor(vec![w])),
            ("layers.0.input_layernorm.weight".to_owned(), tensor(vec![w])),
            ("layers.0.post_attention_layernorm.weight".to_owned(), tensor(vec![w])),
            ("layers.0.self_attn.q_proj.weight".to_owned(), tensor(vec![w, w])),
            ("layers.0.self_attn.k_proj.weight".to_owned(), tensor(vec![kv, w])),
            ("layers.0.self_attn.v_proj.weight".to_owned(), tensor(vec![kv, w])),
            ("layers.0.self_attn.o_proj.weight".to_owned(), tensor(vec![w, w])),
            ("layers.0.mlp.gate_proj.weight".to_owned(), tensor(vec![hidden, w])),
            ("layers.0.mlp.up_proj.weight".to_owned(), tensor(vec![hidden, w])),
            ("layers.0.mlp.down_proj.weight".to_owned(), tensor(vec![w, hidden])),
        ];
        if !tied {
            named.push(("lm_head.weight".to_owned(), tensor(vec![vocab, w])));
        }
        if qk_norm {
            named.push(("layers.0.self_attn.q_norm.weight".to_owned(), tensor(vec![16])));
            named.push(("layers.0.self_attn.k_norm.weight".to_owned(), tensor(vec![16])));
        }
        if sandwich {
            named.push(("layers.0.attn_out_norm.weight".to_owned(), tensor(vec![w])));
            named.push(("layers.0.mlp_out_norm.weight".to_owned(), tensor(vec![w])));
        }
        let refs: Vec<(String, &Tensor)> = named.iter().map(|(n, t)| (n.clone(), t)).collect();
        write_safetensors(&refs, None)
    }

    #[test]
    fn every_gguf_layout_and_precision_is_written() {
        let mut text = TextSpec { thinking: Some(crate::magic::ThinkingSpec::default()), ..TextSpec::default() };
        text.ensure_specials();
        let layer =
            |attention: Value, ffn: Value, norm: &str| json!({ "attention": attention, "ffn": ffn, "norm": norm });
        let cases = [
            ("llama", layer(json!({"kind": "full"}), json!({"kind": "dense"}), "pre"), false, false, false),
            (
                "qwen3",
                layer(json!({"kind": "full", "qk_norm": true}), json!({"kind": "dense"}), "pre"),
                true,
                false,
                true,
            ),
            (
                "gemma3",
                layer(
                    json!({"kind": "full", "qk_norm": true}),
                    json!({"kind": "dense", "activation": "geglu"}),
                    "sandwich",
                ),
                true,
                true,
                true,
            ),
        ];
        for (arch, layer, qk, sandwich, tied) in cases {
            let a = Architecture {
                width: 64,
                heads: 4,
                kv_heads: 2,
                tied,
                embed_scale: arch == "gemma3",
                layers: vec![layer],
                ..Architecture::default()
            };
            assert_eq!(a.gguf_arch(), Some(arch));
            let model = MagicModel {
                name: "Tiny".into(),
                text: Some(text.clone()),
                architecture: Some(a),
                ..MagicModel::default()
            };
            let weights = tiny_weights(&text, qk, sandwich, tied);
            let described = describe(&model, &weights).unwrap();
            assert_eq!(described.arch, arch);
            let mut sizes = Vec::new();
            for quant in Quant::ALL {
                let bytes = gguf_with(&model, &weights, quant).unwrap();
                let (meta, tensors) = read_gguf_metadata(&bytes).unwrap();
                assert_eq!(meta["general.architecture"], arch);
                assert_eq!(meta["general.file_type"], json!(quant.file_type()));
                assert_eq!(meta["general.size_label"], json!(size_label(described.parameters)));
                assert_eq!(tensors as usize, 12 - usize::from(tied) + 2 * usize::from(qk) + 2 * usize::from(sandwich));
                sizes.push(bytes.len());
            }
            assert!(sizes[0] > sizes[1] && sizes[1] > sizes[2], "{arch}: F16 > Q8_0 > Q4_0: {sizes:?}");
        }
    }

    #[test]
    fn lm_studio_keys_are_lowercase_words() {
        assert_eq!(lmstudio_key("My Agent (small)"), "my-agent-small");
        assert_eq!(lmstudio_key("Magi-F1"), "magi-f1");
        let model = MagicModel {
            name: "My Agent".into(),
            text: Some(TextSpec {
                thinking: Some(crate::magic::ThinkingSpec::default()),
                effort: Some(crate::magic::EffortSpec::default()),
                ..TextSpec::default()
            }),
            architecture: Some(Architecture::default()),
            ..MagicModel::default()
        };
        let exported = Exported { arch: "qwen3", parameters: 3_400_000 };
        let yaml = lmstudio_model_yaml(&model, &exported, "MagiGun", "My-Agent-GGUF", 7_000_000).unwrap();
        assert!(yaml.contains("model: magigun/my-agent\n"));
        assert!(yaml.contains("  - key: magigun/my-agent-gguf\n"));
        assert!(yaml.contains("    - 3.4M\n"));
        assert!(yaml.contains("variable: reasoning_effort"));
        assert!(yaml.contains("variable: enable_thinking"));
        let manifest: Value = serde_json::from_str(&lmstudio_manifest(&model, "MagiGun", "My-Agent-GGUF")).unwrap();
        assert_eq!(manifest["dependencies"][0]["modelKeys"][0], "magigun/my-agent-gguf");
    }
}
