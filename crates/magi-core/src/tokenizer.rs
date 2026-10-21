//! Learned tokens: byte pairs. A text model's tokens are its bytes (0–255),
//! then – for a learned tokenizer – pieces learned from its datasets, each
//! joining two tokens into one (ids 256 on, in the order learned), then its
//! special tags. English takes about a third as many tokens as bytes, so a
//! model reads three times the text in the same context and trains on it
//! three times faster.
//!
//! Text is first cut into chunks – a word with the space before it, a digit,
//! a run of punctuation, line breaks, spaces – and pieces never cross a
//! chunk. The trainer (`runtime/python/magi_train/tokens.py`) cuts and joins
//! exactly the same way; this file learns the pieces.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

fn is_letter(b: u8) -> bool {
    b.is_ascii_alphabetic() || b >= 0x80
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn is_punct(b: u8) -> bool {
    !is_letter(b) && !b.is_ascii_digit() && !is_space(b)
}

fn is_line(b: u8) -> bool {
    b == b'\n' || b == b'\r'
}

fn is_blank(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | 0x0b | 0x0c)
}

fn run(bytes: &[u8], from: usize, class: fn(u8) -> bool) -> usize {
    let mut end = from;
    while end < bytes.len() && class(bytes[end]) {
        end += 1;
    }
    end
}

/// Where the chunk starting at `i` ends – the same as Python's
/// `rb" ?[A-Za-z\x80-\xff]+| ?[0-9]| ?[^\sA-Za-z0-9\x80-\xff]+|[\r\n]+|[ \t\x0b\x0c]+(?![^\s])|[ \t\x0b\x0c]+"`.
fn chunk_end(bytes: &[u8], i: usize) -> usize {
    let n = bytes.len();
    let spaced = |class: fn(u8) -> bool| bytes[i] == b' ' && i + 1 < n && class(bytes[i + 1]);
    if spaced(is_letter) {
        return run(bytes, i + 1, is_letter);
    }
    if is_letter(bytes[i]) {
        return run(bytes, i, is_letter);
    }
    if spaced(|b| b.is_ascii_digit()) {
        return i + 2;
    }
    if bytes[i].is_ascii_digit() {
        return i + 1;
    }
    if spaced(is_punct) {
        return run(bytes, i + 1, is_punct);
    }
    if is_punct(bytes[i]) {
        return run(bytes, i, is_punct);
    }
    if is_line(bytes[i]) {
        return run(bytes, i, is_line);
    }
    // Spaces: all of them, but the last stays for the word after it.
    let longest = run(bytes, i, is_blank);
    let mut end = longest;
    while end > i {
        if end == n || is_space(bytes[end]) {
            return end;
        }
        end -= 1;
    }
    longest.max(i + 1)
}

/// The chunks of a text, in order; together they are the text.
pub fn chunks(bytes: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let end = chunk_end(bytes, i);
        out.push(&bytes[i..end]);
        i = end;
    }
    out
}

/// A learned tokenizer's pieces, ready to cut text with.
pub struct Bpe {
    merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>,
    /// The bytes of every token below the tags.
    pieces: Vec<Vec<u8>>,
    cache: Mutex<HashMap<Vec<u8>, Vec<u32>>>,
}

impl Bpe {
    pub fn new(merges: &[(u32, u32)]) -> Self {
        let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for &(a, b) in merges {
            let mut piece = pieces.get(a as usize).cloned().unwrap_or_default();
            piece.extend_from_slice(pieces.get(b as usize).map_or(&[][..], Vec::as_slice));
            pieces.push(piece);
        }
        let ranks = merges.iter().enumerate().map(|(rank, pair)| (*pair, rank as u32)).collect();
        Self { merges: merges.to_vec(), ranks, pieces, cache: Mutex::new(HashMap::new()) }
    }

    /// A chunk's tokens: its bytes, joined pair by pair – always the pair learned
    /// first, every place it appears – until no learned pair is left.
    pub fn encode_chunk(&self, chunk: &[u8]) -> Vec<u32> {
        if let Some(ids) = self.cache.lock().unwrap_or_else(|p| p.into_inner()).get(chunk) {
            return ids.clone();
        }
        let mut ids: Vec<u32> = chunk.iter().map(|b| u32::from(*b)).collect();
        while ids.len() > 1 {
            let best = ids.windows(2).filter_map(|w| self.ranks.get(&(w[0], w[1])).copied()).min();
            let Some(rank) = best else { break };
            let pair = self.merges[rank as usize];
            let joined = 256 + rank;
            let mut out = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
                    out.push(joined);
                    i += 2;
                } else {
                    out.push(ids[i]);
                    i += 1;
                }
            }
            ids = out;
        }
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if cache.len() > 200_000 {
            cache.clear();
        }
        cache.insert(chunk.to_vec(), ids.clone());
        ids
    }

    /// A text without tags, as tokens.
    pub fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(bytes.len() / 3 + 1);
        for chunk in chunks(bytes) {
            ids.extend(self.encode_chunk(chunk));
        }
        ids
    }

    /// The bytes of a token below the tags (`None` for a tag).
    pub fn bytes(&self, id: u32) -> Option<&[u8]> {
        self.pieces.get(id as usize).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.merges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.merges.is_empty()
    }
}

/// The tokenizer of a list of pieces, built once and shared.
pub fn shared(merges: &[(u32, u32)]) -> Arc<Bpe> {
    static BUILT: OnceLock<Mutex<HashMap<u64, Arc<Bpe>>>> = OnceLock::new();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (a, b) in merges {
        for x in [*a, *b] {
            hash = (hash ^ u64::from(x)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    let mut built = BUILT.get_or_init(Default::default).lock().unwrap_or_else(|p| p.into_inner());
    if let Some(bpe) = built.get(&hash).filter(|b| b.merges == merges) {
        return Arc::clone(bpe);
    }
    if built.len() > 8 {
        built.clear();
    }
    let bpe = Arc::new(Bpe::new(merges));
    built.insert(hash, Arc::clone(&bpe));
    bpe
}

/// Learns `count` pieces from texts (tags taken out first): the most frequent
/// pair of neighbouring tokens becomes one token, again and again. The same
/// texts always give the same pieces.
pub fn learn<'a>(texts: impl IntoIterator<Item = &'a str>, tags: &[String], count: usize) -> Vec<(u32, u32)> {
    // How often each chunk comes up.
    let mut seen: HashMap<Vec<u8>, i64> = HashMap::new();
    for text in texts {
        let mut rest = text;
        while !rest.is_empty() {
            let next = tags
                .iter()
                .filter(|t| !t.is_empty())
                .filter_map(|t| rest.find(t.as_str()).map(|at| (at, t.len())))
                .min();
            let (plain, after) = match next {
                Some((at, len)) => (&rest[..at], &rest[at + len..]),
                None => (rest, ""),
            };
            for chunk in chunks(plain.as_bytes()) {
                *seen.entry(chunk.to_vec()).or_default() += 1;
            }
            rest = after;
        }
    }
    let mut chunks: Vec<(Vec<u8>, i64)> = seen.into_iter().collect();
    chunks.sort_unstable();
    let freq: Vec<i64> = chunks.iter().map(|(_, n)| *n).collect();
    let mut words: Vec<Vec<u32>> = chunks.into_iter().map(|(c, _)| c.into_iter().map(u32::from).collect()).collect();

    let mut counts: HashMap<(u32, u32), i64> = HashMap::new();
    let mut places: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
    for (w, word) in words.iter().enumerate() {
        for pair in word.windows(2) {
            let pair = (pair[0], pair[1]);
            *counts.entry(pair).or_default() += freq[w];
            places.entry(pair).or_default().insert(w);
        }
    }
    let mut heap: BinaryHeap<(i64, std::cmp::Reverse<(u32, u32)>)> =
        counts.iter().map(|(pair, n)| (*n, std::cmp::Reverse(*pair))).collect();
    let mut merges = Vec::with_capacity(count);
    while merges.len() < count {
        let Some((n, std::cmp::Reverse(pair))) = heap.pop() else { break };
        if counts.get(&pair).copied() != Some(n) {
            continue; // an old count
        }
        if n < 2 {
            break;
        }
        let joined = 256 + merges.len() as u32;
        merges.push(pair);
        let mut changed: HashSet<(u32, u32)> = HashSet::new();
        let mut list: Vec<usize> = places.remove(&pair).unwrap_or_default().into_iter().collect();
        list.sort_unstable();
        for w in list {
            let word = &words[w];
            if !word.windows(2).any(|p| (p[0], p[1]) == pair) {
                continue;
            }
            for p in word.windows(2) {
                let p = (p[0], p[1]);
                *counts.entry(p).or_default() -= freq[w];
                changed.insert(p);
            }
            let mut out = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && (word[i], word[i + 1]) == pair {
                    out.push(joined);
                    i += 2;
                } else {
                    out.push(word[i]);
                    i += 1;
                }
            }
            for p in out.windows(2) {
                let p = (p[0], p[1]);
                *counts.entry(p).or_default() += freq[w];
                places.entry(p).or_default().insert(w);
                changed.insert(p);
            }
            words[w] = out;
        }
        counts.remove(&pair);
        for p in changed {
            match counts.get(&p).copied() {
                Some(n) if n > 0 => heap.push((n, std::cmp::Reverse(p))),
                _ => {
                    counts.remove(&p);
                }
            }
        }
    }
    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_text_as_python_cuts_it() {
        let cases: [(&str, &[&str]); 5] = [
            ("Hello, world!", &["Hello", ",", " world", "!"]),
            ("x = 12 + 3.5", &["x", " =", " 1", "2", " +", " 3", ".", "5"]),
            ("    def f():\n\n  return", &["   ", " def", " f", "():", "\n\n", " ", " return"]),
            ("café – ok?", &["café", " –", " ok", "?"]),
            ("a  \tb ", &["a", "  ", "\t", "b", " "]),
        ];
        for (text, want) in cases {
            let got: Vec<&str> = chunks(text.as_bytes()).iter().map(|c| std::str::from_utf8(c).unwrap()).collect();
            assert_eq!(got, want, "{text:?}");
        }
    }

    #[test]
    fn learned_pieces_shorten_text_and_decode_back() {
        let text = "the cat sat on the mat. the cat ate the rat. ".repeat(20);
        let merges = learn([text.as_str()], &["<|end|>".into()], 40);
        assert!(merges.len() >= 10, "{merges:?}");
        let bpe = Bpe::new(&merges);
        let ids = bpe.encode(text.as_bytes());
        assert!(ids.len() * 3 < text.len(), "{} tokens for {} bytes", ids.len(), text.len());
        let back: Vec<u8> = ids.iter().flat_map(|id| bpe.bytes(*id).unwrap().to_vec()).collect();
        assert_eq!(back, text.as_bytes());
        // The same text, the same pieces.
        assert_eq!(merges, learn([text.as_str()], &["<|end|>".into()], 40));
    }
}

#[cfg(test)]
mod python_check {
    /// Writes pieces learned from the example datasets, and texts encoded with them, for
    /// the trainer's tokenizer to be checked against (MAGI_TEST_TOKENS_OUT=<file>).
    #[test]
    #[ignore = "writes a file for a check in Python"]
    fn dump_for_python() {
        let Ok(out) = std::env::var("MAGI_TEST_TOKENS_OUT") else { return };
        let data: Vec<(String, String, bool)> = crate::nodes::studio::example_datasets();
        let texts: Vec<&str> = data.iter().map(|(_, c, _)| c.as_str()).collect();
        let mut spec =
            crate::magic::TextSpec { tokenizer: crate::magic::TokenizerSpec::learned_of(4096), ..Default::default() };
        let learned = spec.learn_tokens(texts.iter().copied());
        let samples: Vec<&str> = texts.iter().flat_map(|t| t.lines().take(40)).collect();
        let encoded: Vec<Vec<u32>> = samples.iter().map(|s| spec.encode(s)).collect();
        let bytes: usize = samples.iter().map(|s| s.len()).sum();
        let tokens: usize = encoded.iter().map(Vec::len).sum();
        println!("{learned} pieces; {bytes} bytes as {tokens} tokens");
        let json = serde_json::json!({ "spec": spec, "samples": samples, "encoded": encoded });
        std::fs::write(out, serde_json::to_vec(&json).unwrap()).unwrap();
    }
}

#[cfg(test)]
mod learn_for_spec {
    /// Learns a tokenizer for a text spec (MAGI_LEARN_SPEC, a JSON file) from the example
    /// datasets' texts, as MacroMagi does before training, and writes the spec with its
    /// pieces to MAGI_LEARN_OUT – for training outside MacroMagi.
    #[test]
    #[ignore = "writes a file for training outside MacroMagi"]
    fn learn_tokens_for_spec() {
        let (Ok(input), Ok(out)) = (std::env::var("MAGI_LEARN_SPEC"), std::env::var("MAGI_LEARN_OUT")) else { return };
        let mut spec: crate::magic::TextSpec = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
        if !spec.tokenizer.learned() {
            spec.tokenizer = crate::magic::TokenizerSpec::learned_of(4096);
        }
        let mut texts = Vec::new();
        for (file, content, _) in crate::nodes::studio::example_datasets() {
            let jsonl = crate::nodes::studio::is_jsonl(&file);
            for sample in crate::nodes::studio::split_samples(&file, &content) {
                texts.extend(crate::dataset::texts(&sample, jsonl));
            }
        }
        let count = spec.learn_tokens(texts.iter().map(String::as_str));
        println!("{count} pieces");
        std::fs::write(out, serde_json::to_vec(&spec).unwrap()).unwrap();
    }
}
