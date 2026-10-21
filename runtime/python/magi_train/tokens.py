"""Text as token ids, exactly as MacroMagi reads it (magic.rs, TextSpec; tokenizer.rs).

Ids 0–255 are the bytes of UTF-8 text; with a learned tokenizer, ids from 256
on are its pieces (each joins two tokens, in the order learned); then come the
special tags in `specials`, in order. Special tags are matched first, longest
first; the rest is cut into chunks (a word with the space before it, a digit, a
run of punctuation, line breaks, spaces) and each chunk's bytes are joined into
pieces, the earliest-learned pair first.
"""


import json
import re


def tool_of(tool):
    """A tool as {"name", "description", "parameters"} (OpenAI's form is unwrapped)."""
    tool = tool.get("function", tool) if isinstance(tool, dict) else {}
    return {
        "name": str(tool.get("name", "")),
        "description": str(tool.get("description", "") or ""),
        "parameters": tool.get("parameters", {}),
    }


def signature(parameters):
    """A tool's arguments, short (magic.rs, signature): in the order the tool declares them, `?` after an
    optional one, the type when it is not text."""
    properties = (parameters or {}).get("properties") or {}
    required = (parameters or {}).get("required") or []
    parts = []
    for name in properties:
        kind = (properties[name] or {}).get("type", "string")
        optional = "" if name in required else "?"
        parts.append("%s%s" % (name, optional) if kind == "string" else "%s%s: %s" % (name, optional, kind))
    return ", ".join(parts)


def byte_letters():
    """Each byte as a printable character, as byte-level tokenizers write pieces (GPT-2's table)."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    letters, extra = {}, 0
    for b in range(256):
        if b in keep:
            letters[b] = chr(b)
        else:
            letters[b] = chr(256 + extra)
            extra += 1
    return letters


# CHUNKS on text instead of bytes (for the fast tokenizer): every character past ASCII is part of a
# word, as its UTF-8 bytes are there; whitespace is only the six ASCII ones the bytes pattern knows.
TEXT_CHUNKS = (r" ?[A-Za-z\x{80}-\x{10FFFF}]+| ?[0-9]| ?[^ \t\n\r\x{0B}\x{0C}A-Za-z0-9\x{80}-\x{10FFFF}]+"
               r"|[\r\n]+|[ \t\x{0B}\x{0C}]+(?![^ \t\n\r\x{0B}\x{0C}])|[ \t\x{0B}\x{0C}]+")


# The chunks text is cut into (tokenizer.rs, chunk_end does the same).
CHUNKS = re.compile(rb" ?[A-Za-z\x80-\xff]+| ?[0-9]| ?[^\sA-Za-z0-9\x80-\xff]+|[\r\n]+|[ \t\x0b\x0c]+(?![^\s])|[ \t\x0b\x0c]+")


class Tokenizer:
    def __init__(self, spec):
        self.spec = spec
        self.specials = list(spec.get("specials", []))
        self.merges = [tuple(m) for m in (spec.get("tokenizer") or {}).get("merges", [])]
        self.base = 256 + len(self.merges)
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._pieces = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self._pieces.append(self._pieces[a] + self._pieces[b])
        self._cache = {}
        self._order = sorted(range(len(self.specials)), key=lambda i: -len(self.specials[i]))
        self._encoded = [tag.encode("utf-8") for tag in self.specials]
        # The tags, longest first, matched where they start with "<" – as the loop below does.
        tags = [self.specials[i] for i in self._order if self.specials[i].startswith("<")]
        self._split = re.compile("(%s)" % "|".join(re.escape(t) for t in tags)) if tags else None
        self._ids = {tag: self.base + self.specials.index(tag) for tag in tags}

    @property
    def vocab(self):
        return self.base + len(self.specials)

    def special(self, tag):
        return self.base + self.specials.index(tag) if tag in self.specials else None

    def _chunk(self, chunk):
        """A chunk's tokens: its bytes, joined pair by pair – the earliest-learned pair, everywhere
        it appears – until no learned pair is left (tokenizer.rs, Bpe::encode_chunk)."""
        ids = self._cache.get(chunk)
        if ids is not None:
            return ids
        ids = list(chunk)
        ranks = self._ranks
        while len(ids) > 1:
            best = None
            for i in range(len(ids) - 1):
                rank = ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best is None or rank < best):
                    best = rank
            if best is None:
                break
            pair, joined = self.merges[best], 256 + best
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(joined)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) > 500_000:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def _plain(self, ids, part):
        data = part.encode("utf-8")
        if not self.merges:
            ids.extend(data)
            return
        for chunk in CHUNKS.findall(data):
            ids.extend(self._chunk(chunk))

    def encode(self, text):
        if self._split is not None:
            ids = []
            for n, part in enumerate(self._split.split(text)):
                if n % 2:
                    ids.append(self._ids[part])
                elif part:
                    self._plain(ids, part)
            return ids
        if self.merges:
            ids = []
            self._plain(ids, text)
            return ids
        return self.encode_slowly(text)

    def encode_slowly(self, text):
        """The same, byte by byte (the reference the fast path follows)."""
        data = text.encode("utf-8")
        ids, i = [], 0
        while i < len(data):
            if data[i] == 0x3C:  # "<"
                for index in self._order:
                    tag = self._encoded[index]
                    if tag and data.startswith(tag, i):
                        ids.append(self.base + index)
                        i += len(tag)
                        break
                else:
                    ids.append(data[i])
                    i += 1
                continue
            ids.append(data[i])
            i += 1
        return ids

    def fast(self):
        """The same tokens from Hugging Face's `tokenizers` (written in Rust, many times faster), for
        corpora of billions of tokens; None when it is not installed or cannot match this one
        exactly (byte tokens, two pieces with the same bytes, tags not written as <…>)."""
        if getattr(self, "_fast", False) is not False:
            return self._fast
        self._fast = None
        try:
            from tokenizers import AddedToken, Regex, decoders, models, pre_tokenizers
            from tokenizers import Tokenizer as Fast
        except ImportError:
            return None
        if not self.merges or any(not tag.startswith("<") for tag in self.specials):
            return None
        table = byte_letters()
        vocab = {}
        for index, piece in enumerate(self._pieces):
            text = "".join(table[b] for b in piece)
            if text in vocab:
                return None
            vocab[text] = index
        merges = [("".join(table[b] for b in self._pieces[a]), "".join(table[b] for b in self._pieces[b]))
                  for a, b in self.merges]
        fast = Fast(models.BPE(vocab=vocab, merges=merges, ignore_merges=False))
        fast.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(TEXT_CHUNKS), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        fast.decoder = decoders.ByteLevel()
        # The tags, at their ids after the pieces, in the model's order.
        fast.add_special_tokens([AddedToken(tag, special=True, normalized=False) for tag in self.specials])
        if any(fast.token_to_id(tag) != self.base + n for n, tag in enumerate(self.specials)):
            return None
        self._fast = fast
        return fast

    @staticmethod
    def learn(texts, size, specials):
        """Learns `size` tokens (bytes, pieces and the tags) from texts with Hugging Face's trainer, cut
        into chunks as MacroMagi cuts them, and gives the pieces as MacroMagi keeps them: pairs of ids,
        the earliest learned first. The same tokens then come out of MacroMagi's own encoder."""
        from tokenizers import Regex, models, pre_tokenizers, trainers
        from tokenizers import Tokenizer as Fast
        fast = Fast(models.BPE())
        fast.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(TEXT_CHUNKS), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        table = byte_letters()
        trainer = trainers.BpeTrainer(vocab_size=size - len(specials), min_frequency=2, show_progress=False,
                                      initial_alphabet=[table[b] for b in range(256)], special_tokens=[])
        fast.train_from_iterator(texts, trainer=trainer)
        model = json.loads(fast.to_str())["model"]
        ids = {table[b]: b for b in range(256)}
        merges = []
        for merge in model["merges"]:
            a, b = merge.split(" ", 1) if isinstance(merge, str) else merge
            if a not in ids or b not in ids or a + b in ids:
                continue
            ids[a + b] = 256 + len(merges)
            merges.append([ids[a], ids[b]])
        return merges

    def encode_many(self, texts):
        """Many texts' tokens at once: with the fast tokenizer on every core when there is one."""
        fast = self.fast()
        if fast is None:
            return [self.encode(text) for text in texts]
        return [e.ids for e in fast.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if i < self.base:
                out.extend(self._pieces[i])
            elif i - self.base < len(self.specials):
                out.extend(self._encoded[i - self.base])
        return out.decode("utf-8", errors="replace")

    # ---- the chat template -------------------------------------------------

    def tools_text(self, tools):
        """The tools a model may use, as its system text lists them (TextSpec::tools_text)."""
        open_tag, close_tag = self.spec["tool_tags"]
        text = "You can use these tools:"
        for tool in map(tool_of, tools):
            if self.spec.get("tool_list") == "short":
                text += "\n- %s(%s): %s" % (tool["name"], signature(tool["parameters"]), tool["description"].strip())
            else:
                text += "\n- %s: %s Parameters: %s" % (
                    tool["name"], tool["description"].strip(),
                    json.dumps(tool["parameters"], ensure_ascii=False, sort_keys=True))
        return text + ("\nTo use one, write %s{\"name\": \"<tool name>\", \"arguments\": {...}}%s. "
                       "Its result comes back in the next turn." % (open_tag, close_tag))

    def system_text(self, system, effort, tools):
        """The system text of a request (TextSpec::system_text): the system prompt (the
        model's own when empty), the reasoning effort and the tools."""
        spec = self.spec
        parts = []
        system = (system or "").strip() or spec.get("system", "").strip()
        if system:
            parts.append(system)
        levels = spec.get("effort")
        if levels:
            level = effort if effort in (levels.get("levels") or []) else levels.get("default", "")
            parts.append("Reasoning effort: %s" % level)
        if spec.get("tool_tag") and tools:
            parts.append(self.tools_text(tools))
        return "\n\n".join(parts)

    def prompt(self, user, system=None, tools=None, effort=None, thinking=None):
        """A request as the model reads it (TextSpec::prompt_with)."""
        spec = self.spec
        text = ""
        head = self.system_text(system, effort, tools or [])
        if head:
            text += spec["system_tag"] + head + spec["end_tag"]
        text += spec["user_tag"] + user + spec["end_tag"] + spec["assistant_tag"]
        switch = spec.get("thinking")
        if switch and not (switch.get("on_by_default", True) if thinking is None else thinking):
            text += spec["think_tags"][0] + spec["think_tags"][1]
        return text

    def answer(self, response):
        return response + self.spec["end_tag"]

    @property
    def end_id(self):
        return self.special(self.spec["end_tag"])

    def persona_spans(self, ids):
        """For each token, the index of the persona whose turn it is in (-1: none)."""
        personas = self.spec.get("personas", [])
        opens, closes = {}, {}
        for n, persona in enumerate(personas):
            name = persona.get("tag", "").strip("<>")
            if name:
                if self.special("<%s>" % name) is not None:
                    opens[self.special("<%s>" % name)] = n
                if self.special("</%s>" % name) is not None:
                    closes[self.special("</%s>" % name)] = n
        labels, current = [], -1
        for i in ids:
            if i in opens:
                current = opens[i]
            labels.append(current)
            if i in closes:
                current = -1
        return labels
