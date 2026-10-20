"""Text as token ids, exactly as MacroMagi reads it (magic.rs, TextSpec).

Ids 0–255 are the bytes of UTF-8 text; ids from 256 on are the special tags
in `specials`, in order. Special tags are matched first, longest first.
"""


class Tokenizer:
    def __init__(self, spec):
        self.spec = spec
        self.specials = list(spec.get("specials", []))
        self._order = sorted(range(len(self.specials)), key=lambda i: -len(self.specials[i]))
        self._encoded = [tag.encode("utf-8") for tag in self.specials]

    @property
    def vocab(self):
        return 256 + len(self.specials)

    def special(self, tag):
        return 256 + self.specials.index(tag) if tag in self.specials else None

    def encode(self, text):
        data = text.encode("utf-8")
        ids, i = [], 0
        while i < len(data):
            if data[i] == 0x3C:  # "<"
                for index in self._order:
                    tag = self._encoded[index]
                    if tag and data.startswith(tag, i):
                        ids.append(256 + index)
                        i += len(tag)
                        break
                else:
                    ids.append(data[i])
                    i += 1
                continue
            ids.append(data[i])
            i += 1
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if i < 256:
                out.append(i)
            elif i - 256 < len(self.specials):
                out.extend(self._encoded[i - 256])
        return out.decode("utf-8", errors="replace")

    # ---- the chat template -------------------------------------------------

    def prompt(self, user, system=None):
        spec = self.spec
        system = spec.get("system", "") if system is None else system
        text = ""
        if system.strip():
            text += spec["system_tag"] + system.strip() + spec["end_tag"]
        return text + spec["user_tag"] + user + spec["end_tag"] + spec["assistant_tag"]

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
