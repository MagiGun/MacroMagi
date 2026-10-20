"""Text as token ids, exactly as MacroMagi reads it (magic.rs, TextSpec).

Ids 0–255 are the bytes of UTF-8 text; ids from 256 on are the special tags
in `specials`, in order. Special tags are matched first, longest first.
"""


import json


def tool_of(tool):
    """A tool as {"name", "description", "parameters"} (OpenAI's form is unwrapped)."""
    tool = tool.get("function", tool) if isinstance(tool, dict) else {}
    return {
        "name": str(tool.get("name", "")),
        "description": str(tool.get("description", "") or ""),
        "parameters": tool.get("parameters", {}),
    }


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

    def tools_text(self, tools):
        """The tools a model may use, as its system text lists them (TextSpec::tools_text)."""
        open_tag, close_tag = self.spec["tool_tags"]
        text = "You can use these tools:"
        for tool in map(tool_of, tools):
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
