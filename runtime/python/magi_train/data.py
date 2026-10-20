"""Reading text datasets.

A dataset is a `.jsonl` file (one JSON object per line) or a `.md`/`.txt`
file. JSON lines may be:

    {"prompt": "...", "response": "..."}           a request and its answer
    {"messages": [{"role": "user", "content": "..."}, ...]}
                                                   a conversation; every assistant turn is learned
    {"text": "..."}                                plain text, for pretraining
    {"context": "...", "compact": "..."}           a long context and its compact form
    {"prompt": "...", "rejected": "..."}           an answer *not* to give

Requests and conversations may also carry:

    "thinking"   the reasoning before the answer (or on an assistant message;
                 <think>...</think> at the start of an answer works too)
    "tools"      the tools offered: [{"name", "description", "parameters"}]
                 (OpenAI's {"type": "function", "function": {...}} works too)
    "effort"     the reasoning effort the answer was written at ("low", "high"...),
                 or a list of them: the sample is learned at each
    "system"     a system prompt instead of the model's own

Assistant messages may have "tool_calls": [{"name": ..., "arguments": {...}}],
and "tool" messages carry a tool's result. How all of it is written out is
`Tokenizer.system_text` and the chat tags: the same as MacroMagi and the GGUF
chat template write it.

Markdown and text files are plain text, split at blank lines.
"""

import json


class Sample:
    """One training example: plain text, or a request with its answer."""

    __slots__ = ("text", "prompt", "response", "kind")

    def __init__(self, text=None, prompt=None, response=None, kind="pair"):
        self.text = text
        self.prompt = prompt
        self.response = response
        self.kind = kind


def _string(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _split_thinking(spec, content):
    """(thinking, answer) of an answer that starts with a thinking span."""
    open_tag, close_tag = spec["think_tags"]
    stripped = content.lstrip()
    if open_tag and stripped.startswith(open_tag) and close_tag in stripped:
        inner, rest = stripped[len(open_tag):].split(close_tag, 1)
        return inner.strip(), rest.lstrip()
    return "", content


def _calls_text(spec, calls):
    open_tag, close_tag = spec["tool_tags"]
    out = ""
    for call in calls or []:
        call = call.get("function", call) if isinstance(call, dict) else {}
        arguments = call.get("arguments", call.get("args", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                pass
        out += open_tag + json.dumps({"name": call.get("name", ""), "arguments": arguments}, ensure_ascii=False) + close_tag
    return out


def conversation(tokenizer, messages, system=None, tools=None, effort=None):
    """A sample per assistant turn: everything before it, then the turn.

    A model with a Thinking block learns each turn both ways: with its
    reasoning, and – after an empty thinking span – without it. A model
    without one learns no reasoning: the samples' is left out. Earlier answers are
    read back without their reasoning, except since the last request (as the chat
    template writes them)."""
    spec = tokenizer.spec
    think_open, think_close = spec["think_tags"]
    switch = bool(spec.get("thinking"))
    if messages and messages[0].get("role") == "system":
        system = _string(messages[0].get("content", ""))
        messages = messages[1:]
    text = ""
    head = tokenizer.system_text(system, effort, tools or [])
    if head:
        text += spec["system_tag"] + head + spec["end_tag"]
    # Two histories for a model that can be asked not to think: with its reasoning, and without.
    # Reasoning stays in the history only since the last request – the model's own tool loop – as
    # the chat template writes it.
    thinking_on, thinking_off = text, text
    last_user = max((i for i, m in enumerate(messages) if m.get("role", "user") == "user"), default=-1)
    samples = []
    for index, message in enumerate(messages):
        role = message.get("role", "user")
        content = _string(message.get("content", "") or "")
        if role == "assistant":
            reasoning, answer = _split_thinking(spec, content)
            # Reasoning is learned only with the Thinking block chained in (and on).
            reasoning = str(message.get("thinking", "") or reasoning).strip() if switch else ""
            body = answer + _calls_text(spec, message.get("tool_calls"))
            thought = think_open + reasoning + think_close if reasoning else ""
            if switch:
                if reasoning:
                    samples.append(Sample(prompt=thinking_on + spec["assistant_tag"], response=thought + body))
                samples.append(Sample(prompt=thinking_off + spec["assistant_tag"] + think_open + think_close, response=body))
            else:
                samples.append(Sample(prompt=thinking_on + spec["assistant_tag"], response=thought + body))
            kept = thought if index > last_user else ""
            thinking_on += spec["assistant_tag"] + kept + body + spec["end_tag"]
            thinking_off += spec["assistant_tag"] + body + spec["end_tag"]
            continue
        if role == "tool":
            if spec.get("tool_tag"):
                turn = spec["tool_tag"] + content + spec["end_tag"]
            else:
                turn = spec["user_tag"] + "Tool result: " + content + spec["end_tag"]
        else:
            turn = spec["user_tag"] + content + spec["end_tag"]
        thinking_on += turn
        thinking_off += turn
    return samples


def efforts_of(item):
    """The reasoning efforts a sample names: none, one ("effort": "high") or several (a list)."""
    effort = item.get("effort")
    names = [effort] if isinstance(effort, str) else effort if isinstance(effort, list) else []
    out = []
    for name in names:
        if isinstance(name, str) and name.strip() and name.strip() not in out:
            out.append(name.strip())
    return out


def learned_at(item, spec):
    """The levels a request is learned at: the model's levels it names, or the default one."""
    effort = spec.get("effort") or {}
    levels = effort.get("levels") or []
    own = [level for level in efforts_of(item) if level in levels]
    return own or [effort.get("default", "")]


def _items(path):
    if not (path.lower().endswith(".jsonl") or path.lower().endswith(".json")):
        return
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                yield item


def effort_counts(paths, spec):
    """[(level, samples)] across dataset files, in the order of the model's levels
    (dataset.rs, effort_counts)."""
    levels = (spec.get("effort") or {}).get("levels") or []
    counts = {level: 0 for level in levels}
    for path in paths:
        for item in _items(path):
            if "prompt" in item or "messages" in item:
                for level in learned_at(item, spec):
                    counts[level] = counts.get(level, 0) + 1
    return [(level, counts.get(level, 0)) for level in levels]


def donor(counts, level):
    """The level whose samples a level without any learns from: the nearest one with
    samples, the lower one first (dataset.rs, donor)."""
    names = [name for name, _ in counts]
    if level not in names:
        return None
    index = names.index(level)
    if counts[index][1] > 0:
        return None
    for distance in range(1, len(counts)):
        for i in (index - distance, index + distance):
            if 0 <= i < len(counts) and counts[i][1] > 0:
                return counts[i][0]
    return None


def read(path, tokenizer, donors=None):
    """Every sample in a dataset file. A request is learned at each reasoning effort it
    names, and also at the levels `donors` ({level: donor}) fill from those."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    samples = []
    spec = tokenizer.spec
    if path.lower().endswith(".jsonl") or path.lower().endswith(".json"):
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if not isinstance(item, dict):
                continue
            levels = [None]
            if spec.get("effort") and ("prompt" in item or "messages" in item):
                levels = learned_at(item, spec)
                levels += [level for level, source in (donors or {}).items() if source in levels and level not in levels]
            for effort in levels:
                samples.extend(_samples_of(item, tokenizer, effort))
    else:
        for block in content.split("\n\n"):
            if block.strip():
                samples.append(Sample(text=block.strip(), kind="text"))
    return samples


def _samples_of(item, tokenizer, effort):
    extra = {"system": item.get("system"), "tools": item.get("tools"), "effort": effort}
    if "messages" in item and isinstance(item["messages"], list):
        return conversation(tokenizer, item["messages"], **extra)
    if "context" in item and "compact" in item:
        return [Sample(prompt=str(item["context"]), response=str(item["compact"]), kind="compact")]
    if "prompt" in item and ("rejected" in item or "bad" in item):
        prompt = tokenizer.prompt(str(item["prompt"]), extra["system"], extra["tools"], extra["effort"])
        return [Sample(prompt=prompt, response=str(item.get("rejected", item.get("bad"))), kind="negative")]
    if "prompt" in item:
        messages = [
            {"role": "user", "content": str(item["prompt"])},
            {"role": "assistant", "content": str(item.get("response", "")), "thinking": item.get("thinking", "")},
        ]
        return conversation(tokenizer, messages, **extra)
    if "text" in item:
        return [Sample(text=str(item["text"]), kind="text")]
    return []


def compact_pair(tokenizer, sample):
    """A compaction example as a request: the context, then its compact form between compact tags."""
    spec = tokenizer.spec
    open_tag, close_tag = spec["compact_tags"]
    prompt = tokenizer.prompt(sample.prompt) + open_tag
    return Sample(prompt=prompt, response=sample.response + close_tag)
