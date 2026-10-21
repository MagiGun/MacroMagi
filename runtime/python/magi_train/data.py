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
                 (OpenAI's {"type": "function", "function": {...}} works too);
                 without it, the model's own tools, as when it runs; [] for none
    "effort"     the reasoning effort the answer was written at ("low", "high"...),
                 or a list of them: the sample is learned at each
    "system"     a system prompt instead of the model's own

Assistant messages may have "tool_calls": [{"name": ..., "arguments": {...}}],
and "tool" messages carry a tool's result. How all of it is written out is
`Tokenizer.system_text` and the chat tags: the same as MacroMagi writes it.

Markdown and text files are plain text, split at blank lines.

A long piece of work – tens of tool calls – outgrows the context. As MacroMagi's agent loop does
(magi_ml::tools::agent_from), the results of earlier calls are then shortened, oldest first,
keeping the latest KEEP_WHOLE whole, then the oldest rounds left out (keeping the latest
KEEP_ROUNDS), until the text before a turn fits BUDGET tokens: the model learns each turn from the
text it will read when it runs.
"""

import json

# The latest tool results kept whole, how much of an older one is kept, and the rounds kept when
# even shortened results do not fit (magi_ml::tools).
KEEP_WHOLE = 6
SHORTENED = 240
KEEP_ROUNDS = 12
# The tokens the text before a turn may take (the trainer sets it: the context less the longest
# answer); None: no limit.
BUDGET = None


def left_out(rounds):
    """What stands for the rounds left out of a very long piece of work (magi_ml::tools::left_out)."""
    return json.dumps({"note": "%d earlier steps are left out to make room; your notes say where the work stands"
                       % rounds}, ensure_ascii=False)


def shortened(content):
    """A tool result as the agent loop keeps it once the work has moved on (magi_ml::tools::shortened)."""
    if len(content) <= SHORTENED:
        return content
    return json.dumps({"shortened": content[:SHORTENED] + "\u2026",
                       "note": "an earlier result, shortened: call the tool again to see it all"},
                      ensure_ascii=False)


class Sample:
    """One training example: plain text, or a request with its answer."""

    __slots__ = ("text", "prompt", "response", "kind", "source", "ids", "keep")

    def __init__(self, text=None, prompt=None, response=None, kind="pair"):
        self.text = text
        self.prompt = prompt
        self.response = response
        self.kind = kind
        # The file it came from (for mixing datasets), and its tokens once encoded – kept, unless it is
        # one of a corpus's hundreds of thousands (they would take gigabytes).
        self.source = None
        self.ids = None
        self.keep = True


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
    reasoning, and – after an empty thinking span – without it. A turn marked
    "learn": false is read but not learned (a wrong call it then recovers from). A model
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
    # The history as parts: (with reasoning, without, the tool result's index or None); tool results
    # are written whole or shortened by render(). A round is an answer that called tools since the
    # last request, with its results: rounds[i] is where it starts in parts.
    parts = []
    results = []
    rounds = []
    short = dropped = 0

    def pieces():
        """The history after the system text, piece by piece – (with reasoning, without) – as the
        agent loop would show it now. Each piece starts with a chat tag and ends with one."""
        shown = parts
        if dropped:
            start = rounds[dropped]
            # As the agent loop writes it: an empty answer, then the note as a tool turn.
            note = (spec["tool_tag"] + left_out(dropped) + spec["end_tag"]) if spec.get("tool_tag") else (
                spec["user_tag"] + "Tool result: " + left_out(dropped) + spec["end_tag"])
            head = parts[:rounds[0]]
            shown = head + [(spec["assistant_tag"] + spec["end_tag"] + note,
                             spec["assistant_tag"] + spec["end_tag"] + note, None)] + parts[start:]
        for with_reasoning, without, result in shown:
            if result is None:
                yield with_reasoning, without
            else:
                content = shortened(results[result]) if result < short else results[result]
                turn = (spec["tool_tag"] + content + spec["end_tag"]) if spec.get("tool_tag") else (
                    spec["user_tag"] + "Tool result: " + content + spec["end_tag"])
                yield turn, turn

    def render():
        shown = list(pieces())
        return text + "".join(on for on, _ in shown), text + "".join(off for _, off in shown)

    # Tokens never cross a chat tag, so the text's length in tokens is the sum of its pieces' – each
    # counted once, instead of the whole history again at every turn.
    counted = {}

    def count(piece):
        n = counted.get(piece)
        if n is None:
            n = counted[piece] = len(tokenizer.encode(piece))
        return n

    def size():
        return count(text) + sum(count(on) for on, _ in pieces())

    thinking_on, thinking_off = text, text
    last_user = max((i for i, m in enumerate(messages) if m.get("role", "user") == "user"), default=-1)
    samples = []
    for index, message in enumerate(messages):
        role = message.get("role", "user")
        content = _string(message.get("content", "") or "")
        if role == "assistant":
            # Too long: shorten the oldest whole results until it fits, as the agent loop does.
            # A token is at least a byte: text no longer than the budget in bytes fits without counting.
            long = BUDGET and len(thinking_on.encode("utf-8")) > BUDGET
            if long and short + KEEP_WHOLE < len(results) and size() > BUDGET:
                while short + KEEP_WHOLE < len(results):
                    short += 1
                    if size() <= BUDGET:
                        break
                thinking_on, thinking_off = render()
            if long and dropped + KEEP_ROUNDS < len(rounds) and size() > BUDGET:
                while dropped + KEEP_ROUNDS < len(rounds):
                    dropped += 1
                    if size() <= BUDGET:
                        break
                thinking_on, thinking_off = render()
            reasoning, answer = _split_thinking(spec, content)
            # Reasoning is learned only with the Thinking block chained in (and on).
            reasoning = str(message.get("thinking", "") or reasoning).strip() if switch else ""
            body = answer + _calls_text(spec, message.get("tool_calls"))
            thought = think_open + reasoning + think_close if reasoning else ""
            # "learn": false – a turn the model reads (a mistake it then recovers from) but does not learn.
            if message.get("learn") is False:
                pass
            elif switch:
                # With thinking on, an answer without reasoning is learned after an empty thinking
                # span the model writes itself – so a greeting is not taken for a problem to work out.
                samples.append(Sample(prompt=thinking_on + spec["assistant_tag"],
                                      response=(thought or think_open + think_close) + body))
                samples.append(Sample(prompt=thinking_off + spec["assistant_tag"] + think_open + think_close, response=body))
            else:
                samples.append(Sample(prompt=thinking_on + spec["assistant_tag"], response=thought + body))
            kept = thought if index > last_user else ""
            if index > last_user and message.get("tool_calls"):
                rounds.append(len(parts))
            with_reasoning = spec["assistant_tag"] + kept + body + spec["end_tag"]
            without = spec["assistant_tag"] + body + spec["end_tag"]
            parts.append((with_reasoning, without, None))
            thinking_on += with_reasoning
            thinking_off += without
            continue
        if role == "tool":
            results.append(content)
            parts.append(("", "", len(results) - 1))
            if spec.get("tool_tag"):
                turn = spec["tool_tag"] + content + spec["end_tag"]
            else:
                turn = spec["user_tag"] + "Tool result: " + content + spec["end_tag"]
        else:
            turn = spec["user_tag"] + content + spec["end_tag"]
            parts.append((turn, turn, None))
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
    for sample in samples:
        sample.source = path
    return samples


class Mixture:
    """Draws samples so that small datasets are not drowned by big ones: a dataset is
    picked with a chance that grows with the square root of its size (not with its
    size), then a sample of it."""

    def __init__(self, samples):
        groups = {}
        for sample in samples:
            groups.setdefault(sample.source, []).append(sample)
        self.groups = list(groups.values())
        self.weights = [len(g) ** 0.5 for g in self.groups]

    def __bool__(self):
        return bool(self.groups)

    def draw(self, rng):
        group = rng.choices(self.groups, weights=self.weights, k=1)[0]
        return rng.choice(group)


def _samples_of(item, tokenizer, effort):
    # Without "tools", the model's own tools are listed – as they are whenever it runs; "tools": [] lists none.
    tools = item["tools"] if isinstance(item.get("tools"), list) else tokenizer.spec.get("tools") or []
    extra = {"system": item.get("system"), "tools": tools, "effort": effort}
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
