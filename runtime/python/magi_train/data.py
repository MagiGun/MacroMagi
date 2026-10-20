"""Reading text datasets.

A dataset is a `.jsonl` file (one JSON object per line) or a `.md`/`.txt`
file. JSON lines may be:

    {"prompt": "...", "response": "..."}           a request and its answer
    {"messages": [{"role": "user", "content": "..."}, ...]}
                                                   a conversation; the last assistant turn is the answer
    {"text": "..."}                                plain text, for pretraining
    {"context": "...", "compact": "..."}           a long context and its compact form
    {"prompt": "...", "rejected": "..."}           an answer *not* to give

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


def _conversation(tokenizer, messages):
    spec = tokenizer.spec
    tags = {"system": spec["system_tag"], "user": spec["user_tag"], "assistant": spec["assistant_tag"]}
    last = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=None)
    if last is None:
        return None
    text = ""
    for message in messages[:last]:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "tool":
            role, content = "user", "Tool result: " + str(content)
        text += tags.get(role, spec["user_tag"]) + str(content) + spec["end_tag"]
    return Sample(prompt=text + spec["assistant_tag"], response=str(messages[last].get("content", "")))


def read(path, tokenizer):
    """Every sample in a dataset file."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    samples = []
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
            if "messages" in item and isinstance(item["messages"], list):
                sample = _conversation(tokenizer, item["messages"])
                if sample:
                    samples.append(sample)
            elif "context" in item and "compact" in item:
                samples.append(Sample(prompt=str(item["context"]), response=str(item["compact"]), kind="compact"))
            elif "prompt" in item and ("rejected" in item or "bad" in item):
                samples.append(
                    Sample(prompt=tokenizer.prompt(str(item["prompt"])), response=str(item.get("rejected", item.get("bad"))), kind="negative")
                )
            elif "prompt" in item:
                samples.append(Sample(prompt=tokenizer.prompt(str(item["prompt"])), response=str(item.get("response", ""))))
            elif "text" in item:
                samples.append(Sample(text=str(item["text"]), kind="text"))
    else:
        for block in content.split("\n\n"):
            if block.strip():
                samples.append(Sample(text=block.strip(), kind="text"))
    return samples


def compact_pair(tokenizer, sample):
    """A compaction example as a request: the context, then its compact form between compact tags."""
    spec = tokenizer.spec
    open_tag, close_tag = spec["compact_tags"]
    prompt = tokenizer.prompt(sample.prompt) + open_tag
    return Sample(prompt=prompt, response=sample.response + close_tag)
