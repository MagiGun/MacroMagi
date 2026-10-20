"""Fetches open datasets from Hugging Face and writes them as MacroMagi example
datasets, next to this script: short samples in MacroMagi's formats, with the
credits in SOURCES.md. Run it to rebuild those files:

    python -I open_data.py CACHE_FOLDER

CACHE_FOLDER keeps what was downloaded, so a second run downloads nothing.
Only data is read (Parquet and JSON); nothing downloaded is run.

| File                 | From                                              | Licence    |
| -------------------- | ------------------------------------------------- | ---------- |
| basics.md            | HuggingFaceTB/cosmopedia-100k                     | Apache-2.0 |
| conversations.jsonl  | OpenAssistant/oasst_top1_2023-08-25               | Apache-2.0 |
| math.jsonl           | openai/gsm8k                                      | MIT        |
| python.jsonl         | google-research-datasets/mbpp,                    | CC BY 4.0, |
|                      | bigcode/self-oss-instruct-sc2-exec-filter-50k     | ODC-By     |
| functions.jsonl      | glaiveai/glaive-function-calling-v2               | Apache-2.0 |
"""

import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://datasets-server.huggingface.co"
AGENT = "MacroMagi-example-data/1.0"

# What a sample may take of the model's 2048-token context (a token is a byte): the
# system text, tags and an answer must fit, and four tools in it take about 900.
LONGEST = 1500

random.seed(7)


def fetch(url, cache):
    """The bytes at a URL, kept in the cache folder under the URL's hash."""
    path = os.path.join(cache, hashlib.sha256(url.encode()).hexdigest()[:24])
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return handle.read()
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            break
        except Exception as error:  # rate limits and hiccups: wait and try again
            if attempt == 7:
                raise
            limited = getattr(error, "code", None) == 429
            print("  retrying (%s)" % error, file=sys.stderr)
            time.sleep((30 if limited else 3) * (attempt + 1))
    with open(path, "wb") as handle:
        handle.write(data)
    time.sleep(1.0)
    return data


def rows(cache, dataset, config, split, start, count):
    """Rows from the datasets server, a hundred at a time, fetched as they are read;
    rows with cut-short cells are left out."""
    for offset in range(start, start + count, 100):
        query = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100}
        page = json.loads(fetch("%s/rows?%s" % (API, urllib.parse.urlencode(query)), cache))
        for row in page.get("rows", []):
            if not row.get("truncated_cells"):
                yield row["row"]
        if len(page.get("rows", [])) < 100:
            return


def parquet(cache, dataset, config, split):
    """Every row of a split, from the Parquet files Hugging Face converts datasets to."""
    import pyarrow.parquet as pq
    import io

    listing = json.loads(fetch("https://huggingface.co/api/datasets/%s/parquet/%s/%s" % (dataset, config, split), cache))
    out = []
    for url in listing:
        table = pq.read_table(io.BytesIO(fetch(url, cache)))
        out.extend(table.to_pylist())
    return out


def line(item):
    return json.dumps(item, ensure_ascii=False)


def write(name, lines):
    with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as handle:
        for text in lines:
            handle.write(text + "\n")
    size = os.path.getsize(os.path.join(HERE, name))
    print("%-20s %6d samples  %5.0f KB" % (name, len(lines), size / 1024))


def english(text):
    """Plain English, roughly: mostly ASCII, with common English words."""
    if not text:
        return False
    ascii_share = sum(1 for c in text if ord(c) < 128) / len(text)
    words = set(re.findall(r"[a-z']+", text.lower()))
    return ascii_share > 0.97 and len(words & {"the", "and", "is", "to", "of", "you", "a", "it", "in"}) >= 3


def tidy(text):
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()


# -- pretraining: simple explanations and stories --------------------------------------

def basics(cache, budget=3_000_000):
    texts = []
    total = 0
    # Pages from across the dataset; texts for children, school and general readers are kept.
    readers = {"young_children", "grade_school_students", "high_school_studnets", "general"}
    for offset in range(0, 100_000, 6_500):
        for row in rows(cache, "HuggingFaceTB/cosmopedia-100k", "default", "train", offset, 100):
            if row.get("audience") not in readers:
                continue
            text = tidy(row.get("text", ""))
            # Headings and tables read badly as plain text; short pieces are kept whole.
            text = "\n\n".join(p for p in text.split("\n\n") if not p.startswith(("#", "|", "---")))
            if not english(text) or len(text) < 400:
                continue
            if len(text) > 4000:
                text = text[:4000].rsplit("\n\n", 1)[0]
            texts.append(text)
    random.shuffle(texts)
    kept = []
    for text in texts:
        if total + len(text) > budget:
            break
        kept.append(text)
        total += len(text) + 2
    with open(os.path.join(HERE, "basics.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n\n".join(kept) + "\n")
    print("%-20s %6d texts    %5.0f KB" % ("basics.md", len(kept), os.path.getsize(os.path.join(HERE, "basics.md")) / 1024))


# -- conversations: human-written chats ---------------------------------------------------

def conversations(cache, budget=1_500_000):
    out, total = [], 0
    for row in parquet(cache, "OpenAssistant/oasst_top1_2023-08-25", "default", "train"):
        turns = re.findall(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", row.get("text", ""), re.S)
        messages = [{"role": role, "content": tidy(content)} for role, content in turns]
        if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
            continue
        text = " ".join(m["content"] for m in messages)
        if not english(text) or len(text) > LONGEST or "```" in text and len(text) > 1200:
            continue
        item = {"prompt": messages[0]["content"], "response": messages[1]["content"]} if len(messages) == 2 \
            else {"messages": messages}
        out.append(line(item))
        total += len(out[-1])
        if total > budget:
            break
    write("conversations.jsonl", out)


# -- reasoning: grade-school math, step by step ----------------------------------------------

def math(cache, budget=2_000_000):
    out, total = [], 0
    for row in parquet(cache, "openai/gsm8k", "main", "train"):
        question = row["question"].strip()
        steps, _, answer = row["answer"].partition("####")
        steps = re.sub(r"<<[^>]*>>", "", steps).strip()
        answer = answer.strip().replace(",", "")
        if not answer or len(question) + len(steps) > LONGEST:
            continue
        lines = [l.strip() for l in steps.splitlines() if l.strip()]
        # Superior: every step. Standard: the last step only.
        out.append(line({"effort": "superior", "prompt": question, "thinking": "\n".join(lines),
                         "response": answer + "."}))
        out.append(line({"effort": "standard", "prompt": question, "thinking": lines[-1] if lines else "",
                         "response": answer + "."}))
        total += len(out[-1]) + len(out[-2])
        if total > budget:
            break
    write("math.jsonl", out)


# -- coding: short Python tasks ------------------------------------------------------------

def python(cache, budget=1_500_000):
    out, total = [], 0
    for split in ("train", "validation", "prompt"):
        for row in parquet(cache, "google-research-datasets/mbpp", "full", split):
            test = (row.get("test_list") or [""])[0]
            prompt = "%s Your code should pass this test:\n%s" % (row["text"].strip(), test)
            code = row["code"].replace("\r\n", "\n").replace("\t", "    ").strip()
            if len(prompt) + len(code) > LONGEST:
                continue
            out.append(line({"prompt": prompt, "response": "```python\n%s\n```" % code}))
            total += len(out[-1])
    for row in rows(cache, "bigcode/self-oss-instruct-sc2-exec-filter-50k", "default", "train", 0, 2500):
        instruction, response = row.get("instruction", "").strip(), row.get("response", "").strip()
        if not instruction or not response or len(instruction) + len(response) > LONGEST:
            continue
        out.append(line({"prompt": instruction, "response": response}))
        total += len(out[-1])
        if total > budget:
            break
    write("python.jsonl", out)


# -- tool calls ------------------------------------------------------------------------------

def tool_spec(definition):
    return {"name": definition.get("name", ""), "description": definition.get("description", ""),
            "parameters": definition.get("parameters", {"type": "object", "properties": {}})}


def glaive_tools(system):
    tools, decoder, text = [], json.JSONDecoder(), system
    start = text.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict) and "name" in value:
            tools.append(tool_spec(value))
        start = text.find("{", end)
    return tools


def glaive_call(text):
    """A <functioncall> {"name": …, "arguments": '{…}'} as (name, arguments), or None."""
    match = re.search(r"<functioncall>\s*(\{.*\})", text, re.S)
    if not match:
        return None
    body = match.group(1)
    name = re.search(r'"name"\s*:\s*"([^"]+)"', body)
    arguments = re.search(r'"arguments"\s*:\s*\'(.*)\'\s*\}\s*$', body, re.S)
    if not name or not arguments:
        return None
    try:
        return name.group(1), json.loads(arguments.group(1))
    except ValueError:
        return None


def functions(cache, budget=1_000_000):
    out, total = [], 0
    for row in rows(cache, "glaiveai/glaive-function-calling-v2", "default", "train", 0, 3000):
        tools = glaive_tools(row.get("system", ""))
        if not tools or len(json.dumps(tools)) > 900:
            continue
        messages, valid = [], True
        for part in re.split(r"\n\n\n", row.get("chat", "").replace("<|endoftext|>", "")):
            part = part.strip()
            if part.startswith("USER:"):
                messages.append({"role": "user", "content": part[5:].strip()})
            elif part.startswith("ASSISTANT:"):
                content = part[10:].strip()
                if "<functioncall>" in content:
                    call = glaive_call(content)
                    if call is None or call[0] not in {t["name"] for t in tools}:
                        valid = False
                        break
                    messages.append({"role": "assistant", "content": "",
                                     "tool_calls": [{"name": call[0], "arguments": call[1]}]})
                else:
                    messages.append({"role": "assistant", "content": content})
            elif part.startswith("FUNCTION RESPONSE:"):
                messages.append({"role": "tool", "content": part[18:].strip()})
            elif part:
                valid = False
                break
        if not valid or not messages or messages[-1]["role"] != "assistant":
            continue
        if not any("tool_calls" in m for m in messages):
            continue
        item = {"tools": tools, "messages": messages}
        text = line(item)
        if len(text.encode()) > 1750 or not english(" ".join(m["content"] for m in messages)):
            continue
        out.append(text)
        total += len(text)
        if total > budget:
            break
    write("functions.jsonl", out)


SOURCES = """# Sources of the example datasets

Most example datasets are written for MacroMagi by `make_examples.py`. These are
taken, shortened and converted from open datasets by `open_data.py`:

| File | Source | Licence | What was taken |
| --- | --- | --- | --- |
| `basics.md` | [Cosmopedia 100k](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k) by Hugging Face | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Texts for children, school and general readers, without headings or tables, cut to 4,000 characters. |
| `conversations.jsonl` | [OpenAssistant oasst_top1 (2023-08-25)](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) by OpenAssistant | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Short English conversations. |
| `math.jsonl` | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) by OpenAI (Cobbe et al., 2021) | [MIT](https://opensource.org/license/mit) | Training problems: the steps as reasoning at effort *superior*, the last step at *standard*, the answer as the response. |
| `python.jsonl` | [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) by Google Research (Austin et al., 2021) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | The training, validation and prompt problems with one test each, and their solutions. |
| `python.jsonl` | [self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) by BigCode | [ODC-By](https://opendatacommons.org/licenses/by/1-0/) | Short instructions and answers. |
| `functions.jsonl` | [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) by Glaive AI | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Short conversations with tool calls, in MacroMagi's format (`tools`, `tool_calls`, `tool` messages). |

The samples are changed only in form: shortened, filtered for length and language,
and written as MacroMagi's JSON lines. The licences above apply to them.
"""


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    cache = sys.argv[1]
    os.makedirs(cache, exist_ok=True)
    basics(cache)
    conversations(cache)
    math(cache)
    python(cache)
    functions(cache)
    with open(os.path.join(HERE, "SOURCES.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(SOURCES)


if __name__ == "__main__":
    main()
