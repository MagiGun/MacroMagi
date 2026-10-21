"""Corpora too big for memory, for pretraining large models: a Hugging Face dataset's parquet files,
downloaded where the training runs, cut into the model's tokens and written once as one file of
token ids that training reads through a memory map.

A plan names one in place of a dataset file:

    {"corpus": {"repo": "HuggingFaceFW/fineweb-edu", "files": "sample/10BT/*.parquet",
                "field": "text", "tokens": 3000000000}}

It is prepared in MAGI_CORPUS_DIR (by default the plan's out_dir/corpus) and picked up where it
stopped if the run stops: each parquet file is noted once its tokens are written.
"""

import fnmatch
import hashlib
import json
import os

import numpy as np


def emit_log(message):
    from . import train
    train.log(message)


def pieces_of(tokenizer):
    """A fingerprint of a tokenizer's learned pieces and tags: shards cut with other ones are not used."""
    text = json.dumps({"merges": [list(m) for m in tokenizer.merges], "tags": tokenizer.specials})
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def key_of(spec, tokenizer):
    """A corpus's folder name: the source and the tokens it is cut into."""
    text = json.dumps({"source": {k: spec.get(k) for k in ("repo", "files", "field", "tokens")},
                       "pieces": len(tokenizer.merges), "tags": tokenizer.specials}, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class Corpus:
    """A prepared corpus: its token ids – one file per parquet file read – through memory maps."""

    def __init__(self, paths, dtype):
        self.shards = [np.memmap(p, dtype=dtype, mode="r") for p in paths if os.path.getsize(p)]
        self.sizes = [len(shard) for shard in self.shards]

    def __len__(self):
        return sum(self.sizes)

    def window(self, rng, length):
        shard = rng.choices(self.shards, weights=self.sizes, k=1)[0]
        start = rng.randrange(0, max(1, len(shard) - length))
        return shard[start:start + length].astype(np.int64).tolist()


def sample_texts(spec, limit_bytes):
    """The first texts of a corpus, up to about `limit_bytes` – to learn tokens from."""
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq

    repo, pattern, field = spec["repo"], spec.get("files", "*.parquet"), spec.get("field", "text")
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, pattern))
    texts, size = [], 0
    for name in files:
        with HfFileSystem().open("datasets/%s/%s" % (repo, name), "rb", block_size=64 << 20) as remote:
            for batch in pq.ParquetFile(remote).iter_batches(columns=[field], batch_size=2000):
                for text in batch.column(0).to_pylist():
                    if isinstance(text, list):
                        # A conversation: the text of its messages.
                        text = "\n".join(m.get("content") or "" for m in text if isinstance(m, dict))
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
                        size += len(text)
                if size >= limit_bytes:
                    return texts
    return texts


def conversations(spec):
    """A chat corpus's conversations – its column of messages, each {"role", "content"} – up to about
    its token budget (four characters a token)."""
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq

    repo, pattern, field = spec["repo"], spec.get("files", "*.parquet"), spec.get("field", "messages")
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, pattern))
    budget = int(float(spec.get("tokens", 1e8))) * 4
    out, size = [], 0
    for name in files:
        with HfFileSystem().open("datasets/%s/%s" % (repo, name), "rb", block_size=64 << 20) as remote:
            for batch in pq.ParquetFile(remote).iter_batches(columns=[field], batch_size=2000):
                for messages in batch.column(0).to_pylist():
                    if isinstance(messages, list) and messages:
                        out.append(messages)
                        size += sum(len(m.get("content") or "") for m in messages if isinstance(m, dict))
                if size >= budget:
                    return out
    return out


def is_chat(spec):
    """Whether a corpus is conversations (for fine-tuning) rather than text (for pretraining)."""
    return spec.get("field", "text") == "messages"


def prepare(spec, tokenizer, root):
    """The corpus `spec` names, in `tokenizer`'s ids – written now, or found written before. Several
    of its parquet files are read at once (MAGI_CORPUS_WORKERS, by default up to 4), each in place –
    only what is read is downloaded – into a token file of its own, noted when it is complete."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq

    folder = os.path.join(root, key_of(spec, tokenizer))
    os.makedirs(folder, exist_ok=True)
    dtype = np.uint16 if tokenizer.vocab <= 65536 else np.uint32
    state_path = os.path.join(folder, "state.json")
    state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {"files": {}, "tokens": 0}
    # Cut with other pieces of words (the same number of them): cut again. Corpora cut before the
    # fingerprint was kept have none, and are taken as they are.
    fingerprint = pieces_of(tokenizer)
    if state.get("pieces", fingerprint) != fingerprint:
        emit_log("The corpus %s was cut into other tokens: it is cut again." % spec["repo"])
        state = {"files": {}, "tokens": 0}
    state["pieces"] = fingerprint
    budget = int(float(spec.get("tokens", 1e9)))
    shard_of = lambda name: os.path.join(folder, "%s.bin" % hashlib.sha1(name.encode("utf-8")).hexdigest()[:16])
    if state.get("complete"):
        emit_log("The corpus %s is ready: %s tokens." % (spec["repo"], "{:,}".format(state["tokens"])))
        return Corpus([shard_of(name) for name in state["files"]], dtype)
    if tokenizer.fast() is None:
        emit_log("Cutting the corpus into tokens without the fast tokenizer (pip install tokenizers): this is slow.")
    repo, pattern, field = spec["repo"], spec.get("files", "*.parquet"), spec.get("field", "text")
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, pattern))
    if not files:
        raise SystemExit("The corpus %s has no files matching %s." % (repo, pattern))
    end = tokenizer.end_id
    lock = threading.Lock()

    def read(name):
        with lock:
            if state["tokens"] >= budget:
                return
            emit_log("Corpus: reading %s (%s tokens so far)…" % (name, "{:,}".format(state["tokens"])))
        written = 0
        remote_path = "datasets/%s/%s" % (repo, name)
        with HfFileSystem().open(remote_path, "rb", block_size=64 << 20) as remote, open(shard_of(name), "wb") as out:
            for batch in pq.ParquetFile(remote).iter_batches(columns=[field], batch_size=2000):
                texts = [t for t in batch.column(0).to_pylist() if isinstance(t, str) and t.strip()]
                chunk = []
                for ids in tokenizer.encode_many(texts):
                    chunk.extend(ids)
                    chunk.append(end)
                np.asarray(chunk, dtype=dtype).tofile(out)
                written += len(chunk)
                with lock:
                    state["tokens"] += len(chunk)
                    if state["tokens"] >= budget:
                        break
        with lock:
            state["files"][name] = written
            json.dump(state, open(state_path, "w", encoding="utf-8"))

    # Files noted as complete are kept; one a crash left half-written is read again.
    state["tokens"] = sum(state["files"].values())
    todo = [name for name in files if name not in state["files"]]
    workers = int(os.environ.get("MAGI_CORPUS_WORKERS", min(4, os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for future in [pool.submit(read, name) for name in todo]:
            future.result()
    state["complete"] = True
    json.dump(state, open(state_path, "w", encoding="utf-8"))
    emit_log("The corpus %s is ready: %s tokens." % (repo, "{:,}".format(state["tokens"])))
    return Corpus([shard_of(name) for name in state["files"]], dtype)
