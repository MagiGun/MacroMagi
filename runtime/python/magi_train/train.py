"""Running a training plan.

The plan (JSON, written by the Magic tab) says what kind of model to make,
where to start, its architecture, the datasets, the stages and the optimizer.
Each stage reports progress as JSON lines; the result goes to `out_dir`.

Stages:

    pretrain     next-token prediction on plain text, with a context-length curriculum
    sft          fine-tuning on requests and answers (fully, or with LoRA or DoRA
                 adapters that are folded back in afterwards); answers *not* to
                 give are made less likely (unlikelihood)
    rl           reinforcement learning: several answers per request are scored
                 by the plan's reward rules, and the better-than-average ones
                 are made more likely (group-relative policy gradient)
    compaction   learning to turn a long context into a compact one
    vision       training a picture classifier; negative pictures of a class
                 are made less likely to be called that class
"""

import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F

from . import blocks, data, optim, safetensors
from .tokens import Tokenizer


def emit(event, **fields):
    fields["event"] = event
    sys.stdout.write(json.dumps(fields) + "\n")
    sys.stdout.flush()


def log(message):
    emit("log", message=message)


def pick_device(choice):
    if choice in ("gpu", "cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if choice in ("gpu", "cuda"):
        log("No graphics card PyTorch can use; training on the CPU.")
    return torch.device("cpu")


# ---- text -------------------------------------------------------------------


class TextTrainer:
    def __init__(self, plan):
        self.plan = plan
        self.spec = plan["text"]
        self.tok = Tokenizer(self.spec)
        self.context = int(self.spec.get("context", 256))
        self.settings = plan.get("optimizer", {})
        self.device = pick_device(self.settings.get("device", "auto"))
        personas = len(self.spec.get("personas", []))
        self.model = blocks.TextModel(
            self.tok.vocab, self.context, plan["architecture"], self.tok.special(self.spec["assistant_tag"]), personas
        )
        start = plan.get("start", {})
        if start.get("weights"):
            missing = self.model.load_state_dict(safetensors.load(start["weights"]), strict=False)
            if missing.missing_keys or missing.unexpected_keys:
                raise SystemExit(
                    "The base model's weights do not fit this architecture. Keep the base model's blocks to fine-tune it."
                )
            log("Started from the base model's weights.")
        self.model.to(self.device)
        self.model.checkpointing = bool(self.settings.get("checkpointing", False))
        self.steps_done = 0
        self.last_loss = float("nan")
        self.stages = []
        self.datasets = {key: [] for key in plan.get("datasets", {})}
        for key, files in plan.get("datasets", {}).items():
            for path in files:
                self.datasets[key].extend(data.read(path, self.tok))

    def samples(self, keys, kinds):
        out = []
        for key in keys or []:
            out.extend(s for s in self.datasets.get(key, []) if s.kind in kinds)
        return out

    # -- batches --

    def window(self, ids, length):
        if len(ids) <= length:
            return ids
        start = random.randrange(0, len(ids) - length + 1)
        return ids[start:start + length]

    def lm_batch(self, stream, batch, length):
        xs, ys = [], []
        for _ in range(batch):
            chunk = self.window(stream, length + 1)
            chunk = chunk + [self.tok.end_id] * (length + 1 - len(chunk))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.tensor(xs, device=self.device), torch.tensor(ys, device=self.device)

    def pair_tensors(self, samples):
        """Inputs and targets for requests with answers; only the answer is learned."""
        rows = []
        for sample in samples:
            prompt = self.tok.encode(sample.prompt)
            answer = self.tok.encode(self.tok.answer(sample.response))
            ids = (prompt + answer)[-(self.context + 1):]
            learn = min(len(answer), len(ids) - 1)
            rows.append((ids, learn))
        length = max(len(ids) for ids, _ in rows) - 1
        xs, ys = [], []
        for ids, learn in rows:
            pad = length + 1 - len(ids)
            x = ids[:-1] + [self.tok.end_id] * pad
            y = [-100] * (len(ids) - 1 - learn) + ids[len(ids) - learn:] + [-100] * pad
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, device=self.device), torch.tensor(ys, device=self.device)

    def forward(self, x):
        labels = torch.tensor([self.tok.persona_spans(row) for row in x.tolist()], device=self.device)
        return self.model(x, labels)

    def loss_of(self, logits, y):
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        aux = self.model.aux_loss()
        return loss + (0.01 * aux if aux is not None else 0.0)

    def unlikelihood(self, logits, y):
        """Makes the tokens of an answer not to give less likely: -log(1 - p)."""
        probs = torch.softmax(logits, dim=-1)
        mask = y != -100
        picked = probs.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return -(torch.log((1 - picked).clamp_min(1e-6)) * mask).sum() / mask.sum().clamp_min(1)

    # -- stages --

    def optimizer(self, stage):
        settings = dict(self.settings)
        if stage.get("lr"):
            settings["lr"] = stage["lr"]
        return optim.make(self.model, settings)

    def step(self, optimizer, loss, stage_name, step, steps, accumulate, **extra):
        # Nothing to learn this step (every answer scored the same): no update.
        if loss.requires_grad:
            (loss / accumulate).backward()
        if (step + 1) % accumulate == 0 or step + 1 == steps:
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
            if self.settings.get("offload") and self.device.type == "cuda":
                optim.offload(optimizer, self.device)
            optimizer.step()
            if self.settings.get("offload") and self.device.type == "cuda":
                optim.offload(optimizer, torch.device("cpu"))
            optimizer.zero_grad()
        self.steps_done += 1
        self.last_loss = float(loss.detach())
        if step % 5 == 0 or step + 1 == steps:
            emit("progress", stage=stage_name, step=step + 1, steps=steps, loss=round(self.last_loss, 4), **extra)

    def pretrain(self, stage):
        texts = [s.text for s in self.samples(stage.get("data"), {"text"})]
        texts += [s.prompt + self.tok.answer(s.response) for s in self.samples(stage.get("data"), {"pair"})]
        if not texts:
            raise SystemExit("Pretraining has no text: wire a Text Dataset with text into it.")
        stream = []
        for text in texts:
            stream.extend(self.tok.encode(text) + [self.tok.end_id])
        steps, batch = int(stage.get("steps", 200)), int(stage.get("batch", 8))
        curriculum = self.settings.get("curriculum") or {}
        first = int(curriculum.get("start", self.context))
        last = int(curriculum.get("end", self.context))
        accumulate = max(1, int(self.settings.get("accumulate", 1)))
        optimizer = self.optimizer(stage)
        self.model.train()
        for step in range(steps):
            length = min(self.context, max(8, int(first + (last - first) * step / max(1, steps - 1))))
            x, y = self.lm_batch(stream, batch, length)
            loss = self.loss_of(self.forward(x), y)
            self.step(optimizer, loss, "pretrain", step, steps, accumulate, context=length)

    def sft(self, stage, name="sft", samples=None):
        good = samples if samples is not None else self.samples(stage.get("data"), {"pair"})
        bad = self.samples(stage.get("negatives"), {"pair", "negative"})
        if not good and not bad:
            raise SystemExit("Fine-tuning has no requests and answers: wire a Text Dataset into it.")
        adapter = stage.get("adapter", "full")
        if adapter in ("lora", "dora"):
            blocks.add_adapters(self.model, int(stage.get("rank", 8)), float(stage.get("alpha", 16)), adapter == "dora")
            self.model.to(self.device)
            log("Training %s adapters (rank %s)." % (adapter.upper(), stage.get("rank", 8)))
        steps, batch = int(stage.get("steps", 200)), int(stage.get("batch", 4))
        weight = float(stage.get("negative_weight", 0.5))
        accumulate = max(1, int(self.settings.get("accumulate", 1)))
        optimizer = self.optimizer(stage)
        self.model.train()
        for step in range(steps):
            loss = torch.zeros((), device=self.device)
            if good:
                x, y = self.pair_tensors(random.choices(good, k=batch))
                loss = loss + self.loss_of(self.forward(x), y)
            if bad:
                x, y = self.pair_tensors(random.choices(bad, k=max(1, batch // 2)))
                loss = loss + weight * self.unlikelihood(self.forward(x), y)
                self.model.aux_loss()
            self.step(optimizer, loss, name, step, steps, accumulate)
        if adapter in ("lora", "dora"):
            blocks.merge_adapters(self.model)

    def compaction(self, stage):
        pairs = [data.compact_pair(self.tok, s) for s in self.samples(stage.get("data"), {"compact"})]
        if not pairs:
            raise SystemExit('Compaction needs {"context": ..., "compact": ...} lines in its dataset.')
        self.sft(stage, "compaction", pairs)

    @torch.no_grad()
    def sample(self, prompt_ids, count, max_new, temperature, top_k):
        self.model.eval()
        x = torch.tensor([prompt_ids] * count, device=self.device)
        done = torch.zeros(count, dtype=torch.bool, device=self.device)
        for _ in range(max_new):
            logits = self.model(x[:, -self.context:])[:, -1, :] / max(temperature, 1e-3)
            if top_k:
                kth = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < kth, -1e9)
            nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.tok.end_id), nxt)
            x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
            done = done | (nxt == self.tok.end_id)
            if bool(done.all()):
                break
        return [row[len(prompt_ids):] for row in x.tolist()]

    def reward(self, completion, rules, reference):
        text = self.tok.decode([t for t in completion if t != self.tok.end_id])
        tool_open, tool_close = self.spec["tool_tags"]
        total = 0.0
        for rule in rules:
            kind, score = rule.get("kind"), float(rule.get("score", 1))
            value = str(rule.get("text", ""))
            if kind == "contains" and value and value in text:
                total += score
            elif kind == "not_contains" and value and value in text:
                total += score
            elif kind == "regex" and value and re.search(value, text):
                total += score
            elif kind == "exact" and reference is not None and text.strip() == reference.strip():
                total += score
            elif kind in ("tool_valid", "tool_malformed"):
                calls = re.findall(re.escape(tool_open) + r"(.*?)" + re.escape(tool_close), text, re.S)
                good = 0
                for call in calls:
                    try:
                        parsed = json.loads(call)
                        good += isinstance(parsed, dict) and isinstance(parsed.get("name"), str)
                    except ValueError:
                        pass
                if kind == "tool_valid" and calls and good == len(calls):
                    total += score
                if kind == "tool_malformed" and good < len(calls):
                    total += score
            elif kind == "loop":
                n = int(rule.get("ngram", 4))
                grams = [tuple(completion[i:i + n]) for i in range(max(0, len(completion) - n + 1))]
                if grams and max(grams.count(g) for g in set(grams)) >= int(rule.get("repeats", 3)):
                    total += score
            elif kind == "short" and len(completion) <= int(rule.get("tokens", 32)):
                total += score
            elif kind == "ends" and completion and completion[-1] == self.tok.end_id:
                total += score
        return total

    def rl(self, stage):
        pairs = self.samples(stage.get("data"), {"pair"})
        if not pairs:
            raise SystemExit("Reinforcement learning needs requests: wire a Text Dataset into it.")
        rules = stage.get("rules") or [{"kind": "ends", "score": 1}]
        steps, group = int(stage.get("steps", 50)), max(2, int(stage.get("group", 4)))
        max_new = int(stage.get("max_new", 48))
        temperature, top_k = float(stage.get("temperature", 0.9)), int(stage.get("top_k", 40))
        optimizer = self.optimizer(stage)
        for step in range(steps):
            sample = random.choice(pairs)
            prompt = self.tok.encode(sample.prompt)[-(self.context - max_new):]
            completions = self.sample(prompt, group, max_new, temperature, top_k)
            rewards = torch.tensor([self.reward(c, rules, sample.response) for c in completions], dtype=torch.float)
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            self.model.train()
            loss = torch.zeros((), device=self.device)
            for completion, adv in zip(completions, advantage.tolist()):
                if not completion or abs(adv) < 1e-6:
                    continue
                ids = (prompt + completion)[-(self.context + 1):]
                x = torch.tensor([ids[:-1]], device=self.device)
                logits = self.forward(x)
                self.model.aux_loss()
                logp = torch.log_softmax(logits[0, -len(completion):], dim=-1)
                taken = logp.gather(-1, torch.tensor(ids[-len(completion):], device=self.device).unsqueeze(-1)).sum()
                loss = loss - adv * taken / len(completion)
            loss = loss / group
            self.step(optimizer, loss, "rl", step, steps, 1, reward=round(float(rewards.mean()), 3))

    # -- results --

    def export(self, out_dir):
        self.model.eval()
        self.model.checkpointing = False
        model = self.model.to("cpu")
        safetensors.save(model.state_dict(), os.path.join(out_dir, "weights.safetensors"))
        tokens = torch.zeros(1, min(8, self.context), dtype=torch.long)
        try:
            torch.onnx.export(
                model, (tokens,), os.path.join(out_dir, "model.onnx"),
                input_names=["tokens"], output_names=["logits"],
                dynamic_axes={"tokens": {1: "length"}, "logits": {1: "length"}},
                opset_version=17, dynamo=False,
            )
        except Exception as error:  # noqa: BLE001 – reported, not hidden
            emit("warning", message="The model could not be exported to ONNX (%s). Install onnx (Settings → AI) to run it in macros." % error)
        return blocks.parameter_count(model)

    def run(self, out_dir):
        for stage in self.plan.get("stages", []):
            kind = stage.get("type")
            emit("stage", stage=kind)
            {"pretrain": self.pretrain, "sft": self.sft, "rl": self.rl, "compaction": self.compaction}[kind](stage)
            self.stages.append(kind)
        parameters = self.export(out_dir)
        return {"stages": self.stages, "steps": self.steps_done, "loss": self.last_loss, "parameters": parameters}


# ---- vision -----------------------------------------------------------------


def read_pictures(path):
    """Pictures the Magic tab prepared: a JSON header line, then float32 tensors [3, h, w] each."""
    with open(path, "rb") as handle:
        header = json.loads(handle.readline())
        body = bytearray(handle.read())
    w, h = header["width"], header["height"]
    tensor = torch.frombuffer(body, dtype=torch.float32).reshape(len(header["items"]), 3, h, w).clone()
    return header, tensor


def train_vision(plan, out_dir):
    settings = plan.get("optimizer", {})
    device = pick_device(settings.get("device", "auto"))
    header, pictures = read_pictures(plan["vision"]["pictures"])
    classes = int(plan["vision"]["classes"])
    model = blocks.VisionModel(classes, plan["architecture"])
    if plan.get("start", {}).get("weights"):
        model.load_state_dict(safetensors.load(plan["start"]["weights"]))
        log("Started from the base model's weights.")
    model.to(device)
    labels = torch.tensor([item["class"] for item in header["items"]])
    negative = torch.tensor([bool(item.get("negative")) for item in header["items"]])
    stage = next((s for s in plan.get("stages", []) if s.get("type") == "vision"), {})
    steps, batch = int(stage.get("steps", 300)), int(stage.get("batch", 16))
    if stage.get("lr"):
        settings = dict(settings, lr=stage["lr"])
    optimizer = optim.make(model, settings)
    emit("stage", stage="vision")
    loss_value = float("nan")
    for step in range(steps):
        model.train()
        index = torch.randint(0, len(pictures), (min(batch, len(pictures)),))
        x = pictures[index].to(device)
        # A little shifting, so the model does not learn exact pixel positions.
        x = torch.roll(x, shifts=(random.randint(-2, 2), random.randint(-2, 2)), dims=(2, 3))
        y, neg = labels[index].to(device), negative[index].to(device)
        logits = model(x)
        loss = torch.zeros((), device=device)
        if (~neg).any():
            loss = loss + F.cross_entropy(logits[~neg], y[~neg])
        if neg.any():
            p = torch.softmax(logits[neg], -1).gather(-1, y[neg].unsqueeze(-1)).squeeze(-1)
            loss = loss - torch.log((1 - p).clamp_min(1e-6)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if step % 5 == 0 or step + 1 == steps:
            emit("progress", stage="vision", step=step + 1, steps=steps, loss=round(loss_value, 4))
    model.eval().to("cpu")
    safetensors.save(model.state_dict(), os.path.join(out_dir, "weights.safetensors"))
    sample = torch.zeros(1, 3, header["height"], header["width"])
    torch.onnx.export(model, (sample,), os.path.join(out_dir, "model.onnx"), input_names=["input"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    return {"stages": ["vision"], "steps": steps, "loss": loss_value, "parameters": blocks.parameter_count(model)}


def main(path):
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    out_dir = plan["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    random.seed(plan.get("seed", 0))
    torch.manual_seed(plan.get("seed", 0))
    began = time.time()
    try:
        if plan.get("kind") == "vision":
            report = train_vision(plan, out_dir)
        else:
            report = TextTrainer(plan).run(out_dir)
    except SystemExit as stop:
        emit("error", message=str(stop))
        return 1
    report["seconds"] = round(time.time() - began, 1)
    if isinstance(report.get("loss"), float) and math.isnan(report["loss"]):
        report["loss"] = None
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    emit("done", **report)
    return 0
