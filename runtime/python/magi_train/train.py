"""Running a training plan.

The plan (JSON, written by the Magic tab) says what kind of model to make,
where to start, its architecture, the datasets and the stages, each with the
optimizer wired into it. Progress goes out as JSON lines; the result goes to
`out_dir`: `weights.safetensors`, `model.onnx` and `report.json`.

Stages:

    pretrain     next-token prediction on plain text, optionally growing the
                 context from short to full (a curriculum)
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


def check_packages():
    """Everything training needs, checked before anything is trained."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise SystemExit(
            "The onnx package is not installed in this Python (%s). Install it under Settings › Runtime, "
            "then train again." % sys.executable
        )


def pick_devices(choice):
    """The devices to train on: a list of CUDA indices, "auto", or the CPU."""
    if isinstance(choice, list) and choice:
        if torch.cuda.is_available():
            found = [i for i in choice if i < torch.cuda.device_count()]
            if found:
                return [torch.device("cuda", i) for i in found]
        else:
            emit("warning", message="This PyTorch cannot use graphics cards (it is the CPU build), so training "
                                    "runs on the CPU. Install PyTorch for NVIDIA graphics cards under Settings › Runtime.")
        return [torch.device("cpu")]
    if choice == "auto" and torch.cuda.is_available():
        return [torch.device("cuda", 0)]
    return [torch.device("cpu")]


def describe(devices):
    if devices[0].type == "cuda":
        return "Training on %s." % ", ".join(torch.cuda.get_device_name(d) for d in devices)
    return "Training on the CPU (%d threads)." % torch.get_num_threads()


class Runner:
    """A model on one or more devices."""

    def __init__(self, model, devices):
        self.core = model.to(devices[0])
        self.devices = devices
        self.parallel = None
        if len(devices) > 1:
            self.parallel = torch.nn.DataParallel(self.core, device_ids=[d.index for d in devices])

    def __call__(self, *args, **kwargs):
        return (self.parallel or self.core)(*args, **kwargs)

    @property
    def device(self):
        return self.devices[0]


def run_steps(runner, settings, steps, stage_name, compute, counter, after_backward=None):
    """The training loop of a stage: `compute(step)` gives (loss, extra fields) for one batch."""
    optimizer = optim.make(runner.core, settings, steps)
    accumulate = max(1, int(settings.get("accumulate", 1)))
    clip = float(settings.get("clip", 1.0))
    offload = bool(settings.get("offload")) and runner.device.type == "cuda"
    runner.core.checkpointing = bool(settings.get("checkpointing"))
    runner.core.train()
    last = float("nan")
    for step in range(steps):
        loss, extra = compute(step)
        if loss is not None and loss.requires_grad:
            (loss / accumulate).backward()
        if after_backward:
            after_backward()
        if (step + 1) % accumulate == 0 or step + 1 == steps:
            if clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in runner.core.parameters() if p.requires_grad], clip)
            if offload:
                optim.offload(optimizer, runner.device)
            optimizer.step(step)
            if offload:
                optim.offload(optimizer, torch.device("cpu"))
            optimizer.zero_grad()
        if loss is not None:
            last = float(loss.detach())
        counter[0] += 1
        if step % 5 == 0 or step + 1 == steps:
            emit("progress", stage=stage_name, step=step + 1, steps=steps,
                 loss=None if math.isnan(last) else round(last, 4), **(extra or {}))
    return last


# ---- text -------------------------------------------------------------------


class TextTrainer:
    def __init__(self, plan):
        self.plan = plan
        self.spec = plan["text"]
        self.tok = Tokenizer(self.spec)
        architecture = dict(plan["architecture"])
        self.context = int(architecture.get("context", self.spec.get("context", 256)))
        architecture["context"] = self.context
        self.architecture = architecture
        self.personas = len(self.spec.get("personas", []))
        self.uses_personas = any((l.get("ffn") or {}).get("kind") == "persona" for l in architecture.get("layers", []))
        model = blocks.TextModel(self.tok.vocab, architecture, self.tok.special(self.spec["assistant_tag"]), self.personas)
        start = plan.get("start", {})
        if start.get("weights"):
            missing = model.load_state_dict(safetensors.load(start["weights"]), strict=False)
            if missing.missing_keys or missing.unexpected_keys:
                raise SystemExit(
                    "The base model's weights do not fit this architecture (%d missing, %d left over). Keep the "
                    "base model's architecture blocks to train it further."
                    % (len(missing.missing_keys), len(missing.unexpected_keys))
                )
            log("Started from the base model's weights.")
        devices = pick_devices(plan.get("devices", "auto"))
        log(describe(devices))
        self.runner = Runner(model, devices)
        log("%s parameters, %d layers, %d tokens of context." % (
            "{:,}".format(blocks.parameter_count(model)), len(model.layers), self.context))
        self.counter = [0]
        self.pending = []
        self.last_loss = float("nan")
        self.stages = []
        self.datasets = {key: [] for key in plan.get("datasets", {})}
        for key, files in plan.get("datasets", {}).items():
            for path in files:
                self.datasets[key].extend(data.read(path, self.tok))

    @property
    def device(self):
        return self.runner.device

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
            pad = length + 1 - len(chunk)
            xs.append(chunk[:-1] + [self.tok.end_id] * pad)
            ys.append(chunk[1:] + [-100] * pad)
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
            xs.append(ids[:-1] + [self.tok.end_id] * pad)
            ys.append([-100] * (len(ids) - 1 - learn) + ids[len(ids) - learn:] + [-100] * pad)
        return torch.tensor(xs, device=self.device), torch.tensor(ys, device=self.device)

    def forward(self, x, y=None):
        labels = None
        if self.uses_personas:
            labels = torch.tensor([self.tok.persona_spans(row) for row in x.tolist()], device=self.device)
        out = self.runner(x, labels, y)
        if out["load"].numel():
            # Applied after the backward pass, so recomputed layers route as they did.
            self.pending.append(out["load"].sum(0).detach())
        return out["logits"], out["aux"].mean()

    def balance(self):
        for load in self.pending:
            self.runner.core.balance(load)
        self.pending = []

    def loss_of(self, x, y):
        logits, aux = self.forward(x, y)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        return loss + aux

    def unlikelihood(self, x, y):
        """Makes the tokens of an answer not to give less likely: -log(1 - p)."""
        logits, _ = self.forward(x, None)
        probs = torch.softmax(logits, dim=-1)
        mask = y != -100
        picked = probs.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return -(torch.log((1 - picked).clamp_min(1e-6)) * mask).sum() / mask.sum().clamp_min(1)

    @staticmethod
    def settings(stage):
        return stage.get("optimizer") or {}

    # -- stages --

    def pretrain(self, stage):
        texts = [s.text for s in self.samples(stage.get("data"), {"text"})]
        texts += [s.prompt + self.tok.answer(s.response) for s in self.samples(stage.get("data"), {"pair"})]
        if not texts:
            raise SystemExit("Pretraining has no text: wire a Text Dataset with files into its Text input.")
        stream = []
        for text in texts:
            stream.extend(self.tok.encode(text) + [self.tok.end_id])
        steps, batch = int(stage.get("steps", 200)), int(stage.get("batch", 8))
        first = int(stage.get("curriculum_start", 0) or self.context)
        first = max(8, min(first, self.context))
        log("Pretraining on %s tokens of text." % "{:,}".format(len(stream)))

        def compute(step):
            length = min(self.context, int(first + (self.context - first) * step / max(1, steps - 1)))
            x, y = self.lm_batch(stream, batch, length)
            return self.loss_of(x, y), ({"context": length} if first < self.context else {})

        self.last_loss = run_steps(self.runner, self.settings(stage), steps, "pretrain", compute, self.counter, self.balance)

    def sft(self, stage, name="sft", samples=None):
        good = samples if samples is not None else self.samples(stage.get("data"), {"pair"})
        bad = self.samples(stage.get("negatives"), {"pair", "negative"})
        if not good and not bad:
            raise SystemExit("Fine-tuning has no requests and answers: wire a Text Dataset into its “What to do” input.")
        adapter = stage.get("adapter", "full")
        core = self.runner.core
        if adapter in ("lora", "dora"):
            blocks.add_adapters(core, int(stage.get("rank", 8)), float(stage.get("alpha", 16)), adapter == "dora")
            self.runner = Runner(core, self.runner.devices)
            log("Training %s adapters (rank %s) on %d requests." % (adapter.upper(), stage.get("rank", 8), len(good)))
        else:
            log("Training all weights on %d requests%s." % (len(good), ", %d answers to avoid" % len(bad) if bad else ""))
        steps, batch = int(stage.get("steps", 200)), int(stage.get("batch", 4))
        weight = float(stage.get("negative_weight", 0.5))

        def compute(step):
            loss = torch.zeros((), device=self.device)
            if good:
                x, y = self.pair_tensors(random.choices(good, k=batch))
                loss = loss + self.loss_of(x, y)
            if bad:
                x, y = self.pair_tensors(random.choices(bad, k=max(1, batch // 2)))
                loss = loss + weight * self.unlikelihood(x, y)
            return loss, {}

        self.last_loss = run_steps(self.runner, self.settings(stage), steps, name, compute, self.counter, self.balance)
        if adapter in ("lora", "dora"):
            blocks.merge_adapters(core)
            self.runner = Runner(core, self.runner.devices)

    def compaction(self, stage):
        pairs = [data.compact_pair(self.tok, s) for s in self.samples(stage.get("data"), {"compact"})]
        if not pairs:
            raise SystemExit('Context Compaction needs {"context": …, "compact": …} lines in its dataset.')
        self.sft(stage, "compaction", pairs)

    @torch.no_grad()
    def sample(self, prompt_ids, count, max_new, temperature, top_k):
        model = self.runner.core
        model.eval()
        x = torch.tensor([prompt_ids] * count, device=self.device)
        done = torch.zeros(count, dtype=torch.bool, device=self.device)
        for _ in range(max_new):
            logits = model(x[:, -self.context:])["logits"][:, -1, :] / max(temperature, 1e-3)
            if top_k:
                kth = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < kth, -1e9)
            nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.tok.end_id), nxt)
            x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
            done = done | (nxt == self.tok.end_id)
            if bool(done.all()):
                break
        model.train()
        out = []
        for row in x.tolist():
            completion = row[len(prompt_ids):]
            if self.tok.end_id in completion:
                completion = completion[: completion.index(self.tok.end_id) + 1]
            out.append(completion)
        return out

    def reward(self, completion, rules, reference):
        text = self.tok.decode([t for t in completion if t != self.tok.end_id])
        tool_open, tool_close = self.spec["tool_tags"]
        total = 0.0
        for rule in rules:
            kind, score = rule.get("kind"), float(rule.get("score", 1))
            value = str(rule.get("text", ""))
            if kind == "contains" and value and value in text:
                total += score
            elif kind == "not_contains" and value and value not in text:
                total += score
            elif kind == "regex" and value:
                try:
                    if re.search(value, text):
                        total += score
                except re.error:
                    pass
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
                broken = text.count(tool_open) != text.count(tool_close)
                if kind == "tool_valid" and calls and good == len(calls) and not broken:
                    total += score
                if kind == "tool_malformed" and (good < len(calls) or broken):
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
            raise SystemExit("Reinforcement Learning needs requests: wire a Text Dataset into its Requests input.")
        rules = stage.get("rules") or [{"kind": "ends", "score": 1}]
        steps, group = int(stage.get("steps", 50)), max(2, int(stage.get("group", 4)))
        max_new = int(stage.get("max_new", 48))
        temperature, top_k = float(stage.get("temperature", 0.9)), int(stage.get("top_k", 40))

        def compute(step):
            sample = random.choice(pairs)
            prompt = self.tok.encode(sample.prompt)[-(self.context - max_new):]
            completions = self.sample(prompt, group, max_new, temperature, top_k)
            rewards = torch.tensor([self.reward(c, rules, sample.response) for c in completions], dtype=torch.float)
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            loss = torch.zeros((), device=self.device)
            for completion, adv in zip(completions, advantage.tolist()):
                if not completion or abs(adv) < 1e-6:
                    continue
                ids = (prompt + completion)[-(self.context + 1):]
                x = torch.tensor([ids[:-1]], device=self.device)
                logits, _ = self.forward(x)
                logp = torch.log_softmax(logits[0, -len(completion):], dim=-1)
                taken = logp.gather(-1, torch.tensor(ids[-len(completion):], device=self.device).unsqueeze(-1)).sum()
                loss = loss - adv * taken / len(completion)
            return loss / group, {"reward": round(float(rewards.mean()), 3)}

        self.last_loss = run_steps(self.runner, self.settings(stage), steps, "rl", compute, self.counter, self.balance)

    # -- results --

    def export(self, out_dir):
        model = self.runner.core.to("cpu")
        model.eval()
        model.checkpointing = False
        safetensors.save(model.state_dict(), os.path.join(out_dir, "weights.safetensors"))
        tokens = torch.zeros(1, self.context, dtype=torch.long)
        log("Exporting the model.")
        torch.onnx.export(
            blocks.Exported(model), (tokens,), os.path.join(out_dir, "model.onnx"),
            input_names=["tokens"], output_names=["logits"], opset_version=17, dynamo=False,
        )
        return blocks.parameter_count(model)

    def run(self, out_dir):
        for stage in self.plan.get("stages", []):
            kind = stage.get("type")
            emit("stage", stage=kind)
            {"pretrain": self.pretrain, "sft": self.sft, "rl": self.rl, "compaction": self.compaction}[kind](stage)
            self.stages.append(kind)
        parameters = self.export(out_dir)
        return {"stages": self.stages, "steps": self.counter[0], "loss": self.last_loss, "parameters": parameters,
                "gguf_compatible": blocks.gguf_compatible(self.architecture)}


# ---- vision -----------------------------------------------------------------


def read_pictures(path):
    """Pictures the Magic tab prepared: a JSON header line, then float32 tensors [3, h, w] each."""
    with open(path, "rb") as handle:
        header = json.loads(handle.readline())
        body = bytearray(handle.read())
    w, h = header["width"], header["height"]
    tensor = torch.frombuffer(body, dtype=torch.float32).reshape(len(header["items"]), 3, h, w).clone()
    return header, tensor


class VisionWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return {"logits": self.model(x)}


def train_vision(plan, out_dir):
    devices = pick_devices(plan.get("devices", "auto"))
    log(describe(devices))
    header, pictures = read_pictures(plan["vision"]["pictures"])
    classes = int(plan["vision"]["classes"])
    model = blocks.VisionModel(classes, plan["architecture"])
    if plan.get("start", {}).get("weights"):
        model.load_state_dict(safetensors.load(plan["start"]["weights"]))
        log("Started from the base model's weights.")
    runner = Runner(VisionWrap(model), devices)
    labels = torch.tensor([item["class"] for item in header["items"]])
    negative = torch.tensor([bool(item.get("negative")) for item in header["items"]])
    stage = next((s for s in plan.get("stages", []) if s.get("type") == "vision"), {})
    steps, batch = int(stage.get("steps", 300)), int(stage.get("batch", 16))
    emit("stage", stage="vision")
    log("%d pictures of %d answers." % (len(pictures), classes))
    counter = [0]

    def compute(step):
        index = torch.randint(0, len(pictures), (min(batch, len(pictures)),))
        x = pictures[index].to(runner.device)
        # A little shifting, so the model does not learn exact pixel positions.
        x = torch.roll(x, shifts=(random.randint(-2, 2), random.randint(-2, 2)), dims=(2, 3))
        y, neg = labels[index].to(runner.device), negative[index].to(runner.device)
        logits = runner(x)["logits"]
        loss = torch.zeros((), device=runner.device)
        if (~neg).any():
            loss = loss + F.cross_entropy(logits[~neg], y[~neg])
        if neg.any():
            p = torch.softmax(logits[neg], -1).gather(-1, y[neg].unsqueeze(-1)).squeeze(-1)
            loss = loss - torch.log((1 - p).clamp_min(1e-6)).mean()
        return loss, {}

    loss_value = run_steps(runner, stage.get("optimizer") or {}, steps, "vision", compute, counter)
    model.eval().to("cpu")
    safetensors.save(model.state_dict(), os.path.join(out_dir, "weights.safetensors"))
    sample = torch.zeros(1, 3, header["height"], header["width"])
    log("Exporting the model.")
    torch.onnx.export(model, (sample,), os.path.join(out_dir, "model.onnx"), input_names=["input"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    return {"stages": ["vision"], "steps": counter[0], "loss": loss_value, "parameters": blocks.parameter_count(model)}


def main(path):
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    out_dir = plan["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    random.seed(plan.get("seed", 0))
    torch.manual_seed(plan.get("seed", 0))
    began = time.time()
    try:
        check_packages()
        if plan.get("kind") == "vision":
            report = train_vision(plan, out_dir)
        else:
            report = TextTrainer(plan).run(out_dir)
    except SystemExit as stop:
        emit("error", message=str(stop))
        return 1
    except torch.cuda.OutOfMemoryError:
        emit("error", message="The graphics card ran out of memory. Make the model or the batch smaller, turn on "
                              "“Recompute activations” in the Optimizer block, or train on the CPU.")
        return 1
    report["seconds"] = round(time.time() - began, 1)
    if isinstance(report.get("loss"), float) and math.isnan(report["loss"]):
        report["loss"] = None
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    emit("done", **report)
    return 0
