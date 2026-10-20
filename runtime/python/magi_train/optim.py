"""Optimizers, learning-rate schedules and memory helpers.

Each training stage gets its optimizer from the Optimizer block wired into it
(or the defaults). Muon (momentum orthogonalized by a Newton–Schulz
iteration, as DeepSeek-V4 trains with it) updates the 2-D weight matrices of
the layers; AdamW updates everything else – embeddings, the head, norms,
routers' biases, hyper-connection gates – as Muon is meant to be used.
"""

import math

import torch

# DeepSeek-V4's hybrid Newton–Schulz: eight fast steps, then two that settle.
FAST, SETTLE = (3.4445, -4.7750, 2.0315), (2.0, -1.5, 0.5)


def orthogonalize(g, steps=10):
    """An approximately orthogonal matrix with the same "direction" as `g`."""
    x = g / (g.norm() + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for i in range(steps):
        a, b, c = FAST if i < steps - 2 else SETTLE
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, orthogonalized, rescaled to a steady size, with decoupled weight decay."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, rms=0.2):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rms=rms))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.reshape(p.shape[0], -1)
                buf = self.state[p].setdefault("momentum", torch.zeros_like(g))
                buf.mul_(group["momentum"]).add_(g)
                update = orthogonalize(g + group["momentum"] * buf)
                update = update * math.sqrt(max(update.shape)) * group["rms"]
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.view_as(p), alpha=-group["lr"])


class Combined:
    """Several optimizers stepped as one, with one learning-rate schedule."""

    def __init__(self, optimizers, schedule):
        self.optimizers = [o for o in optimizers if o is not None]
        self.schedule = schedule
        self.base = [[g["lr"] for g in o.param_groups] for o in self.optimizers]

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad(set_to_none=True)

    def step(self, progress):
        factor = self.schedule(progress)
        for o, base in zip(self.optimizers, self.base):
            for group, lr in zip(o.param_groups, base):
                group["lr"] = lr * factor
        for o in self.optimizers:
            o.step()

    def state_tensors(self):
        for o in self.optimizers:
            for state in o.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        yield state, key, value


def schedule_of(settings, steps):
    """The learning rate's factor at step i: linear warmup, then constant or cosine decay to 10%."""
    warmup = max(0, int(settings.get("warmup", 0)))
    kind = settings.get("schedule", "constant")

    def factor(i):
        if warmup and i < warmup:
            return (i + 1) / warmup
        if kind == "cosine" and steps > warmup:
            t = (i - warmup) / max(1, steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        return 1.0

    return factor


def is_matrix(name, p):
    # Embeddings, the head and gates stay with AdamW.
    return p.ndim == 2 and not any(part in name for part in ("embed", "position", "lm_head", "w_pre", "w_res", "w_post", "gate.weight"))


def make(model, settings, steps):
    """The optimizer of a stage, for the parameters that train."""
    name = settings.get("name", "adamw")
    lr = float(settings.get("lr", 3e-3))
    decay = float(settings.get("weight_decay", 0.01))
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    schedule = schedule_of(settings, steps)
    if name == "muon":
        matrices = [p for n, p in params if is_matrix(n, p)]
        ids = {id(p) for p in matrices}
        rest = [p for n, p in params if id(p) not in ids]
        return Combined([
            Muon(matrices, lr=float(settings.get("muon_lr", 0.02)), weight_decay=decay) if matrices else None,
            torch.optim.AdamW(rest, lr=lr, betas=(0.9, 0.95), weight_decay=decay) if rest else None,
        ], schedule)
    if name == "sgd":
        return Combined([torch.optim.SGD([p for _, p in params], lr=lr, momentum=0.9, weight_decay=decay)], schedule)
    return Combined([torch.optim.AdamW([p for _, p in params], lr=lr, betas=(0.9, 0.95), weight_decay=decay)], schedule)


def offload(optimizer, device):
    """Keeps optimizer state in system memory between steps (less graphics memory, slower)."""
    for state, key, value in optimizer.state_tensors():
        state[key] = value.to(device, non_blocking=True)
