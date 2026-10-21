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
    """An approximately orthogonal matrix with the same "direction" as `g` – or, for a
    stack of matrices [n, rows, columns], one for each, in one go."""
    # bfloat16 on graphics cards, as Muon is usually run: the iteration does not need more.
    x = g.bfloat16() if g.is_cuda else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    flat = x.ndim == 2
    if flat:
        x = x.unsqueeze(0)
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.mT
    for i in range(steps):
        a, b, c = FAST if i < steps - 2 else SETTLE
        s = x @ x.mT
        # a·x + (b·s + c·s²)·x as two fused products: faster, and the sums stay in float32 (about a third
        # of the rounding error of separate steps in bfloat16).
        x = torch.baddbmm(x, torch.baddbmm(s, s, s, beta=b, alpha=c), x, beta=a)
    if tall:
        x = x.mT
    return x[0] if flat else x


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, orthogonalized, rescaled to a steady size, with decoupled weight decay."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, rms=0.2):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rms=rms))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            # Matrices of the same shape (one per layer) are orthogonalized together: far fewer
            # small steps on the graphics card than one matrix at a time.
            # The elementwise parts for every matrix at once (a few kernels, not a few per matrix).
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad.reshape(p.shape[0], -1) for p in params]
            bufs = [self.state[p].setdefault("momentum", torch.zeros_like(g)) for p, g in zip(params, grads)]
            torch._foreach_mul_(bufs, group["momentum"])
            torch._foreach_add_(bufs, grads)
            nesterov = torch._foreach_add(grads, bufs, alpha=group["momentum"])
            if group["weight_decay"]:
                torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
            shapes = {}
            for p, g in zip(params, nesterov):
                shapes.setdefault(tuple(g.shape), []).append((p, g))
            for shape, items in shapes.items():
                updates = orthogonalize(torch.stack([g for _, g in items])).unbind(0)
                scale = math.sqrt(max(shape)) * group["rms"]
                torch._foreach_add_([p for p, _ in items], [u.view_as(p) for (p, _), u in zip(items, updates)],
                                    alpha=-group["lr"] * scale)


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


def adamw(params, lr, decay):
    fused = bool(params) and all(p.is_cuda for p in params)
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=decay, fused=fused)


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
            Muon(matrices, lr=float(settings.get("muon_lr", 0.004)), weight_decay=decay) if matrices else None,
            adamw(rest, float(settings.get("embed_lr", lr)), decay) if rest else None,
        ], schedule)
    if name == "sgd":
        return Combined([torch.optim.SGD([p for _, p in params], lr=lr, momentum=0.9, weight_decay=decay)], schedule)
    return Combined([adamw([p for _, p in params], lr, decay)], schedule)


def offload(optimizer, device):
    """Keeps optimizer state in system memory between steps (less graphics memory, slower)."""
    for state, key, value in optimizer.state_tensors():
        state[key] = value.to(device, non_blocking=True)
