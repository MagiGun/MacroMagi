"""Optimizers and memory helpers.

Muon (momentum orthogonalized by a Newton–Schulz iteration) updates the 2-D
weights of the blocks; everything else (embeddings, the head, norms, biases)
uses AdamW, as Muon is meant to be used.
"""

import torch


def orthogonalize(g, steps=5):
    """An approximately orthogonal matrix with the same "direction" as `g`."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                buf = state.setdefault("momentum", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                update = orthogonalize(p.grad + group["momentum"] * buf)
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(update, alpha=-group["lr"] * scale)


class Combined:
    """Several optimizers stepped as one."""

    def __init__(self, optimizers):
        self.optimizers = [o for o in optimizers if o is not None]

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad(set_to_none=True)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_tensors(self):
        for o in self.optimizers:
            for state in o.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        yield state, key, value


def make(model, settings):
    """The optimizer the Optimizer block chose, for the parameters that train."""
    name = settings.get("name", "adamw")
    lr = float(settings.get("lr", 3e-3))
    decay = float(settings.get("weight_decay", 0.01))
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if name == "muon":
        matrices = [p for n, p in params if p.ndim == 2 and "embed" not in n and "position" not in n and not n.startswith("head")]
        ids = {id(p) for p in matrices}
        rest = [p for n, p in params if id(p) not in ids]
        return Combined([
            Muon(matrices, lr=float(settings.get("muon_lr", 0.02))) if matrices else None,
            torch.optim.AdamW(rest, lr=lr, weight_decay=decay) if rest else None,
        ])
    if name == "sgd":
        return Combined([torch.optim.SGD([p for _, p in params], lr=lr, momentum=0.9)])
    return Combined([torch.optim.AdamW([p for _, p in params], lr=lr, weight_decay=decay)])


def offload(optimizer, device):
    """Keeps optimizer state in system memory between steps (less graphics memory, slower)."""
    for state, key, value in optimizer.state_tensors():
        state[key] = value.to(device, non_blocking=True)
