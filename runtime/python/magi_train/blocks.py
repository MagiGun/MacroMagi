"""The building blocks of models trained in MacroMagi.

A text model is an embedding, the blocks the Magic tab's canvas lists (each
a stack of layers), and a head that gives a score per token. The blocks:

    transformer        attention and a feed-forward layer: the standard block
    sparse_attention   attention that keeps keys and values in a small latent
                       (low-rank, optionally shared by the block's layers) and
                       looks at a sliding window plus every `stride`-th token
    moe                a mixture of experts: shared experts every token goes
                       through, and routed experts of which each token uses `top_k`
    ssm                a state-space layer in the quadratic ("state space
                       duality") form of Mamba-2: per head, a learned decay
                       over time instead of attention scores
    persona_router     feed-forward experts, one per persona of the reasoning
                       (and one neutral), with routing taught by the persona tags
    prefix_split       not layers: the request (up to the assistant tag) is read
                       both ways at once, and only the answer causally (a
                       prefix language model, or encoder/decoder split)

A vision model is a stack of convolutions (`conv`) and a classifier.

Everything traces to ONNX with a dynamic sequence length, so MacroMagi can run
the models it trains.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

NEG = -1e4


# ---- attention --------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, width, heads, kv_rank=0):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.kv_rank = kv_rank
        if kv_rank:
            # Keys and values come from a small latent: the cache stores only it.
            self.kv_down = nn.Linear(width, kv_rank, bias=False)
            self.k_up = nn.Linear(kv_rank, width, bias=False)
            self.v_up = nn.Linear(kv_rank, width, bias=False)
        else:
            self.k = nn.Linear(width, width, bias=False)
            self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def latent(self, x):
        return self.kv_down(x)

    def forward(self, x, mask, latent=None):
        b, s, w = x.shape
        q = self.q(x).view(b, s, self.heads, self.dim).transpose(1, 2)
        if self.kv_rank:
            c = self.kv_down(x) if latent is None else latent
            k, v = self.k_up(c), self.v_up(c)
        else:
            k, v = self.k(x), self.v(x)
        k = k.view(b, s, self.heads, self.dim).transpose(1, 2)
        v = v.view(b, s, self.heads, self.dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dim)
        scores = scores.masked_fill(~mask, NEG)
        y = torch.softmax(scores, dim=-1) @ v
        return self.out(y.transpose(1, 2).reshape(b, s, w))


class FeedForward(nn.Module):
    def __init__(self, width, mult=4):
        super().__init__()
        self.up = nn.Linear(width, width * mult)
        self.down = nn.Linear(width * mult, width)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MoE(nn.Module):
    """Shared experts for every token, and routed experts of which each uses `top_k`."""

    def __init__(self, width, experts=8, shared=1, top_k=2, mult=2):
        super().__init__()
        self.shared = nn.ModuleList(FeedForward(width, mult) for _ in range(shared))
        self.experts = nn.ModuleList(FeedForward(width, mult) for _ in range(experts))
        self.router = nn.Linear(width, experts, bias=False)
        self.top_k = max(1, min(top_k, experts))
        self.aux = None

    def forward(self, x):
        y = sum((expert(x) for expert in self.shared), torch.zeros_like(x))
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)
        top, index = probs.topk(self.top_k, dim=-1)
        weights = torch.zeros_like(probs).scatter(-1, index, top)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        # Small models: every expert runs, weighted by the router (unused ones by 0).
        outs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        y = y + (weights.unsqueeze(-1) * outs).sum(-2)
        if self.training:
            # Load balancing: tokens spread over the experts.
            share = (weights > 0).float().mean(dim=(0, 1))
            mean_prob = probs.mean(dim=(0, 1))
            self.aux = len(self.experts) * (share * mean_prob).sum()
        return y


class PersonaFFN(nn.Module):
    """One feed-forward expert per persona, plus a neutral one; routing learns the persona tags."""

    def __init__(self, width, personas, mult=4):
        super().__init__()
        self.experts = nn.ModuleList(FeedForward(width, mult) for _ in range(personas + 1))
        self.router = nn.Linear(width, personas + 1)
        self.aux = None

    def forward(self, x, labels=None):
        logits = self.router(x)
        weights = torch.softmax(logits, dim=-1)
        outs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        if self.training and labels is not None:
            # Inside a persona's turn its expert should be chosen; elsewhere the neutral one.
            target = (labels + 1).clamp_min(0)
            self.aux = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        return (weights.unsqueeze(-1) * outs).sum(-2)


class SSM(nn.Module):
    """A Mamba-2 style state-space layer in its quadratic (state space duality) form."""

    def __init__(self, width, heads, state=16):
        super().__init__()
        self.heads, self.dim, self.state = heads, width // heads, state
        self.in_proj = nn.Linear(width, width * 2 + state * 2 + heads)
        self.a_log = nn.Parameter(torch.zeros(heads))
        # As in Mamba-2: the output is normalized before its gate, and steps start small.
        self.y_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, width)

    def forward(self, x, mask):
        b, s, w = x.shape
        z, xs, bmat, cmat, dt = torch.split(self.in_proj(x), [w, w, self.state, self.state, self.heads], dim=-1)
        dt = F.softplus(dt - 3.0).clamp(max=4.0)  # (b, s, h): how much each step forgets and takes in
        log_a = -dt * torch.exp(self.a_log)  # <= 0
        cum = torch.cumsum(log_a, dim=1).transpose(1, 2)  # (b, h, s)
        # decay[t, s] = exp(L_t - L_s) for s <= t: how much of step s is left at step t.
        diff = cum.unsqueeze(-1) - cum.unsqueeze(-2)
        decay = torch.exp(diff.masked_fill(~mask, NEG))
        scores = (cmat @ bmat.transpose(-1, -2)).unsqueeze(1) / math.sqrt(self.state)  # (b, 1, s, s)
        weights = (scores * decay).masked_fill(~mask, 0.0)
        xs = (xs.view(b, s, self.heads, self.dim) * dt.unsqueeze(-1)).transpose(1, 2)  # (b, h, s, d)
        y = (weights @ xs).transpose(1, 2).reshape(b, s, w)
        return self.out(self.y_norm(y) * F.silu(z))


# ---- layers -----------------------------------------------------------------


class Layer(nn.Module):
    """A residual layer: a mixer (attention or SSM) and a feed-forward part."""

    def __init__(self, width, heads, mixer="attention", ffn="dense", params=None, personas=0):
        super().__init__()
        p = params or {}
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.kind = mixer
        if mixer == "ssm":
            self.mixer = SSM(width, heads, int(p.get("state", 16)))
        else:
            self.mixer = Attention(width, heads, int(p.get("kv_rank", 0)) if mixer == "sparse" else 0)
        if ffn == "moe":
            self.ffn = MoE(width, int(p.get("experts", 8)), int(p.get("shared", 1)), int(p.get("top_k", 2)), int(p.get("mult", 2)))
        elif ffn == "persona":
            self.ffn = PersonaFFN(width, personas, int(p.get("mult", 4)))
        else:
            self.ffn = FeedForward(width, int(p.get("mult", 4)))
        self.window = int(p.get("window", 0)) if mixer == "sparse" else 0
        self.stride = int(p.get("stride", 0)) if mixer == "sparse" else 0
        self.share = bool(p.get("share_kv", False)) and mixer == "sparse"

    def mask(self, base):
        if not (self.window or self.stride):
            return base
        s = base.shape[-1]
        rows = torch.arange(s, device=base.device).unsqueeze(1)
        cols = torch.arange(s, device=base.device).unsqueeze(0)
        keep = torch.zeros(s, s, dtype=torch.bool, device=base.device)
        if self.window:
            keep = keep | ((rows - cols) < self.window)
        if self.stride:
            keep = keep | ((cols % self.stride) == (self.stride - 1))
        return base & keep

    def forward(self, x, mask, latent=None, labels=None):
        h = self.norm1(x)
        if self.kind == "ssm":
            x = x + self.mixer(h, mask)
        else:
            x = x + self.mixer(h, self.mask(mask), latent)
        if isinstance(self.ffn, PersonaFFN):
            return x + self.ffn(self.norm2(x), labels)
        return x + self.ffn(self.norm2(x))


def layers_of(block, width, heads, personas):
    kind, p = block.get("kind"), block.get("params", {}) or {}
    count = max(1, int(p.get("layers", 1)))
    mixer, ffn = {
        "transformer": ("attention", "dense"),
        "sparse_attention": ("sparse", "dense"),
        "moe": ("attention", "moe"),
        "ssm": ("ssm", "dense"),
        "persona_router": ("attention", "persona"),
    }.get(kind, (None, None))
    if mixer is None:
        return []
    return [Layer(width, heads, mixer, ffn, p, personas) for _ in range(count)]


class TextModel(nn.Module):
    """Tokens in, a score per possible next token out: `tokens` [1, s] -> `logits` [1, s, vocab]."""

    def __init__(self, vocab, context, architecture, prefix_id=None, personas=0):
        super().__init__()
        width, heads = int(architecture.get("width", 128)), int(architecture.get("heads", 4))
        blocks = architecture.get("blocks", []) or [{"kind": "transformer", "params": {"layers": 2}}]
        self.prefix_split = any(b.get("kind") == "prefix_split" for b in blocks)
        self.prefix_id = prefix_id
        self.context = context
        self.embed = nn.Embedding(vocab, width)
        self.position = nn.Embedding(context, width)
        layers = []
        for block in blocks:
            group = layers_of(block, width, heads, personas)
            # A block's own share of the KV latent starts afresh.
            if group:
                group[0].first_of_block = True
            layers.extend(group)
        if not layers:
            layers = layers_of({"kind": "transformer", "params": {"layers": 2}}, width, heads, personas)
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.checkpointing = False

    def base_mask(self, tokens):
        b, s = tokens.shape
        positions = torch.arange(s, device=tokens.device)
        mask = (positions.unsqueeze(0) <= positions.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # (1, 1, s, s)
        if self.prefix_split and self.prefix_id is not None:
            # The request (up to the last assistant tag) sees itself both ways.
            marks = (tokens == self.prefix_id).long() * (positions + 1).unsqueeze(0)
            prefix = marks.max(dim=1).values.view(b, 1, 1, 1)
            inside = (positions.view(1, 1, s, 1) < prefix) & (positions.view(1, 1, 1, s) < prefix)
            mask = mask | inside
        return mask

    def forward(self, tokens, labels=None):
        # Recomputation for checkpointing sets these again after they were read: start afresh.
        for module in self.modules():
            if getattr(module, "aux", None) is not None:
                module.aux = None
        s = tokens.shape[1]
        positions = torch.arange(s, device=tokens.device)
        x = self.embed(tokens) + self.position(positions).unsqueeze(0)
        mask = self.base_mask(tokens)
        latent = None
        for layer in self.layers:
            if getattr(layer, "first_of_block", False):
                latent = None
            # Layers of a block that share keys and values use the first one's latent.
            use = None
            if layer.share and layer.mixer.kv_rank:
                if latent is None:
                    latent = layer.mixer.latent(layer.norm1(x))
                use = latent
            if self.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask, use, labels, use_reentrant=False)
            else:
                x = layer(x, mask, use, labels)
        return self.head(self.norm(x))

    def aux_loss(self):
        total = None
        for module in self.modules():
            aux = getattr(module, "aux", None)
            if aux is not None:
                total = aux if total is None else total + aux
                module.aux = None
        return total



class VisionModel(nn.Module):
    """A picture in, a score per class out: `input` [1, 3, h, w] -> `logits` [1, classes]."""

    def __init__(self, classes, architecture):
        super().__init__()
        convs = [b for b in architecture.get("blocks", []) if b.get("kind") == "conv"] or [{"params": {}}]
        p = convs[0].get("params", {}) or {}
        layers, channels = int(p.get("layers", 3)), int(p.get("channels", 32))
        parts, c_in = [], 3
        for i in range(max(1, layers)):
            c_out = channels * (2 ** min(i, 3))
            parts += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*parts)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(c_in, classes)

    def forward(self, x):
        return self.classify(self.pool(self.features(x)).flatten(1))


# ---- adapters: LoRA and DoRA ------------------------------------------------


class Adapted(nn.Module):
    """A frozen linear layer with a low-rank update; DoRA also learns each output's magnitude."""

    def __init__(self, base, rank, alpha, dora=False):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * (1.0 / math.sqrt(base.in_features)))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        self.dora = dora
        if dora:
            self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1))

    def weight(self):
        w = self.base.weight + self.scale * (self.b @ self.a)
        if self.dora:
            w = self.magnitude.unsqueeze(1) * w / w.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return w

    def forward(self, x):
        return F.linear(x, self.weight(), self.base.bias)

    def merged(self):
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def add_adapters(model, rank=8, alpha=16, dora=False):
    """Freezes the model and puts adapters on the linear layers of its blocks."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.layers.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, Adapted(child, rank, alpha, dora))


def merge_adapters(model):
    """Folds adapters back into plain linear layers, so the model is as before, updated."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, Adapted):
                setattr(module, name, child.merged())
    for p in model.parameters():
        p.requires_grad_(True)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
