"""A tiny model in Magi's shape – block windows, experts after a plain layer, grouped key/value heads,
queries and keys normalized, a tied head – with random weights, and the scores PyTorch gives at the
last token of several prompts that cross many windows. MacroMagi's native engine must give the same:

    python tools/make_parity.py <folder>
    MAGI_TEST_PARITY_DIR=<folder> cargo test -p magi-ml native_engine_matches_pytorch -- --ignored
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from magi_train import blocks, safetensors  # noqa: E402

VOCAB = 300


def main(folder):
    os.makedirs(folder, exist_ok=True)
    torch.manual_seed(0)
    moe = {"kind": "moe", "experts": 4, "shared": 1, "top_k": 2, "mult": 0.5, "gating": "sqrt_softplus",
           "balance": "bias"}
    layers = []
    for index in range(4):
        attention = {"kind": "full", "qk_norm": True}
        if (index + 1) % 4:
            attention["window"] = 16
        layers.append({"attention": attention, "ffn": moe if index else {"kind": "dense", "mult": 2.75}})
    architecture = {"width": 64, "heads": 4, "kv_heads": 2, "context": 256, "positions": "rope",
                    "rope_theta": 10000.0, "tied": True, "layers": layers}
    model = blocks.TextModel(VOCAB, architecture, 5, 0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() > 1:
                p.normal_(0, 0.08)
        for m in model.modules():
            if isinstance(m, blocks.MoE):
                m.bias.normal_(0, 0.05)
    model.eval()
    cases = []
    tokens = torch.randint(0, VOCAB, (1, 120), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = model(tokens)["logits"][0]
    for length in (10, 17, 33, 48, 70, 120):
        cases.append({"tokens": tokens[0, :length].tolist(), "scores": logits[length - 1].tolist()})
    # The chunked kernel against the mask, in PyTorch itself.
    pos = torch.arange(120)
    base = (pos[None, :] <= pos[:, None])[None, None]
    q, k, v = (torch.randn(1, 4, 120, 16) for _ in range(3))
    masked = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=blocks.window_mask(base, 16))
    print("chunked vs masked:", (blocks.window_attention(q, k, v, 16) - masked).abs().max().item())
    safetensors.save(model.state_dict(), os.path.join(folder, "weights.safetensors"))
    with open(os.path.join(folder, "architecture.json"), "w", encoding="utf-8") as handle:
        json.dump(architecture, handle)
    with open(os.path.join(folder, "cases.json"), "w", encoding="utf-8") as handle:
        json.dump(cases, handle)
    print("wrote", len(cases), "cases to", folder)


if __name__ == "__main__":
    main(sys.argv[1])
