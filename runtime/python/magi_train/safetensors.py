"""Reading and writing weights in the safetensors format, without the
`safetensors` package: an 8-byte header length, a JSON header, then the raw
little-endian tensors. Only data, never code, so a shared `.magic` file cannot
run anything when its weights are loaded."""

import json
import struct

import torch

_DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.int64: "I64", torch.int32: "I32"}
_TORCH = {v: k for k, v in _DTYPES.items()}


def save(tensors, path):
    header, chunks, offset = {}, [], 0
    for name, tensor in tensors.items():
        tensor = tensor.detach().to("cpu").contiguous()
        if tensor.dtype not in _DTYPES:
            tensor = tensor.float()
        flat = tensor.reshape(-1).view(torch.uint8)
        data = flat.numpy().tobytes() if _has_numpy() else bytes(flat.tolist())
        header[name] = {"dtype": _DTYPES[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for chunk in chunks:
            handle.write(chunk)


def load(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        body = handle.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = bytearray(body[start:end])
        tensor = torch.frombuffer(raw, dtype=_TORCH[info["dtype"]]) if raw else torch.empty(0, dtype=_TORCH[info["dtype"]])
        tensors[name] = tensor.reshape(info["shape"]).clone()
    return tensors


def _has_numpy():
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False

