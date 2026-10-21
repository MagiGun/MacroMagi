"""Describes a picture for view_image with a small vision model, so Magi reads about a picture
instead of seeing it: SmolVLM-256M-Instruct (Apache-2.0), installed under Settings > Runtime
(Image descriptions) into MAGI_VISION_MODEL. MacroMagi reads the picture's text itself (the
system's text recognition) and adds it.

Reads {"path": …} as JSON on standard input; prints {"description": …} or {"error": …}.
"""
import json
import os
import sys

PROMPT = ("Describe this picture for someone who cannot see it: what it shows, what is where, "
          "and anything that looks like a screen, a window, a chart or a diagram.")


def main(args):
    path = os.path.expanduser(str(args.get("path", "")).strip())
    if not os.path.isfile(path):
        return {"error": "no such picture: " + path}
    model_dir = os.environ.get("MAGI_VISION_MODEL", "").strip()
    if not model_dir or not os.path.isdir(model_dir):
        return {"error": "no image description model: install Image descriptions under Settings > Runtime"}
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as missing:
        return {"error": "image descriptions need %s: install Image descriptions under Settings > Runtime"
                % missing.name}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(model_dir, torch_dtype=dtype, local_files_only=True).to(device)
    image = Image.open(path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return {"description": text.strip()}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
