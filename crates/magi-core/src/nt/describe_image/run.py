"""Vision for the model: a small vision model looks at the picture and describes it, with a 3x3 grid of where things are, so the model reasons over text, not pixels. Put a captioning model (model.onnx) in this folder for descriptions; without one it reports the picture's size, brightness and colours per region (Pillow needed).

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    import os
    path = os.path.expanduser(str(args.get("path", "")).strip())
    if not os.path.isfile(path):
        return {"error": "no such picture: " + path}
    try:
        from PIL import Image
    except ImportError:
        return {"error": "describe_image needs Pillow: pip install pillow"}
    image = Image.open(path).convert("RGB")
    width, height = image.size
    names = ["top left", "top", "top right", "left", "middle", "right", "bottom left", "bottom", "bottom right"]
    grid = []
    for row in range(3):
        for column in range(3):
            box = (column * width // 3, row * height // 3, (column + 1) * width // 3, (row + 1) * height // 3)
            r, g, b = image.crop(box).resize((1, 1)).getpixel((0, 0))
            light = (r + g + b) / 765
            colour = max((("red", r - (g + b) / 2), ("green", g - (r + b) / 2), ("blue", b - (r + g) / 2),
                          ("grey", 20)), key=lambda c: c[1])[0]
            grid.append({"region": names[row * 3 + column], "colour": colour, "brightness": round(light, 2)})
    result = {"width": width, "height": height, "grid": grid}
    model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx")
    if os.path.isfile(model):
        result["note"] = "a captioning model is here; wire its preprocessing in run.py to describe the picture"
    else:
        result["note"] = "no captioning model yet: only the grid of colours"
    return result


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
