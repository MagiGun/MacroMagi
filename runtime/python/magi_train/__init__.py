"""MacroMagi's model trainer.

The Magic tab turns its canvas into a *plan* (JSON) and runs this package in
the Python macros use, which needs PyTorch (and `onnx` to export models):

    python -u magi_train_run.py plan.json

It trains text (agent) and vision models from scratch or from a model's
weights, through the stages the plan lists, and writes `model.onnx`,
`weights.safetensors` and `report.json` into the plan's output folder. It
reports progress as JSON lines on standard output.

Unlike the macro runtime (package `magi`), this package needs third-party
packages: PyTorch and onnx, which Settings › Runtime installs.
"""
