"""Exact math for the model.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return {"error": "no expression"}
    try:
        import sympy  # optional: algebra and calculus
        x, y, z, t = sympy.symbols("x y z t")
        names = {name: getattr(sympy, name) for name in dir(sympy) if not name.startswith("_")}
        names.update({"x": x, "y": y, "z": z, "t": t})
        value = sympy.sympify(expression.replace("^", "**"), locals=names)
        value = sympy.simplify(value) if not value.is_number else value
        number = value.evalf() if value.is_number else None
        return {"result": str(value), "decimal": str(number) if number is not None else None}
    except ImportError:
        pass
    import ast
    import math
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max})
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Lambda, ast.Subscript)) or (
            isinstance(node, ast.Name) and node.id not in allowed
        ):
            return {"error": "not a plain math expression: " + expression}
    value = eval(compile(tree, "<expression>", "eval"), {"__builtins__": {}}, allowed)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"result": value}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
