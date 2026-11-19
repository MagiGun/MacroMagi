"""The AI block. The app learns the pictures and looks at the screen; this only asks it to."""

from . import value_node


@value_node("ai.vision")
def ai_vision(run, node):
    # A picture wired in is looked at instead of the screen.
    wired, picture = run.input(node, "picture")
    return run.host.call("ai_vision", params=node["params"], picture=picture if wired else None, node=node["id"])
