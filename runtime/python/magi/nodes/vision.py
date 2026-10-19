"""Vision blocks. The app captures the screen and does the matching."""

from . import value_node


@value_node("vision.find_image")
def find_image(run, node):
    # A picture wired in is looked for instead of the block's list.
    wired, picture = run.input(node, "picture")
    if wired:
        return run.host.call("find_image", params=node["params"], picture=picture)
    return run.host.find_image(node["params"])


@value_node("vision.screenshot")
def screenshot(run, node):
    return run.host.call("screenshot", params=node["params"])


@value_node("vision.read_text")
def read_text(run, node):
    return run.host.call("read_text", params=node["params"])


@value_node("vision.find_color")
def find_color(run, node):
    return run.host.find_color(node["params"])
