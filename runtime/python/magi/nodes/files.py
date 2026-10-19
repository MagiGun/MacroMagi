"""File blocks: reading, writing and searching files, and folders.

The app does the reading and writing; these functions only ask for it.
"""

from . import flow_node, value_node

OUT = "out"


def _path(run, node):
    wired, path = run.input(node, "path")
    return path if wired else None


@value_node("files.read")
def read_file(run, node):
    return run.host.call("read_file", params=node["params"], path=_path(run, node))


@flow_node("files.write")
def write_file(run, node):
    wired, value = run.input(node, "value")
    run.waited = True
    run.host.call("write_file", params=node["params"], value=value if wired else None, path=_path(run, node))
    return OUT


@value_node("files.find")
def find_text(run, node):
    wired, text = run.input(node, "text")
    return run.host.call("find_text", params=node["params"], text=text if wired else None)


@value_node("files.list")
def list_folder(run, node):
    return run.host.call("list_folder", params=node["params"], path=_path(run, node))


@flow_node("files.make_folder")
def make_folder(run, node):
    run.host.call("make_folder", params=node["params"], path=_path(run, node))
    return OUT
