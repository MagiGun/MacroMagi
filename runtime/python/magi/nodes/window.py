"""Window blocks. The app finds and activates windows."""

from . import flow_node, value_node


@flow_node("window.focus")
def focus_window(run, node):
    run.waited = True
    found = run.host.call("focus_window", params=node["params"])
    run.outputs[node["id"]] = {"screen": found["kind"] if found else None}
    if found is None:
        run.log("No window matching “%s” was found" % node["params"]["target"], level="warning", node=node["id"])
        return "missing"
    return "out"


@flow_node("window.display")
def switch_display(run, node):
    run.waited = True
    run.host.call("switch_display", params=node["params"])
    return "out"


@value_node("window.current")
def current_window(run, node):
    return run.host.call("current_window")
