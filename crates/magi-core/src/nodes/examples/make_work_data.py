"""work.jsonl: long pieces of work for a Magi model – tens of tool calls each, thinking before
every step, MagiDB before code.

Every result is real. Each piece of work happens in a temporary folder standing for a folder on
the user's computer: MacroMagi's own native tools (crates/magi-core/src/nt) read, search and
change its files, look things up in MagiDB (the repository's magidb/ folder), and run its tests and
scripts with Python – so the model learns from exactly what the tools give back. Only what cannot
run here is written out: the web, pictures, .docx/.xlsx/.pdf files and git.

    python make_work_data.py            writes work.jsonl beside it

What it teaches:

- plan first, as steps or pseudocode; keep a running list in the thinking on long work
  ("Done: … Next: …");
- look up exact code and commands in MagiDB (db_search, db_read) instead of writing them from
  memory; the web for what MagiDB does not have;
- read before changing; change one thing at a time; run the tests; read failures and fix them;
- recover when a tool fails (wrong path, text not found or there twice, a refused call);
- documents through open_document and edit_document, pictures through view_image, arithmetic
  through the calculator;
- answer briefly at the end: what was done and how it was checked.
"""

import contextlib
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.normpath(os.path.join(HERE, "..", "..", "nt"))
MAGIDB = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "magidb"))
OUT = os.path.join(HERE, "work.jsonl")

STANDARD, SUPERIOR = "standard", "superior"


def load_tool(name):
    spec = importlib.util.spec_from_file_location("nt_" + name, os.path.join(NT, name, "run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


TOOLS = {}


def tools():
    if not TOOLS:
        os.environ["MAGI_DB_DIR"] = MAGIDB
        for name in ["db_search", "db_read", "list_files", "read_file", "search_files", "write_file", "edit_file",
                     "run_command", "open_document", "edit_document", "calculator"]:
            TOOLS[name] = load_tool(name)
    return TOOLS


def dumps(value):
    """A tool result as MacroMagi writes it into the model's text."""
    return json.dumps(value, ensure_ascii=False)


class World:
    """The files a piece of work happens in: a real temporary folder standing for `root` on the
    user's computer. Tool calls name `root`; the tools run on the temporary folder; their results
    name `root` again."""

    def __init__(self, root, files):
        self.root = root.rstrip("/")
        self.base = tempfile.mkdtemp(prefix="magi-work-")
        self.real = os.path.join(self.base, os.path.basename(self.root) or "work")
        os.makedirs(self.real)
        for path, text in files.items():
            self.put(path, text)

    def put(self, path, text):
        full = os.path.join(self.real, *path.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def get(self, path):
        with open(os.path.join(self.real, *path.split("/")), encoding="utf-8") as handle:
            return handle.read()

    def exists(self, path):
        return os.path.exists(os.path.join(self.real, *path.split("/")))

    def path(self, rel=""):
        return self.root + ("/" + rel if rel else "")

    def to_real(self, value):
        if isinstance(value, str):
            return value.replace(self.root, self.real.replace("\\", "/"))
        if isinstance(value, list):
            return [self.to_real(v) for v in value]
        if isinstance(value, dict):
            return {k: self.to_real(v) for k, v in value.items()}
        return value

    def to_virtual(self, value):
        if isinstance(value, str):
            real_forward = self.real.replace("\\", "/")
            value = value.replace(real_forward, self.root).replace(self.real, self.root.replace("/", "\\"))
            # Python's temporary folder can be written with its short (8.3) name in tracebacks.
            return value.replace(os.path.realpath(self.real), self.root.replace("/", "\\"))
        if isinstance(value, list):
            return [self.to_virtual(v) for v in value]
        if isinstance(value, dict):
            return {k: self.to_virtual(v) for k, v in value.items()}
        return value

    def run(self, name, args):
        with contextlib.redirect_stdout(sys.stderr):
            result = tools()[name](self.to_real(args))
        return self.to_virtual(result)

    def close(self):
        shutil.rmtree(self.base, ignore_errors=True)


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


class Work:
    """One piece of work being written down: the request, then thought, call, result… answer."""

    def __init__(self, rng, world=None, effort=STANDARD):
        self.rng = rng
        self.world = world
        self.effort = effort
        self.messages = []
        self.calls = 0

    def ask(self, text):
        self.messages.append({"role": "user", "content": text})

    def step(self, thinking, name, simulated=None, learn=True, **arguments):
        """A thought and a tool call; the tool runs (or `simulated` is its result). Returns the result."""
        message = {"role": "assistant", "content": "", "thinking": thinking,
                   "tool_calls": [call(name, **arguments)]}
        if not learn:
            message["learn"] = False
        self.messages.append(message)
        result = simulated if simulated is not None else self.world.run(name, arguments)
        self.messages.append({"role": "tool", "content": dumps(result)})
        self.calls += 1
        return result

    def answer(self, text, thinking=None):
        message = {"role": "assistant", "content": text}
        if thinking:
            message["thinking"] = thinking
        self.messages.append(message)

    def sample(self):
        return {"messages": self.messages, "effort": self.effort}


def pick(rng, *options):
    return rng.choice(options)


def plural(n, word, words=None):
    return "%d %s" % (n, word if n == 1 else (words or word + "s"))


def listing(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---- projects ------------------------------------------------------------------------------------
#
# Small Python projects about one kind of record each: a CSV file of rows, a package that loads it
# and works out totals, a report, a command line and unittest tests.

THEMES = [
    {"pkg": "grades", "title": "grades", "csv": "grades.csv", "key": "student", "group": "subject",
     "value": "score", "keys": ["Ana", "Ben", "Chloe", "Dev", "Ema", "Finn", "Gus", "Hana", "Ivo", "Jun"],
     "groups": ["maths", "history", "art", "biology"], "range": (40, 100)},
    {"pkg": "shop", "title": "shop sales", "csv": "sales.csv", "key": "product", "group": "region",
     "value": "amount", "keys": ["kettle", "lamp", "mug", "rug", "shelf", "stool", "teapot", "vase"],
     "groups": ["north", "south", "east", "west"], "range": (5, 400)},
    {"pkg": "weather", "title": "weather readings", "csv": "readings.csv", "key": "day", "group": "station",
     "value": "temperature", "keys": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
     "groups": ["harbour", "airport", "hill"], "range": (-5, 31)},
    {"pkg": "library", "title": "library loans", "csv": "loans.csv", "key": "title", "group": "genre",
     "value": "days", "keys": ["Dune", "Emma", "Ulysses", "Beloved", "Matilda", "Persuasion", "Kindred"],
     "groups": ["fiction", "classics", "children"], "range": (1, 40)},
    {"pkg": "fitness", "title": "workouts", "csv": "workouts.csv", "key": "date", "group": "activity",
     "value": "minutes", "keys": ["2026-03-0%d" % d for d in range(1, 10)],
     "groups": ["run", "swim", "cycle", "yoga"], "range": (10, 95)},
    {"pkg": "expenses", "title": "expenses", "csv": "expenses.csv", "key": "item", "group": "category",
     "value": "cost", "keys": ["rent", "bus pass", "groceries", "phone", "gym", "books", "cinema"],
     "groups": ["home", "travel", "food", "fun"], "range": (3, 900)},
    {"pkg": "stock", "title": "warehouse stock", "csv": "stock.csv", "key": "part", "group": "warehouse",
     "value": "quantity", "keys": ["bolt", "nut", "washer", "hinge", "bracket", "spring", "gear"],
     "groups": ["A", "B", "C"], "range": (0, 500)},
    {"pkg": "energy", "title": "energy use", "csv": "meter.csv", "key": "month", "group": "building",
     "value": "kwh", "keys": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug"],
     "groups": ["office", "lab", "depot"], "range": (120, 2400)},
]

ROOTS = ["C:/Users/me/Projects/{name}", "C:/Users/me/code/{name}", "D:/work/{name}", "C:/dev/{name}",
         "C:/Users/me/Documents/{name}", "D:/projects/{name}"]

DOCSTRINGS = {
    "load_rows": "Every row of the CSV file, with {value} as a number.",
    "total": "The sum of field over the rows.",
    "average": "The mean of field over the rows; 0 when there are none.",
    "top": "The n rows with the largest field, largest first.",
    "count_by": "How many rows have each value of field.",
    "total_by": "The sum of field for each value of group.",
    "summary": "The report's lines: counts, totals, averages and the top rows.",
    "main": "Reads the CSV file named on the command line and prints its summary.",
}


def csv_text(theme, rng, rows=None):
    low, high = theme["range"]
    count = rows or rng.randint(10, 24)
    lines = ["%s,%s,%s" % (theme["key"], theme["group"], theme["value"])]
    for _ in range(count):
        lines.append("%s,%s,%s" % (rng.choice(theme["keys"]), rng.choice(theme["groups"]), rng.randint(low, high)))
    return "\n".join(lines) + "\n"


BUGS = {
    # bug: (function, the wrong lines, the right ones, what was wrong)
    "total_skips_first": ("total", "    return sum(row[field] for row in rows[1:])\n",
                          "    return sum(row[field] for row in rows)\n",
                          "it summed rows[1:], leaving out the first row"),
    "average_empty": ("average", "    return total(rows, field) / len(rows)\n",
                      "    if not rows:\n        return 0\n    return total(rows, field) / len(rows)\n",
                      "it divided by len(rows) even when there were no rows"),
    "average_off_by_one": ("average", "    return total(rows, field) / (len(rows) + 1)\n",
                           "    return total(rows, field) / len(rows)\n",
                           "it divided by len(rows) + 1 instead of len(rows)"),
    "top_smallest": ("top", "    return sorted(rows, key=lambda row: row[field])[:n]\n",
                     "    return sorted(rows, key=lambda row: row[field], reverse=True)[:n]\n",
                     "it sorted smallest first, so it gave the smallest rows"),
    "count_by_one": ("count_by", "        counts[row[field]] = 1\n",
                     "        counts[row[field]] = counts.get(row[field], 0) + 1\n",
                     "it set each count to 1 instead of adding 1"),
    "total_by_last": ("total_by", "        totals[row[group]] = row[field]\n",
                      "        totals[row[group]] = totals.get(row[group], 0) + row[field]\n",
                      "it kept only the last row's value for each group instead of adding them up"),
}


def stats_source(docstrings=True, bug=None):
    def doc(name):
        return '    """%s"""\n' % DOCSTRINGS[name] if docstrings else ""
    parts = {
        "total": "    return sum(row[field] for row in rows)\n",
        "average": "    if not rows:\n        return 0\n    return total(rows, field) / len(rows)\n",
        "top": "    return sorted(rows, key=lambda row: row[field], reverse=True)[:n]\n",
        "count_by": "        counts[row[field]] = counts.get(row[field], 0) + 1\n",
        "total_by": "        totals[row[group]] = totals.get(row[group], 0) + row[field]\n",
    }
    if bug:
        function, wrong, right, _ = BUGS[bug]
        parts[function] = parts[function].replace(right, wrong) if right in parts[function] else wrong
    return (
        '"""Totals and counts over the rows."""\n\n\n'
        "def total(rows, field):\n" + doc("total") + parts["total"] + "\n\n"
        "def average(rows, field):\n" + doc("average") + parts["average"] + "\n\n"
        "def top(rows, field, n=3):\n" + doc("top") + parts["top"] + "\n\n"
        "def count_by(rows, field):\n" + doc("count_by") + "    counts = {}\n    for row in rows:\n" + parts["count_by"]
        + "    return counts\n\n\n"
        "def total_by(rows, group, field):\n" + doc("total_by") + "    totals = {}\n    for row in rows:\n"
        + parts["total_by"] + "    return totals\n"
    )


REPORT_NAMES = ["month", "week", "person", "place", "kind", "source", "team", "day", "size", "status", "owner",
                "level", "label", "batch", "shift", "route", "zone", "tier", "channel", "segment", "quarter", "year",
                "client", "vendor", "season", "branch", "desk", "lane", "wing", "floor"]

TODO_JOBS = ["handle rows with an empty {value}", "round {value} to whole numbers in the report",
             "warn when the CSV file has no header", "sort the {group} names case-insensitively",
             "skip rows whose {value} is negative", "cache the loaded rows", "add a --top option to main.py",
             "write the report to a file as well", "check the {key} column exists",
             "support semicolon-separated files", "show percentages next to the totals",
             "add a test for count_by with no rows", "log how many rows were read", "make n a keyword argument",
             "use pathlib for the path", "translate the report's labels", "add a --since date filter",
             "move the format strings into constants", "print the largest {key}", "add type hints",
             "explain the report in the README", "check {value} is a number before adding it"]


def project(theme, rng, docstrings=True, bug=None, extra=0, todos=0):
    """A project's files: {path: text}. `extra` adds that many report modules that use total_by;
    `todos` puts that many TODO comments in."""
    p, key, group, value = theme["pkg"], theme["key"], theme["group"], theme["value"]

    def doc(name, indent="    "):
        return '%s"""%s"""\n' % (indent, DOCSTRINGS[name].format(value=value)) if docstrings else ""

    files = {
        "README.md": "# %s\n\nReads `data/%s` and reports on it.\n\n    python main.py data/%s\n    python -m unittest\n"
                     % (theme["title"].capitalize(), theme["csv"], theme["csv"]),
        "data/" + theme["csv"]: csv_text(theme, rng),
        p + "/__init__.py": "",
        p + "/load.py": ('"""Reading the %s rows."""\nimport csv\n\n\ndef load_rows(path):\n%s'
                         '    with open(path, newline="", encoding="utf-8") as f:\n'
                         "        rows = list(csv.DictReader(f))\n    for row in rows:\n"
                         '        row["%s"] = float(row["%s"])\n    return rows\n'
                         % (theme["csv"], doc("load_rows"), value, value)),
        p + "/stats.py": stats_source(docstrings, bug),
        p + "/report.py": ('"""The report printed by main.py."""\nfrom .stats import average, top, total, total_by\n\n\n'
                           "def summary(rows):\n%s"
                           '    lines = [f"{len(rows)} rows"]\n'
                           "    lines.append(f\"total %s: {total(rows, '%s'):.1f}\")\n"
                           "    lines.append(f\"average %s: {average(rows, '%s'):.1f}\")\n"
                           '    for row in top(rows, "%s"):\n'
                           "        lines.append(f\"top: {row['%s']} ({row['%s']:.1f})\")\n"
                           '    for name, amount in sorted(total_by(rows, "%s", "%s").items()):\n'
                           '        lines.append(f"{name}: {amount:.1f}")\n    return lines\n'
                           % (doc("summary"), value, value, value, value, value, key, value, group, value)),
        "main.py": ("import argparse\n\nfrom %s.load import load_rows\nfrom %s.report import summary\n\n\n"
                    'def main():\n%s    parser = argparse.ArgumentParser(description="Reports on %s.")\n'
                    '    parser.add_argument("path")\n    args = parser.parse_args()\n    rows = load_rows(args.path)\n'
                    "    for line in summary(rows):\n        print(line)\n\n\n"
                    'if __name__ == "__main__":\n    main()\n' % (p, p, doc("main"), theme["csv"])),
        "tests/__init__.py": "",
        "tests/test_stats.py": (
            "import unittest\n\nfrom %(p)s.stats import average, count_by, top, total, total_by\n\n"
            "ROWS = [\n"
            '    {"%(k)s": "a", "%(g)s": "x", "%(v)s": 2.0},\n'
            '    {"%(k)s": "b", "%(g)s": "y", "%(v)s": 3.0},\n'
            '    {"%(k)s": "c", "%(g)s": "x", "%(v)s": 4.0},\n'
            "]\n\n\nclass StatsTests(unittest.TestCase):\n"
            '    def test_total(self):\n        self.assertEqual(total(ROWS, "%(v)s"), 9.0)\n\n'
            '    def test_average(self):\n        self.assertEqual(average(ROWS, "%(v)s"), 3.0)\n\n'
            '    def test_average_of_nothing_is_zero(self):\n        self.assertEqual(average([], "%(v)s"), 0)\n\n'
            "    def test_top_is_largest_first(self):\n"
            '        self.assertEqual([row["%(k)s"] for row in top(ROWS, "%(v)s", 2)], ["c", "b"])\n\n'
            '    def test_count_by(self):\n        self.assertEqual(count_by(ROWS, "%(g)s"), {"x": 2, "y": 1})\n\n'
            "    def test_total_by(self):\n"
            '        self.assertEqual(total_by(ROWS, "%(g)s", "%(v)s"), {"x": 6.0, "y": 3.0})\n'
            % {"p": p, "k": key, "g": group, "v": value}),
    }
    # Report modules that use total_by: rename and docstring work grows with them.
    names = list(REPORT_NAMES)
    rng.shuffle(names)
    for name in names[:extra]:
        files["%s/reports/by_%s.py" % (p, name)] = (
            '"""Totals of %s for each %s, for the %s report."""\nfrom ..stats import total_by\n\n\n'
            "def by_%s_lines(rows):\n%s"
            '    totals = total_by(rows, "%s", "%s")\n'
            '    return [f"{name}: {amount:.1f}" for name, amount in sorted(totals.items())]\n'
            % (value, group, name, name, ('    """One line per %s: its total %s."""\n' % (group, value))
                                         if docstrings else "", group, value))
    if extra:
        files[p + "/reports/__init__.py"] = ""
    # TODO comments, before a def or a return.
    jobs = [j.format(key=key, group=group, value=value) for j in TODO_JOBS]
    rng.shuffle(jobs)
    targets = sorted(path for path in files if path.endswith(".py") and not path.endswith("__init__.py")
                     and not path.startswith("tests/"))
    for job in jobs[:todos]:
        path = rng.choice(targets)
        lines = files[path].split("\n")
        spots = [i for i, line in enumerate(lines) if line.startswith("def ") or line.startswith("    return")]
        at = rng.choice(spots) if spots else 0
        indent = "    " if lines[at].startswith("    ") else ""
        lines.insert(at, indent + "# TODO: " + job)
        files[path] = "\n".join(lines)
    return files


def project_root(theme, rng):
    return rng.choice(ROOTS).format(name=theme["pkg"])


def unittest_result(output):
    """(the failing tests as (FAIL or ERROR, name), how many ran, whether all passed)."""
    bad = []
    ran = 0
    for line in output.splitlines():
        if line.startswith(("FAIL: ", "ERROR: ")):
            kind, rest = line.split(": ", 1)
            bad.append((kind, rest.split(" ")[0]))
        if line.startswith("Ran "):
            ran = int(line.split()[1])
    return bad, ran, output.rstrip().endswith("OK")


def error_line(output):
    """The line that says what went wrong: an AssertionError, or the exception's last line."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("AssertionError"):
            return line
    for line in reversed(lines):
        if "Error" in line.split(":")[0]:
            return line
    return lines[-1] if lines else ""


# ---- code: fixing a failing test -------------------------------------------------------------------

DIAGNOSES = {
    "total_skips_first": "total() sums rows[1:]: the first row is left out, so total gives 7.0 instead of 9.0 – and "
                         "average() uses total(), so it is wrong too. The sum must go over every row:\n"
                         "  total = sum of row[field] for each row\nFix: rows[1:] -> rows.",
    "average_empty": "average([]) divides by len(rows), which is 0: ZeroDivisionError. The test wants 0 when there "
                     "are no rows:\n  if no rows: return 0\n  return total / count",
    "average_off_by_one": "average() divides by len(rows) + 1: 9 / 4 = 2.25, not 9 / 3 = 3. The mean is the total "
                          "divided by the number of rows:\n  return total(rows, field) / len(rows)",
    "top_smallest": "top() sorts smallest first, so it gives a and b instead of c and b. Largest first needs "
                    "reverse=True:\n  sort rows by field, largest first; keep the first n",
    "count_by_one": "count_by() sets each count to 1 instead of adding one: x appears twice but counts 1.\n"
                    "  counts[value] = (counts[value] or 0) + 1",
    "total_by_last": "total_by() keeps only the last row's value for a group (x: 4.0) instead of adding them "
                     "(2.0 + 4.0 = 6.0):\n  totals[group] = (totals[group] or 0) + row[field]",
}

AFTER = {
    "total_skips_first": "sums every row",
    "average_empty": "gives 0 when there are no rows",
    "average_off_by_one": "divides the total by the number of rows",
    "top_smallest": "sorts largest first",
    "count_by_one": "adds one for each row",
    "total_by_last": "adds up the values of each group",
}


def fix_bug(rng):
    theme = rng.choice(THEMES)
    bug = rng.choice(list(BUGS))
    effort = SUPERIOR if rng.random() < 0.4 else STANDARD
    p = theme["pkg"]
    world = World(project_root(theme, rng), project(theme, rng, rng.random() < 0.6, bug, extra=rng.choice([0, 0, 2, 4])))
    work = Work(rng, world, effort)
    root = world.path()
    work.ask(pick(rng, "The tests in %s fail. Can you fix it?" % root,
                  "`python -m unittest` fails in my %s project (%s). Please find out why and fix the code."
                  % (theme["title"], root),
                  "Something broke in %s – the tests don't pass any more. Fix it please." % root,
                  "Fix the failing tests in %s." % root,
                  "My %s code at %s has a bug, the tests show it. Can you sort it out?" % (theme["title"], root)))
    if effort == SUPERIOR:
        plan = ("Failing tests in a project I have not seen. Plan:\n1. Look at the project.\n2. Run the tests to see "
                "what fails.\n3. Read the failing test and the code it calls.\n4. Fix the code, not the test.\n"
                "5. Run the tests again, and the program once.")
    else:
        plan = "Failing tests: see the project, run the tests, read what fails, fix the code, run them again."
    work.step(plan, "list_files", folder=root)
    out = work.step("%s/ is the code, tests/ the tests. Run them to see what fails." % p,
                    "run_command", command="python -m unittest", folder=root)
    bad, ran, _ = unittest_result(out["output"])
    names = [name for _, name in bad]
    function = BUGS[bug][0]
    seen = "%s of %d fail: %s. %s" % (plural(len(bad), "test"), ran, listing(names), error_line(out["output"]))
    if effort == SUPERIOR or rng.random() < 0.5:
        work.step(seen + "\nRead the tests to see what is expected.", "read_file",
                  path=world.path("tests/test_stats.py"))
        lead = "The test calls %s() with three rows (2.0, 3.0, 4.0). Find %s in the code." % (function, function)
    else:
        lead = seen + "\nIt comes from %s(). Find it." % function
    found = work.step(lead, "search_files", text="def %s" % function, folder=root)
    match = found["matches"][0]
    line = int(match.split(":")[1])
    work.step("%s is in %s at line %d. Read it." % (function, match.split(":")[0], line), "read_file",
              path=world.path(p + "/stats.py"), **{"from": line, "lines": 8})
    _, wrong, right, what = BUGS[bug]
    denied = rng.random() < 0.06
    result = work.step(DIAGNOSES[bug], "edit_file", simulated={"error": "the user did not allow this call"} if denied
                       else None, path=world.path(p + "/stats.py"), find=wrong.rstrip("\n"),
                       replace=right.rstrip("\n"))
    if denied:
        work.answer("The bug is in `%s()` in `%s/stats.py`: %s. You did not allow me to change the file, so here is "
                    "the fix to make yourself – replace\n\n```python\n%s```\n\nwith\n\n```python\n%s```"
                    % (function, p, what, wrong, right),
                    "The change was not allowed: do not try again; give the fix instead.")
        world.close()
        return work.sample()
    out = work.step("Changed. Run the tests again.", "run_command", command="python -m unittest", folder=root)
    _, ran, ok = unittest_result(out["output"])
    checked = ""
    if effort == SUPERIOR:
        report = work.step("All %d pass. Run the program once on the real data, as the README says." % ran,
                           "run_command", command="python main.py data/%s" % theme["csv"], folder=root)
        first = report["output"].strip().splitlines()[:2]
        checked = " The report runs too (%s)." % "; ".join(first)
    world.close()
    work.answer("Fixed `%s()` in `%s/stats.py`: %s; now it %s. All %d tests pass.%s"
                % (function, p, what, AFTER[bug], ran, checked),
                "%d tests, all OK. Say what was wrong and what changed." % ran if ok else None)
    return work.sample()


# ---- code: adding an option, with MagiDB ---------------------------------------------------------------


def add_json_option(rng):
    theme = rng.choice(THEMES)
    effort = SUPERIOR if rng.random() < 0.4 else STANDARD
    world = World(project_root(theme, rng), project(theme, rng, rng.random() < 0.6))
    work = Work(rng, world, effort)
    root = world.path()
    work.ask(pick(rng, "Add a --json option to main.py in %s that prints the summary as JSON instead of lines." % root,
                  "Can you make %s/main.py print JSON when I pass --json?" % root,
                  "I need the %s report as JSON. Add a --json flag to main.py (%s)." % (theme["title"], root)))
    work.step("A --json flag for main.py. Plan:\n1. Read main.py.\n2. Look up argparse flags and json in MagiDB.\n"
              "3. Add the flag, and print json.dumps(summary) when it is set.\n4. Run it with and without --json; "
              "run the tests.", "read_file", path=world.path("main.py"))
    found = work.step("main.py parses one argument (path) with argparse and prints summary(rows) line by line. "
                      "I need the exact way to add an on/off flag: MagiDB.", "db_search",
                      query="python argparse flag option")
    page = found["results"][0]["page"]
    work.step("%s should have it." % page, "db_read", page=page)
    found = work.step("A flag is add_argument(\"--json\", action=\"store_true\"). Now the exact json call.",
                      "db_search", query="python json dumps")
    page = found["results"][0]["page"]
    work.step("%s." % page, "db_read", page=page)
    work.step("json.dumps(data, indent=2) gives the text. Changes:\n- import json\n- the --json argument\n"
              "- if args.json: print(json.dumps(summary(rows), indent=2)) else: print each line\n"
              "First the import.", "edit_file", path=world.path("main.py"), find="import argparse\n",
              replace="import argparse\nimport json\n")
    work.step("Next: the argument.", "edit_file", path=world.path("main.py"), find='    parser.add_argument("path")',
              replace='    parser.add_argument("path")\n    parser.add_argument("--json", action="store_true", '
                      'help="print the summary as JSON")')
    work.step("Last: print JSON when it is set.", "edit_file", path=world.path("main.py"),
              find="    for line in summary(rows):\n        print(line)",
              replace="    if args.json:\n        print(json.dumps(summary(rows), indent=2))\n    else:\n"
                      "        for line in summary(rows):\n            print(line)")
    out = work.step("Try it with --json.", "run_command", command="python main.py data/%s --json" % theme["csv"],
                    folder=root)
    start = out["output"].strip().splitlines()[:2]
    if effort == SUPERIOR:
        work.step("JSON comes out (%s …). Without --json it must still print lines." % " ".join(start),
                  "run_command", command="python main.py data/%s" % theme["csv"], folder=root)
    out = work.step("Run the tests too.", "run_command", command="python -m unittest", folder=root)
    _, ran, ok = unittest_result(out["output"])
    world.close()
    work.answer("Added `--json` to `main.py`: `python main.py data/%s --json` prints the summary as a JSON list "
                "(with `json.dumps(..., indent=2)`); without it, the lines print as before. The %d tests still pass."
                % (theme["csv"], ran))
    return work.sample()


# ---- code: renaming across many files -----------------------------------------------------------------


def rename(rng):
    theme = rng.choice(THEMES)
    effort = SUPERIOR if rng.random() < 0.3 else STANDARD
    extra = rng.choice([4, 6, 9, 12, 16, 20, 24, 28, 30])
    world = World(project_root(theme, rng), project(theme, rng, rng.random() < 0.5, extra=extra))
    work = Work(rng, world, effort)
    root, p = world.path(), theme["pkg"]
    old = "total_by"
    new = rng.choice(["sum_by", "totals_by", "sum_per", "group_totals"])
    work.ask(pick(rng, "Rename %s to %s everywhere in %s." % (old, new, root),
                  "In %s, can you rename the function %s to %s – every use of it, tests included?" % (root, old, new),
                  "Please rename `%s` → `%s` across the whole project at %s and make sure nothing breaks."
                  % (old, new, root)))
    found = work.step("A rename across the project. Plan:\n1. Find every line with %s.\n2. Change each one.\n"
                      "3. Search again: nothing must be left.\n4. Run the tests." % old,
                      "search_files", text=old, folder=root)
    matches = found["matches"]
    files = []
    for match in matches:
        name = match.split(":")[0]
        if name not in files:
            files.append(name)
    todo = [(m.split(":")[0], m.split(":", 2)[2].strip()) for m in matches]
    work_list = "%s in %s:\n%s" % (plural(len(todo), "line"), plural(len(files), "file"),
                                   "\n".join("- " + f for f in files))
    for index, (path, text) in enumerate(todo):
        done = [f for f in files[:files.index(path)]]
        if index == 0:
            thinking = work_list + "\nOne edit per line, the line copied exactly. First: %s." % path
        elif rng.random() < 0.3 or effort == SUPERIOR and index % 3 == 0:
            left = len(todo) - index
            thinking = "Done: %s. Next: %s (%s left)." % (
                listing(done[-3:]) if done else todo[index - 1][0], path, plural(left, "line"))
        else:
            thinking = "Next: %s." % path
        if effort == SUPERIOR and (index == 0 or todo[index - 1][0] != path) and rng.random() < 0.4:
            work.step(thinking + " Read it first.", "read_file", path=world.path(path))
            thinking = "The line is there as search_files showed it."
        result = work.step(thinking, "edit_file", path=world.path(path), find=text, replace=text.replace(old, new))
        if "error" in result:
            # There twice: read it and give more of the text.
            content = world.get(path)
            work.step("It is in %s more than once: read the file and take more of it." % path, "read_file",
                      path=world.path(path))
            at = content.index(text)
            start = content.rfind("\n", 0, at) + 1
            end = content.find("\n", at + len(text))
            longer = content[start:content.find("\n", end + 1)]
            work.step("Take the line with the one after it.", "edit_file", path=world.path(path), find=longer,
                      replace=longer.replace(old, new, 1))
    left = work.step("Every line is changed. Search again: nothing with %s may be left." % old, "search_files",
                     text=old, folder=root)
    out = work.step("None left. Run the tests." if not left["matches"] else "Some left: change them too.",
                    "run_command", command="python -m unittest", folder=root)
    _, ran, ok = unittest_result(out["output"])
    world.close()
    work.answer("Renamed `%s` to `%s` in %s – %s – and checked that no `%s` is left. All %d tests pass."
                % (old, new, plural(len(files), "file"), plural(len(todo), "line"), old, ran))
    return work.sample()


# ---- code: a docstring for every function ----------------------------------------------------------------


def docstrings(rng):
    theme = rng.choice(THEMES)
    effort = SUPERIOR if rng.random() < 0.3 else STANDARD
    extra = rng.choice([2, 4, 6, 9, 12, 16, 20, 24, 28])
    world = World(project_root(theme, rng), project(theme, rng, docstrings=False, extra=extra))
    work = Work(rng, world, effort)
    root, p, value, group = world.path(), theme["pkg"], theme["value"], theme["group"]
    work.ask(pick(rng, "Add a one-line docstring to every function in %s (not the tests)." % root,
                  "None of the functions in %s have docstrings. Please add a short one to each, tests excepted." % root,
                  "Document every function in %s with a docstring." % root))
    found = work.step("A docstring for every function but the tests. Plan:\n1. Find every def.\n2. For each file: "
                      "read it, then add a one-line docstring under each def, saying what it gives.\n"
                      "3. Run the tests.", "search_files", text="def ", folder=root)
    defs = []
    for match in found["matches"]:
        path, line, text = match.split(":", 2)
        if path.startswith("tests/"):
            continue
        defs.append((path, int(line), text.strip()))
    files = []
    for path, _, _ in defs:
        if path not in files:
            files.append(path)
    first = ("%s in %s (leaving out tests/: test names say what they check):\n%s"
             % (plural(len(defs), "function"), plural(len(files), "file"), "\n".join("- " + f for f in files)))
    count = 0
    for number, path in enumerate(files):
        lead = first if number == 0 else ("Done: %s. Next: %s (%s left)." % (
            listing(files[max(0, number - 2):number]), path, plural(len(files) - number, "file"))
            if rng.random() < 0.5 else "Next: %s." % path)
        work.step(lead, "read_file", path=world.path(path))
        for _, _, text in [d for d in defs if d[0] == path]:
            name = text[4:text.index("(")]
            if name in DOCSTRINGS:
                doc = DOCSTRINGS[name].format(value=value)
            else:
                doc = "One line per %s: its total %s." % (group, value)
            work.step("%s: %s" % (name, doc), "edit_file", path=world.path(path), find=text + "\n",
                      replace='%s\n    """%s"""\n' % (text, doc))
            count += 1
    out = work.step("Every function has one. Run the tests: a docstring in the wrong place would break the code.",
                    "run_command", command="python -m unittest", folder=root)
    _, ran, ok = unittest_result(out["output"])
    world.close()
    work.answer("Added a one-line docstring to all %d functions in %s (the tests left as they are). The %d tests "
                "still pass." % (count, plural(len(files), "file"), ran))
    return work.sample()


# ---- code: a list of the TODO comments ----------------------------------------------------------------


def todo_list(rng):
    theme = rng.choice(THEMES)
    effort = SUPERIOR if rng.random() < 0.3 else STANDARD
    world = World(project_root(theme, rng),
                  project(theme, rng, rng.random() < 0.6, extra=rng.choice([2, 4, 6, 9]), todos=rng.randint(5, 22)))
    work = Work(rng, world, effort)
    root = world.path()
    work.ask(pick(rng, "Go through %s, find every TODO comment and put them in a TODO.md with where each one is." % root,
                  "Make me a TODO.md listing all the TODOs in the code at %s (file and line, and what each is about)."
                  % root,
                  "Collect the TODO comments of %s into TODO.md." % root))
    found = work.step("Every TODO into TODO.md. Plan:\n1. search_files for TODO.\n2. Read around each to see what it "
                      "is about.\n3. Write TODO.md: file, line, what to do.", "search_files", text="TODO", folder=root)
    items = []
    for match in found["matches"]:
        path, line, text = match.split(":", 2)
        items.append((path, int(line), text.strip()[len("# TODO: "):]))
    rows = []
    for index, (path, line, job) in enumerate(items):
        thinking = ("%s. Read around each." % plural(len(items), "TODO") if index == 0 else
                    "Next: %s line %d (%d left)." % (path, line, len(items) - index))
        read = work.step(thinking, "read_file", path=world.path(path), **{"from": max(1, line - 1), "lines": 4})
        after = [l.split("| ", 1)[1].strip() for l in read["text"].splitlines()[1:] if "| " in l]
        where = after[1] if len(after) > 1 else ""
        rows.append("| `%s` | %d | %s | `%s` |" % (path, line, job[0].upper() + job[1:], where[:60]))
    text = ("# TODO\n\nThe TODO comments in the code, as of now.\n\n| File | Line | To do | At |\n| --- | --- | --- | --- |\n"
            + "\n".join(rows) + "\n")
    work.step("All read. Write TODO.md.", "write_file", path=world.path("TODO.md"), text=text)
    world.close()
    work.answer("Wrote `TODO.md` with the %s in the code: for each, the file, the line, what to do and the code it "
                "is next to." % plural(len(items), "TODO comment"))
    return work.sample()


# ---- code: a new script from a data file -------------------------------------------------------------

SCRIPTS = {
    "totals": {
        "ask": "Write a script in {root} that prints the total {value} for each {group}, largest first.",
        "file": "totals_by_{group}.py",
        "plan": "A script over {csv}:\n  read the rows\n  for each row: totals[{group}] += {value} (as a number)\n"
                "  sort the totals, largest first; print each",
        "lookups": [("python read csv file", "python/csv.md"), ("group and total by key", "algorithms/counting.md"),
                    ("sort dictionary by value largest first", "python/collections.md")],
        "code": "import csv\nfrom collections import defaultdict\n\ntotals = defaultdict(float)\n"
                'with open("{csv}", newline="", encoding="utf-8") as f:\n    for row in csv.DictReader(f):\n'
                '        totals[row["{GROUP}"]] += float(row["{value}"])\n\n'
                "for name, total in sorted(totals.items(), key=lambda item: item[1], reverse=True):\n"
                '    print(f"{{name}}: {{total:.1f}}")\n',
        "says": "prints each {group}'s total {value}, largest first",
    },
    "averages": {
        "ask": "Make a script in {root} that works out the average {value} per {group} from {csv} and saves it "
               "as averages.json.",
        "file": "averages.py",
        "plan": "A script over {csv}:\n  read the rows\n  collect the {value}s of each {group} (as numbers)\n"
                "  average = sum / count for each\n  write them to averages.json",
        "lookups": [("python read csv file", "python/csv.md"), ("python write json file", "python/json.md")],
        "code": "import csv\nimport json\nfrom collections import defaultdict\n\nvalues = defaultdict(list)\n"
                'with open("{csv}", newline="", encoding="utf-8") as f:\n    for row in csv.DictReader(f):\n'
                '        values[row["{GROUP}"]].append(float(row["{value}"]))\n\n'
                "averages = {{name: round(sum(v) / len(v), 2) for name, v in values.items()}}\n"
                'with open("averages.json", "w", encoding="utf-8") as f:\n    json.dump(averages, f, indent=2)\n'
                'print(f"averages of {{len(averages)}} {group} values written to averages.json")\n',
        "says": "writes the average {value} of each {group} to averages.json",
        "check": "averages.json",
    },
    "above": {
        "ask": "I need a script in {root} that copies the rows of {csv} whose {value} is above {limit} into big.csv.",
        "file": "above_{limit}.py",
        "plan": "A script over {csv}:\n  read the rows with their header\n  keep those whose {value} (as a number) "
                "> {limit}\n  write them, with the same header, to big.csv",
        "lookups": [("python csv read and write rows", "python/csv.md")],
        "code": "import csv\n\n"
                'with open("{csv}", newline="", encoding="utf-8") as f:\n    reader = csv.DictReader(f)\n'
                "    fields = reader.fieldnames\n"
                '    rows = [row for row in reader if float(row["{value}"]) > {limit}]\n\n'
                'with open("big.csv", "w", newline="", encoding="utf-8") as f:\n'
                "    writer = csv.DictWriter(f, fieldnames=fields)\n    writer.writeheader()\n"
                "    writer.writerows(rows)\n"
                'print(f"{{len(rows)}} rows written to big.csv")\n',
        "says": "copies the rows with {value} above {limit} into big.csv",
        "check": "big.csv",
    },
    "common": {
        "ask": "Write a script in {root} that shows the 3 most common {key}s in {csv} with how often each appears.",
        "file": "common_{key}s.py",
        "plan": "A script over {csv}:\n  read the rows\n  count each {key}\n  print the 3 most common with their counts",
        "lookups": [("python read csv file", "python/csv.md"), ("count most common items", "python/collections.md")],
        "code": "import csv\nfrom collections import Counter\n\n"
                'with open("{csv}", newline="", encoding="utf-8") as f:\n'
                '    counts = Counter(row["{KEY}"] for row in csv.DictReader(f))\n\n'
                "for name, count in counts.most_common(3):\n"
                '    print(f"{{name}}: {{count}}")\n',
        "says": "prints the 3 most common {key}s with their counts",
    },
}


def new_script(rng):
    theme = rng.choice(THEMES)
    kind = rng.choice(list(SCRIPTS))
    recipe = SCRIPTS[kind]
    effort = SUPERIOR if rng.random() < 0.35 else STANDARD
    folder = rng.choice(["C:/Users/me/Documents/{name}", "C:/Users/me/Desktop/{name}", "D:/data/{name}",
                         "C:/Users/me/Downloads/{name}"]).format(name=theme["pkg"] + "-data")
    low, high = theme["range"]
    limit = rng.randint(low + (high - low) // 3, high - (high - low) // 4)
    fill = {"root": folder, "csv": theme["csv"], "group": theme["group"], "value": theme["value"],
            "key": theme["key"], "limit": limit}
    files = {theme["csv"]: csv_text(theme, rng, rng.randint(14, 40))}
    if rng.random() < 0.5:
        files["notes.txt"] = "Exported from the %s sheet.\n" % theme["title"]
    world = World(folder, files)
    work = Work(rng, world, effort)
    work.ask(recipe["ask"].format(**fill))
    work.step("A new script. First see the data.", "list_files", folder=folder)
    head = work.step("%s is there. Read its first lines for the columns." % theme["csv"], "read_file",
                     path=world.path(theme["csv"]), lines=4)
    work_plan = recipe["plan"].format(**fill)
    looked = []
    for index, (query, wanted) in enumerate(recipe["lookups"]):
        thinking = ("Columns: %s, %s, %s. Plan:\n%s\nThe exact code from MagiDB, not from memory."
                    % (theme["key"], theme["group"], theme["value"], work_plan) if index == 0
                    else "Next I need: %s." % query)
        found = work.step(thinking, "db_search", query=query)
        pages = [r["page"] for r in found.get("results", [])]
        page = wanted if wanted in pages else (pages[0] if pages else None)
        if page:
            work.step("%s." % page, "db_read", page=page)
            looked.append(page)
    # A first try that names a column wrong (a capital letter), now and then: the error leads to the header.
    mistake = rng.random() < 0.3
    upper = {"GROUP": theme["group"], "KEY": theme["key"]}
    if mistake:
        upper = {"GROUP": theme["group"].capitalize(), "KEY": theme["key"].capitalize()}
    code = recipe["code"].format(**fill, **upper)
    name = recipe["file"].format(**fill)
    # The wrong first try is read, not learned: the model learns the recovery, not the mistake.
    work.step("From %s:\n%s" % (listing(looked), work_plan) if looked else "Write it.", "write_file",
              learn=not mistake, path=world.path(name), text=code)
    out = work.step("Run it.", "run_command", command="python " + name, folder=folder)
    if out["exit"] != 0:
        error = error_line(out["output"])
        wrong = upper["GROUP"] if upper["GROUP"] in code and upper["GROUP"] != theme["group"] else upper["KEY"]
        right = theme["group"] if wrong == upper["GROUP"] else theme["key"]
        work.step("%s: no column named %s. The header said %s – lower case." % (error, wrong, right), "edit_file",
                  path=world.path(name), find='row["%s"]' % wrong, replace='row["%s"]' % right)
        out = work.step("Run it again.", "run_command", command="python " + name, folder=folder)
    shown = out["output"].strip().splitlines()
    if recipe.get("check") and (effort == SUPERIOR or rng.random() < 0.5):
        work.step("It ran: %s. Check what it wrote." % (shown[-1] if shown else "no output"), "read_file",
                  path=world.path(recipe["check"]), lines=8)
    world.close()
    says = recipe["says"].format(**fill)
    text = "Wrote `%s` in %s: it %s. Run it with `python %s` there." % (name, folder, says, name)
    if not recipe.get("check") and shown:
        text += " On your data:\n\n```\n%s\n```" % "\n".join(shown[:8])
    elif shown:
        text += " On your data: %s." % shown[-1]
    work.answer(text)
    return work.sample()


# ---- documents ---------------------------------------------------------------------------------------

ITEMS = ["kettle", "desk lamp", "mug", "rug", "shelf", "stool", "teapot", "vase", "chair", "mirror", "clock",
         "pillow", "blanket", "candle", "frame", "basket", "bowl", "jug", "tray", "planter", "lantern", "coaster set",
         "bookend", "doormat", "hook rail", "towel", "apron", "napkin set", "bread bin", "spice rack", "cushion",
         "side table", "coat hook", "plant pot", "cutting board", "salad bowl", "wine rack", "key tray", "bath mat",
         "shower curtain", "tea towel", "egg cup", "butter dish", "cake stand", "fruit bowl", "storage box",
         "picture ledge", "floor lamp", "wall clock", "throw", "laundry basket"]


def document_result(path, kind, parts):
    lines = ["%s: %s" % (address, text) for address, text in parts]
    return {"path": path, "kind": kind, "parts": len(parts), "text": "\n".join(lines), "editable": kind != "pdf"}


def raise_prices(rng):
    effort = SUPERIOR if rng.random() < 0.3 else STANDARD
    folder = rng.choice(["C:/Users/me/Documents/shop", "D:/business", "C:/Users/me/Desktop"])
    path = folder + "/" + rng.choice(["prices.xlsx", "price list.xlsx", "catalogue.xlsx"])
    sheet = rng.choice(["Prices", "Sheet1", "2026"])
    percent = rng.choice([3, 5, 8, 10, 12, 15])
    count = rng.choice([6, 8, 12, 16, 20, 25, 30, 40, 50])
    items = rng.sample(ITEMS, count)
    prices = [rng.choice([rng.randint(2, 90) + rng.choice([0, 0.5, 0.99, 0.25]), rng.randint(5, 250)])
              for _ in items]
    parts = [("%s!A1" % sheet, "Item"), ("%s!B1" % sheet, "Price")]
    for row, (item, price) in enumerate(zip(items, prices), 2):
        parts += [("%s!A%d" % (sheet, row), item.capitalize()), ("%s!B%d" % (sheet, row), str(price))]
    work = Work(rng, effort=effort)
    work.ask(pick(rng, "Raise every price in %s by %d%%." % (path, percent),
                  "Prices go up %d%% next month – can you update %s?" % (percent, path),
                  "Put all the prices in %s up by %d%%, rounded to cents." % (path, percent)))
    work.step("An .xlsx file: open_document, not read_file. Then each price times %s, from the calculator, "
              "rounded to cents, and one edit_document with all of them." % (1 + percent / 100),
              "open_document", simulated=document_result(path, "xlsx", parts), path=path)
    edits = []
    factor = "%g" % (1 + percent / 100)
    for index, (item, price) in enumerate(zip(items, prices)):
        thinking = ("%d prices, in B2 to B%d. %s first." % (count, count + 1, item.capitalize()) if index == 0
                    else ("%s (%d left)." % (item.capitalize(), count - index) if rng.random() < 0.4
                          else item.capitalize() + "."))
        result = tools()["calculator"]({"expression": "%s * %s" % (price, factor)})
        work.step(thinking, "calculator", simulated=result, expression="%s * %s" % (price, factor))
        new = round(float(result["result"]) + 1e-9, 2)
        edits.append({"at": "%s!B%d" % (sheet, index + 2), "text": "%.2f" % new})
    work.step("All %d worked out. Write them in one go." % count, "edit_document",
              simulated={"path": path, "changed": [e["at"] for e in edits], "backup": path + ".magi-backup"},
              path=path, edits=edits)
    sample_rows = "\n".join("| %s | %s | %s |" % (item.capitalize(), price, e["text"])
                            for item, price, e in list(zip(items, prices, edits))[:5])
    work.answer("Raised all %d prices in `%s` by %d%%, rounded to cents (a copy of the file as it was is kept as "
                "`%s.magi-backup`). For example:\n\n| Item | Was | Now |\n| --- | --- | --- |\n%s"
                % (count, path, percent, path.rsplit("/", 1)[1], sample_rows))
    return work.sample()


TYPOS = {"recieve": "receive", "seperate": "separate", "definately": "definitely", "occured": "occurred",
         "untill": "until", "wich": "which", "acommodate": "accommodate", "goverment": "government",
         "tommorow": "tomorrow", "begining": "beginning", "existance": "existence", "beleive": "believe"}

SENTENCES = ["The team will {w} the new samples on Monday.", "We kept the two projects {w} this year.",
             "The results were {w} better than in March.", "The delay {w} because the parts arrived late.",
             "Nothing changes {w} the board agrees.", "The budget, {w} was approved in May, covers travel.",
             "The hotel can {w} twenty guests.", "The {w} grant pays for half of the costs.",
             "The review starts {w} at nine.", "From the {w}, the plan was to grow slowly.",
             "The {w} of the old contract makes this harder.", "We {w} the new design is clearer."]
TYPO_SLOT = {"recieve": 0, "seperate": 1, "definately": 2, "occured": 3, "untill": 4, "wich": 5, "acommodate": 6,
             "goverment": 7, "tommorow": 8, "begining": 9, "existance": 10, "beleive": 11}
PLAIN = ["This report covers the first half of the year.", "Sales grew in every region but the north.",
         "Costs stayed close to the plan.", "Two new people joined the support team.",
         "The website moved to a faster host in April.", "Customer questions are answered within a day.",
         "The warehouse lease runs until 2028.", "Next steps are listed at the end."]


def fix_typos(rng):
    effort = SUPERIOR if rng.random() < 0.3 else STANDARD
    path = rng.choice(["C:/Users/me/Documents/report.docx", "D:/work/minutes.docx", "C:/Users/me/Desktop/letter.docx"])
    wrong = rng.sample(list(TYPOS), rng.randint(2, 6))
    paragraphs = [("p1", rng.choice(["Half-year report", "Minutes of the March meeting", "Project update"]))]
    body = [SENTENCES[TYPO_SLOT[w]].format(w=w) for w in wrong] + rng.sample(PLAIN, rng.randint(3, 7))
    rng.shuffle(body)
    paragraphs += [("p%d" % i, text) for i, text in enumerate(body, 2)]
    work = Work(rng, effort=effort)
    work.ask(pick(rng, "Fix the spelling mistakes in %s." % path, "Can you correct the typos in %s?" % path,
                  "Please proofread %s and fix any misspelled words." % path))
    work.step("A .docx file: open_document gives its paragraphs with their numbers.", "open_document",
              simulated=document_result(path, "docx", paragraphs), path=path)
    edits, fixed = [], []
    for address, text in paragraphs:
        for typo in wrong:
            if typo in text:
                edits.append({"at": address, "text": text.replace(typo, TYPOS[typo])})
                fixed.append("%s → %s (%s)" % (typo, TYPOS[typo], address))
    work.step("Misspelled: %s. Each paragraph rewritten whole with the word fixed." % "; ".join(fixed),
              "edit_document", simulated={"path": path, "changed": [e["at"] for e in edits],
                                          "backup": path + ".magi-backup"}, path=path, edits=edits)
    work.answer("Fixed %s in `%s`: %s. The rest of the document is unchanged, and the original is kept as a "
                "`.magi-backup` copy beside it." % (plural(len(edits), "spelling mistake"), path.rsplit("/", 1)[1],
                                                     ", ".join(f.split(" (")[0] for f in fixed)))
    return work.sample()


MEETINGS = ["budget", "hiring", "launch", "office move", "safety review", "supplier", "website", "training"]


def summarize_pdfs(rng):
    effort = SUPERIOR if rng.random() < 0.3 else STANDARD
    folder = rng.choice(["C:/Users/me/Documents/minutes", "D:/board/2026", "C:/Users/me/Downloads/meetings"])
    count = rng.choice([3, 4, 6, 8, 10, 14, 20])
    topics = rng.sample(MEETINGS, min(count, len(MEETINGS)))
    while len(topics) < count:
        topics.append(rng.choice(MEETINGS))
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]
    docs = []
    for index, topic in enumerate(topics):
        month = months[index % len(months)]
        name = "%02d-%s.pdf" % (index + 1, topic.replace(" ", "-"))
        owner = rng.choice(["Priya", "Tom", "Lena", "Omar", "Yuki", "Mateo", "Grace", "Ivan"])
        date = "%d %s" % (rng.randint(1, 28), rng.choice(months))
        decision = rng.choice(["approved", "postponed", "approved with changes", "sent back for more figures"])
        pages = [("page1", "Minutes – %s meeting, %s 2026. Present: %s and four others. The %s plan was %s."
                  % (topic, month, owner, topic, decision)),
                 ("page2", "Actions: %s writes the next draft by %s. Open question: the cost of the %s."
                  % (owner, date, rng.choice(["contractors", "new tools", "travel", "extra hours"])))]
        docs.append((name, pages, topic, decision, owner, date))
    world = World(folder, {name: "" for name, *_ in docs})
    work = Work(rng, world, effort)
    work.ask(pick(rng, "Summarize the meeting minutes in %s into one summary.md: the decision and the actions of each."
                  % folder,
                  "Read all the PDFs in %s and write me a short summary.md of what was decided." % folder))
    work.step("Every PDF in the folder, then one summary. First, the list.", "list_files", folder=folder)
    lines = []
    for index, (name, pages, topic, decision, owner, date) in enumerate(docs):
        thinking = ("%s. PDFs open with open_document. First %s." % (plural(count, "PDF"), name) if index == 0
                    else "Noted %s: %s. Next: %s (%d left)." % (docs[index - 1][2], docs[index - 1][3], name,
                                                                count - index))
        work.step(thinking, "open_document", simulated=document_result(folder + "/" + name, "pdf", pages),
                  path=folder + "/" + name)
        lines.append("## %s\n\n- Decision: the %s plan was %s.\n- Action: %s writes the next draft by %s.\n"
                     % (topic.capitalize(), topic, decision, owner, date))
    text = "# Meeting summary\n\n" + "\n".join(lines)
    work.step("All %d read. Write summary.md." % count, "write_file", path=world.path("summary.md"), text=text)
    world.close()
    work.answer("Wrote `summary.md` in %s with the decision and the action of each of the %d meetings (%s)."
                % (folder, count, listing(t for _, _, t, *_ in docs[:4]) + (" …" if count > 4 else "")))
    return work.sample()


# ---- pictures ----------------------------------------------------------------------------------------

SCREENS = [
    ("a settings window with a dark theme; a list of options on the left, Notifications selected",
     "Settings\nNotifications\nShow previews\nDo not disturb"),
    ("a login page in a web browser with an e-mail field, a password field and a blue Sign in button",
     "Sign in\nE-mail\nPassword\nForgot your password?"),
    ("a spreadsheet with a bar chart of monthly sales; March is the tallest bar",
     "Sales 2026\nJan Feb Mar Apr\nTotal 48,210"),
    ("an error dialog over a code editor, with a red cross icon and an OK button",
     "Error\nThe file config.json could not be found.\nOK"),
    ("a terminal window with green text; the last line ends with a blinking cursor",
     "$ npm test\n12 passing\n1 failing"),
    ("a photo of a whiteboard with a hand-drawn flow chart of three boxes joined by arrows",
     "Plan -> Build -> Check"),
    ("a mobile app's home screen with four round icons and a search bar at the top",
     "Search\nPhotos  Maps  Mail  Notes"),
    ("a calendar week view with three coloured meetings on Tuesday and one on Friday",
     "Tue 10:00 Stand-up\nTue 14:00 Design review\nFri 16:00 Retro"),
    ("a file manager window listing folders named src, docs and tests",
     "src\ndocs\ntests\nREADME.md"),
    ("a chat window with two messages and a text box at the bottom",
     "Ana: Can you send the slides?\nYou: Sure, in five minutes."),
    ("a dashboard with three number cards and a line chart going up",
     "Users 1,284\nOrders 312\nRevenue $18,940"),
    ("a photo of a parked red bicycle against a brick wall", "(no text)"),
]


def view_result(path, description, text, rng):
    width, height = rng.choice([(1920, 1080), (1280, 720), (2560, 1440), (1080, 1920), (800, 600)])
    return {"path": path, "width": width, "height": height, "text": text, "description": description}


def catalog_screenshots(rng):
    effort = SUPERIOR if rng.random() < 0.3 else STANDARD
    folder = rng.choice(["C:/Users/me/Pictures/Screenshots", "D:/bugs/screens", "C:/Users/me/Desktop/shots"])
    count = rng.choice([5, 8, 12, 16, 24, 32, 40, 50, 60])
    shots = []
    for index in range(count):
        description, text = rng.choice(SCREENS)
        name = rng.choice(["Screenshot %d.png", "screen-%03d.png", "capture_%d.jpg"]) % (index + 1)
        if name in [s[0] for s in shots]:
            name = "shot-%d.png" % (index + 1)
        shots.append((name, description, text))
    world = World(folder, {name: "" for name, _, _ in shots})
    work = Work(rng, world, effort)
    work.ask(pick(rng, "Go through the screenshots in %s and make an index.md that says what each one shows." % folder,
                  "I have a folder of screenshots at %s. Write index.md with one line per picture describing it."
                  % folder))
    work.step("Each picture through view_image, then one index. First the list.", "list_files", folder=folder)
    rows = []
    for index, (name, description, text) in enumerate(shots):
        if index == 0:
            thinking = "%s. One at a time: %s." % (plural(count, "picture"), name)
        elif index % 6 == 0:
            thinking = "%d done, %d left. Next: %s." % (index, count - index, name)
        else:
            thinking = "Next: %s." % name
        work.step(thinking, "view_image", simulated=view_result(folder + "/" + name, description, text, rng),
                  path=folder + "/" + name)
        words = text.replace("\n", " · ") if text != "(no text)" else "no text"
        rows.append("| %s | %s | %s |" % (name, description[0].upper() + description[1:], words))
    content = "# Screenshots\n\n| File | Shows | Text in it |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n"
    work.step("All %d seen. Write index.md." % count, "write_file", path=world.path("index.md"), text=content)
    world.close()
    work.answer("Wrote `index.md` in %s: one row for each of the %d screenshots – what it shows and the text in it."
                % (folder, count))
    return work.sample()


def attached_picture(rng):
    description, text = rng.choice(SCREENS[:5] + SCREENS[8:11])
    path = rng.choice(["C:/Users/me/Pictures/", "C:/Users/me/Desktop/", "D:/temp/"]) + rng.choice(
        ["image.png", "screenshot.png", "photo.jpg", "error.png", "capture.png"])
    question = rng.choice(["What does this say?", "What is in this picture?", "Can you read this for me?",
                           "What am I looking at here?", "Explain what this shows."])
    work = Work(rng)
    work.ask("%s\n\nAttached: %s (picture)" % (question, path))
    result = work.step("A picture is attached: view_image first – I do not see pixels.", "view_image",
                       simulated=view_result(path, description, text, rng), path=path)
    answer = "It shows %s." % description
    if text != "(no text)":
        answer += " The text in it reads:\n\n> " + text.replace("\n", "\n> ")
    if "could not be found" in text:
        answer += ("\n\nThe program is looking for `config.json` and cannot find it: check that the file is in the "
                   "folder the program runs from, or that its path is spelled right.")
    work.answer(answer, "%s; the text: %s." % (result["description"], text.replace("\n", " / ")))
    return work.sample()


RECEIPT_SHOPS = ["Green Grocer", "City Pharmacy", "Book Nook", "Bike & Co", "Corner Cafe", "Hardware Hub",
                 "Fresh Market", "Paper Store"]


def receipts(rng):
    effort = SUPERIOR if rng.random() < 0.3 else STANDARD
    folder = rng.choice(["C:/Users/me/Documents/receipts", "D:/expenses/march", "C:/Users/me/Pictures/receipts"])
    count = rng.choice([4, 6, 8, 10, 14, 20, 30])
    items = []
    for index in range(count):
        shop = rng.choice(RECEIPT_SHOPS)
        total = round(rng.uniform(2, 120), 2)
        date = "2026-03-%02d" % rng.randint(1, 28)
        items.append(("receipt-%02d.jpg" % (index + 1), shop, total, date))
    world = World(folder, {name: "" for name, *_ in items})
    work = Work(rng, world, effort)
    work.ask(pick(rng, "Read the receipt photos in %s, put each one (shop, date, total) in expenses.csv and tell me "
                       "the grand total." % folder,
                  "Can you go through my receipts in %s and make an expenses.csv? I also need the total." % folder))
    work.step("Photos of receipts: view_image reads their text. Then a CSV, and the sum from the calculator.",
              "list_files", folder=folder)
    for index, (name, shop, total, date) in enumerate(items):
        text = "%s\n%s\nItems: %d\nTOTAL %.2f\nThank you!" % (shop.upper(), date, rng.randint(1, 6), total)
        thinking = ("%s. First %s." % (plural(count, "receipt"), name) if index == 0 else
                    "%s, %s, %.2f. Next: %s." % (items[index - 1][1], items[index - 1][3], items[index - 1][2], name))
        work.step(thinking, "view_image", path=folder + "/" + name, simulated=view_result(
            folder + "/" + name, "a photo of a printed shop receipt on a table", text, rng))
    expression = " + ".join("%.2f" % total for *_, total, _ in items)
    result = tools()["calculator"]({"expression": expression})
    work.step("All read. The grand total:", "calculator", simulated=result, expression=expression)
    grand = round(float(result["result"]) + 1e-9, 2)
    csv = "file,shop,date,total\n" + "".join("%s,%s,%s,%.2f\n" % (n, s, d, t) for n, s, t, d in items)
    work.step("%.2f. Write expenses.csv." % grand, "write_file", path=world.path("expenses.csv"), text=csv)
    world.close()
    work.answer("Wrote `expenses.csv` in %s with the %d receipts (shop, date, total). The grand total is %.2f."
                % (folder, count, grand))
    return work.sample()


# ---- MagiDB: questions, errors and git ---------------------------------------------------------------

QUESTIONS = [
    ("How do I read a CSV file in Python?", "python read csv file"),
    ("What's the right way to read a whole text file in Python?", "python read file"),
    ("How can I write JSON to a file in Python?", "python write json file"),
    ("How do I make an HTTP request in Python without installing anything?", "python http request urllib"),
    ("How do I run another program from Python and get its output?", "python run program subprocess output"),
    ("How do I format a number with two decimals in an f-string?", "python f-string format decimals"),
    ("How do I find all email addresses in a text with Python?", "python regex find all"),
    ("How do I sort a list of dicts by a key in Python?", "python sort list of dicts by key"),
    ("How do I add 7 days to today's date in Python?", "python date add days timedelta"),
    ("How do I add a --verbose flag to a Python script?", "python argparse flag"),
    ("How do I write a unit test in Python without pytest?", "python unittest test"),
    ("How do I install a package for the Python I'm running?", "python pip install package"),
    ("How do I log messages in Python instead of print?", "python logging"),
    ("How do I list every .py file in a folder and its subfolders?", "python list files folder glob"),
    ("How do I fetch JSON from an API in JavaScript?", "javascript fetch json"),
    ("How do I remove duplicates from an array in JavaScript?", "javascript array unique"),
    ("How do I read a file in Node.js?", "node read file"),
    ("How do I read a file to a string in Rust?", "rust read file"),
    ("How do I handle errors with ? in Rust?", "rust errors result"),
    ("How do I count words with a HashMap in Rust?", "rust hashmap count"),
    ("How do I undo my last git commit but keep the changes?", "git undo last commit keep changes"),
    ("How do I create a new branch in git and switch to it?", "git new branch"),
    ("How do I see what changed before committing?", "git diff status"),
    ("How do I find text in all files from the Windows command line?", "windows find text in files"),
    ("How do I get the 10 biggest orders with SQL?", "sql select order by limit"),
    ("How do I use SQLite from Python safely?", "sqlite python placeholders"),
    ("How do I read an Excel file in Python?", "python excel openpyxl"),
    ("How do I create a Word document from Python?", "python word docx"),
    ("How do I do a binary search in Python?", "binary search sorted list"),
    ("What's the easiest way to count how often each item appears?", "count items frequency"),
]


def first_code(text):
    if "```" not in text:
        return ""
    block = text.split("```", 2)[1]
    return "```" + block + "```"


def magidb_question(rng):
    question, query = rng.choice(QUESTIONS)
    world = World("C:/Users/me", {})
    work = Work(rng, world, rng.choice([STANDARD, STANDARD, SUPERIOR]))
    work.ask(question)
    found = work.step("A how-to: the exact code is in MagiDB, not in my memory.", "db_search", query=query)
    results = found.get("results", [])
    if not results:
        world.close()
        return None
    page = results[0]["page"]
    read = work.step("%s answers it." % results[0]["title"], "db_read", page=page)
    world.close()
    code = first_code(read.get("text", ""))
    summary = results[0]["summary"]
    work.answer("From MagiDB (%s): %s\n\n%s" % (page, summary, code) if code
                else "From MagiDB (%s): %s" % (page, summary))
    return work.sample()


TRACEBACKS = [
    ("Traceback (most recent call last):\n  File \"app.py\", line 3, in <module>\n    import requests\n"
     "ModuleNotFoundError: No module named 'requests'", "ModuleNotFoundError No module named",
     "The `requests` package is not installed in the Python you ran. Install it into that same Python with "
     "`python -m pip install requests`, then run the script again."),
    ("Traceback (most recent call last):\n  File \"report.py\", line 12, in <module>\n    total += row['Amount']\n"
     "KeyError: 'Amount'", "KeyError python",
     "The row has no key `'Amount'` – the column is probably spelled differently (keys are case-sensitive). Print "
     "the row's keys (`print(row.keys())`) and use the exact name, or `row.get('amount')`."),
    ("  File \"main.py\", line 8\n    print(total)\nIndentationError: unexpected indent", "IndentationError python",
     "Line 8 is indented more than the code around it. Python uses indentation for blocks: line it up with the "
     "lines above (four spaces per level, no tabs)."),
    ("TypeError: can only concatenate str (not \"int\") to str", "TypeError can only concatenate str",
     "Text and a number are being joined with `+`. Turn the number into text first – `f\"Total: {total}\"` or "
     "`\"Total: \" + str(total)`."),
    ("TypeError: Cannot read properties of undefined (reading 'name')", "javascript Cannot read properties of undefined",
     "Something before `.name` is `undefined` – the object you expect is not there yet. Find where it comes from "
     "and check it exists first, or write `obj?.name`."),
    ("fatal: not a git repository (or any of the parent directories): .git", "git not a git repository",
     "Git is being run outside a repository. Run it in the project's folder (the one with the `.git` folder), or "
     "create a repository there with `git init`."),
    ("'python' is not recognized as an internal or external command, operable program or batch file.",
     "is not recognized as an internal or external command",
     "Windows cannot find Python: it is not installed, or not on the PATH. Try `py` instead of `python`, or install "
     "Python and tick *Add python.exe to PATH*."),
    ("PermissionError: [Errno 13] Permission denied: 'report.xlsx'", "PermissionError python",
     "The file is open in another program (often Excel) or protected. Close it there and run the script again."),
]


def explain_error(rng):
    trace, query, explanation = rng.choice(TRACEBACKS)
    world = World("C:/Users/me", {})
    work = Work(rng, world, rng.choice([STANDARD, SUPERIOR]))
    work.ask(rng.choice(["What does this error mean?", "I get this, how do I fix it?", "Help, what's wrong here:",
                         "Why does this happen?"]) + "\n\n```\n" + trace + "\n```")
    found = work.step("An error message: its last line, in MagiDB.", "db_search", query=query)
    results = found.get("results", [])
    errors = [r["page"] for r in results if r["page"].startswith("errors/")]
    page = errors[0] if errors else (results[0]["page"] if results else None)
    if page:
        work.step("%s lists it." % page, "db_read", page=page)
    world.close()
    work.answer(explanation)
    return work.sample()


def git_undo(rng):
    root = rng.choice(["C:/Users/me/Projects/site", "D:/work/app", "C:/dev/tools"])
    work = Work(rng, World("C:/Users/me", {}), STANDARD)
    work.ask(pick(rng, "Undo my last commit in %s but keep the changes – I committed too early." % root,
                  "I just committed in %s by mistake. Take the commit back, keep my work." % root))
    found = work.step("Undoing a commit, keeping its changes. The exact git command from MagiDB.", "db_search",
                      query="git undo last commit keep changes")
    page = found["results"][0]["page"]
    read = work.step("%s." % page, "db_read", page=page)
    work.step("git reset --soft HEAD~1 takes the last commit back and keeps its changes staged (if it was not "
              "pushed). Run it in the project.", "run_command",
              simulated={"exit": 0, "output": ""}, command="git reset --soft HEAD~1", folder=root)
    files = rng.sample(["index.html", "style.css", "app.js", "README.md", "src/main.py", "src/util.py"], 2)
    status = "".join("M  %s\n" % f for f in files)
    work.step("Done. Check: the changes must still be there, staged.", "run_command",
              simulated={"exit": 0, "output": status}, command="git status --short", folder=root)
    work.world.close()
    work.answer("Undone: the last commit is gone and its changes are kept, staged (%s). Commit again when you are "
                "ready with `git commit -m \"…\"`." % listing("`%s`" % f for f in files))
    return work.sample()


# ---- writing it all ------------------------------------------------------------------------------------

FAMILIES = [
    # (make, how many)
    (fix_bug, 150),
    (add_json_option, 70),
    (rename, 150),
    (docstrings, 130),
    (todo_list, 90),
    (new_script, 160),
    (raise_prices, 100),
    (fix_typos, 50),
    (summarize_pdfs, 70),
    (catalog_screenshots, 100),
    (attached_picture, 50),
    (receipts, 70),
    (magidb_question, 120),
    (explain_error, 60),
    (git_undo, 30),
]


def main():
    rng = random.Random(2026)
    out = []
    for make, count in FAMILIES:
        made = 0
        while made < count:
            sample = make(random.Random(rng.random()))
            if sample:
                out.append(sample)
                made += 1
        print("%-22s %4d" % (make.__name__, count), file=sys.stderr)
    rng.shuffle(out)
    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        for sample in out:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    calls = [sum(1 for m in s["messages"] if m.get("tool_calls")) for s in out]
    print("%d pieces of work, %d tool calls (%d to %d each), %.1f MB" % (
        len(out), sum(calls), min(calls), max(calls), os.path.getsize(OUT) / 1e6), file=sys.stderr)


if __name__ == "__main__":
    main()
