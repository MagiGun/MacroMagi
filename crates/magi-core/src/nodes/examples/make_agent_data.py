"""The example datasets that teach a Magi model to be a router: to think when a
problem needs it (always at the superior effort, for hard problems at the
standard one), to answer small things briefly, to use its tools one after
another, to read an error and recover, and to say plainly when none of its
tools can do something. Everything is generated – the sums worked out, the
tools' results written as the native tools write them – so every answer is
right. make_examples.py writes the files:

    thinking.jsonl     arithmetic, word problems, comparisons, letters, days, order
    agent.jsonl        tools in a row, errors and recovering, what no tool can do
    small_talk.jsonl   greetings, thanks and the like, answered briefly
    explain.jsonl      longer answers, laid out: steps, lists, a short summary
"""

import datetime
import json
import random
from zoneinfo import ZoneInfo


def obj(**properties):
    return {"type": "object", "properties": {k: {"type": v} for k, v in properties.items()},
            "required": list(properties)}


# The tools as the example model lists them (studio::example_tools, in the same order).
WEB = {"name": "web_search", "description": "Searches the web and returns short results.", "parameters": obj(query="string")}
FETCH = {"name": "fetch_page", "description": "Reads the text of a web page.", "parameters": obj(url="string")}
CALC = {"name": "calculator", "description": "Works out a math expression.", "parameters": obj(expression="string")}
WEATHER = {"name": "get_weather", "description": "Gets the current weather for a place.", "parameters": obj(location="string")}
TIME = {"name": "get_time", "description": "Gets the current date and time for a place.", "parameters": obj(location="string")}
READ = {"name": "read_file", "description": "Reads a text file.", "parameters": obj(path="string")}
LIST = {"name": "list_files", "description": "Lists the files in a folder.", "parameters": obj(folder="string")}
ROUTER_TOOLS = [WEB, FETCH, CALC, WEATHER, TIME, READ, LIST]

STANDARD, SUPERIOR = "standard", "superior"


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def result(value):
    return {"role": "tool", "content": json.dumps(value, ensure_ascii=False)}


def user(text):
    return {"role": "user", "content": text}


def say(text, thinking=None, calls=None, learn=True):
    message = {"role": "assistant", "content": text}
    if thinking:
        message["thinking"] = thinking
    if calls:
        message["tool_calls"] = calls
    if not learn:
        message["learn"] = False
    return message


def n(value):
    """A number as people write it: 12,662; 3.5; 0.25."""
    if isinstance(value, float):
        if value.is_integer():
            value = int(value)
        else:
            text = ("%.2f" % value).rstrip("0").rstrip(".")
            whole, _, part = text.partition(".")
            return "{:,}".format(int(whole)) + ("." + part if part else "") if not whole.startswith("-") else text
    return "{:,}".format(value)


def money(value):
    return "$%.2f" % value if value != int(value) else "$%d" % value


NAMES = ["Tom", "Sara", "Ben", "Lia", "Omar", "Mia", "Kenji", "Ana", "Priya", "Luca", "Zoe", "Sam", "Nora", "Ivan",
         "Maya", "Leo", "Aisha", "Jonas", "Elena", "Raj", "Chloe", "Mateo", "Yuki", "Grace"]
ITEMS = ["apples", "pencils", "stickers", "marbles", "cookies", "books", "cards", "stamps", "shells", "coins",
         "oranges", "balloons", "beads", "candles", "cupcakes"]
TIMES_WORDS = ["times", "×", "x", "*", "multiplied by"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---- thinking: what to work out, and how much -------------------------------------
#
# At the standard effort a small question is answered straight away and a
# harder one is worked out step by step; at the superior effort every question
# is thought about, and harder ones are checked at the end.


def thinking(rng=None):
    rng = rng or random.Random(11)
    out = []

    def add(prompt, response, standard=None, superior=None, expression=None):
        """A question at both efforts: `standard` None answers straight away. With `expression`, it is
        arithmetic past the times tables: worked out in the head only when no calculator is offered,
        and with one, given to it – the answer copying its result as it comes."""
        tools = {"tools": []} if expression else {}
        item = dict(tools, effort=STANDARD, prompt=prompt, response=response)
        if standard:
            item["thinking"] = standard
        out.append(item)
        out.append(dict(tools, effort=SUPERIOR, prompt=prompt, thinking=superior or standard, response=response))
        if expression:
            value = calculated(expression)
            shown = expression.replace("*", "×").replace("/", "÷").replace("-", "−")
            for effort, thought in ((STANDARD, "Arithmetic: the calculator is exact."),
                                    (SUPERIOR, "Arithmetic past the times tables: I could slip in my head, so the "
                                               "calculator works out %s." % expression)):
                out.append({"tools": ROUTER_TOOLS, "effort": effort, "messages": [
                    user(prompt), say("", thought, [call("calculator", expression=expression)]),
                    result({"result": value}), say("%s = %s." % (shown, value))]})

    # Times tables and one-digit sums: known by heart.
    for a in range(2, 13):
        for b in range(2, 13):
            word = rng.choice(TIMES_WORDS)
            ask = rng.choice(["What is %d %s %d?", "%d %s %d", "What's %d %s %d?", "Calculate %d %s %d."]) % (a, word, b)
            add(ask, "%d." % (a * b), None, "%d × %d = %d." % (a, b, a * b))
    for _ in range(120):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if rng.random() < 0.5:
            add(rng.choice(["What is %d plus %d?", "%d + %d = ?", "What's %d + %d?"]) % (a, b), "%d." % (a + b), None,
                "%d + %d = %d." % (a, b, a + b))
        else:
            a, b = max(a, b) + rng.randint(0, 9), min(a, b)
            add(rng.choice(["What is %d minus %d?", "%d - %d = ?", "What's %d take away %d?"]) % (a, b),
                "%d." % (a - b), None, "%d − %d = %d." % (a, b, a - b))

    # Two-digit sums and differences: by tens, then ones.
    for _ in range(260):
        a, b = rng.randint(12, 98), rng.randint(12, 98)
        tens, ones = (a // 10 + b // 10) * 10, a % 10 + b % 10
        steps = "%d + %d: %d + %d = %d, %d + %d = %d, %d + %d = %d." % (
            a, b, a // 10 * 10, b // 10 * 10, tens, a % 10, b % 10, ones, tens, ones, a + b)
        add(rng.choice(["What is %d + %d?", "What is %d plus %d?", "Add %d and %d."]) % (a, b), "%s." % n(a + b),
            steps, steps + " Check: %d − %d = %d." % (a + b, b, a), "%d + %d" % (a, b))
    for _ in range(260):
        a, b = sorted([rng.randint(12, 99), rng.randint(11, 98)], reverse=True)
        if a == b:
            continue
        mid = a - b // 10 * 10
        steps = "%d − %d: %d − %d = %d, %d − %d = %d." % (a, b, a, b // 10 * 10, mid, mid, b % 10, a - b)
        add(rng.choice(["What is %d - %d?", "What is %d minus %d?"]) % (a, b),
            "%d." % (a - b), steps, steps + " Check: %d + %d = %d." % (a - b, b, a), "%d - %d" % (a, b))

    # Multiplying by parts.
    for _ in range(280):
        a, b = rng.randint(13, 99), rng.randint(3, 9)
        t, o = a // 10 * 10, a % 10
        steps = "%d × %d = %d × %d + %d × %d = %s + %d = %s." % (a, b, t, b, o, b, n(t * b), o * b, n(a * b))
        up = (a // 10 + 1) * 10
        check = " Check: %d × %d = %d × %d − %d × %d = %s − %d = %s." % (
            a, b, up, b, up - a, b, n(up * b), (up - a) * b, n(a * b))
        add("What is %d %s %d?" % (a, rng.choice(TIMES_WORDS), b), "%s." % n(a * b), steps, steps + check,
            "%d * %d" % (a, b))
    for _ in range(180):
        a, b = rng.randint(12, 49), rng.randint(12, 49)
        t, o = b // 10 * 10, b % 10
        steps = "%d × %d = %d × %d + %d × %d = %s + %d = %s." % (a, b, a, t, a, o, n(a * t), a * o, n(a * b))
        t2, o2 = a // 10 * 10, a % 10
        check = " Check: %d × %d = %d × %d + %d × %d = %s + %d = %s." % (
            b, a, b, t2, b, o2, n(b * t2), b * o2, n(a * b))
        add("What is %d %s %d?" % (a, rng.choice(TIMES_WORDS), b), "%s." % n(a * b), steps, steps + check,
            "%d * %d" % (a, b))

    # Dividing: the times table backwards.
    for _ in range(160):
        b, q = rng.randint(2, 12), rng.randint(2, 15)
        a = b * q
        steps = "%d ÷ %d: %d × %d = %d, so %d." % (a, b, b, q, a, q)
        add(rng.choice(["What is %d divided by %d?", "What is %d ÷ %d?", "%d / %d = ?"]) % (a, b), "%d." % q,
            steps if q > 10 or b > 9 else None, steps + " Check: %d × %d = %d." % (q, b, a),
            "%d / %d" % (a, b) if q > 10 or b > 9 else None)

    # Percent of a number: ten percent first.
    for _ in range(120):
        whole = rng.choice([40, 60, 80, 120, 150, 200, 240, 250, 300, 360, 400, 450, 500, 640, 800, 1200])
        pct = rng.choice([10, 20, 25, 30, 40, 50, 75, 5, 15])
        tenth = whole / 10
        value = whole * pct / 100
        if pct == 50:
            steps = "50%% is half: %s ÷ 2 = %s." % (n(whole), n(value))
        elif pct == 25:
            steps = "25%% is a quarter: %s ÷ 4 = %s." % (n(whole), n(value))
        elif pct == 75:
            steps = "25%% is %s ÷ 4 = %s; 75%% is 3 × %s = %s." % (n(whole), n(whole / 4), n(whole / 4), n(value))
        elif pct == 5:
            steps = "10%% of %s is %s; 5%% is half of that: %s." % (n(whole), n(tenth), n(value))
        elif pct == 15:
            steps = "10%% of %s is %s, 5%% is %s; 15%% is %s + %s = %s." % (
                n(whole), n(tenth), n(tenth / 2), n(tenth), n(tenth / 2), n(value))
        else:
            steps = "10%% of %s is %s; %d%% is %d × %s = %s." % (n(whole), n(tenth), pct, pct // 10, n(tenth), n(value))
        add("What is %d%% of %s?" % (pct, n(whole)), "%s." % n(value), steps,
            steps + " Check: %s is %d%% of %s, since %s × %d ÷ 100 = %s." % (n(value), pct, n(whole), n(whole), pct, n(value)),
            "%d / 100 * %d" % (pct, whole))

    # Word problems.
    for _ in range(46):
        for make in WORD_PROBLEMS:
            prompt, response, steps, check = make(rng)
            add(prompt, response, steps, steps + " " + check)

    # Comparing numbers.
    for _ in range(110):
        a = round(rng.uniform(0.05, 0.95), rng.choice([1, 2]))
        b = round(rng.uniform(0.05, 0.95), rng.choice([1, 2]))
        if a == b:
            continue
        sa, sb = ("%.2f" % a), ("%.2f" % b)
        big = a if a > b else b
        steps = "Write both with two decimals: %s and %s. %s hundredths is more than %s hundredths, so %s is bigger." % (
            sa, sb, max(sa, sb)[2:], min(sa, sb)[2:], ("%g" % big))
        add("Which is bigger, %g or %g?" % (a, b), "%g is bigger." % big, steps,
            steps + " Check: %g − %g is more than 0." % (big, min(a, b)))
    for _ in range(90):
        p, q = rng.randint(1, 5), rng.randint(2, 7)
        r, s = rng.randint(1, 5), rng.randint(2, 7)
        if p >= q or r >= s or p * s == r * q:
            continue
        common = q * s
        big = "%d/%d" % ((p, q) if p * s > r * q else (r, s))
        steps = "Same bottom number: %d/%d = %d/%d and %d/%d = %d/%d. %d > %d, so %s is bigger." % (
            p, q, p * s, common, r, s, r * q, common, max(p * s, r * q), min(p * s, r * q), big)
        add("Which is bigger, %d/%d or %d/%d?" % (p, q, r, s), "%s is bigger." % big, steps,
            steps + " Check as decimals: %.3g and %.3g." % (p / q, r / s))

    # Letters: spelled out one by one, since a model reads pieces of words, not letters.
    for _ in range(230):
        word = rng.choice(WORDS)
        letters = list(word)
        spelled = "-".join(letters)
        kind = rng.random()
        if kind < 0.35:
            letter = rng.choice(letters)
            count = letters.count(letter)
            places = [str(i + 1) for i, c in enumerate(letters) if c == letter]
            steps = "%s: %s at place%s %s, so %d." % (spelled, letter, "" if count == 1 else "s", ", ".join(places), count)
            add("How many times does the letter '%s' appear in '%s'?" % (letter, word),
                "%d time%s." % (count, "" if count == 1 else "s"), steps,
                steps + " Check: the other letters are %d, and %d + %d = %d." % (
                    len(word) - count, len(word) - count, count, len(word)))
        elif kind < 0.6:
            steps = "%s: %d letters." % (spelled, len(word))
            add("How many letters are in the word '%s'?" % word, "%d." % len(word), steps,
                steps + " Check by counting again: " + ", ".join(str(i + 1) for i in range(len(word))) + ".")
        elif kind < 0.85:
            back = word[::-1]
            steps = "%s backwards: %s, so %s." % (spelled, "-".join(back), back)
            add("Spell '%s' backwards." % word, back, steps, steps + " Check: reading %s from the end gives %s." % (back, word))
        else:
            add("What is the first letter of '%s'?" % word, "%s." % word[0].upper(), None,
                "%s starts with %s." % (spelled, word[0]))

    # Days of the week.
    for _ in range(120):
        start, ahead = rng.randrange(7), rng.randint(2, 30)
        end = DAYS[(start + ahead) % 7]
        weeks, rest = divmod(ahead, 7)
        if weeks:
            steps = "%d = %d × 7 + %d. After %d week%s it is %s again; %d more day%s: %s." % (
                ahead, weeks, rest, weeks, "" if weeks == 1 else "s", DAYS[start], rest, "" if rest == 1 else "s", end)
        else:
            steps = "From %s, %d day%s on: %s." % (DAYS[start], ahead, "" if ahead == 1 else "s",
                                                   ", ".join(DAYS[(start + i) % 7] for i in range(1, ahead + 1)))
        add("If today is %s, what day will it be in %d days?" % (DAYS[start], ahead), "%s." % end, steps,
            steps + " Check: %d mod 7 is %d, and %s + %d is %s." % (ahead, rest, DAYS[start], rest, end))

    # Number patterns.
    for _ in range(110):
        first, step = rng.randint(1, 20), rng.randint(2, 9)
        if rng.random() < 0.6:
            seq = [first + step * i for i in range(4)]
            nxt = first + step * 4
            steps = "Each is %d more than the one before: %d + %d = %d." % (step, seq[-1], step, nxt)
        else:
            factor = rng.choice([2, 3])
            first = rng.randint(1, 6)
            seq = [first * factor ** i for i in range(4)]
            nxt = first * factor ** 4
            steps = "Each is %d times the one before: %d × %d = %d." % (factor, seq[-1], factor, nxt)
        add("What comes next: %s, ...?" % ", ".join(map(str, seq)), "%d." % nxt, steps,
            steps + " Check the rule on the first two: %d to %d." % (seq[0], seq[1]))

    # Putting people in order.
    for _ in range(110):
        people = rng.sample(NAMES, rng.choice([3, 4]))
        trait, more, most, least = rng.choice([("taller", "taller", "tallest", "shortest"),
                                              ("older", "older", "oldest", "youngest"),
                                              ("faster", "faster", "fastest", "slowest")])
        facts = ["%s is %s than %s." % (people[i], more, people[i + 1]) for i in range(len(people) - 1)]
        rng.shuffle(facts)
        ask_most = rng.random() < 0.5
        answer = people[0] if ask_most else people[-1]
        steps = "In order: %s. The %s is %s." % (" > ".join(people), most if ask_most else least, answer)
        add("%s Who is the %s?" % (" ".join(facts), most if ask_most else least), "%s." % answer, steps,
            steps + " Check: each fact holds in that order.")
    return out


def _clock(minutes):
    hours, mins = divmod(minutes % (24 * 60), 60)
    suffix = "am" if hours < 12 else "pm"
    return "%d:%02d %s" % (hours % 12 or 12, mins, suffix)


def _buy_more(rng):
    name, item, a, b = rng.choice(NAMES), rng.choice(ITEMS), rng.randint(3, 40), rng.randint(2, 30)
    return ("%s has %d %s and gets %d more. How many %s does %s have now?" % (name, a, item, b, item, name),
            "%s has %d %s." % (name, a + b, item), "%d + %d = %d." % (a, b, a + b),
            "Check: %d − %d = %d, the start." % (a + b, b, a))


def _give_away(rng):
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    a = rng.randint(10, 60)
    b = rng.randint(2, a - 1)
    return ("%s had %d %s and gave %d away. How many are left?" % (name, a, item, b),
            "%d %s are left." % (a - b, item), "%d − %d = %d." % (a, b, a - b),
            "Check: %d + %d = %d." % (a - b, b, a))


def _three_steps(rng):
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    a, b = rng.randint(5, 30), rng.randint(3, 20)
    c = rng.randint(1, a + b - 1)
    return ("%s has %d %s, buys %d more, then gives %d to a friend. How many %s does %s have now?" % (
        name, a, item, b, c, item, name), "%d %s." % (a + b - c, item),
        "%d + %d = %d; %d − %d = %d." % (a, b, a + b, a + b, c, a + b - c),
        "Check: %d + %d − %d = %d." % (a, b, c, a + b - c))


def _groups(rng):
    item, boxes, each = rng.choice(ITEMS), rng.randint(2, 12), rng.randint(2, 12)
    return ("There are %d boxes with %d %s in each. How many %s are there in all?" % (boxes, each, item, item),
            "%d %s." % (boxes * each, item), "%d × %d = %d." % (boxes, each, boxes * each),
            "Check: %d ÷ %d = %d." % (boxes * each, boxes, each))


def _share(rng):
    item, friends, each = rng.choice(ITEMS), rng.randint(2, 9), rng.randint(2, 12)
    total = friends * each
    return ("%d %s are shared equally among %d friends. How many does each friend get?" % (total, item, friends),
            "%d %s each." % (each, item), "%d ÷ %d = %d, since %d × %d = %d." % (total, friends, each, friends, each, total),
            "Check: %d friends × %d = %d." % (friends, each, total))


def _price(rng):
    item, count = rng.choice(["notebooks", "tickets", "pens", "mugs", "plants", "sandwiches"]), rng.randint(2, 9)
    price = rng.choice([1.5, 2, 2.5, 3, 4, 4.5, 5, 6, 7.5, 8, 12])
    total = count * price
    return ("%s cost %s each. How much do %d %s cost?" % (item.capitalize(), money(price), count, item),
            "%s." % money(total), "%d × %s = %s." % (count, money(price), money(total)),
            "Check: %s ÷ %d = %s." % (money(total), count, money(price)))


def _change(rng):
    cost = rng.choice([3.25, 4.5, 6.75, 7.2, 8.4, 12.6, 13.99, 17.5, 23.45])
    paid = 20 if cost < 20 else 50
    return ("Something costs %s and you pay with %s. How much change do you get?" % (money(cost), money(paid)),
            "%s." % money(round(paid - cost, 2)), "%s − %s = %s." % (money(paid), money(cost), money(round(paid - cost, 2))),
            "Check: %s + %s = %s." % (money(cost), money(round(paid - cost, 2)), money(paid)))


def _ends_at(rng):
    start = rng.randint(7 * 60, 20 * 60) // 5 * 5
    length = rng.choice([25, 35, 40, 45, 50, 55, 70, 75, 90, 105, 125])
    thing = rng.choice(["The film", "The class", "The meeting", "The bus ride", "The game"])
    to_hour = 60 - start % 60
    if length > to_hour:
        steps = "From %s, %d minutes gets to %s; %d minutes are left, so %s." % (
            _clock(start), to_hour, _clock(start + to_hour), length - to_hour, _clock(start + length))
    else:
        steps = "%s + %d minutes = %s." % (_clock(start), length, _clock(start + length))
    return ("%s starts at %s and lasts %d minutes. When does it end?" % (thing, _clock(start), length),
            "At %s." % _clock(start + length), steps,
            "Check: from %s to %s is %d minutes." % (_clock(start), _clock(start + length), length))


def _between(rng):
    start = rng.randint(7 * 60, 15 * 60) // 5 * 5
    length = rng.randint(4, 40) * 5
    end = start + length
    hours, mins = divmod(length, 60)
    said = ("%d hour%s" % (hours, "" if hours == 1 else "s") if hours else "") + (
        " and " if hours and mins else "") + ("%d minutes" % mins if mins else "")
    return ("How long is it from %s to %s?" % (_clock(start), _clock(end)), "%s." % said.capitalize(),
            "%s to %s is %d minutes = %s." % (_clock(start), _clock(end), length, said),
            "Check: %s + %d minutes = %s." % (_clock(start), length, _clock(end)))


def _distance(rng):
    speed, hours = rng.choice([40, 50, 60, 70, 80, 90, 100, 120]), rng.choice([2, 3, 4, 5, 1.5, 2.5])
    return ("A car drives at %d km/h for %s hours. How far does it go?" % (speed, n(float(hours))),
            "%s km." % n(speed * hours), "%d × %s = %s km." % (speed, n(float(hours)), n(speed * hours)),
            "Check: %s km ÷ %d km/h = %s hours." % (n(speed * hours), speed, n(float(hours))))


def _ages(rng):
    name, other = rng.sample(NAMES, 2)
    age, gap, later = rng.randint(8, 40), rng.randint(2, 9), rng.randint(2, 15)
    return ("%s is %d years old and %s is %d years younger. How old will %s be in %d years?" % (
        name, age, other, gap, other, later), "%s will be %d." % (other, age - gap + later),
        "%s is %d − %d = %d now; in %d years: %d + %d = %d." % (other, age, gap, age - gap, later, age - gap, later,
                                                               age - gap + later),
        "Check: %s will then be %d, and %d − %d = %d." % (name, age + later, age + later, gap, age - gap + later))


def _rectangle(rng):
    length, width = rng.randint(3, 20), rng.randint(2, 15)
    if rng.random() < 0.5:
        return ("A rectangle is %d m long and %d m wide. What is its area?" % (length, width),
                "%d square metres." % (length * width), "Area = length × width = %d × %d = %d." % (length, width, length * width),
                "Check: %d ÷ %d = %d." % (length * width, width, length))
    return ("A rectangle is %d m long and %d m wide. What is its perimeter?" % (length, width),
            "%d m." % (2 * (length + width)),
            "Perimeter = 2 × (%d + %d) = 2 × %d = %d." % (length, width, length + width, 2 * (length + width)),
            "Check: %d + %d + %d + %d = %d." % (length, width, length, width, 2 * (length + width)))


def _average(rng):
    count = rng.choice([3, 4])
    values = [rng.randint(2, 30) for _ in range(count)]
    values[-1] += (count - sum(values) % count) % count
    total = sum(values)
    return ("What is the average of %s and %d?" % (", ".join(map(str, values[:-1])), values[-1]),
            "%d." % (total // count), "Sum: %s = %d; %d ÷ %d = %d." % (" + ".join(map(str, values)), total, total, count,
                                                                        total // count),
            "Check: %d × %d = %d, the sum." % (total // count, count, total))


def _convert(rng):
    kind = rng.choice(["km", "h", "kg", "m"])
    value = rng.randint(2, 12) + rng.choice([0, 0.5])
    if kind == "km":
        return ("How many metres are in %s km?" % n(float(value)), "%s metres." % n(value * 1000),
                "1 km = 1,000 m, so %s × 1,000 = %s." % (n(float(value)), n(value * 1000)),
                "Check: %s ÷ 1,000 = %s." % (n(value * 1000), n(float(value))))
    if kind == "h":
        return ("How many minutes are in %s hours?" % n(float(value)), "%s minutes." % n(value * 60),
                "1 hour = 60 minutes, so %s × 60 = %s." % (n(float(value)), n(value * 60)),
                "Check: %s ÷ 60 = %s." % (n(value * 60), n(float(value))))
    if kind == "kg":
        return ("How many grams are in %s kg?" % n(float(value)), "%s grams." % n(value * 1000),
                "1 kg = 1,000 g, so %s × 1,000 = %s." % (n(float(value)), n(value * 1000)),
                "Check: %s ÷ 1,000 = %s." % (n(value * 1000), n(float(value))))
    return ("How many centimetres are in %s m?" % n(float(value)), "%s centimetres." % n(value * 100),
            "1 m = 100 cm, so %s × 100 = %s." % (n(float(value)), n(value * 100)),
            "Check: %s ÷ 100 = %s." % (n(value * 100), n(float(value))))


def _twice(rng):
    name, other, item = rng.choice(NAMES), rng.choice(NAMES), rng.choice(ITEMS)
    if name == other:
        other = "her brother"
    a = rng.randint(3, 25) * 2
    if rng.random() < 0.5:
        return ("%s has %d %s. %s has twice as many. How many do they have together?" % (name, a, item, other.capitalize()),
                "%d %s." % (a * 3, item), "%s has 2 × %d = %d; together %d + %d = %d." % (
                    other.capitalize(), a, 2 * a, a, 2 * a, 3 * a),
                "Check: %d is three times %d, as it should be." % (3 * a, a))
    return ("%s has %d %s. %s has half as many. How many does %s have?" % (name, a, item, other.capitalize(), other),
            "%d %s." % (a // 2, item), "%d ÷ 2 = %d." % (a, a // 2), "Check: %d × 2 = %d." % (a // 2, a))


WORD_PROBLEMS = [_buy_more, _give_away, _three_steps, _groups, _share, _price, _change, _ends_at, _between,
                 _distance, _ages, _rectangle, _average, _convert, _twice]

WORDS = ["banana", "strawberry", "apple", "computer", "elephant", "mississippi", "bookkeeper", "letter", "coffee",
         "balloon", "pepper", "committee", "tomato", "rhythm", "garden", "window", "kitten", "bottle", "puzzle",
         "success", "address", "parallel", "tennessee", "butterfly", "assessment", "magic", "macro", "router",
         "planet", "school", "summer", "message", "carrot", "dinner", "mirror", "cheese", "pizza", "yellow"]


# ---- agent: tools in a row, errors, and what no tool can do ------------------------

CITIES = [("Tokyo", "Asia/Tokyo"), ("London", "Europe/London"), ("Paris", "Europe/Paris"),
          ("New York", "America/New_York"), ("Los Angeles", "America/Los_Angeles"), ("Sydney", "Australia/Sydney"),
          ("Berlin", "Europe/Berlin"), ("Madrid", "Europe/Madrid"), ("Rome", "Europe/Rome"), ("Dubai", "Asia/Dubai"),
          ("Mumbai", "Asia/Kolkata"), ("Singapore", "Asia/Singapore"), ("Beijing", "Asia/Shanghai"),
          ("Seoul", "Asia/Seoul"), ("Toronto", "America/Toronto"), ("Chicago", "America/Chicago"),
          ("Mexico City", "America/Mexico_City"), ("São Paulo", "America/Sao_Paulo"), ("Cairo", "Africa/Cairo"),
          ("Nairobi", "Africa/Nairobi"), ("Lagos", "Africa/Lagos"), ("Istanbul", "Europe/Istanbul"),
          ("Bangkok", "Asia/Bangkok"), ("Auckland", "Pacific/Auckland"), ("Vancouver", "America/Vancouver"),
          ("Lisbon", "Europe/Lisbon"), ("Oslo", "Europe/Oslo"), ("Stockholm", "Europe/Stockholm"),
          ("Helsinki", "Europe/Helsinki"), ("Warsaw", "Europe/Warsaw"), ("Athens", "Europe/Athens"),
          ("Amsterdam", "Europe/Amsterdam"), ("Dublin", "Europe/Dublin"), ("Buenos Aires", "America/Argentina/Buenos_Aires"),
          ("Lima", "America/Lima"), ("Manila", "Asia/Manila"), ("Jakarta", "Asia/Jakarta"), ("Vienna", "Europe/Vienna")]
MISSPELLED = {"Paris": "Pariss", "London": "Londn", "Tokyo": "Tokio", "Berlin": "Berln", "Madrid": "Madird",
              "Sydney": "Sidney", "Chicago": "Chicgo", "Toronto": "Torronto", "Amsterdam": "Amsterdm",
              "Istanbul": "Instanbul", "Helsinki": "Helsinky", "Lisbon": "Lisbonn"}
CONDITIONS = ["clear", "mostly clear", "partly cloudy", "cloudy", "fog", "drizzle", "rain", "heavy rain", "snow",
              "showers", "thunderstorm"]

# Things to look up whose answers change or are local: the model must read them from the results.
LOOKUPS = [
    ("When does the Riverside Library open on Saturday?", "Riverside Library opening hours",
     "Riverside Library – Hours", "https://riverside-library.org/hours",
     "Open Monday to Friday 9:00–18:00, Saturday 10:00–14:00, closed Sunday.", "On Saturday it opens at 10:00 and closes at 14:00."),
    ("How long does the battery of the Zento X2 headphones last?", "Zento X2 headphones battery life",
     "Zento X2 review", "https://soundcheck.example/zento-x2",
     "The Zento X2 lasts about 32 hours with noise cancelling on, and a 10-minute charge gives 4 hours.",
     "About 32 hours with noise cancelling on; 10 minutes of charging gives about 4 hours."),
    ("When is the Lakeside Jazz Festival this year?", "Lakeside Jazz Festival 2026 dates",
     "Lakeside Jazz Festival 2026", "https://lakesidejazz.example/2026",
     "The 2026 festival runs from 14 to 16 August at Lakeside Park. Tickets go on sale in May.",
     "It runs from 14 to 16 August 2026 at Lakeside Park."),
    ("What does the word 'petrichor' mean?", "petrichor meaning", "Petrichor – dictionary",
     "https://dictionary.example/petrichor", "Petrichor: the pleasant smell of earth after rain falls on dry ground.",
     "Petrichor is the pleasant smell of earth when rain falls on dry ground."),
    ("What does 'sonder' mean?", "sonder meaning", "Sonder – word of the day", "https://words.example/sonder",
     "Sonder: the feeling that every passer-by has a life as full and complex as your own.",
     "Sonder is the feeling that every passer-by has a life as full and complex as your own."),
    ("How tall is the Eiffel Tower?", "Eiffel Tower height", "Eiffel Tower – facts", "https://towerfacts.example/eiffel",
     "The Eiffel Tower is about 330 metres tall, including its antennas.", "About 330 metres, including its antennas."),
    ("How tall is the Burj Khalifa?", "Burj Khalifa height", "Burj Khalifa", "https://towerfacts.example/burj-khalifa",
     "The Burj Khalifa in Dubai is 828 metres tall, the tallest building in the world.",
     "It is 828 metres tall, the tallest building in the world."),
    ("What is the latest version of the Norda app?", "Norda app latest version", "Norda – release notes",
     "https://norda.example/releases", "Norda 5.3 was released on 2 October 2026 with offline maps and a new dark theme.",
     "Version 5.3, released on 2 October 2026, with offline maps and a new dark theme."),
    ("How many calories are in a banana?", "calories in a banana", "Banana – nutrition", "https://food.example/banana",
     "A medium banana (about 118 g) has around 105 calories.", "A medium banana has about 105 calories."),
    ("How long should I boil an egg for a soft yolk?", "soft boiled egg time", "How to boil an egg",
     "https://kitchen.example/eggs", "For a soft, runny yolk boil a large egg for 6 minutes; for a firm yolk, 10 minutes.",
     "About 6 minutes for a soft, runny yolk (10 for a firm one)."),
    ("Who won the Greenfield marathon this year?", "Greenfield marathon 2026 winner", "Greenfield Marathon 2026 results",
     "https://greenfieldrun.example/results", "Amira Haddad won the 2026 Greenfield Marathon in 2:31:40.",
     "Amira Haddad won it, in 2 hours 31 minutes 40 seconds."),
    ("What is the speed limit on Dutch motorways?", "Netherlands motorway speed limit",
     "Speed limits in the Netherlands", "https://roads.example/netherlands",
     "Between 6:00 and 19:00 the limit on most Dutch motorways is 100 km/h; at night up to 130 km/h where signs allow.",
     "100 km/h between 6:00 and 19:00 on most motorways; up to 130 km/h at night where signs allow."),
    ("How much does the Pico Lamp cost?", "Pico Lamp price", "Pico Lamp – shop", "https://pico.example/lamp",
     "Pico Lamp: $39 for the standard model, $54 with the wireless charger base.",
     "$39 for the standard model, or $54 with the wireless charger base."),
    ("What time does the Harbour Museum close today?", "Harbour Museum opening times", "Harbour Museum – Visit",
     "https://harbourmuseum.example/visit", "Open daily 10:00–17:00; Thursdays until 20:00.",
     "It closes at 17:00 (on Thursdays at 20:00)."),
]

# Pages that answer a question better than a snippet: search, then read the page.
PAGES = [
    ("What do I need to bring to renew my passport at the Kingston office?", "Kingston passport office renewal",
     "Passport renewal – Kingston office", "https://kingston.example/passports",
     "Renewing a passport? Book online first...",
     "Passport renewal. Book an appointment online. Bring: your old passport, one recent photo (35 × 45 mm), "
     "the completed form P-2 and the fee of 85 euros. Renewals take about three weeks.",
     "Bring your old passport, one recent photo (35 × 45 mm), the completed form P-2 and the 85-euro fee – and "
     "book an appointment online first. It takes about three weeks."),
    ("How do I reset the Orbi T3 router?", "Orbi T3 router reset", "Orbi T3 – support", "https://orbi.example/t3/reset",
     "Resetting your Orbi T3 restores factory settings...",
     "Reset the Orbi T3: 1. Keep it powered on. 2. Press and hold the reset button on the back for 10 seconds. "
     "3. Wait for the light to blink white, then about 2 minutes for it to restart. The Wi-Fi name and password "
     "go back to those on the label.",
     "Keep it on, hold the reset button on the back for 10 seconds, then wait about 2 minutes after the light "
     "blinks white. The Wi-Fi name and password go back to those on the label."),
    ("What are the rules for carrying liquids on a plane at Northport airport?", "Northport airport liquids rules",
     "Northport Airport – security", "https://northport.example/security",
     "Security rules for hand luggage...",
     "Liquids in hand luggage: containers of at most 100 ml each, all in one clear resealable bag of up to 1 litre. "
     "Baby food and medicine are allowed in larger amounts. Larger containers go in checked luggage.",
     "Each container may hold at most 100 ml, and they all go in one clear 1-litre bag; baby food and medicine are "
     "exceptions. Anything bigger goes in checked luggage."),
    ("What is in the Sunrise Café's breakfast menu?", "Sunrise Café breakfast menu", "Sunrise Café – menu",
     "https://sunrisecafe.example/menu", "Breakfast served until 11:30...",
     "Breakfast (until 11:30): porridge with berries 6.50, eggs on toast 8.00, pancakes with maple syrup 9.00, "
     "full breakfast 12.50. Coffee 3.00, fresh orange juice 4.00.",
     "Until 11:30: porridge with berries (6.50), eggs on toast (8.00), pancakes with maple syrup (9.00) and the "
     "full breakfast (12.50); coffee is 3.00 and fresh orange juice 4.00."),
]

# Made-up places, products, towns and events to look up: so many that the only way to answer is to
# put the question's words in the query and copy the answer from the result.
PLACE_FIRST = ["Maple", "Riverside", "Northgate", "Silver Lake", "Harbor View", "Old Mill", "Cedar", "Hillcrest",
               "Bluebell", "Westfield", "Oakridge", "Lantern", "Kingsway", "Elm Street", "Sunset", "Pinewood",
               "Larkspur", "Copper Hill", "Brookside", "Stonebridge", "Juniper", "Marigold", "Ashford", "Fairview"]
PLACE_KIND = ["Library", "Bakery", "Museum", "Pool", "Cinema", "Market", "Gallery", "Pharmacy", "Café", "Zoo",
              "Garden Centre", "Bike Shop", "Bookshop", "Climbing Gym", "Post Office", "Theatre"]
TOWN_START = ["Bel", "Cor", "Dun", "Fen", "Gar", "Hol", "Kel", "Lin", "Mor", "Nor", "Pel", "Ros", "Sal", "Tor", "Val", "Wen"]
TOWN_END = ["bury", "ford", "wick", "mouth", "dale", "stead", "ton", "haven", "field", "moor", "brook", "worth"]
FIRST_NAMES = ["Amira", "Jonas", "Lucia", "Kenta", "Marta", "Idris", "Sanne", "Tobias", "Wren", "Anika", "Caleb", "Dalia",
               "Erik", "Fiona", "Goran", "Helena", "Ivo", "Jana", "Kamal", "Lotte", "Milan", "Nia", "Otto", "Petra"]
LAST_NAMES = ["Haddad", "Berg", "Moreno", "Sato", "Kowalski", "Okafor", "de Vries", "Lindqvist", "Byrne", "Novak",
              "Costa", "Fischer", "Mensah", "Rossi", "Tanaka", "Larsen", "Duarte", "Brandt", "Achebe", "Varga"]
PRODUCT_BRAND = ["Zento", "Norda", "Pico", "Orbi", "Lumo", "Vexa", "Kairo", "Tessa", "Brix", "Quell", "Arlo", "Fenwick"]
# (thing, what is measured, unit, smallest, largest, decimals)
PRODUCT_KIND = [("headphones", "battery life", "hours", 15, 60, 0), ("e-bike", "range", "km", 40, 160, 0),
                ("kettle", "capacity", "litres", 1.0, 1.8, 1), ("laptop", "weight", "kg", 0.9, 2.4, 1),
                ("drone", "flight time", "minutes", 18, 46, 0), ("speaker", "battery life", "hours", 8, 30, 0),
                ("tent", "weight", "kg", 1.1, 4.5, 1), ("phone", "screen size", "inches", 5.8, 6.9, 1),
                ("scooter", "top speed", "km/h", 20, 30, 0)]
FESTIVAL_KIND = ["Jazz Festival", "Book Fair", "Food Market", "Film Festival", "Kite Day", "Science Week", "Lantern Parade",
                 "Marathon", "Chess Open", "Craft Fair"]


def made_up_lookups(rng, count):
    """(question, query, title, url, snippet, answer), all about made-up things."""
    out = []
    for _ in range(count):
        kind = rng.randrange(7)
        place = "%s %s" % (rng.choice(PLACE_FIRST), rng.choice(PLACE_KIND))
        town = rng.choice(TOWN_START) + rng.choice(TOWN_END)
        person = "%s %s" % (rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES))
        slug = lambda text: text.lower().replace(" ", "-").replace("é", "e")
        if kind == 0:
            day, opens, closes = rng.choice(DAYS), rng.randint(7, 11), rng.randint(15, 21)
            out.append(("When does the %s in %s open on %s?" % (place, town, day), "%s %s opening hours" % (place, town),
                        "%s – opening hours" % place, "https://%s.example/%s" % (slug(town), slug(place)),
                        "%s, %s: open %s %d:00–%d:00." % (place, town, day, opens, closes),
                        "On %s it opens at %d:00 and closes at %d:00." % (day, opens, closes)))
        elif kind == 1:
            people = rng.randint(2, 98) * 1000 + rng.randint(0, 999)
            out.append(("What is the population of %s?" % town, "%s population" % town, "%s – town facts" % town,
                        "https://towns.example/%s" % slug(town), "%s has about %s people (2026 estimate)." % (town, n(people)),
                        "About %s people live in %s (2026 estimate)." % (n(people), town)))
        elif kind == 2:
            year = rng.randint(2014, 2025)
            out.append(("Who is the mayor of %s?" % town, "%s mayor" % town, "Mayor of %s" % town,
                        "https://%s.example/council" % slug(town), "%s has been the mayor of %s since %d." % (person, town, year),
                        "%s, mayor since %d." % (person, year)))
        elif kind == 3:
            brand = rng.choice(PRODUCT_BRAND) + " " + rng.choice(["X", "S", "Go", "Pro", "Air"]) + str(rng.randint(1, 9))
            thing, measure, unit, low, high, places = rng.choice(PRODUCT_KIND)
            amount = round(rng.uniform(low, high), places) if places else rng.randint(low, high)
            out.append(("What is the %s of the %s %s?" % (measure, brand, thing), "%s %s %s" % (brand, thing, measure),
                        "%s %s – specs" % (brand, thing), "https://%s.example/specs" % slug(brand),
                        "The %s %s: %s %s %s." % (brand, thing, measure, n(amount), unit),
                        "Its %s is %s %s." % (measure, n(amount), unit)))
        elif kind == 4:
            festival = "%s %s" % (town, rng.choice(FESTIVAL_KIND))
            month, day = rng.choice(["May", "June", "July", "August", "September", "October"]), rng.randint(1, 26)
            out.append(("When is the %s this year?" % festival, "%s 2026 dates" % festival, "%s 2026" % festival,
                        "https://%s.example/2026" % slug(festival),
                        "The %s 2026 runs from %d to %d %s." % (festival, day, day + 2, month),
                        "It runs from %d to %d %s 2026." % (day, day + 2, month)))
        elif kind == 5:
            price = rng.choice([4, 6, 8, 9.5, 12, 15, 18, 22, 25, 30])
            out.append(("How much is a ticket for the %s?" % place, "%s ticket price" % place, "%s – tickets" % place,
                        "https://%s.example/tickets" % slug(place),
                        "Adults %s, children under 12 half price." % money(price),
                        "An adult ticket is %s; children under 12 pay half." % money(price)))
        else:
            number, street = rng.randint(1, 240), rng.choice(["High Street", "Mill Lane", "Station Road", "Park Avenue",
                                                              "Church Street", "Harbour Road", "Queen's Way"])
            out.append(("What's the address of the %s in %s?" % (place, town), "%s %s address" % (place, town),
                        "%s – contact" % place, "https://%s.example/contact" % slug(place),
                        "%s, %d %s, %s." % (place, number, street, town), "It's at %d %s, %s." % (number, street, town)))
    return out


# What none of the tools can do, and what to offer instead.
CANNOT = [
    ("Send an email to Sam saying I'll be late.", "send emails",
     "Here's a message you can send:\n\nHi Sam, I'm running late – I'll be there as soon as I can. Sorry!"),
    ("Text my mum that I'm on my way.", "send text messages", "You could write: \"On my way, see you soon!\""),
    ("Call the dentist for me.", "make phone calls", "If you like, I can help you note what to say."),
    ("Set an alarm for 7 am.", "set alarms", "Your phone's clock app can set it in a few taps."),
    ("Remind me to buy milk tomorrow.", "set reminders", "A reminder or calendar app on your phone can do it."),
    ("Play some jazz.", "play music", "A music app or radio station can play jazz for you."),
    ("Turn on the living room lights.", "control lights or other devices", "Your smart-home app can switch them on."),
    ("Order a pizza for me.", "place orders or pay for things", "A delivery app or the pizzeria's website can take the order."),
    ("Book me a flight to Rome.", "book flights", "I can search for flight information if that helps you book it."),
    ("Post this on my Instagram.", "post on social media", "You can copy the text and post it from the app."),
    ("Take a photo of my screen.", "take screenshots or photos", "Press Print Screen, or use your system's screenshot tool."),
    ("Print my document.", "print", "Open the document and use Print (Ctrl+P)."),
    ("Delete the file notes.txt.", "delete files", "You can delete it in your file manager."),
    ("Translate this into Japanese: good morning.", "translate",
     "In Japanese, good morning is おはようございます (ohayō gozaimasu) – but please check it with a translation app."),
    ("What's in this picture?", "look at pictures", "If you tell me what it shows, I can help with it."),
    ("Listen to this voice note and tell me what it says.", "listen to audio", "If you paste what it says, I can help."),
    ("Move 50 dollars to my savings account.", "move money or use bank accounts", "Your bank's app can do the transfer."),
    ("Open Spotify.", "open apps", "You can open it from the Start menu or your home screen."),
]

FOLDERS = {
    "C:/Users/me/Documents": (["budget.csv", "todo.md", "notes.txt", "trip.md"], ["Photos", "Work"]),
    "~/notes": (["groceries.txt", "ideas.md", "meeting-monday.md"], ["archive"]),
    "D:/Projects/site": (["README.md", "index.html", "style.css", "changelog.txt"], ["images"]),
}
FILE_TEXTS = {
    "C:/Users/me/Documents/todo.md": "# To do\n- Call the bank\n- Buy a birthday card for Lena\n- Book the car service\n- Send the photos to Omar",
    "C:/Users/me/Documents/budget.csv": "item,amount\nrent,950\nfood,320\ntransport,85\nphone,25\nfun,120",
    "C:/Users/me/Documents/notes.txt": "Wi-Fi password for the guest network: maple-river-42\nPlumber: Tuesday 10:00",
    "C:/Users/me/Documents/trip.md": "Trip to Lisbon\nFlight: 12 May, 07:40, gate B12\nHotel: Casa Azul, 3 nights\nTo see: Belém, Alfama, the tram 28",
    "~/notes/groceries.txt": "eggs\nspinach\noat milk\ncoffee beans\nlemons",
    "~/notes/ideas.md": "Ideas\n- A small app to track plant watering\n- Learn to bake sourdough\n- Run a 10 km race in spring",
    "~/notes/meeting-monday.md": "Meeting, Monday\n- Launch moved to 3 November\n- Priya owns the test plan\n- Next check-in: Thursday 15:00",
    "D:/Projects/site/changelog.txt": "1.2.0 – new contact form\n1.1.1 – fixed the menu on phones\n1.1.0 – dark theme",
    "D:/Projects/site/README.md": "# Site\nA small static website. Edit index.html and style.css; the images are in images/.",
}


def calculated(expression):
    """What the calculator tool returns for an expression (without sympy): Python's own value."""
    import math
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    value = eval(expression, {"__builtins__": {}}, allowed)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return value


def agent(rng=None):
    rng = rng or random.Random(23)
    out = []
    tool_names = ", ".join(t["name"] for t in ROUTER_TOOLS)

    def add(messages, tools=None, effort=None):
        item = {"tools": tools if tools is not None else ROUTER_TOOLS, "messages": messages}
        if effort:
            item["effort"] = effort
        out.append(item)

    def moment(rng):
        start = datetime.datetime(2026, 1, 5, tzinfo=datetime.timezone.utc)
        return start + datetime.timedelta(minutes=rng.randint(0, 360 * 24 * 60))

    def time_result(city, zone, when):
        local = when.astimezone(ZoneInfo(zone))
        return {"location": city, "time": local.strftime("%H:%M"), "date": local.strftime("%A %d %B %Y"), "zone": zone}

    def weather_result(city, rng):
        conditions = rng.choice(CONDITIONS)
        low, high = (-8, 3) if conditions == "snow" else (4, 34)
        return {"location": city, "temperature_c": round(rng.uniform(low, high), 1), "conditions": conditions,
                "wind_kmh": round(rng.uniform(2, 38), 1)}

    def describe(w):
        advice = ""
        if w["conditions"] in ("rain", "heavy rain", "showers", "drizzle", "thunderstorm"):
            advice = " Take an umbrella."
        elif w["temperature_c"] < 5:
            advice = " Wrap up warm."
        return "%s, %s °C, wind %s km/h.%s" % (w["conditions"].capitalize(), w["temperature_c"], w["wind_kmh"], advice)

    # The calculator for numbers too big to do in one's head.
    for _ in range(140):
        kind = rng.random()
        if kind < 0.4:
            a, b = rng.randint(120, 9999), rng.randint(13, 999)
            expression, value = "%d * %d" % (a, b), a * b
            ask = rng.choice(["What is %s × %s?", "Calculate %s times %s.", "%s * %s = ?"]) % (n(a), n(b))
            answer = "%d × %d = %d." % (a, b, value)
        elif kind < 0.6:
            pct, whole = rng.choice([7.5, 12.5, 17.5, 18, 23, 35, 62.5]), round(rng.uniform(80, 5000), 2)
            expression, value = "%s / 100 * %s" % (pct, whole), round(pct / 100 * whole, 2)
            ask = "What is %s%% of %s?" % (n(float(pct)), n(whole))
            answer = "%s%% of %s is %s." % (n(float(pct)), n(whole), n(value))
        elif kind < 0.75:
            bill, people, tip = round(rng.uniform(24, 240), 2), rng.randint(2, 6), rng.choice([10, 15, 18, 20])
            expression, value = "%s * %s / %d" % (bill, 1 + tip / 100, people), round(bill * (1 + tip / 100) / people, 2)
            ask = "The bill is %s. With a %d%% tip, how much does each of %d people pay?" % (money(bill), tip, people)
            answer = "Each pays %s (the bill with a %d%% tip is %s)." % (money(value), tip, money(round(bill * (1 + tip / 100), 2)))
        elif kind < 0.9:
            a = rng.randint(2, 40) ** 2 * rng.choice([1, 1, 1, 2, 3])
            expression, value = "sqrt(%d)" % a, round(a ** 0.5, 4)
            ask = "What is the square root of %s?" % n(a)
            answer = "The square root of %s is %s." % (n(a), n(value) if value != int(value) else n(int(value)))
        else:
            a, b, c = rng.randint(100, 999), rng.randint(10, 99), rng.randint(1000, 9999)
            expression, value = "(%d + %d) * %d" % (a, b, c), (a + b) * c
            ask = "What is (%d + %d) × %s?" % (a, b, n(c))
            answer = "(%d + %d) × %d = %d." % (a, b, c, value)
        result_value = calculated(expression)
        for effort, thought in ((STANDARD, "Big numbers: the calculator is exact."),
                                (SUPERIOR, "This needs exact arithmetic I could get wrong in my head. I'll use the "
                                           "calculator with %s and read the result back." % expression)):
            if effort == SUPERIOR and rng.random() < 0.5:
                continue
            add([user(ask), say("", thought, [call("calculator", expression=expression)]),
                 result({"result": result_value}), say(answer)], effort=effort)

    # A calculator error: the expression in words, then again in symbols.
    for _ in range(40):
        a, b = rng.randint(3, 99), rng.randint(3, 99)
        words = rng.choice(["%d times %d", "%d multiplied by %d", "%d plus %d", "%d minus %d"]) % (a, b)
        symbol = {"times": "*", "multiplied by": "*", "plus": "+", "minus": "-"}[words.split(" ", 1)[1].rsplit(" ", 1)[0]]
        value = eval("%d %s %d" % (a, symbol, b))
        add([user("Use the calculator: %s." % words),
             say("", "I'll pass it to the calculator.", [call("calculator", expression=words)]),
             result({"error": "not a plain math expression: " + words}),
             say("", "It wants symbols, not words: %d %s %d." % (a, symbol, b),
                 [call("calculator", expression="%d %s %d" % (a, symbol, b))]),
             result({"result": value}), say("%s = %s." % (words.capitalize(), n(value)))])
    for _ in range(10):
        a = rng.randint(1, 500)
        add([user("What is %d divided by 0?" % a),
             say("", "Division by zero: I'll check with the calculator.", [call("calculator", expression="%d / 0" % a)]),
             result({"error": "division by zero"}),
             say("It can't be done: dividing by zero has no answer, because no number times 0 gives %d." % a)])

    # The time somewhere, and the difference between two places.
    for _ in range(90):
        (city, zone), when = rng.choice(CITIES), moment(rng)
        found = time_result(city, zone, when)
        add([user(rng.choice(["What time is it in %s?", "What's the time in %s right now?", "Time in %s?"]) % city),
             say("", "The time changes: I'll ask the time tool.", [call("get_time", location=city)]),
             result(found), say("It's %s in %s (%s)." % (found["time"], city, found["date"]))],
            effort=rng.choice([STANDARD, None]))
    for _ in range(60):
        (a, za), (b, zb) = rng.sample(CITIES, 2)
        when = moment(rng)
        ra, rb = time_result(a, za, when), time_result(b, zb, when)
        offset = (when.astimezone(ZoneInfo(zb)).utcoffset() - when.astimezone(ZoneInfo(za)).utcoffset()).total_seconds() / 3600
        hours = n(abs(offset)) if offset != int(offset) else "%d" % abs(offset)
        if offset == 0:
            gap = "%s and %s have the same time right now: %s." % (a, b, ra["time"])
        else:
            gap = "%s is %s hour%s %s of %s: it's %s there and %s in %s." % (
                b, hours, "" if abs(offset) == 1 else "s", "ahead" if offset > 0 else "behind", a, rb["time"], ra["time"], a)
        add([user("How many hours ahead or behind is %s from %s right now?" % (b, a)),
             say("", "I need both clocks: the time tool for each city, then the difference.",
                 [call("get_time", location=a), call("get_time", location=b)]),
             result(ra), result(rb),
             say(gap, "%s in %s, %s in %s: %s hours apart." % (ra["time"], a, rb["time"], b, hours))],
            effort=SUPERIOR if rng.random() < 0.5 else STANDARD)

    # The weather, two places compared, and weather with the time.
    for _ in range(90):
        city = rng.choice(CITIES)[0]
        found = weather_result(city, rng)
        add([user(rng.choice(["What's the weather in %s?", "Is it raining in %s?", "How warm is it in %s right now?",
                              "weather %s"]) % city),
             say("", "Current weather: the weather tool.", [call("get_weather", location=city)]),
             result(found), say("In %s: %s" % (city, describe(found)))])
    for _ in range(50):
        a, b = rng.sample([c[0] for c in CITIES], 2)
        wa, wb = weather_result(a, rng), weather_result(b, rng)
        if wa["temperature_c"] == wb["temperature_c"]:
            continue
        warmer = a if wa["temperature_c"] > wb["temperature_c"] else b
        add([user("Is it warmer in %s or in %s right now?" % (a, b)),
             say("", "Two places: the weather tool for each, then compare.",
                 [call("get_weather", location=a), call("get_weather", location=b)]),
             result(wa), result(wb),
             say("%s is warmer: %s °C there, against %s °C in %s." % (
                 warmer, max(wa["temperature_c"], wb["temperature_c"]), min(wa["temperature_c"], wb["temperature_c"]),
                 b if warmer == a else a),
                 "%s: %s °C; %s: %s °C. %s is warmer." % (a, wa["temperature_c"], b, wb["temperature_c"], warmer))],
            effort=rng.choice([STANDARD, SUPERIOR]))
    for _ in range(40):
        city, zone = rng.choice(CITIES)
        w, t = weather_result(city, rng), time_result(city, zone, moment(rng))
        add([user("What's the weather and the time in %s?" % city),
             say("", "Two things: the weather tool and the time tool, both for %s." % city,
                 [call("get_weather", location=city), call("get_time", location=city)]),
             result(w), result(t), say("In %s it's %s. %s" % (city, t["time"], describe(w)))])

    # A misspelled place: not found, then the right spelling – or asking which one.
    for right, wrong in MISSPELLED.items():
        found = weather_result(right, rng)
        add([user("What's the weather in %s?" % wrong),
             say("", "Weather: the weather tool.", [call("get_weather", location=wrong)]),
             result({"error": "location not found"}),
             say("", "Not found: probably a typo for %s. I'll try that." % right, [call("get_weather", location=right)]),
             result(found), say("In %s: %s" % (right, describe(found)))])
    for place in ["Springfield", "San Jose", "Georgetown", "Victoria", "Richmond", "Kingston"]:
        add([user("What time is it in %s?" % place),
             say("", "The time tool can tell me.", [call("get_time", location=place)]),
             result({"error": "no time zone is known for %s; try a large city or a zone like Asia/Tokyo" % place}),
             say("There are several places called %s, and the time tool couldn't tell which one you mean. Which "
                 "country or state is it in?" % place,
                 "The tool doesn't know which %s. I'll ask instead of guessing." % place)])
    # The weather tool fails: try once more, then search the web instead, or say it plainly.
    for _ in range(30):
        city = rng.choice(CITIES)[0]
        w = weather_result(city, rng)
        snippet = "%s today: %s, around %d °C." % (city, w["conditions"], round(w["temperature_c"]))
        add([user("What's the weather like in %s?" % city),
             say("", "Weather: the weather tool.", [call("get_weather", location=city)]),
             result({"error": "<urlopen error timed out>"}),
             say("", "The weather tool timed out. A web search can find a forecast instead.",
                 [call("web_search", query="%s weather today" % city)]),
             result({"results": [{"title": "%s weather" % city, "url": "https://weather.example/%s" % city.lower().replace(" ", "-"),
                                  "snippet": snippet}]}),
             say("A weather site says %s is %s today, around %d °C. (The weather tool didn't answer, so this is "
                 "from a web search.)" % (city, w["conditions"], round(w["temperature_c"])))])
    for _ in range(12):
        city = rng.choice(CITIES)[0]
        add([user("Weather in %s?" % city),
             say("", "The weather tool.", [call("get_weather", location=city)]),
             result({"error": "<urlopen error [Errno 11001] getaddrinfo failed>"}),
             say("I couldn't get the weather: the weather tool can't reach the internet right now. Check your "
                 "connection, or try again in a moment.", "No connection. I say so plainly rather than guess.")],
            tools=[CALC, WEATHER, TIME])

    # Looking things up: search, then answer from the result – copying names and numbers exactly.
    for ask, query, title, url, snippet, answer in LOOKUPS:
        for effort, thought in ((STANDARD, "Something to look up: I'll search."),
                                (SUPERIOR, "I don't know this reliably, and it can change. I'll search for \"%s\" "
                                           "and answer from the results." % query)):
            add([user(ask), say("", thought, [call("web_search", query=query)]),
                 result({"results": [{"title": title, "url": url, "snippet": snippet},
                                     {"title": "More about " + query, "url": url.rsplit("/", 1)[0], "snippet": "See the full page for details."}]}),
                 say(answer)], effort=effort)
        # Nothing found at first: a better query.
        add([user(ask),
             say("", "Something to look up.", [call("web_search", query=ask.rstrip("?"))]),
             result({"results": []}),
             say("", "No results. A shorter query might work: \"%s\"." % query, [call("web_search", query=query)]),
             result({"results": [{"title": title, "url": url, "snippet": snippet}]}),
             say(answer)])
    # Many more, about made-up things: the query from the question's words, the answer copied from the result.
    for ask, query, title, url, snippet, answer in made_up_lookups(rng, 320):
        effort = rng.choice([STANDARD, STANDARD, SUPERIOR])
        thought = ("Something to look up: I'll search." if effort == STANDARD else
                   "I don't know this; it's local and can change. I'll search for \"%s\" and answer from the result." % query)
        add([user(ask), say("", thought, [call("web_search", query=query)]),
             result({"results": [{"title": title, "url": url, "snippet": snippet}]}), say(answer)], effort=effort)
    # Search, then read the page for the details.
    for ask, query, title, url, snippet, page, answer in PAGES:
        for effort in (STANDARD, SUPERIOR):
            add([user(ask),
                 say("", "Something to look up: I'll search.", [call("web_search", query=query)]),
                 result({"results": [{"title": title, "url": url, "snippet": snippet}]}),
                 say("", "The snippet is cut short. I'll read the page.", [call("fetch_page", url=url)]),
                 result({"text": page, "more": False}),
                 say(answer)], effort=effort)
        # The page is gone: the result's snippet, or another page.
        add([user(ask),
             say("", "Something to look up: I'll search.", [call("web_search", query=query)]),
             result({"results": [{"title": title, "url": url, "snippet": snippet},
                                 {"title": title + " (archive)", "url": url + "-archive", "snippet": snippet}]}),
             say("", "The snippet is cut short. I'll read the first page.", [call("fetch_page", url=url)]),
             result({"error": "HTTP Error 404: Not Found"}),
             say("", "That page is gone. I'll try the second result.", [call("fetch_page", url=url + "-archive")]),
             result({"text": page, "more": False}),
             say(answer)])
    # Searching, then working something out with the calculator.
    for _ in range(30):
        rate = round(rng.uniform(0.8, 1.6), 3)
        amount = rng.choice([25, 40, 75, 120, 250, 480, 1000])
        pair = rng.choice([("euros", "dollars", "EUR", "USD"), ("pounds", "euros", "GBP", "EUR"),
                           ("dollars", "Canadian dollars", "USD", "CAD"), ("francs", "euros", "CHF", "EUR")])
        value = round(amount * rate, 2)
        add([user("How much is %d %s in %s?" % (amount, pair[0], pair[1])),
             say("", "Exchange rates change: search for today's, then multiply.",
                 [call("web_search", query="%s to %s exchange rate" % (pair[2], pair[3]))]),
             result({"results": [{"title": "%s to %s" % (pair[2], pair[3]), "url": "https://rates.example/%s-%s" % (pair[2], pair[3]),
                                  "snippet": "1 %s = %s %s (live rate)" % (pair[2], rate, pair[3])}]}),
             say("", "1 %s = %s %s. Now %d × %s." % (pair[2], rate, pair[3], amount, rate),
                 [call("calculator", expression="%d * %s" % (amount, rate))]),
             result({"result": calculated("%d * %s" % (amount, rate))}),
             say("%d %s is about %s %s at today's rate (1 %s = %s %s). Rates change, so the bank's may differ a little." % (
                 amount, pair[0], n(value), pair[1], pair[2], rate, pair[3]))], effort=rng.choice([STANDARD, SUPERIOR]))

    # Files: list a folder, read a file, and find the right name after a wrong one.
    for folder, (files, folders) in FOLDERS.items():
        add([user("What's in %s?" % folder),
             say("", "The files tool lists a folder.", [call("list_files", folder=folder)]),
             result({"folder": folder, "files": files, "folders": folders}),
             say("%s has %d files – %s – and the folder%s %s." % (
                 folder, len(files), ", ".join(files), "" if len(folders) == 1 else "s", ", ".join(folders)))])
    todo = "C:/Users/me/Documents/todo.md"
    add([user("What's on my to-do list? It's todo.md in my Documents."),
         say("", "I'll read the file.", [call("read_file", path=todo)]),
         result({"path": todo, "text": FILE_TEXTS[todo]}),
         say("Your to-do list:\n- Call the bank\n- Buy a birthday card for Lena\n- Book the car service\n- Send the photos to Omar")])
    add([user("Read my to-do list at C:/Users/me/Documents/todo.txt"),
         say("", "I'll read the file.", [call("read_file", path="C:/Users/me/Documents/todo.txt")]),
         result({"error": "no such file: C:/Users/me/Documents/todo.txt"}),
         say("", "No such file. I'll list the folder to find the right name.",
             [call("list_files", folder="C:/Users/me/Documents")]),
         result({"folder": "C:/Users/me/Documents", "files": FOLDERS["C:/Users/me/Documents"][0],
                 "folders": FOLDERS["C:/Users/me/Documents"][1]}),
         say("", "It's todo.md, not todo.txt.", [call("read_file", path=todo)]),
         result({"path": todo, "text": FILE_TEXTS[todo]}),
         say("It's called todo.md. On it:\n- Call the bank\n- Buy a birthday card for Lena\n- Book the car service\n- Send the photos to Omar")])
    budget = "C:/Users/me/Documents/budget.csv"
    add([user("Add up my budget in C:/Users/me/Documents/budget.csv."),
         say("", "Read the file, then add the amounts with the calculator.", [call("read_file", path=budget)]),
         result({"path": budget, "text": FILE_TEXTS[budget]}),
         say("", "Amounts: 950, 320, 85, 25, 120.", [call("calculator", expression="950 + 320 + 85 + 25 + 120")]),
         result({"result": 1500}),
         say("The total is 1,500: rent 950, food 320, transport 85, phone 25 and fun 120.")], effort=SUPERIOR)
    for path in ["C:/Users/me/Documents/notes.txt", "C:/Users/me/Documents/trip.md", "~/notes/groceries.txt",
                 "~/notes/meeting-monday.md", "~/notes/ideas.md", "D:/Projects/site/changelog.txt",
                 "D:/Projects/site/README.md"]:
        text = FILE_TEXTS[path]
        summary = {
            "C:/Users/me/Documents/notes.txt": "Two notes: the guest Wi-Fi password is maple-river-42, and the plumber comes on Tuesday at 10:00.",
            "C:/Users/me/Documents/trip.md": "Your Lisbon trip: the flight is on 12 May at 07:40 from gate B12, you stay 3 nights at Casa Azul, and you want to see Belém, Alfama and tram 28.",
            "~/notes/groceries.txt": "Five things: eggs, spinach, oat milk, coffee beans and lemons.",
            "~/notes/meeting-monday.md": "From Monday's meeting: the launch moved to 3 November, Priya owns the test plan, and the next check-in is Thursday at 15:00.",
            "~/notes/ideas.md": "Three ideas: a small app to track plant watering, learning to bake sourdough, and running a 10 km race in spring.",
            "D:/Projects/site/changelog.txt": "The latest version is 1.2.0, which added a contact form; before it, 1.1.1 fixed the menu on phones and 1.1.0 added a dark theme.",
            "D:/Projects/site/README.md": "It's a small static website: you edit index.html and style.css, and the images are in images/.",
        }[path]
        add([user(rng.choice(["What does %s say?", "Summarize %s.", "Read %s for me."]) % path),
             say("", "I'll read the file.", [call("read_file", path=path)]),
             result({"path": path, "text": text}), say(summary)])

    # What no tool can do: say so, and offer what can be done.
    for ask, what, offer in CANNOT:
        thought = "They want me to %s. My tools are %s; none can %s. I say so and offer what I can." % (
            what.replace("send ", "send ").split(" or ")[0], tool_names, what)
        reply = "I can't %s – none of my tools can. %s" % (what, offer)
        add([user(ask), say(reply, thought)], effort=SUPERIOR)
        add([user(ask), say(reply, "None of my tools can %s." % what)], effort=STANDARD)
        # A tool it does not have, called anyway (not learned), and the error read and answered.
        made_up = {"send emails": "send_email", "send text messages": "send_sms", "set alarms": "set_alarm",
                   "set reminders": "create_reminder", "play music": "play_music", "translate": "translate"}.get(what)
        if made_up:
            add([user(ask),
                 say("", "I'll do it.", [call(made_up, text=ask)], learn=False),
                 result({"error": "there is no tool named %s; the tools here are: %s" % (made_up, tool_names)}),
                 say(reply, "There's no %s tool. I say so and offer what I can." % made_up)])
    # A tool that is not in the list today: the weather without a weather tool, a search without a search tool.
    for _ in range(14):
        city = rng.choice(CITIES)[0]
        add([user("What's the weather in %s?" % city),
             say("I can't check the weather here: I have no weather or search tool. A weather app or website will show it.",
                 "No weather tool and no search tool in my list. I say so.")], tools=[CALC, TIME])
    for ask, query, title, url, snippet, answer in LOOKUPS[:8]:
        add([user(ask),
             say("I can't look that up here: I have no search tool, and I don't know it reliably. A quick web search "
                 "will find it.", "I don't know this reliably and there's no search tool. I say so rather than guess.")],
            tools=[CALC, WEATHER, TIME])
    # And when a question needs no tool at all, none is used.
    for ask, answer in [("What is the capital of France?", "Paris."), ("How many days are in a week?", "Seven."),
                        ("What is 6 times 7?", "42."), ("Who are you?", "I'm Magi, a small assistant made in MacroMagi. I think things through and use my tools for facts and exact results."),
                        ("What's the opposite of cold?", "Hot."), ("Say hi!", "Hi!"),
                        ("Is the sun a star?", "Yes, the sun is a star."), ("What is water made of?", "Hydrogen and oxygen: H₂O.")]:
        add([user(ask), say(answer)], effort=STANDARD)
        add([user(ask), say(answer, "I know this; no tool is needed.")], effort=SUPERIOR)
    rng.shuffle(out)
    return out


# ---- small talk, answered briefly ---------------------------------------------------

GREETINGS = (["hi", "hello", "hey", "hiya", "hey there", "hello there", "hi there", "yo", "howdy", "heyy", "hii",
              "helo", "hi!", "Hello!", "Hey!", "Hi :)", "hello 👋", "good morning", "Good morning!", "good afternoon",
              "Good evening", "morning", "hey magi", "Hi Magi!", "hello magi", "sup", "what's up", "whats up"],
             ["Hi! How can I help?", "Hello! What can I do for you?", "Hey! What do you need?",
              "Hi there! How can I help you today?", "Hello! What would you like to do?"],
             "A greeting. I greet back and offer help.")
HOW_ARE_YOU = (["how are you?", "How are you", "how are you doing", "how's it going?", "How are things?",
                "how r u", "you ok?", "How do you do?", "how have you been?", "how is your day", "Are you well?",
                "hi, how are you?", "Hello, how are you today?", "hey how's it going"],
               ["I'm doing well, thanks! How can I help?", "Good, thank you! What can I do for you?",
                "All good here – how about you?", "I'm well, thanks for asking. What do you need?"],
               "They ask how I am. A short friendly answer, then back to them.")
THANKS = (["thanks", "thank you", "Thanks!", "thank you so much", "thx", "ty", "Thanks a lot!", "cheers",
           "much appreciated", "thanks, that helped", "great, thank you", "perfect, thanks", "Thank you very much."],
          ["You're welcome!", "Happy to help!", "Anytime!", "Glad it helped!", "You're welcome. Anything else?"],
          "They thank me. I say they're welcome.")
BYE = (["bye", "goodbye", "Bye!", "see you", "see ya", "good night", "Goodnight!", "talk later", "I have to go",
        "that's all, bye", "catch you later"],
       ["Goodbye!", "Bye! Have a good day.", "See you!", "Good night!", "Take care!"],
       "They're leaving. I say goodbye.")
OKAY = (["ok", "okay", "cool", "got it", "alright", "nice", "great", "sounds good", "I see", "makes sense", "awesome",
         "k", "fine", "sure"],
        ["Great. Anything else?", "Good! Let me know if you need anything.", "Glad that's clear.", "Okay!"],
        "They acknowledge. A short reply.")
LAUGH = (["lol", "haha", "hahaha", "😂", "that's funny", "lmao"],
         ["Glad that made you smile!", "😄", "Ha! Glad you liked it."],
         "They laugh. A light reply.")
PRAISE = (["you're great", "good job", "well done", "you're smart", "nice work", "you're awesome", "that was helpful"],
          ["Thank you! Glad I could help.", "Thanks – happy to help!", "That's kind of you. Anything else?"],
          "A compliment. I thank them.")
WHO = (["what's your name?", "who are you", "Who are you?", "what are you?", "are you a robot?", "are you an AI?",
        "are you human?", "what should I call you?", "introduce yourself", "tell me about yourself"],
       ["I'm Magi, a small AI assistant made in MacroMagi. I think things through and use tools for facts and exact results.",
        "I'm Magi – an AI assistant made in MacroMagi. I can answer questions, work things out step by step and use tools.",
        "My name is Magi. I'm a small AI assistant made in MacroMagi."],
       "They ask about me. I'm Magi, a small AI assistant made in MacroMagi.")
MADE = (["who made you?", "who created you", "who built you?", "where do you come from?"],
        ["I was made in MacroMagi, an app for building and training AI models and macros.",
         "I was trained in MacroMagi."],
        "They ask who made me: MacroMagi.")
CAN_HELP = (["can you help me?", "I need help", "help", "can you do something for me?", "I have a question",
             "quick question"],
            ["Of course. What do you need?", "Sure – what's the question?", "Happy to. What can I do?"],
            "They want help. I ask what they need.")
FEELINGS = (["I'm sad", "I had a bad day", "I'm tired", "I'm stressed", "I'm bored", "I feel lonely"],
            ["I'm sorry to hear that. Do you want to talk about it, or would a distraction help?",
             "That sounds hard. A short break, some water or fresh air can help a little. Want to tell me what happened?",
             "Sorry you're feeling that way. Is there something I can help with?"],
            "They share a feeling. I answer kindly and briefly, and offer to help.")
SMALL_TALK = [GREETINGS, HOW_ARE_YOU, THANKS, BYE, OKAY, LAUGH, PRAISE, WHO, MADE, CAN_HELP, FEELINGS]


def small_talk(rng=None):
    rng = rng or random.Random(31)
    out = []
    for asks, replies, thought in SMALL_TALK:
        for ask in asks:
            for k in range(2):
                reply = rng.choice(replies)
                out.append({"effort": STANDARD, "prompt": ask, "response": reply})
                if k == 0:
                    out.append({"effort": SUPERIOR, "prompt": ask, "thinking": thought, "response": reply})
            # With tools offered, small talk still needs none.
            out.append({"tools": ROUTER_TOOLS, "effort": STANDARD, "prompt": ask, "response": rng.choice(replies)})
    # A greeting, then a real question: the greeting is short, the question answered.
    follow = [("What is 12 times 7?", "84.", None), ("What is the capital of Japan?", "Tokyo.", None),
              ("Can you tell me a fun fact?", "Octopuses have three hearts.", None),
              ("What's 15% of 80?", "12.", "10% of 80 is 8, 5% is 4; 8 + 4 = 12.")]
    for greet in GREETINGS[0][:14]:
        question, answer, thought = rng.choice(follow)
        out.append({"effort": STANDARD, "messages": [
            user(greet), say(rng.choice(GREETINGS[1])), user(question), say(answer, thought)]})
    rng.shuffle(out)
    return out


# ---- longer answers, laid out --------------------------------------------------------
#
# Most real questions want more than a word: steps, a list, or an explanation
# with an example. The answer leads with the point, then the detail.

EXPLAIN = [
    ("How do I make a good cup of tea?",
     "Plan: kettle, warm the pot, steep time, extras.",
     "1. Boil fresh water. For black tea it should be just off the boil; for green tea, let it cool for a minute or two.\n"
     "2. Warm the cup or pot with a little hot water, then pour it out.\n"
     "3. Add one tea bag or a teaspoon of loose tea per cup.\n"
     "4. Pour the water over and steep: 3–5 minutes for black tea, 2–3 for green.\n"
     "5. Remove the tea, then add milk, lemon or honey if you like.\n\n"
     "Steeping too long makes tea bitter, so set a timer the first few times."),
    ("Explain how photosynthesis works.",
     "Plan: what it is, inputs, outputs, where, why it matters.",
     "Photosynthesis is how plants make their own food from light.\n\n"
     "- **Inputs:** carbon dioxide from the air, water from the roots, and sunlight.\n"
     "- **Where:** in the leaves, inside tiny parts called chloroplasts, which hold the green pigment chlorophyll.\n"
     "- **Outputs:** sugar (glucose), which the plant uses for energy and growth, and oxygen, which it releases.\n\n"
     "In short: light + water + carbon dioxide → sugar + oxygen. It's also where almost all the oxygen we breathe comes from."),
    ("What's the difference between weather and climate?",
     "Plan: define each, the time scale, an example.",
     "**Weather** is what the air is doing right now or over a few days: sunny, rainy, 18 °C, windy.\n\n"
     "**Climate** is the usual weather of a place over many years – typically 30 or more.\n\n"
     "An example: a cold, snowy day in Madrid is weather. Madrid having hot, dry summers and mild winters is its climate. "
     "Weather tells you whether to take an umbrella today; climate tells you what clothes to pack for a season."),
    ("How do I write a for loop in Python?",
     "Plan: the syntax, an example over a list, range, a tip.",
     "A `for` loop runs the same code once for each item in a sequence:\n\n"
     "```python\nfruits = [\"apple\", \"banana\", \"cherry\"]\nfor fruit in fruits:\n    print(fruit)\n```\n\n"
     "To repeat something a number of times, loop over `range`:\n\n"
     "```python\nfor i in range(5):\n    print(i)   # 0, 1, 2, 3, 4\n```\n\n"
     "The indented lines are the loop's body. Use `enumerate(fruits)` when you need the position as well as the item."),
    ("Give me tips for a job interview.",
     "Plan: before, during, after – short and practical.",
     "**Before**\n- Read about the company and the role, and prepare two or three questions to ask them.\n"
     "- Practise short stories about your work: a problem, what you did, and the result.\n\n"
     "**During**\n- Arrive a few minutes early (or test your camera and sound for a video call).\n"
     "- Listen to the whole question, then answer it directly, with an example.\n"
     "- It's fine to pause and think before answering.\n\n"
     "**After**\n- Send a short thank-you note the same day."),
    ("How does the internet work, simply?",
     "Plan: computers, addresses, packets, routers, an example.",
     "The internet is a huge network of computers connected by cables, radio and fibre.\n\n"
     "1. Every device gets an address (an IP address), like a postal address.\n"
     "2. When you open a website, your request is cut into small pieces called packets.\n"
     "3. Routers pass the packets from network to network until they reach the website's server.\n"
     "4. The server sends the page back the same way, and your browser puts the pieces together.\n\n"
     "It all happens in a fraction of a second, which is why a page appears almost at once."),
    ("How can I save money each month?",
     "Plan: see spending, plan, automate, cut, small habits.",
     "1. **See where it goes.** Write down a month of spending, or check your bank statements.\n"
     "2. **Make a simple plan.** Decide how much goes to needs, wants and savings – 50/30/20 is a common start.\n"
     "3. **Save first.** Move the savings to a separate account on payday, automatically.\n"
     "4. **Cut the leaks.** Cancel subscriptions you don't use and compare prices on big bills.\n"
     "5. **Wait before buying.** Give non-essential purchases a day or two.\n\n"
     "Small, steady amounts add up faster than occasional big ones."),
    ("What is a CPU and what does it do?",
     "Plan: definition, what it does, cores, speed, analogy.",
     "The CPU (central processing unit) is the part of a computer that carries out instructions – the computer's brain.\n\n"
     "- It fetches an instruction from memory, works out what it means, and does it: adding numbers, comparing them, moving data.\n"
     "- It repeats this billions of times a second (its clock speed, measured in GHz).\n"
     "- Modern CPUs have several **cores**, each able to work on a task at the same time.\n\n"
     "If a program is a recipe, the CPU is the cook following it step by step."),
    ("How do I get better at sleeping?",
     "Plan: schedule, light, screens, food and drink, the room.",
     "- **Keep a schedule:** go to bed and get up at the same times, weekends too.\n"
     "- **Get daylight** in the morning and keep the evening dim.\n"
     "- **Put screens away** 30–60 minutes before bed.\n"
     "- **Watch caffeine and alcohol:** no coffee after early afternoon; alcohol makes sleep lighter.\n"
     "- **Make the room** dark, quiet and cool (around 18 °C).\n\n"
     "If you can't sleep after about 20 minutes, get up and do something calm until you feel sleepy. If problems go on for weeks, see a doctor."),
    ("What is the difference between a virus and bacteria?",
     "Plan: size, alive or not, how they spread, treatment.",
     "**Bacteria** are tiny single-celled living things. Many are harmless or useful, like those in your gut; some cause illness, such as strep throat. Antibiotics can kill bacteria.\n\n"
     "**Viruses** are much smaller and can't live or multiply on their own: they get inside a host's cells and use them to make copies. Colds and flu are viruses. Antibiotics don't work on viruses; vaccines and some antiviral medicines do.\n\n"
     "So: bacteria are cells and can be treated with antibiotics; viruses need a host and can't."),
    ("How do I start learning to code?",
     "Plan: pick a language, a small goal, practise, projects, help.",
     "1. **Pick one language.** Python is a friendly first choice.\n"
     "2. **Learn the basics:** variables, if, loops and functions – a free online course or book is enough.\n"
     "3. **Code a little every day.** Twenty minutes daily beats three hours once a week.\n"
     "4. **Build something small** you care about: a quiz, a to-do list, a budget calculator.\n"
     "5. **Read errors calmly.** They tell you the line and the problem; searching the message usually helps.\n\n"
     "Expect to feel stuck often – that's part of learning, not a sign you can't do it."),
    ("Why is the sky blue?",
     "Plan: sunlight's colours, scattering, why blue, sunsets.",
     "Sunlight is a mix of all colours. When it passes through the air, it bumps into tiny gas molecules and gets scattered in all directions.\n\n"
     "Blue light has a shorter wavelength than red, so it's scattered much more. That scattered blue light reaches your eyes from every part of the sky, so the sky looks blue.\n\n"
     "At sunset, light travels through much more air. Most of the blue is scattered away before it reaches you, leaving the reds and oranges."),
    ("How do I write a good email to my manager asking for a day off?",
     "Plan: subject, the ask with the date, cover, polite close; then a sample.",
     "Keep it short and clear: the date, a reason if you want to give one, and how your work is covered.\n\n"
     "**Subject:** Day off on Friday 14 November\n\n"
     "Hi Anna,\n\nCould I take a day off on Friday 14 November? I'll finish the weekly report by Thursday, and Omar has agreed to cover any urgent requests.\n\nThanks,\nSam\n\n"
     "Ask early, and mention the date in the subject so it's easy to find."),
    ("What are the planets in our solar system?",
     "Plan: list in order, inner and outer, a fact.",
     "There are eight planets. In order from the Sun:\n\n"
     "1. Mercury\n2. Venus\n3. Earth\n4. Mars\n5. Jupiter\n6. Saturn\n7. Uranus\n8. Neptune\n\n"
     "The first four are small and rocky; the outer four are giants made mostly of gas and ice. Jupiter is the largest. Pluto was reclassified as a dwarf planet in 2006."),
    ("How do I change a flat tyre?",
     "Plan: safety, tools, the steps in order, check after.",
     "1. **Get safe:** pull over on flat ground away from traffic, turn on the hazard lights and apply the handbrake.\n"
     "2. **Get the tools:** spare tyre, jack and wheel wrench (usually under the boot floor).\n"
     "3. **Loosen the nuts** a little while the wheel is still on the ground.\n"
     "4. **Jack up the car** at the jacking point shown in the manual, until the tyre is off the ground.\n"
     "5. **Swap the wheel:** remove the nuts and the flat, put on the spare, and hand-tighten the nuts.\n"
     "6. **Lower the car** and tighten the nuts fully in a star pattern.\n\n"
     "Check the spare's pressure soon, and drive gently if it's a thin temporary spare (often limited to 80 km/h)."),
    ("Explain what an API is.",
     "Plan: definition, analogy, example, why it matters.",
     "An API (application programming interface) is a set of rules that lets one program ask another for data or actions.\n\n"
     "Think of a restaurant: you don't go into the kitchen; you order from a menu, and the waiter brings the food. The menu and the waiter are the API.\n\n"
     "For example, a weather app doesn't measure the weather itself. It sends a request like \"weather for Paris\" to a weather service's API and gets back the temperature and conditions.\n\n"
     "APIs let programs work together without knowing how the other one is built inside."),
    ("How should I prepare for an exam?",
     "Plan: plan early, active study, practice tests, rest.",
     "1. **Start early** and spread study over days; a little each day sticks better than cramming.\n"
     "2. **Study actively:** test yourself, explain topics out loud, and make short summaries instead of re-reading.\n"
     "3. **Do past papers** under timed conditions, then review every mistake.\n"
     "4. **Focus on weak spots** – the topics you'd rather avoid are usually the ones to practise.\n"
     "5. **Sleep and eat well** the days before; tired brains forget.\n\n"
     "On the day, read every question carefully and answer the easy ones first."),
    ("What is compound interest?",
     "Plan: definition, example with numbers, why it matters.",
     "Compound interest is interest earned on your interest, not just on the money you put in.\n\n"
     "An example: put $1,000 in an account at 5% a year.\n"
     "- After year 1: $1,000 + $50 = $1,050.\n"
     "- After year 2: 5% of $1,050 is $52.50, so $1,102.50.\n"
     "- After year 10: about $1,629.\n\n"
     "The growth speeds up over time, which is why starting to save early makes such a difference – and why debt that compounds grows fast too."),
    ("How do vaccines work?",
     "Plan: immune system, what a vaccine shows it, memory, result.",
     "Vaccines train your immune system to recognise a germ before you meet the real thing.\n\n"
     "1. A vaccine contains a harmless piece or copy of a germ, or instructions to make one.\n"
     "2. Your immune system reacts to it and makes antibodies and memory cells.\n"
     "3. If the real germ arrives later, the memory cells recognise it and respond quickly, often before you get ill.\n\n"
     "That's why vaccinated people usually get no illness or a much milder one. Some vaccines need boosters to keep the memory strong."),
    ("How do I make my writing clearer?",
     "Plan: point first, short sentences, plain words, cut, read aloud.",
     "- **Put the main point first.** Readers should know what you want after one sentence.\n"
     "- **Use short sentences** – one idea each.\n"
     "- **Choose plain words:** \"use\" instead of \"utilise\", \"help\" instead of \"facilitate\".\n"
     "- **Cut what doesn't add anything:** \"in order to\" → \"to\", \"at this point in time\" → \"now\".\n"
     "- **Read it out loud.** Where you stumble, the reader will too.\n\n"
     "Clear writing isn't simple-minded; it's respectful of the reader's time."),
]


def explain():
    out = []
    for ask, plan, answer in EXPLAIN:
        out.append({"effort": STANDARD, "prompt": ask, "response": answer})
        out.append({"effort": SUPERIOR, "prompt": ask, "thinking": plan, "response": answer})
        out.append({"tools": ROUTER_TOOLS, "effort": STANDARD, "prompt": ask, "response": answer})
    return out


if __name__ == "__main__":
    for name, items in (("thinking", thinking()), ("agent", agent()), ("small_talk", small_talk()), ("explain", explain())):
        print(name, len(items))
