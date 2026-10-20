"""The current date and time, from this computer's clock.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    import datetime
    location = str(args.get("location", "")).strip()
    zone = None
    if location:
        from zoneinfo import ZoneInfo, available_timezones
        wanted = location.lower().replace(" ", "_")
        for name in sorted(available_timezones()):
            if name.lower() == wanted or name.lower().endswith("/" + wanted):
                zone = ZoneInfo(name)
                break
        if zone is None:
            return {"error": "no time zone is known for " + location + "; try a large city or a zone like Asia/Tokyo"}
    now = datetime.datetime.now(zone) if zone else datetime.datetime.now().astimezone()
    return {"location": location or "here", "time": now.strftime("%H:%M"), "date": now.strftime("%A %d %B %Y"),
            "zone": str(now.tzinfo)}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
