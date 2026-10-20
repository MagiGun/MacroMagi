"""The current weather, from Open-Meteo's free service (online).

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    import urllib.parse
    import urllib.request
    location = str(args.get("location", "")).strip()
    if not location:
        return {"error": "no location"}
    get = lambda url: json.loads(urllib.request.urlopen(url, timeout=15).read())
    found = get("https://geocoding-api.open-meteo.com/v1/search?count=1&name=" + urllib.parse.quote(location))
    if not found.get("results"):
        return {"error": "location not found"}
    place = found["results"][0]
    weather = get("https://api.open-meteo.com/v1/forecast?current=temperature_2m,weather_code,wind_speed_10m"
                  "&latitude=%s&longitude=%s" % (place["latitude"], place["longitude"]))["current"]
    codes = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "fog", 48: "fog", 51: "drizzle",
             61: "rain", 63: "rain", 65: "heavy rain", 71: "snow", 73: "snow", 75: "heavy snow", 80: "showers",
             81: "showers", 82: "heavy showers", 95: "thunderstorm"}
    return {"location": place.get("name", location), "temperature_c": weather["temperature_2m"],
            "conditions": codes.get(weather["weather_code"], "code %s" % weather["weather_code"]),
            "wind_kmh": weather["wind_speed_10m"]}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
