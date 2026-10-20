"""Speech to text with a small Whisper model, on this computer (pip install faster-whisper).

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    import os
    path = os.path.expanduser(str(args.get("path", "")).strip())
    if not os.path.isfile(path):
        return {"error": "no such file: " + path}
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"error": "transcribe_audio needs faster-whisper: pip install faster-whisper"}
    model = WhisperModel(os.environ.get("MAGI_WHISPER_MODEL", "tiny"), device="auto", compute_type="int8")
    segments, info = model.transcribe(path, language=(args.get("language") or None))
    parts = [{"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()} for s in segments]
    return {"language": info.language, "text": " ".join(p["text"] for p in parts), "segments": parts[:200]}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
