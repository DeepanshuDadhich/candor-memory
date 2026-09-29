"""One tiny LLM client for any OpenAI compatible endpoint (Gemini, DeepSeek, BytePlus, OpenAI).
Settings come from .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (optional: LLM_MIN_INTERVAL seconds
between calls, default 1.0, to stay inside free tier rate limits). Standard library only.
Returns None when no key is set or a call keeps failing, so callers can fall back to the keyword baseline."""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def _load_env():
    p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
_last_call = 0.0
_pace = threading.Lock()


def available():
    key = os.environ.get("LLM_API_KEY", "")
    return bool(key) and "paste_your_key" not in key


def model_name():
    return os.environ.get("LLM_MODEL", "gemini-2.5-flash")


def chat(system, user, temperature=0.0, retries=None):
    retries = retries or int(os.environ.get("LLM_RETRIES", "4"))
    global _last_call
    if not available():
        return None
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    gap = float(os.environ.get("LLM_MIN_INTERVAL", "1.0"))
    body = json.dumps({
        "model": model_name(),
        "temperature": temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    for attempt in range(retries):
        with _pace:                                    # space out call starts, even across threads
            wait = gap - (time.time() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                pause = min(4 * 2 ** attempt, 40)          # back off harder on rate limits
                print(f"LLM {e.code}, retry {attempt + 1}/{retries - 1} in {pause}s", file=sys.stderr, flush=True)
                time.sleep(pause)
                continue
            print(f"LLM error {e.code}: {e.read()[:300]!r}", file=sys.stderr, flush=True)
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            print(f"LLM error: {e}", file=sys.stderr, flush=True)
            return None


if __name__ == "__main__":
    if not available():
        print("No key found. Put your real key in .env (LLM_API_KEY=...) and try again.")
    else:
        out = chat("Reply with exactly one word.", "Say: ready")
        print("Model replied:", out if out else "(no reply, see the error above)")
