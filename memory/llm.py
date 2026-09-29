"""One tiny LLM client for any OpenAI compatible endpoint (Gemini, DeepSeek, BytePlus, OpenAI).
Settings come from .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. Standard library only.
Returns None when no key is set or a call fails, so callers can fall back to the keyword baseline."""
import json
import os
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


def available():
    key = os.environ.get("LLM_API_KEY", "")
    return bool(key) and "paste_your_key" not in key


def chat(system, user, temperature=0.0, retries=3):
    if not available():
        return None
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = json.dumps({
        "model": os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
        "temperature": temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            print(f"LLM error {e.code}: {e.read()[:200]!r}")
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            print(f"LLM error: {e}")
            return None


if __name__ == "__main__":
    if not available():
        print("No key found. Put your real key in .env (LLM_API_KEY=...) and try again.")
    else:
        out = chat("Reply with exactly one word.", "Say: ready")
        print("Model replied:", out if out else "(no reply, see the error above)")
