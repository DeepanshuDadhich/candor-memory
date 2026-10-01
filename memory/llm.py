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
_extra_ok = True                       # set to False if the provider rejects LLM_EXTRA_BODY
USAGE_FILE = Path(__file__).resolve().parent.parent / "out" / "token_usage.json"
_usage_lock = threading.Lock()


def _used():
    """{model: {"total", "prompt", "completion", "reasoning", "calls"}} (older int-only files are migrated)."""
    try:
        raw = json.loads(USAGE_FILE.read_text())
    except Exception:
        return {}
    return {m: (v if isinstance(v, dict) else {"total": int(v)}) for m, v in raw.items()}


def _budget_left():
    """LLM_TOKEN_BUDGET in .env caps total tokens per model across all runs (0 or unset = no cap)."""
    cap = int(os.environ.get("LLM_TOKEN_BUDGET", "0") or 0)
    return None if cap <= 0 else cap - _used().get(model_name(), {}).get("total", 0)


def _record(usage):
    usage = usage or {}
    add = {"total": int(usage.get("total_tokens") or 0),
           "prompt": int(usage.get("prompt_tokens") or 0),
           "completion": int(usage.get("completion_tokens") or 0),
           "reasoning": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
           "calls": 1}
    with _usage_lock:
        u = _used()
        cur = u.setdefault(model_name(), {})
        for k, v in add.items():
            cur[k] = cur.get(k, 0) + v
        USAGE_FILE.parent.mkdir(exist_ok=True)
        USAGE_FILE.write_text(json.dumps(u))


def available():
    key = os.environ.get("LLM_API_KEY", "")
    return bool(key) and "paste_your_key" not in key


def model_name():
    return os.environ.get("LLM_MODEL", "gemini-2.5-flash")


def chat(system, user, temperature=0.0, retries=None):
    retries = retries or int(os.environ.get("LLM_RETRIES", "4"))
    global _last_call, _extra_ok
    if not available():
        return None
    left = _budget_left()
    if left is not None and left <= 0:
        print(f"LLM token budget used up for {model_name()}; skipping call (falls back to keywords)",
              file=sys.stderr, flush=True)
        return None
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    gap = float(os.environ.get("LLM_MIN_INTERVAL", "1.0"))
    payload = {
        "model": model_name(),
        "temperature": temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    extra = {}
    try:                                    # optional provider specific fields, e.g. {"thinking": {"type": "disabled"}}
        extra = json.loads(os.environ.get("LLM_EXTRA_BODY", "") or "{}")
    except ValueError:
        print("LLM_EXTRA_BODY is not valid JSON; ignoring it", file=sys.stderr, flush=True)

    def request():
        body = dict(payload, **(extra if _extra_ok else {}))
        return urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})

    for attempt in range(retries):
        with _pace:                                    # space out call starts, even across threads
            wait = gap - (time.time() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.time()
        try:
            with urllib.request.urlopen(request(), timeout=180) as r:
                data = json.loads(r.read())
            _record(data.get("usage"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and extra and _extra_ok:
                _extra_ok = False                  # another provider: drop LLM_EXTRA_BODY once and retry plainly
                print(f"LLM {e.code} with LLM_EXTRA_BODY; retrying without it for the rest of the run",
                      file=sys.stderr, flush=True)
                continue
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
        u = _used().get(model_name(), {})
        left = _budget_left()
        print(f"Usage on {model_name()}: total {u.get('total', 0)}, prompt {u.get('prompt', 0)}, "
              f"completion {u.get('completion', 0)} (of which reasoning {u.get('reasoning', 0)}), calls {u.get('calls', 0)}"
              + ("" if left is None else f", budget left {left}"))
