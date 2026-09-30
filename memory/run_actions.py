"""Dry run:  python3 memory/run_actions.py --commands C.jsonl --out P.jsonl [--data data] [--no-llm]

Input lines: {"id", "command", "as_of"}. Output lines: {"id", "actions": [...]}. Nothing is executed."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm
from actions import Directory, normalize, plan, rule_fallback
from loader import dt, load
from pipeline import Memory, build_glossary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    dr = Directory(a.data)
    use_llm = llm.available() and not a.no_llm
    mem = None
    if use_llm:
        units, deleted, edits = load(a.data)
        mem = Memory(units, deleted, edits, build_glossary(a.data))
    print(f"mode: {'LLM planner (' + llm.model_name() + ')' if use_llm else 'rule fallback (no LLM)'}", file=sys.stderr, flush=True)

    rows = [json.loads(l) for l in open(a.commands) if l.strip()]

    def one(r):
        try:
            acts = plan(r["command"], r["as_of"], dr, mem)[0] if use_llm else normalize(rule_fallback(r["command"]), dr, dt(r["as_of"]), r["command"])
        except Exception as e:
            print(f"{r['id']}: error ({e}); using rule fallback", file=sys.stderr, flush=True)
            acts = normalize(rule_fallback(r["command"]), dr, dt(r["as_of"]), r["command"])
        print(f"{r['id']} -> {[x['type'] for x in acts]}", file=sys.stderr, flush=True)
        return {"id": r["id"], "actions": acts}

    with ThreadPoolExecutor(max_workers=max(1, a.workers if use_llm else 1)) as pool:
        res = list(pool.map(one, rows))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"done: {len(res)} commands -> {a.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
