"""One command:  python3 memory/run_memory.py --questions Q.jsonl --out A.jsonl [--data data] [--no-llm]
                 [--only ID1,ID2] [--workers N]

With an API key in .env it runs the full pipeline (rewrite -> fuse -> rerank -> answer).
Without a key, or with --no-llm, or if a call fails for a question, it falls back to the keyword baseline.
Questions run in parallel (LLM_WORKERS in .env, default 6); output keeps the input order."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm
from loader import dt, load, visible
from pipeline import Memory, build_glossary
from retrieve import BM25

ABSTAIN_BELOW = 6.0     # baseline only: top BM25 score under this means nothing relevant


def baseline(units, deleted, edits, q):
    vis = visible(units, deleted, edits, dt(q["as_of"]))
    hits = BM25(vis).search(q["question"], k=20)
    top = hits[0][1] if hits else 0.0
    ids = [u.id for u, _ in hits]
    if top < ABSTAIN_BELOW:
        return {"id": q["id"], "answer": "I don't know. I don't have anything in memory about that.",
                "sources": [], "retrieved": ids, "abstained": True}
    return {"id": q["id"], "answer": " ".join(hits[0][0].text.split()[:35]), "sources": ids[:3],
            "retrieved": ids, "abstained": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--only", default="", help="comma separated question ids to run (for quick tests)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LLM_WORKERS", "6")))
    a = ap.parse_args()

    units, deleted, edits = load(a.data)
    use_llm = llm.available() and not a.no_llm
    mem = Memory(units, deleted, edits, build_glossary(a.data)) if use_llm else None
    print(f"mode: {'LLM pipeline (' + llm.model_name() + f'), {a.workers} parallel' if use_llm else 'keyword baseline (no LLM)'}",
          file=sys.stderr, flush=True)

    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        qs = [q for q in qs if q["id"] in keep]

    def one(q):
        t0 = time.time()
        res = None
        if use_llm:
            try:
                res = mem.run(q)
            except Exception as e:
                print(f"{q['id']}: pipeline error ({e}); using baseline", file=sys.stderr, flush=True)
        tag = "llm" if res else "baseline"
        res = res or baseline(units, deleted, edits, q)
        print(f"{q['id']} [{tag}] abstained={res['abstained']} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        return res

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers if use_llm else 1)) as pool:
        results = list(pool.map(one, qs))
    with open(a.out, "w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")
    print(f"done: {len(results)} questions in {time.time() - start:.0f}s -> {a.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
