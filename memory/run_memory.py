"""One command: python3 memory/run_memory.py --questions Q.jsonl --out A.jsonl [--data data]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from loader import dt, load, visible
from retrieve import BM25

ABSTAIN_BELOW = 6.0     # top BM25 score under this => nothing relevant in memory (tuned on train, revisit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    units, deleted, edits = load(a.data)
    with open(a.questions) as f, open(a.out, "w") as out:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            vis = visible(units, deleted, edits, dt(q["as_of"]))
            hits = BM25(vis).search(q["question"], k=20)
            top = hits[0][1] if hits else 0.0
            abstain = top < ABSTAIN_BELOW
            ids = [u.id for u, _ in hits]
            if abstain:
                ans = "I don't know. I don't have anything in memory about that."
            else:
                ans = " ".join(hits[0][0].text.split()[:35])      # placeholder until the LLM answerer exists
            out.write(json.dumps({"id": q["id"], "answer": ans, "sources": [] if abstain else ids[:3],
                                  "retrieved": ids, "abstained": abstain}) + "\n")


if __name__ == "__main__":
    main()
