#!/usr/bin/env bash
# One command: ./run.sh                  -> answers the train questions and scores retrieval
#              ./run.sh Q.jsonl A.jsonl  -> answers any question file
set -e
cd "$(dirname "$0")"

Q="${1:-evals/memory_train.jsonl}"
A="${2:-out/memory_train_answers.jsonl}"

# Pick the first Python 3.10+ interpreter available.
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 "$HOME/.local/bin/python3.13" python3; do
  if command -v "$cand" >/dev/null 2>&1 &&
     "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$cand"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "error: need Python 3.10 or newer" >&2
  exit 1
fi
echo "using $("$PY" --version) ($PY)"

mkdir -p out "$(dirname "$A")"
"$PY" memory/run_memory.py --questions "$Q" --out "$A" --data data
(cd eval_harness && "$PY" score_retrieval.py --gold ../evals/memory_train.jsonl --answers "../$A" | tail -6)
