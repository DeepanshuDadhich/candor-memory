#!/usr/bin/env bash
# One command:  bash run.sh
#   Answers the memory train questions and the action train commands (dry run), then scores both.
#   Other inputs: bash run.sh memory  QUESTIONS.jsonl [OUT.jsonl]
#                 bash run.sh actions COMMANDS.jsonl  [OUT.jsonl]
# Needs Python 3.10+ and nothing else. Model settings come from .env (see .env.example); without a key
# it still runs, in keyword / rule mode, with lower scores.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3.10 "$HOME/.local/bin/python3.13" python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is required (for example: brew install python@3.13)." >&2
  exit 1
fi
mkdir -p out

case "${1:-}" in
  memory)
    $PY memory/run_memory.py --questions "$2" --out "${3:-out/memory_answers.jsonl}" --data data ;;
  actions)
    $PY memory/run_actions.py --commands "$2" --out "${3:-out/action_predictions.jsonl}" --data data ;;
  "")
    echo "== memory (train)"
    $PY memory/run_memory.py --questions evals/memory_train.jsonl --out out/memory_train_answers.jsonl --data data
    (cd eval_harness && $PY score_retrieval.py --gold ../evals/memory_train.jsonl --answers ../out/memory_train_answers.jsonl --out ../out/results_retrieval.json | tail -6)
    (cd eval_harness && $PY score_memory.py --gold ../evals/memory_train.jsonl --answers ../out/memory_train_answers.jsonl --judge none --out ../out/results_memory.json | tail -4)
    echo "== actions (train, dry run)"
    $PY memory/run_actions.py --commands evals/actions_train.jsonl --out out/actions_train_predictions.jsonl --data data
    (cd eval_harness && $PY score_actions.py --gold ../evals/actions_train.jsonl --predictions ../out/actions_train_predictions.jsonl --out ../out/results_actions.json | tail -15) ;;
  *)
    echo "usage: bash run.sh [memory QUESTIONS.jsonl [OUT] | actions COMMANDS.jsonl [OUT]]" >&2; exit 2 ;;
esac
