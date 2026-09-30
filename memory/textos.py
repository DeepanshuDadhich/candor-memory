"""TextOS: type a command, see what it would do, confirm, and (for a few safe actions) do it locally.

    python3 memory/textos.py [--as-of 2026-09-18T09:00:00-07:00]

Slack, Gmail and Calendar are not connected in this project, so those actions are only shown as a plan.
Two actions run for real, and only after you type y: app.open (macOS `open -a`) and reminder.create
(saved to out/reminders.json). memory.ask runs the memory pipeline and prints the answer.
confirm and clarify never act: they ask you first.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm
from actions import Directory, normalize, plan, rule_fallback
from loader import dt, load
from pipeline import Memory, build_glossary

DEFAULT_AS_OF = "2026-09-18T09:00:00-07:00"


def describe(a):
    t, g = a["type"], a["args"]
    return {
        "slack.send_message": lambda: f"Slack message to {g['to']}: {g['text']}",
        "gmail.send": lambda: f"Email to {', '.join(g['to'])} | {g['subject']}: {g['body']}",
        "calendar.create_event": lambda: f"New event '{g['title']}' {g['start']} to {g['end']} with {', '.join(g['attendees']) or 'no one else'}",
        "calendar.update_event": lambda: f"Change event {g['event_id']}: " + ", ".join(f"{k} -> {v}" for k, v in g.items() if k != "event_id"),
        "reminder.create": lambda: f"Reminder at {g['due']}: {g['text']}",
        "memory.ask": lambda: f"Ask memory: {g['question']}",
        "app.open": lambda: f"Open app: {g['app']}",
        "clarify": lambda: f"Question: {g['question']}",
        "confirm": lambda: f"Needs your OK first: {g['summary']}",
    }[t]()


def execute(a, mem, as_of):
    t, g = a["type"], a["args"]
    if t == "app.open":
        r = subprocess.run(["open", "-a", g["app"]], capture_output=True, text=True)
        print("  opened" if r.returncode == 0 else f"  could not open: {r.stderr.strip()}")
    elif t == "reminder.create":
        p = Path("out/reminders.json")
        p.parent.mkdir(exist_ok=True)
        items = json.loads(p.read_text()) if p.exists() else []
        p.write_text(json.dumps(items + [g], indent=2))
        print(f"  saved to {p}")
    elif t == "memory.ask" and mem:
        res = mem.run({"id": "TEXTOS", "question": g["question"], "as_of": as_of})
        print("  " + (res["answer"] if res else "no answer"))
    else:
        print("  (not connected here, shown as a plan only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=DEFAULT_AS_OF)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    dr = Directory(a.data)
    use_llm = llm.available()
    mem = None
    if use_llm:
        units, deleted, edits = load(a.data)
        mem = Memory(units, deleted, edits, build_glossary(a.data))
    print(f"TextOS ({'model: ' + llm.model_name() if use_llm else 'no model key, simple rules only'}). "
          f"Pretending it is {a.as_of}. Type a command, or 'quit'.")
    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd.lower() in ("quit", "exit", "q"):
            break
        if not cmd:
            continue
        acts = plan(cmd, a.as_of, dr, mem)[0] if use_llm else normalize(rule_fallback(cmd), dr, dt(a.as_of), cmd)
        for x in acts:
            print(" - " + describe(x))
        if any(x["type"] in ("clarify", "confirm") for x in acts):
            continue
        if input("Do it? [y/N] ").strip().lower() == "y":
            for x in acts:
                execute(x, mem, a.as_of)


if __name__ == "__main__":
    main()
