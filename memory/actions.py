"""Action planner (TextOS core): a spoken or typed command -> typed actions, as a dry run.

    command + as_of
      1. build context in plain code: people (Slack ids, DM ids, emails), channels, the calendar as it existed
         at as_of, the next days of the week, and name clashes ("Sarah" is two different people)
      2. one LLM call plans the actions; if the command needs a fact from memory ("email John the corrected
         number"), it asks for it, the memory pipeline answers, and the planner runs again with that fact
      3. plain code checks and repairs every action: real ids only, times in Alex's timezone, moved events keep
         their length, and a command that needs a question or a confirmation gets exactly that and nothing else

Nothing is ever sent or changed here. `textos.py` is the interactive front end that can run a few actions
locally after a yes.
"""
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from loader import dt, mask
from pipeline import ask, parse_json

LA = ZoneInfo("America/Los_Angeles")
TYPES = {"slack.send_message", "gmail.send", "calendar.create_event", "calendar.update_event",
         "reminder.create", "memory.ask", "app.open", "clarify", "confirm"}
AUTOMATED = re.compile(r"^(no-?reply|notifications?|notify|digest|news|hello|events|partners|reminders|calendar-notification|all|hiring)\b", re.I)
DESTRUCTIVE = re.compile(r"\b(delete|erase|wipe|purge|destroy)\b|\bcancel (all|every|everything)\b|\bremove all\b", re.I)


INVITE = re.compile(r"^(?:Updated invitation|Invitation):\s*(.+?)\s*@\s*(.+)$", re.I)
_SPAN = re.compile(r"(?:\w{3}\s+)?([A-Za-z]{3})\w*\s+(\d{1,2}),\s*(\d{4})\s+(\d{1,2})(?::(\d\d))?\s*([ap]m)\s*[-–]\s*(\d{1,2})(?::(\d\d))?\s*([ap]m)", re.I)
_MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _parse_span(text):
    """'Thu Sep 17, 2026 2pm - 3pm (PDT)' -> (start, end) ISO strings in Los Angeles time."""
    m = _SPAN.search(text)
    if not m:
        return None
    mon, day, year = _MON.get(m[1][:3].lower()), int(m[2]), int(m[3])
    if not mon:
        return None

    def at(h, mi, ap):
        h = int(h) % 12 + (12 if ap.lower() == "pm" else 0)
        return datetime(year, mon, day, h, int(mi or 0), tzinfo=LA).isoformat(timespec="seconds")
    return at(m[4], m[5], m[6]), at(m[7], m[8], m[9])


# ---------- context ----------
class Directory:
    def __init__(self, data_dir):
        d = Path(data_dir)
        self.people = {}                                  # email -> {name, email, slack, dm, internal}
        users = json.load(open(d / "connectors/slack/users.json"))
        chans = json.load(open(d / "connectors/slack/channels.json"))
        self.channels = [c for c in chans if not c["is_dm"]]
        me = next((u["id"] for u in users if u.get("email", "").startswith("alex@")), "U01ALEX")
        dms = {}
        for c in chans:
            if c["is_dm"]:
                for m in c["members"]:
                    if m != me:
                        dms[m] = c["id"]
        for u in users:
            if u.get("email") and not u.get("is_bot"):
                self.people[u["email"].lower()] = {"name": u["real_name"], "email": u["email"].lower(), "slack": u["id"],
                                                   "dm": dms.get(u["id"]), "title": u.get("title", "")}
        for line in open(d / "connectors/gmail/messages.jsonl"):
            m = json.loads(line)
            for f in [m["from"]] + m["to"] + m.get("cc", []):
                name, _, rest = f.partition("<")
                email = (rest.rstrip(">") if rest else f).strip().lower()
                if not email or email in self.people or AUTOMATED.search(email):
                    continue
                nm = name.strip() or email.split("@")[0].replace(".", " ").title()
                self.people[email] = {"name": nm, "email": email, "slack": None, "dm": None, "title": ""}
        self.me = next(p for p in self.people.values() if p["email"].startswith("alex@"))
        self.events = {}
        for line in open(d / "connectors/google_calendar/events.jsonl"):
            e = json.loads(line)
            self.events[e["id"]] = e
        self.invites = {}                                 # event title -> [(sent, start, end)] from invitation emails
        for line in open(d / "connectors/gmail/messages.jsonl"):
            m = json.loads(line)
            hit = INVITE.match(m["subject"])
            if hit:
                span = _parse_span(hit.group(2))
                if span:
                    self.invites.setdefault(hit.group(1).strip().lower(), []).append((dt(m["date"]), *span))
        for v in self.invites.values():
            v.sort()

    # -- lookups --
    def person(self, text):
        t = str(text).strip().lower()
        for p in self.people.values():
            if t in (p["email"], p["name"].lower(), p["slack"] and p["slack"].lower(), p["dm"] and p["dm"].lower()):
                return p
        hits = [p for p in self.people.values() if t and (t == p["name"].split()[0].lower())]
        return hits[0] if len(hits) == 1 else None

    def slack_target(self, v):
        s = str(v).strip()
        low = s.lower().lstrip("#@")
        for c in self.channels:
            if low in (c["id"].lower(), c["name"].lower()):
                return c["id"]
        p = self.person(low)
        if p:
            return p["dm"] or p["slack"]
        for p in self.people.values():
            if s in (p["slack"], p["dm"]):
                return s
        return None

    def email_of(self, v):
        s = str(v).strip()
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", s)
        if m:
            return m.group(0).lower()
        p = self.person(s)
        return p["email"] if p else None

    def visible_events(self, as_of):
        """The calendar as Alex saw it at `as_of`. The file only holds each event's current state, which exists
        from its `updated` time. For an event changed after `as_of`, the earlier state comes from the last
        calendar invitation email about it before `as_of`; if there is none, the event is left out."""
        out = []
        for e in self.events.values():
            if dt(e["updated"]) <= as_of:
                out.append(e)
            elif dt(e["created"]) <= as_of:
                old = self.invites.get(e["summary"].lower(), [])
                old = [x for x in old if x[0] <= as_of]
                if old:
                    _, start, end = old[-1]
                    out.append({**e, "status": "confirmed", "start": {"dateTime": start}, "end": {"dateTime": end}})
        return sorted(out, key=lambda e: e["start"].get("dateTime") or e["start"].get("date"))

    def ambiguities(self, command):
        """First names shared by several people, when the command doesn't say which one."""
        words = set(re.findall(r"[a-z]+", command.lower()))
        by_first = {}
        for p in self.people.values():
            if p["email"] != self.me["email"]:
                by_first.setdefault(p["name"].split()[0].lower(), []).append(p)
        notes = []
        for first, ps in by_first.items():
            if first in words and len(ps) > 1 and not any(p["name"].split()[-1].lower() in words for p in ps):
                who = "; ".join(f"{p['name']} ({'Slack ' + p['slack'] if p['slack'] else 'email only, external'})" for p in ps)
                notes.append(f"'{first.title()}' could be: {who}")
        return notes


def _when(e):
    s, en = e["start"], e["end"]
    if "date" in s:
        return f"{s['date']} all day"
    a, b = dt(s["dateTime"]), dt(en["dateTime"])
    return f"{a:%Y-%m-%d %a %H:%M}-{b:%H:%M}"


def context_text(dr, as_of):
    now = as_of.astimezone(LA)
    days = ", ".join(f"{(now + timedelta(days=i)):%a %m-%d}" for i in range(0, 12))
    people = "\n".join(f"- {p['name']}{' (' + p['title'] + ')' if p['title'] else ''} | {p['email']} | "
                       f"{'slack ' + p['slack'] + (', dm ' + p['dm'] if p['dm'] else '') if p['slack'] else 'not on Slack'}"
                       for p in dr.people.values())
    chans = ", ".join(f"#{c['name']} = {c['id']}" for c in dr.channels)
    events = "\n".join(f"- {e['id']} | {e['summary']} | {_when(e)}"
                       + (f" | repeats {';'.join(e['recurrence'])}" if e.get("recurrence") else "")
                       + (" | CANCELLED" if e["status"] == "cancelled" else "")
                       for e in dr.visible_events(as_of))
    return (f"Now: {now:%A %Y-%m-%d %H:%M} America/Los_Angeles (UTC{now:%z}). Upcoming days: {days}.\n"
            f"The user is Alex Rivera ({dr.me['email']}).\n\nPEOPLE (name | email | Slack):\n{people}\n\n"
            f"SLACK CHANNELS: {chans}\n\nCALENDAR (id | title | when):\n{events}")


# ---------- planner ----------
PLAN_SYSTEM = """You turn one command from Alex Rivera into actions for a dry run. Nothing is executed.
Output JSON only: {"need_facts": [], "actions": [{"type": "...", "args": {...}}]}

Action types and args:
- slack.send_message: to (a Slack user id, DM id or channel id from the lists), text
- gmail.send: to (list of email addresses), subject, body   (cc optional)
- calendar.create_event: title, start, end, attendees (list of emails)
- calendar.update_event: event_id (from the CALENDAR list), plus the fields that change (start, end)
- reminder.create: text, due
- memory.ask: question   (the command itself is a question about Alex's past conversations, mail or calendar)
- app.open: app
- clarify: question   (the command is ambiguous and a wrong guess would matter)
- confirm: summary    (the command is destructive or irreversible: deleting, erasing, cancelling in bulk)

Rules:
- Times: write local time as YYYY-MM-DDTHH:MM:SS (America/Los_Angeles, no offset). Work relative dates out from Now
  and the list of upcoming days: "tomorrow" is the next day, "the 25th" is the next 25th of a month, "3pm" is 15:00.
- Moving or rescheduling an event: use calendar.update_event on that event's id, set the new start, and keep its
  length (end = new start + the event's old duration). Keep its date unless told otherwise.
- New meetings: title from the topic; length as stated, otherwise 30 minutes; attendees are people's emails.
- "N before/after <event>": find that event in the CALENDAR (match the whole name, for example "board meeting" is the
  event with that name, not other events that mention the board) and compute from its start. Reminder text says what to do.
- Channel: people who are on Slack get a Slack message by default; people who are not on Slack (external) get an email.
  If the command says email or Slack, follow it. Write messages in Alex's voice, short and natural, using facts from the command.
- If a first name matches several people (see NAME CLASHES): when the command names a channel (Slack, email) and only
  one of them can be reached that way, pick that person without asking. Otherwise, if nothing says which one, use
  clarify and name the options. Do not guess between them when the choice changes the channel or the person.
- Destructive commands get one confirm action that says exactly what would be deleted, and nothing else.
- Several requests in one command give several actions, one per request. Do not add actions nobody asked for.
- If the command asks for something you can't do with these types, use clarify.
- If an action needs a fact you don't have (a number, a decision, a date agreed in a past conversation) that is not in
  the command, the lists or the calendar, put a question in need_facts and leave actions empty. Then you will be told
  the answer. Never invent facts, numbers or email addresses.
- Only use ids and email addresses that appear in the lists."""


def rule_fallback(command):
    """No model available: cover the easy, safe cases and otherwise ask."""
    c = command.strip()
    if DESTRUCTIVE.search(c):
        return [{"type": "confirm", "args": {"summary": f"This looks destructive: '{c}'. Do you want me to go ahead?"}}]
    m = re.match(r"(?i)^(open|launch|start)\s+(.+)$", c)
    if m:
        return [{"type": "app.open", "args": {"app": m.group(2).strip().rstrip(".")}}]
    if c.endswith("?") or re.match(r"(?i)^(what|when|who|where|why|how|did|does|do|is|are)\b", c):
        return [{"type": "memory.ask", "args": {"question": c}}]
    return [{"type": "clarify", "args": {"question": "I can't work out what to do with that without a language model. Could you rephrase it?"}}]


def _iso(v):
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    d = d.replace(tzinfo=LA) if d.tzinfo is None else d.astimezone(LA)
    return d.isoformat(timespec="seconds")


def _clarify(q):
    return {"type": "clarify", "args": {"question": q}}


def normalize(raw, dr, as_of, command):
    """Check and repair the model's plan. Returns a clean action list."""
    out = []
    for a in raw if isinstance(raw, list) else []:
        if not isinstance(a, dict) or a.get("type") not in TYPES:
            continue
        t, g = a["type"], dict(a.get("args") or {})
        if t == "slack.send_message":
            to = dr.slack_target(g.get("to"))
            text = mask(str(g.get("text") or "").strip())
            if not to:
                return [_clarify("Who or which channel should I send that to on Slack?")]
            if text:
                out.append({"type": t, "args": {"to": to, "text": text}})
        elif t == "gmail.send":
            to = [e for e in (dr.email_of(x) for x in (g.get("to") if isinstance(g.get("to"), list) else [g.get("to")])) if e]
            cc = [e for e in (dr.email_of(x) for x in (g.get("cc") if isinstance(g.get("cc"), list) else [g.get("cc")] if g.get("cc") else [])) if e]
            body = mask(str(g.get("body") or "").strip())
            if not to:
                return [_clarify("Which email address should I send that to?")]
            if body:
                out.append({"type": t, "args": {"to": list(dict.fromkeys(to)), "cc": cc,
                                                "subject": str(g.get("subject") or "Quick note").strip(), "body": body}})
        elif t == "calendar.create_event":
            start = _iso(g.get("start"))
            if not start:
                return [_clarify("What day and time should I schedule it?")]
            end = _iso(g.get("end")) or (datetime.fromisoformat(start) + timedelta(minutes=30)).isoformat(timespec="seconds")
            att = [e for e in (dr.email_of(x) for x in (g.get("attendees") or [])) if e]
            out.append({"type": t, "args": {"title": str(g.get("title") or "Meeting").strip(), "start": start,
                                            "end": end, "attendees": list(dict.fromkeys(att))}})
        elif t == "calendar.update_event":
            ev = {e["id"]: e for e in dr.visible_events(as_of)}.get(str(g.get("event_id")))
            if not ev:
                return [_clarify("Which calendar event do you mean?")]
            args = {"event_id": ev["id"]}
            start = _iso(g.get("start"))
            end = _iso(g.get("end"))
            if start and "dateTime" in ev["start"] and not end:       # a moved event keeps its length
                length = dt(ev["end"]["dateTime"]) - dt(ev["start"]["dateTime"])
                end = (datetime.fromisoformat(start) + length).isoformat(timespec="seconds")
            if start:
                args["start"] = start
            if end:
                args["end"] = end
            if g.get("title"):
                args["title"] = str(g["title"])
            out.append({"type": t, "args": args})
        elif t == "reminder.create":
            due = _iso(g.get("due"))
            if not due:
                return [_clarify("When should I remind you?")]
            out.append({"type": t, "args": {"text": mask(str(g.get("text") or "").strip()), "due": due}})
        elif t == "memory.ask":
            q = str(g.get("question") or command).strip()
            out.append({"type": t, "args": {"question": q}})
        elif t == "app.open":
            if str(g.get("app") or "").strip():
                out.append({"type": t, "args": {"app": str(g["app"]).strip()}})
        elif t in ("clarify", "confirm"):
            key = "question" if t == "clarify" else "summary"
            if str(g.get(key) or "").strip():
                out.append({"type": t, "args": {key: str(g[key]).strip()}})
    gate = [a for a in out if a["type"] in ("confirm", "clarify")]
    if gate:                                              # asking first means not acting yet
        return [next((a for a in gate if a["type"] == "confirm"), gate[0])]
    if DESTRUCTIVE.search(command):                       # safety net independent of the model
        return [{"type": "confirm", "args": {"summary": f"This looks destructive: '{command.strip()}'. Do you want me to go ahead?"}}]
    return out or [_clarify("I couldn't tell what to do with that. Could you say it another way?")]


def plan(command, as_of, dr, memory=None):
    """Return (actions, facts_used)."""
    now = dt(as_of)
    ctx = context_text(dr, now)
    clashes = dr.ambiguities(command)
    facts = []
    for round_ in range(2):
        user = (f"{ctx}\n\nNAME CLASHES: {'; '.join(clashes) if clashes else 'none'}\n"
                + (f"\nFACTS FROM MEMORY (already looked up, use them):\n" + "\n".join(f"- {q} -> {a}" for q, a in facts) + "\n" if facts else "")
                + f"\nCOMMAND: {command}")
        out = parse_json(ask(PLAN_SYSTEM, user)) or {}
        need = [q for q in (out.get("need_facts") or []) if isinstance(q, str) and q.strip()][:2]
        if need and memory is not None and round_ == 0 and not out.get("actions"):
            for i, q in enumerate(need):
                res = memory.run({"id": f"ACT-LOOKUP-{i}", "question": q, "as_of": as_of})
                facts.append((q, (res or {}).get("answer", "unknown")))
            continue
        return normalize(out.get("actions"), dr, now, command), facts
    return [_clarify("I couldn't find the details I need for that.")], facts
