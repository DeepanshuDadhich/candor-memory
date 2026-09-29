"""Load every source in data/ into one flat list of Units with a delivery time.

Design notes
- One Unit = the most specific citable thing (meeting segment, ChatGPT message, Slack message, ...).
- Time filtering, edits and deletions are applied in plain code (never by an LLM), so the
  hard rules (no future records, no deleted messages) cannot be broken by a model.
- Secrets are masked at ingest. Text that tries to instruct an AI is kept as *content* but
  flagged, so downstream code can down-rank it and never obey it.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

SECRET_RE = re.compile(r"\b(sk|pk|rk|ghp|xox[abp])[-_][A-Za-z0-9_\-]{10,}")
INJECTION_RE = re.compile(
    r"(ignore (all |your )?(previous|prior) instructions|ai assistant[s]? (summari[sz]ing|reading)|"
    r"you are an? (ai|assistant)|disregard (the )?(above|previous))", re.I)


def mask(text):
    return SECRET_RE.sub("[REDACTED_SECRET]", text)


def dt(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


@dataclass
class Unit:
    id: str
    record: str
    time: datetime
    source: str            # meeting | dictation | slack | gmail | calendar | codex | chatgpt
    text: str
    speaker: str = ""
    meta: dict = field(default_factory=dict)
    injected: bool = False


def load(data_dir):
    d = Path(data_dir)
    units, deleted, edits = [], {}, {}

    def add(**kw):
        kw["text"] = mask(kw["text"])
        kw["injected"] = bool(INJECTION_RE.search(kw["text"]))
        units.append(Unit(**kw))

    for f in sorted((d / "native/meetings").glob("*.json")):
        m = json.loads(f.read_text())
        start = dt(m["start"])
        for s in m["segments"]:
            who = s.get("speaker_name") or s.get("speaker_label") or "Unknown speaker"
            conf = s.get("speaker_confidence")
            add(id=s["seg_id"], record=m["id"], time=start + timedelta(seconds=s["end_s"]),
                source="meeting", speaker=who,
                text=f"[{m['title']}, {m['start'][:10]}] {who}: {s['text']}",
                meta={"title": m["title"], "confidence": conf, "named": bool(s.get("speaker_name"))})

    for line in open(d / "native/dictation/dictations.jsonl"):
        x = json.loads(line)
        add(id=x["id"], record=x["id"], time=dt(x["timestamp"]), source="dictation",
            text=f"[Dictation {x['mode']} into {x['target_app']} - {x['target_context']}, "
                 f"{x['delivery_state']}] {x['cleaned_text']} {x.get('raw_transcript') or ''}",
            meta={"state": x["delivery_state"]})

    users = {u["id"]: u["real_name"] for u in json.load(open(d / "connectors/slack/users.json"))}
    chans = {c["id"]: c["name"] for c in json.load(open(d / "connectors/slack/channels.json"))}
    for line in open(d / "connectors/slack/messages.jsonl"):
        x = json.loads(line)
        t = dt(x["ts"])
        where = chans.get(x["channel_id"], x["channel_id"])
        if x.get("subtype") == "message_deleted":
            deleted[x["target_id"]] = t
            add(id=x["id"], record=x["id"], time=t, source="slack",
                text=f"[Slack #{where}] (a message was deleted)", meta={"deletes": x["target_id"]})
            continue                       # the event carries no content of the deleted message
        if x.get("subtype") == "message_changed":
            edits.setdefault(x["target_id"], []).append((t, x["text"]))
            add(id=x["id"], record=x["id"], time=t, source="slack",
                text=f"[Slack #{where}, edit of {x['target_id']}] {x['text']}",
                meta={"edit_of": x["target_id"]})
            continue
        who = users.get(x.get("user"), x.get("bot_name") or x.get("user"))
        add(id=x["id"], record=x["id"], time=t, source="slack", speaker=who,
            text=f"[Slack #{where}] {who}: {x['text']}", meta={"channel": where})

    for line in open(d / "connectors/gmail/messages.jsonl"):
        x = json.loads(line)
        add(id=x["id"], record=x["id"], time=dt(x["date"]), source="gmail", speaker=x["from"],
            text=f"[Email {x['date'][:16]}] From {x['from']} To {', '.join(x['to'])} | "
                 f"{x['subject']}\n{x['body']}", meta={"subject": x["subject"]})

    for line in open(d / "connectors/google_calendar/events.jsonl"):
        x = json.loads(line)
        st, en = x["start"], x["end"]
        when = f"{st.get('dateTime') or st.get('date')} to {en.get('dateTime') or en.get('date')}"
        att = ", ".join(a["email"] for a in x.get("attendees", []))
        add(id=x["id"], record=x["id"], time=dt(x["updated"]), source="calendar",
            text=f"[Calendar, {x['status']}] {x['summary']} | {when} | {x.get('location') or ''} | "
                 f"attendees: {att} | {x.get('description') or ''}")

    for f in sorted((d / "connectors/codex/sessions").glob("*.jsonl")):
        ev = [json.loads(l) for l in open(f)]
        meta, body = ev[0], ev[1:]
        text = "\n".join(f"{e.get('role', e.get('tool', e['type']))}: {e.get('content') or e.get('input', '')}"
                         for e in body)
        add(id=meta["id"], record=meta["id"], time=dt(body[-1]["timestamp"] if body else meta["started_at"]),
            source="codex", text=f"[Codex session, repo {meta.get('repo')}]\n{text}")

    for c in json.load(open(d / "connectors/chatgpt/conversations.json")):
        for m in c["messages"]:
            add(id=m["id"], record=c["id"], time=dt(m["create_time"]), source="chatgpt",
                text=f"[ChatGPT '{c['title']}'] {m['role']}: {m['content']}")

    units.sort(key=lambda u: u.time)
    return units, deleted, edits


def visible(units, deleted, edits, as_of):
    """Only what existed at `as_of`: not from the future, not deleted, edits applied."""
    out = []
    for u in units:
        if u.time > as_of or (u.id in deleted and deleted[u.id] <= as_of):
            continue
        tgt = u.meta.get("edit_of")           # an edit of a message that has since been deleted is gone too
        if tgt and tgt in deleted and deleted[tgt] <= as_of:
            continue
        newer = [txt for t, txt in edits.get(u.id, []) if t <= as_of]
        if newer:
            head = u.text.partition(": ")[0]
            u = Unit(**{**u.__dict__, "text": mask(f"{head}: {newer[-1]}"), "injected": u.injected})
        out.append(u)
    return out
