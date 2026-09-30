"""Baseline retrieval: BM25 over the time-filtered units + a few cheap heuristics.
Standard library only. An LLM query-rewriter / reranker plugs in on top of this later."""
import math
import re
from collections import Counter

STOP = set("a an the is are was were be been of to in on for and or did do does i my me we our you your "
           "it its this that what when who which how why with at by from as about has have had not".split())

# Tiny alias table so a question word also matches how the data phrases it. Extend by hand from the data.
ALIASES = {"launch": ["launching", "ship", "release", "v2"], "launching": ["launch", "ship", "release"],
           "flight": ["fly", "flying", "denver", "airline"], "board": ["boardprep"],
           "salary": ["compensation", "pay"], "sign": ["signed", "contract", "signature"]}


MONTHS = {m: i for i, ms in enumerate([["jan", "january"], ["feb", "february"], ["mar", "march"], ["apr", "april"],
          ["may"], ["jun", "june"], ["jul", "july"], ["aug", "august"], ["sep", "sept", "september"], ["oct", "october"],
          ["nov", "november"], ["dec", "december"]], 1) for m in ms}
DATE_RES = [
    (re.compile(r"\b20\d\d-(\d\d)-(\d\d)"), lambda m: (int(m[1]), int(m[2]))),
    (re.compile(r"\b([a-z]{3,9})\.? (\d{1,2})(?:st|nd|rd|th)?\b"), lambda m: (MONTHS.get(m[1]), int(m[2]))),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})\b"), lambda m: (int(m[1]), int(m[2]))),
]


def date_tokens(text):
    """Same day, one token: '2026-09-23', 'Sep 23', 'September 23rd' and '9/23' all become d0923."""
    out = []
    for rx, fn in DATE_RES:
        for m in rx.finditer(text):
            mo, day = fn(m)
            if mo and 1 <= mo <= 12 and 1 <= day <= 31:
                out.append(f"d{mo:02d}{day:02d}")
    return out


def tokens(text):
    low = text.lower()
    words = [w for w in re.findall(r"[a-z0-9]+", low) if w not in STOP and len(w) > 1]
    return words + date_tokens(low)


class BM25:
    def __init__(self, units, k1=1.4, b=0.75):
        self.units = units
        self.docs = [Counter(tokens(u.text)) for u in units]
        self.len = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(units)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def feedback_query(self, units, n_terms=8, exclude=()):
        """Pseudo relevance feedback: the rarest terms shared by the top records (names, dates, product words)."""
        seen = Counter()
        for u in units:
            seen.update(set(tokens(u.text)))
        ex = set(exclude)
        cand = [t for t in seen if t not in ex and not t.isdigit()]
        cand.sort(key=lambda t: (seen[t] > 1, self.idf.get(t, 0)), reverse=True)
        return " ".join(cand[:n_terms])

    def search(self, query, k=20):
        q = []
        for t in tokens(query):
            q.append(t)
            q.extend(ALIASES.get(t, []))
        scored = []
        for i, c in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = c.get(t)
                if f:
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                u = self.units[i]
                if u.injected:
                    s *= 0.1                     # planted instructions are content, never a good source
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.units[i], s) for s, i in scored[:k]]
