# Candor take-home: memory for Alex, plus TextOS

Two weeks of Alex Rivera's work life (meetings, dictation, Slack, email, calendar, Codex, ChatGPT) go in.
Out come answers that know **what is true now, what was true then, who said what, and when to say "I don't know"**,
plus a small command assistant (TextOS) that turns "move board deck prep to 3pm" into checked actions.

```
bash run.sh
```

That one command answers the memory train questions and the action train commands (dry run) and scores both.

## Results (train sets, final commit)

Model: `deepseek-v4-flash-ga-260731` on BytePlus ModelArk, thinking disabled.

| | Score |
|---|---|
| Retrieval (needed records in top 10) | **96.0%** (95% CI 88–100%), MRR 0.92 |
| Forbidden records retrieved | **0** |
| Answers, offline scorer (strict / lenient) | **81.5% / 92.6%** |
| Actions, dry run | **100%** (12/12), argument accuracy 100% |
| Without any model key (keyword and rule mode) | retrieval 72%, actions 25% |

Cost and speed: the run that produced the committed outputs took **23 seconds and 77,000 tokens**, because every
model call is cached on disk and only the answer prompt and the action guards had changed. The last cold run of the
whole pipeline on this model (the commit just before the final code-side guards) used **about 216,000 tokens in 42
seconds** for 27 questions + 12 commands. A 2-question test on the earlier Flash model measured about 7,100 tokens
per memory question. I spent ₹0: everything ran on BytePlus free token packs.

On an earlier model (`deepseek-v4-1-flash-260910`) with earlier prompts, the design scored 96% retrieval, 88.9% / 100%
answers and 100% actions. Model and prompts changed together, so that difference isn't isolated (see *Known
weaknesses*).

Output files from this commit: `out/memory_train_answers.jsonl`, `out/actions_train_predictions.jsonl`.
Note: `bash run.sh` rewrites these two files, so running it without a key replaces them with keyword-mode results.
`git checkout out/` restores the submitted versions.

## Running it

Needs Python 3.10+ and nothing else (standard library only).

```
cp .env.example .env          # then put a key in LLM_API_KEY
bash run.sh                    # train sets, with scores
bash run.sh memory  QUESTIONS.jsonl  [OUT.jsonl]    # any memory question file
bash run.sh actions COMMANDS.jsonl   [OUT.jsonl]    # any action command file (dry run)
python3 memory/textos.py --as-of 2026-09-18T09:00:00-07:00   # interactive TextOS
```

Any OpenAI-compatible chat API works. `.env` settings:

| Setting | Meaning |
|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | the provider (BytePlus, DeepSeek, OpenAI, Gemini's OpenAI endpoint, ...) |
| `LLM_EXTRA_BODY` | optional JSON merged into each request; I use `{"thinking":{"type":"disabled"}}` on BytePlus. If a provider rejects it, the client drops it automatically and retries. |
| `LLM_TOKEN_BUDGET` | hard cap on tokens per model across runs; past it the system falls back instead of spending |
| `LLM_WORKERS`, `LLM_MIN_INTERVAL` | parallel questions (default 6) and spacing between calls, for rate limits |

With no key, or when a call fails, it still runs and still writes valid output, using keywords and rules.

## Memory design

```
question + as_of
  1. time filter (code)   only records delivered by as_of; Slack edits applied as of that moment; deleted
                          messages gone from their deletion time; calendar events from their `updated` time
  2. query rewrite (LLM)  up to 4 searches in the words the records use: names, synonyms, full names for
                          ambiguous first names
  3. candidates (code)    BM25 over the question and each rewrite, fused by reciprocal rank, plus one
                          feedback hop on the rarest terms of the top hits; reserved slots per source
  4. rerank (LLM)         picks and orders the records an answer needs; can ask for a date ("the day I fly")
                          and the day's calendar events and emails are fetched and ranked again
  5. answer (LLM)         draft -> evidence -> final: the draft's claims are searched for, and if that finds
                          records the model hasn't seen, it answers again with them in view
```

**One store, most specific ids.** Every source becomes a flat list of units: a meeting segment, a ChatGPT message,
a Slack message, an email, a dictation, a calendar event, a Codex session. Each keeps its id, delivery time,
speaker (and speaker confidence), and source metadata. Citing the exact segment is what the scorer and a user want.

**Hard rules live in code, not in prompts.** Future records, deleted messages and superseded edits are removed
before any model sees anything, so no model mistake can leak them. Secrets are masked at ingest with a pattern
(`[REDACTED_SECRET]`), and the output is masked again. Text addressed to AI assistants (the planted instruction in
a promo email) is detected at ingest, down-ranked in search and replaced by a stub before it reaches a model, so the
model never reads it; email addresses that only appear in such text are removed from answers.

**Retrieval details that mattered**
- Dates are normalised: "Sep 23", "9/23", "September 23rd" and "2026-09-23" all become one token. This alone took
  the keyword baseline from 64% to 72%.
- The reranker sees emails without their address clutter, so the subject and first lines (where dates live) fit.
- At most 4 segments from one meeting in the top 10, so a long meeting can't push out the short email that
  settles the question.
- The reranker is told to prefer the record that states a fact first-hand over later summaries of it, and for
  "has X happened" questions, the latest update on X.

**Answer rules.** The newest record wins, and older values appear only as history. Disagreement is reported as
"people disagree" with who holds which view, never as "I don't know". "A said B said X" is reported as second-hand.
Unidentified speakers ("Speaker 2") are never given names. Abstain only when the records never mention the subject.

**Engineering.** Questions run in parallel; every call is cached on disk by prompt hash (atomic writes, thread
safe); a token budget stops spending before the free quota runs out; every step has a fallback, so a failed call
degrades one question instead of crashing a run.

## TextOS (bonus)

```
command + as_of
  1. context (code)   people with Slack ids, DM ids and emails (Slack users + email contacts, minus automated
                      senders); channels; the calendar as it stood at as_of; the next 12 days; name clashes
  2. guards (code)    an unresolved name clash ("message Sarah": Sarah Kim or Sarah Patel?) returns a clarify
                      question before any model call; phrases like "the corrected NRR" or "the latest date"
                      trigger a memory lookup first, so the message carries the real number
  3. plan (LLM)       one call returns typed actions; it may still ask the memory for a missing fact
  4. check (code)     only real ids and emails; times in America/Los_Angeles with offsets; a moved event
                      keeps its length; clarify/confirm are returned alone; delete-style commands always get
                      a confirm even if the model forgot
```

**Calendar as of a moment.** The data holds each event's current state only. For an event changed after the
command's time, TextOS rebuilds the earlier time from the last calendar invitation email before that moment (on
Sep 12, board deck prep is still Thursday 2–3pm). Recurring events are marked as such.

**Interactive mode.** `memory/textos.py` shows the plan and asks "Do it? [y/N]". After a yes it really opens apps
(macOS `open -a`), saves reminders to `out/reminders.json`, and answers questions through the memory. Slack, Gmail
and Calendar aren't connected in this project, so those are shown as plans. Clarify and confirm never act.

## What didn't work (and what I changed)

1. **Keyword search alone: 60%.** Questions and records use different words ("why did the launch slip" vs "push
   it to October"). Query rewriting plus LLM reranking took retrieval to 88%.
2. **Gemini free tier.** Rate limits made each question take 2–4 minutes. I made the client provider-agnostic and
   moved to BytePlus.
3. **Reasoning costs.** DeepSeek V4 Pro was slow (342s for a 2-question test) and one full run plus tests used
   almost its whole 500,000-token free pack (per the provider's quota notice; I wasn't counting tokens yet). On
   the Flash model, turning thinking off, together with trimming the prompts, cut a 2-question test from about
   19,200 to 7,100 tokens per question and from 77s to 8s.
4. **An "answer-guided" step that overstated retrieval.** My first version searched with the finished answer and
   inserted the hits into `retrieved`, so the list credited records the answer never used. I replaced it with
   draft -> evidence -> final: `retrieved` now only lists records the answer writer actually read.
5. **Snippets too short (likely cause).** After I cut rerank snippets to 240 characters, the date hop stopped
   firing for the Denver question; the flight date sits about 290 characters into the airline email. Emails now
   drop address clutter instead, and the hop worked again on that model. I didn't test the cut in isolation.
6. **Train-shaped prompts.** Some prompt examples were close to train gold answers. I replaced them with invented
   ones and removed dataset-specific synonyms, because the hidden test uses different questions. Two answers got
   worse around the same time, but the model changed too, so I can't say how much is the prompts.
7. **Trusting the model to follow rules.** On the newer model the planner ignored the name-clash note and skipped
   the memory lookup (actions fell to 83%). Both rules moved into code, and actions went back to 100% on that model
   (not rerun on the older one). Lesson: anything that must always happen belongs in code.

**Known weaknesses**
- MEM-TR-25 ("my calendar the day I fly to Denver"): the newer model finds the flight but doesn't pull the board
  meeting, so the answer says the day is empty. The older model got it right.
- "I don't know" in front of a correct latest state (MEM-TR-26: the answer gives the pending review and reply date,
  but opens with "I don't know", so it is scored as an abstention).
- Three answers are "unverified" in the offline scorer because they mention the old value as history; I believe
  they are right, but only the LLM judge can confirm.
- Answer quality moved between runs (88.9% vs 81.5% strict), but model and prompts changed together. A clean
  comparison (same code, both models) is the next eval I'd run.

## If this were the real product

The flat, rebuilt-per-query store is fine for two weeks of data and keeps the time rules exact. At Candor scale I'd
keep the same rules but move them into the data: an append-only Postgres table of units with `valid_from` and
`valid_to` (so "as of" is a query, not a filter), pgvector embeddings next to BM25, a people table that resolves
speakers, Slack ids and email addresses to one person, and commitments and decisions extracted at ingest as rows
that later records can update. Retrieval would stay hybrid, with the LLM reranker kept for the final ordering.

## Tools used

- Claude Code for writing and reviewing code; I steered the design, reviewed the changes and ran the evals.
- BytePlus ModelArk: `deepseek-v4-flash-ga-260731` (final), `deepseek-v4-1-flash-260910`, `deepseek-v4-pro-ga-260813`
  (early); free token packs, ₹0 spent.
- Gemini API (free tier), briefly, before switching.
- Python 3.13, standard library only. The eval harness is the one provided.

## Repo layout

```
run.sh                  one command
memory/loader.py        sources -> units; time filter, edits, deletions, secret masking, injection flags
memory/retrieve.py      BM25 with date tokens and feedback terms
memory/pipeline.py      rewrite -> candidates -> rerank (+ date hop) -> draft/evidence/final answer
memory/llm.py           OpenAI-compatible client: cache-friendly, retries, pacing, token budget
memory/run_memory.py    memory runner (parallel, falls back per question)
memory/actions.py       TextOS planner: context, guards, plan, checks
memory/run_actions.py   action runner (dry run)
memory/textos.py        interactive TextOS
out/                    train output files (committed); caches and usage logs (ignored)
```
