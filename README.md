# Voice Bot — a speech-to-speech banking assistant

Speak to it, and it speaks back. A voice agent that handles balances,
transfers, bills, spending and cards — where a language model does the
understanding and the talking, and a database holds every fact it is allowed
to state.

```
your voice -> VAD -> Whisper -> LLM <-> tools -> your browser speaks
                                         |
                                     PostgreSQL
```

The model is the brain. It is deliberately **not** the memory and **not** the
hands: it cannot know a balance without asking the database, and it cannot
move money without going through code that can refuse.

<!-- Save a screenshot of the running UI to docs/images/ui.png, then delete
     these two comment lines to show it.
![The interface, with the inspector panel open](docs/images/ui.png)
-->

---

## How it is stopped from inventing your balance

A language model asked for your balance will eventually produce a confident,
plausible, invented number, because producing plausible text is what it does.
Three mechanisms sit between that tendency and the customer, and none of them
is a prompt instruction.

**It has to ask.** The model has no account data. Nine tools are its only
route to one, and each returns JSON from the database. Asking for a balance
means calling `get_balance` and reading what comes back.

**It cannot act alone.** `transfer_money`, `pay_bill`, `freeze_card` and
`order_card` return a summary and a token on the first call and change
nothing. Only a second call carrying that token executes, and the Yes and
Cancel buttons act on the pending action directly rather than sending the word
"yes" back through the model. A model talked past its instructions still
cannot move money; so a misheard "yes" cannot either.

**Every spoken figure is checked.** Before the reply reaches your ears, each
money figure in it is compared by value against the integer cents the bank
reported handing over. A mismatch discards the sentence and reads back the
stored value instead. The check is scoped to money, so "five working days" and
"the card ending 4417" pass untouched.

What the guard trusts comes from the executor, never from the text of a tool
result. An earlier version parsed results for figures and could be defeated by
laundering: calling `get_balance(account="$31,415.00")` makes the bank's error
echo that string back, and a guard reading result text then treats the figure
as sourced. A figure spelled out in words, or quoted in another currency, is a
violation outright — neither can be checked.

The cost of that last rule is real and worth stating: **the bot cannot do
arithmetic out loud.** "That's about £30 less than last month" is a derived
figure no tool returned, so it is suppressed. A guard that admits
correct-looking totals admits incorrect ones by the same rule.

---

## What it is like to use

```
YOU   hi there, how are you?
BOT   I'm doing well, thanks for asking! How can I help you with your accounts today?
      tools: none

YOU   what's my savings balance?
BOT   Your savings balance is $19,020.00.
      tools: get_balance{account: savings}          figures: verified

YOU   how much did I spend this week?
BOT   You spent $41.99 this week.
      tools: get_spending{period: this week}        figures: verified

YOU   send fifty dollars to sarah from checking
BOT   I'll transfer $50.00 to Sarah from your Checking account. Does that sound right?
      tools: transfer_money{amount_cents: 5000, ...}   -> waiting
      [Yes, confirm]  [Cancel]

                                                   balance still $2,483.50
→ Yes
BOT   Done! $50.00 sent to Sarah. Your Checking balance is now $2,433.50.
```

Transcribed from a real session against the live model.

---

## Measured latency

On the machine this runs on — an Intel i5-6200U, two cores, no CUDA — with the
model served remotely on another host.

| Stage | Time | Where it runs |
|---|---|---|
| Whisper `base.en` (3 s of speech) | **2.3 s** | this CPU |
| LLM, conversational reply | **3.4 – 5.5 s** | remote |
| LLM, reply after a tool call | **4 – 10 s** | remote, two round trips |
| Speech out | **~0 s** | the browser |
| **A whole spoken turn** | **≈ 6 – 12 s** | |

Whisper's model loads once at startup and takes 49 s to do it.

**This is slower than the version it replaced**, which answered in about 0.6 s
on a GPU. The LLM buys accuracy and natural conversation; it does not buy
speed, and it is remote, so faster local hardware does not help it. Speech
synthesis moved into the browser precisely because it was the one stage that
could be removed outright.

---

## Quick start

### Run the tests — no GPU, no model, no network

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,serve]" -c constraints.txt
.venv/Scripts/python -m pytest
```

`constraints.txt` pins the versions this was built and tested against, so an
install months from now resolves to the same set rather than to whatever is
newest.

275 tests in about 15 seconds, plus 16 for the barge-in detector under
`node --test "tests/js/*.test.js"`. Every external dependency sits behind an
abstract base class with a fake — including the language model, whose
`FakeLLMClient` replays scripted replies and tool calls. The whole agent loop,
the confirmation gate and the numeric guard are exercised with no endpoint.

### Run it for real

```bash
VOICEBOT_CONFIG=configs/serve.yaml \
  .venv/Scripts/python -m uvicorn voicebot.api.app:app --port 8000
```

Open <http://localhost:8000>. The page is served by the same process as the
API, which is what lets the browser grant it a microphone.

`configs/serve.yaml` points at the model endpoint, selects browser speech and
keeps the bank in PostgreSQL, which needs setting up once — see
[The database](#the-database). Without `VOICEBOT_CONFIG` the service starts on
fakes with an in-memory bank — useful for working on the front end.

Tests that need the real endpoint are marked `live` and deselected by default:

```bash
.venv/Scripts/python -m pytest -m live
```

---

## The API

| Endpoint | Purpose |
|---|---|
| `POST /v1/converse` | Audio in, a whole turn out. |
| `POST /v1/chat` | The same turn, typed instead of spoken. |
| `POST /v1/confirm` | Agree to the pending action. |
| `POST /v1/cancel` | Decline it. |
| `POST /v1/reset` | Restore the demo balances; keep the audit log. |
| `POST /v1/transcribe` | ASR alone. |
| `POST /v1/speak` | TTS alone. |
| `DELETE /v1/session/{id}` | Forget a conversation. |
| `GET /health` | Liveness. Answers before the models load. |
| `GET /ready` | Readiness, naming the engines actually loaded. |
| `GET /metrics` | Prometheus text: turns, tool calls by name, guard hits. |

A turn returns the reply, every tool call with its arguments and result,
whether the guard fired, and per-stage timings — which is exactly what the
inspector panel draws.

Interactive docs at `/docs`.

**`voicebot_guard_hits_total` is the metric worth watching.** A rising one
means the model is reaching for figures it was not given.

---

## Layout

| Path | Contents |
|---|---|
| `src/voicebot/llm/` | The client, its offline fake, the fifteen tool schemas, the system prompt |
| `src/voicebot/agent/` | The LLM ⇄ tool loop, the executor that owns the confirmation gate, the numeric guard |
| `src/voicebot/bank/` | The bank in PostgreSQL and in SQLite, the ledger, the audit log, the seed data |
| `src/voicebot/bank/migrations/` | The PostgreSQL schema, as numbered SQL files applied in order |
| `scripts/` | `migrate_sqlite_to_postgres.py`: copies the SQLite bank across and proves the copy |
| `src/voicebot/asr/` | Whisper engine |
| `src/voicebot/tts/` | VITS, and the null engine that hands synthesis to the browser |
| `src/voicebot/pipeline/` | Orchestrator, audio plumbing, per-stage timing |
| `src/voicebot/api/` | FastAPI app, routes, sessions, published payload shapes |
| `src/voicebot/api/static/` | The front end: `index.html`, one module per screen, one stylesheet each |
| `configs/` | YAML run configuration, validated by pydantic at startup |
| `tests/` | Unit and integration tests |
| `docs/design/` | The design documents and implementation plans |
| `docs/migration/` | The move from SQLite to PostgreSQL: what changed, how it was verified, how to undo it |
| `docs/experiments.md` | v1's method, results and the measurement bugs that changed them |
| `docs/model_card.md` | v1's trained model: what it was, and what it was not for |

Every engine sits behind an abstract base class with a registry, so swapping
Whisper for another ASR, or the endpoint for a hosted provider, is a config
change.

The front end is ES modules loaded straight by the browser — no bundler, no
build step, nothing to install. `app.js` is the entry point and holds only
the wiring; the work is in `core` (state and the API client), `audio`,
`conversation`, `cards`, `screens` and `mic`, and each has a stylesheet of
the same name. `vad.js` stays a classic script because it publishes one
constructor on `window` and is tested under Node.

The dependency runs one way, which is why `conversation.js` does not import
any screen: a screen asks to hear about a turn with `onTurn`, and `app.js`
is where they say so. Adding a screen is a line in that list.

---

## The database

Eight tables in PostgreSQL, reached through psycopg and a connection pool —
no ORM. Money is integer cents throughout.

`transactions` is a ledger: money out is stored negative, every balance change
writes a row in the same transaction as the update, so an interruption leaves
neither. Spending periods are a closed vocabulary (`today`, `this week`,
`last month`, …) mapped onto date ranges, and an unrecognised one returns an
error listing the valid ones. v1 stored periods as free text, which meant "how
much did I spend in October?" matched nothing, summed to zero, and the bot
answered "nothing" to a question it had not understood.

### Setting it up

It was built and tested against PostgreSQL 18. The connection comes from
five variables, read from the environment or from `.env`:

| Variable | If unset |
|---|---|
| `VOICEBOT_DB_HOST` | `localhost` |
| `VOICEBOT_DB_PORT` | `5432` |
| `VOICEBOT_DB_NAME` | `voicebot` |
| `VOICEBOT_DB_USER` | your login name, as for `psql` |
| `VOICEBOT_DB_PASSWORD` | none |

```bash
cp .env.example .env    # then fill in VOICEBOT_DB_PASSWORD
```

`.env` is gitignored, and no committed file holds a password. Anything
already exported wins over it.

The bank makes its own tables. On startup it applies any numbered file in
`src/voicebot/bank/migrations/` it has not applied before — all in one
transaction, each recorded in `schema_migrations` — and seeds the demo data
if the tables are empty. It does not make the database. The migration script
below does; without it, run this once in `psql` or pgAdmin's Query Tool:

```sql
CREATE DATABASE voicebot ENCODING 'UTF8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0;
```

`C` sorts text by its bytes, which is what SQLite does, so names come back in
the same order from either bank on any operating system.

### Moving the SQLite data across

```bash
.venv/Scripts/python scripts/migrate_sqlite_to_postgres.py --dry-run   # all of it, then roll back
.venv/Scripts/python scripts/migrate_sqlite_to_postgres.py --report docs/migration/report.md
```

It copies `data/bank.db` into the database `VOICEBOT_DB_NAME` names, creating
it if need be, and proves the copy:

- **The file is only read.** It is opened read-only, and its SHA-256 is
  compared before and after.
- **All or nothing.** The copy is one transaction, and the checks run inside
  it — row counts, checksums, and every field of every row compared with its
  source, type included. It commits only if all of them pass.
- **Safe to run again.** A database that already matches is reported up to
  date and left alone. One holding anything else is refused unless
  `--replace` is given.

`--source` copies another file, such as a backup. It exits 0 when the data is
there, 1 when it refused or a check failed, and 2 when it could not connect.
The report from the migration of 29 October 2026 is in `docs/migration/`.

### Going back to SQLite

Set `engine: sqlite` in `configs/serve.yaml` and restart. Its `path` is
`data/bank.db`, the file the data was copied from, which the migration only
ever read; a backup taken before it began is kept in `data/backups/`. The
SQLite bank is still in the code, and `python -m voicebot.bank` follows the
configuration, so nothing else needs changing.

Two things do not come back with it. Whatever was written in PostgreSQL after
the switch — payments, deposits, PIN changes, the audit log since — stays
there: the file is the bank as it stood on the day it was copied. And the
PostgreSQL bank is the one that holds up when requests arrive together. The
SQLite bank shares one connection between threads, and under simultaneous
requests it can charge a bill twice, let a burst of wrong PINs through the
three-try lockout, and refuse a transfer that fits —
`tests/unit/test_bank_concurrency.py` shows each.

### Running the tests against PostgreSQL

```bash
VOICEBOT_TEST_DATABASE=postgres .venv/Scripts/python -m pytest
```

By default the suite runs on in-memory SQLite and needs no server. With this
set, the bank every test is given is a PostgreSQL one: in a database of its
own (`voicebot_test`, or `VOICEBOT_TEST_DB_NAME`), with a fresh schema for
each test, dropped afterwards. It refuses to run against the database the app
uses. It takes minutes rather than seconds, and it is the only run that
includes the concurrency tests, which send the same payment or PIN from
several threads at once.

### Looking after the demo bank

```bash
python -m voicebot.bank --show                          # balances and turn count
python -m voicebot.bank --deposit 500 --account checking # pay money in
python -m voicebot.bank --reset                          # opening balances, audit log kept
```

They act on the bank the server opens — the one `VOICEBOT_CONFIG` names, or
`configs/serve.yaml` — and say which it is. `--config` picks another
configuration, and `--path` names a SQLite file outright.

Editing the opening balances in `bank/seed.py` does nothing to a database
that already exists: the seed runs once, when the tables are empty. Use these.

Paying money in is administrative and has no tool, so the assistant cannot do
it. An assistant that can create money is not an assistant.

`turns` is the audit log — what was said, which tools ran, whether the guard
fired and what it suppressed. For a banking demo it is worth more than the
account tables.

One bank per process, one pending action per session: this is one customer's
assistant, but two browser tabs mid-confirmation must not be able to confirm
each other's transfer.

Money leaving an account takes two steps, not one. The yes is agreement; the
PIN for the account it leaves is authority. Paying a bill, moving between the
customer's own accounts and sending to a payee all go through it, and the
agreement survives a wrong PIN or a cancelled keypad — they try again, they
do not agree again. Freezing a card and filing a dispute do not: both are
reversible and neither moves a penny.

The check lives in the executor, so it holds whether the customer pressed the
button or the model carried its own confirmation token back. The PIN itself
goes from the keypad to the bank and nowhere else: not through the model, not
into the transcript, and not into `turns`.

---

## Quality gates

```bash
.venv/Scripts/python -m ruff check src tests scripts     # lint
.venv/Scripts/python -m black --check src tests scripts  # formatting
.venv/Scripts/python -m mypy src scripts                 # types
.venv/Scripts/python -m pytest                           # 472 tests
.venv/Scripts/python -m pytest --cov=voicebot            # with coverage
node --test tests/js/vad.test.js                         # 16 barge-in tests
```

All of these run on every push: see `.github/workflows/ci.yml`. So does the
suite against PostgreSQL, on a server the job starts for itself. The `live`
tests are not among them, because the model endpoint is on a private network
no runner can reach — run those by hand with `pytest -m live`.

Coverage is gated at 90%, measured across the SQLite and PostgreSQL runs
together — each bank's code runs only in its own — and currently sits at
95%. To measure it the same way locally, run the PostgreSQL pass after the
SQLite one with `--cov=voicebot --cov-append`.

To have the same checks run before a commit rather than after a push:

```bash
pip install pre-commit && pre-commit install
```

---

## Known limitations

Stated plainly, because they change how the system should be judged.

- **No arithmetic.** The numeric guard rejects any figure a tool did not
  return, including correct sums. Deliberate, and it means the bot cannot
  compare two balances out loud.
- **One currency.** USD. Asked in pounds, it says so rather than converting.
- **One endpoint, no fallback.** The trained model that used to run locally is
  gone. If the model server is unreachable, the bot says so and can do nothing
  else. It also has no authentication.
- **Slower than v1.** Six to twelve seconds a turn against roughly one. See
  the latency table.
- **Sessions are in-process**, with a TTL and a cap, and a conversation does
  not survive a restart — the money does. `SessionStore` is an interface for
  exactly that reason.
- **The model's judgement is not tested.** The suite proves that no model
  behaviour can move money without a confirmation or state an unsourced
  figure. It cannot prove the model will choose the right tool.
- **English only**, and accuracy through a real microphone across accents is
  unmeasured.
- **Barge-in is an energy gate, not a speech classifier.** It measures the
  microphone against a moving estimate of the room, which is enough to tell a
  person from the bot's own voice through the speakers. A sustained noise at
  speaking volume — a television, someone else talking nearby — will still
  interrupt it. A real classifier would not, at the cost of a model download.

---

## What's next

Streaming the model's reply and beginning to speak at the first sentence
boundary, which would hide most of the LLM's time behind the speech that
follows it. A spoken benchmark measuring task success against word error
rate. An authenticated endpoint.

---

## History

v1 was the same product built without a language model: a RoBERTa intent and
slot model trained here, a finite-state dialogue manager, and a file of reply
templates. It reached 0.8963 test accuracy and 0.9241 macro-F1, and it could
not invent a balance because every sentence it could say was written by hand.

That work is kept in [`docs/model_card.md`](docs/model_card.md) and
[`docs/experiments.md`](docs/experiments.md) — including the two measurement
bugs that changed its conclusions, one of which turned a slot F1 of 1.0000
into an honest 0.6866.

v2 trades that rigidity for conversation, and rebuilds the safety property
with tools, a confirmation gate and a numeric guard rather than with
templates.

---

## Licences

This project is MIT licensed — see `LICENSE`.

The models it can load are not all permissive. Check these before any
commercial use — they are not all permissive:

| Component | Licence |
|---|---|
| `openai/whisper-*` | MIT |
| `facebook/mms-tts-eng` *(optional, non-default)* | **CC BY-NC 4.0 — non-commercial** |

VITS is no longer the default voice; the browser's own synthesiser is. If you
switch it back on with `tts.engine: vits`, that licence applies again.
