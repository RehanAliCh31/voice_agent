# Voice Bot — a speech-to-speech banking assistant

Speak to it in English or Urdu and it speaks back in the same language. A
voice agent for balances, transfers, bills, spending and cards that also
answers general questions — a language model does the understanding and the
talking, and a database holds every fact it is allowed to state.

```
your voice -> VAD -> qwen3-asr + whisper-urdu -> LLM <-> tools -> qwen3-tts / OmniVoice -> your browser plays it
                       (both, in parallel)             |
                                                   PostgreSQL
```

The model is the brain. It is deliberately **not** the memory and **not** the
hands: it cannot know a balance without asking the database, and it cannot
move money without going through code that can refuse.

This file is the project's only document. It covers how to run it, how it
works, how to configure and tune it, how it fails, and what is still open.

## Contents

1. [Quick start](#1-quick-start)
2. [How it is stopped from inventing your balance](#2-how-it-is-stopped-from-inventing-your-balance)
3. [What it is like to use](#3-what-it-is-like-to-use)
4. [The office services](#4-the-office-services)
5. [Language, voice and gender](#5-language-voice-and-gender)
6. [Hands-free live voice](#6-hands-free-live-voice)
7. [Speaking a reply, and speed](#7-speaking-a-reply-and-speed)
8. [When something fails](#8-when-something-fails)
9. [Configuration](#9-configuration)
10. [The database](#10-the-database)
11. [The API and the live protocol](#11-the-api-and-the-live-protocol)
12. [Layout](#12-layout)
13. [Tests and quality gates](#13-tests-and-quality-gates)
14. [Known limitations and open decisions](#14-known-limitations-and-open-decisions)
15. [History](#15-history)
16. [Licences](#16-licences)

---

## 1. Quick start

### Run the tests — no GPU, no model, no network

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,serve,assistant]" -c constraints.txt
.venv/Scripts/python -m pytest
```

`constraints.txt` pins the versions this was built and tested against, so an
install months from now resolves to the same set. Every external dependency
sits behind an abstract base class with a fake — including the language
model, whose `FakeLLMClient` replays scripted replies and tool calls — so the
whole agent loop, the confirmation gate and the guards are exercised with no
endpoint.

### Run it for real

```bash
.venv/Scripts/python -m voicebot.assistant setup    # the voice activity model, 2 MB, hash-checked
./run.sh                                             # checks the model, the database and the speech services, then starts
```

or directly:

```bash
VOICEBOT_CONFIG=configs/serve.yaml \
  .venv/Scripts/python -m uvicorn voicebot.api.app:app --port 8000
```

Open <http://localhost:8000>. The page is served by the same process as the
API, which is what lets the browser grant it a microphone. Click the
microphone once to start a hands-free conversation, again to end it, or type
in the box below it.

`configs/serve.yaml` points at the model endpoint, selects the office
recognisers and voices listed in `configs/assistant.yaml`, and keeps the bank
in PostgreSQL (set up once — see [The database](#10-the-database)).

`./run.sh --fake` (or no `VOICEBOT_CONFIG`) starts on fakes with an in-memory
bank: no model, no endpoint, no downloads — for working on the front end.
`./run.sh --test` runs the test suite instead.

### Docker

```bash
docker build -t voicebot .
docker run --rm -p 8000:8000 -e VOICEBOT_CONFIG=configs/serve.yaml \
  --env-file .env -e VOICEBOT_DB_HOST=host.docker.internal \
  -v voicebot-data:/app/data voicebot
```

The image runs as a non-root user and writes only to `/app/data`, a volume.
It has not been built on this machine (Docker is not installed here).

---

## 2. How it is stopped from inventing your balance

A language model asked for your balance will eventually produce a confident,
plausible, invented number, because producing plausible text is what it does.
Three mechanisms sit between that tendency and the customer, and none of them
is a prompt instruction.

**It has to ask.** The model has no account data. Its tools are its only
route to any, and each returns JSON from the database. Asking for a balance
means calling `get_balance` and reading what comes back.

There are 15 tools: `list_accounts`, `list_bills`, `list_cards`,
`get_balance`, `get_spending`, `list_transactions`, `hand_off`,
`transfer_money`, `dispute_transaction`, `move_money`, `pay_bill`,
`freeze_card`, `ask_for_pin`, `set_card_control`, `order_card`.

**It cannot act alone.** `transfer_money`, `move_money`, `pay_bill`,
`freeze_card`, `order_card`, `set_card_control` and `dispute_transaction`
return a summary and a token on the first call and change nothing. Only a
second call carrying that token executes, and the Yes and Cancel buttons act
on the pending action directly rather than sending the word "yes" back
through the model.

Money leaving an account takes two steps. The yes is agreement; the **PIN**
for the account it leaves is authority. Paying a bill, moving between the
customer's own accounts and sending to a payee all go through it, and the
agreement survives a wrong PIN or a cancelled keypad — they try again, they
do not agree again. Freezing a card and filing a dispute need no PIN: both
are reversible and neither moves a penny. The check lives in the executor, so
it holds whether the customer pressed the button or the model carried its own
token back. The PIN goes from the keypad to the bank and nowhere else: not
through the model, not into the transcript, not into the audit log.

**Every spoken figure is checked.** Before a reply reaches your ears, each
money figure in it is compared by value against the integer cents the bank
reported handing over. A mismatch discards the sentence and reads back the
stored value instead. The check is scoped to money, so "five working days"
and "the card ending 4417" pass untouched. What the guard trusts comes from
the executor, never from the text of a tool result — an earlier version could
be defeated by laundering a figure through a tool's error message. A figure
spelled out in words, or quoted in another currency, is a violation outright.

The cost of that last rule is real: **the bot cannot do arithmetic out loud.**
"That's about £30 less than last month" is a derived figure no tool returned,
so it is suppressed. A guard that admits correct-looking totals admits
incorrect ones by the same rule.

Other guards: a promise to look something up that is never made sends the
model back to make it (`not_looked_up` if it still does not); references and
phone numbers no tool returned are refused; a list too long to read aloud is
summed up by the executor (`too_many_to_read`) rather than read in full.

Paying money in is administrative and has no tool, so the assistant cannot
do it — an assistant that can create money is not an assistant. One bank per
process and one pending action per session, so two browser tabs
mid-confirmation cannot confirm each other's transfer. Asking for a person
opens a case with a reference (`hand_off`, six reasons); a rule fires in
Python whether or not the model agrees.

---

## 3. What it is like to use

```
YOU   hi there, how are you?
BOT   I'm doing well, thanks for asking! How can I help you with your accounts today?
      tools: none

YOU   what's my savings balance?
BOT   Your savings balance is $19,020.00.
      tools: get_balance{account: savings}          figures: verified

YOU   send fifty dollars to sarah from checking
BOT   I'll transfer $50.00 to Sarah from your Checking account. Does that sound right?
      tools: transfer_money{amount_cents: 5000, ...}   -> waiting
      [Yes, confirm]  [Cancel]  then the PIN keypad

→ Yes, PIN
BOT   Done! $50.00 sent to Sarah. Your Checking balance is now $2,433.50.
```

The same conversation works in Urdu. Each turn, spoken or typed, is answered
in the language it was asked in and read by that language's voice; figures
stay in digits on screen, and the guard checks them exactly as in English:

```
YOU   kia ap muje sun skte hain?
BOT   جی ہاں، میں آپ کو سن سکتی ہوں۔ میں آپ کی کس طرح مدد کر سکتی ہوں؟

YOU   میرے سیونگز اکاؤنٹ کا بیلنس کیا ہے؟
BOT   آپ کے سیونگز اکاؤنٹ کا بیلنس $19,020.00 ہے۔
      tools: list_accounts                          figures: verified
```

Urdu typed in Latin letters, as it often is on a phone, is recognised by its
words and answered in Urdu script, which is what the Urdu voice reads. It
answers questions about the world from what it knows, in one or two short
sentences, and explains the bank from its data.

The page has four screens — Assistant, Transactions, Cards and Monthly — in a
dark theme lit by teal, violet and magenta. The Assistant screen has three
columns: the balance and bills on the left, the voice in the middle (one
large button whose light shows listening, hearing you, thinking or
speaking), and the conversation on the right. Urdu is set in Nastaliq. A turn returns the
reply, every tool call with its arguments and result, whether a guard fired,
and per-stage timings.

---

## 4. The office services

The recognisers, the voices and the model are served on the office network.
Their addresses are kept in `.env`, which is never committed: the configs name
them (`VOICEBOT_LLM_URL`, `VOICEBOT_QWEN_ASR_URL`, `VOICEBOT_WHISPER_URDU_URL`,
`VOICEBOT_QWEN_TTS_URL`, `VOICEBOT_OMNIVOICE_URL`) and `.env.example` lists
them. The voices and who the bot is are in `configs/assistant.yaml`; none is in
code.

```bash
.venv/Scripts/python -m voicebot.assistant setup    # the voice activity model
.venv/Scripts/python -m voicebot.assistant check    # the config, the model, every endpoint and voice
```

Measured 29 October 2026, with real requests:

| Service | Model | Latency | Notes |
|---|---|---|---|
| English ASR | qwen3-asr | 0.32 s for 4 s of speech; 0.77 s for 11 s | Any rate, mono or stereo. No confidence or language output. With no language, Urdu comes back as Hindi in Devanagari. |
| Urdu ASR | whisper-urdu | 0.67 s for 5 s; 1.3 s for 15 s | **16 kHz only.** Names the language it heard and gives `avg_logprob`; also transcribes English, in English. |
| English TTS | qwen3-tts (Qwen3-TTS-12Hz-0.6B-Base) | 0.9–1.2 s for 4 words; ~0.25 s a word | One built-in female voice; ignores voice names. Now and then says words that were not in its text. |
| Urdu TTS | k2-fsa/OmniVoice | ~1.8 s per request plus ~0.12 s a word | 17 named voices (`GET /v1/audio/voices`), `seed`, `speed`. |
| LLM | qwen3.6-35b-a3b-fp8 | first token ~0.3 s; a short reply 0.4–0.8 s (en), 0.5–2 s (ur) | Reasoning is asked off explicitly; a stray `</think>` is stripped. |

What the services do that shapes the code:

- **Neither voice streams.** Both accept `stream: true`, but the first byte
  arrives only when the whole clip is done. A reply can only be spoken a piece
  at a time.
- **Both voices handle one request at a time**: two at once take twice as long
  (and qwen3-tts gets slower than serial), so pieces are made one after
  another.
- **Both voices misread digits**, so every number is written out as words
  before it reaches them.
- **On silence and noise the recognisers make things up** — qwen3-asr "I." or
  "Okay.", whisper-urdu "ملتے ہیں". Whisper's own confidence separates these
  from speech (real speech −0.08 to −0.22, silence −0.44, noise −1.03), and
  short low-confidence results are dropped as noise.

**Troubleshooting.** `check` says an endpoint is DOWN: connect to the office
network or the VPN. Urdu comes out as gibberish: Urdu copied from a PDF or a
terminal can paste with its letters reversed — type it fresh; the bot never
reverses, reshapes or "fixes" Urdu. The model file is missing or damaged:
`python -m voicebot.assistant setup`.

---

## 5. Language, voice and gender

### Where things live

| What | Where |
|---|---|
| Who the bot is: gender, name, a voice per gender and language, backups | `configs/assistant.yaml`, `agent:` |
| Every sentence the bot says without the model, English and Urdu, female and male | `src/voicebot/messages.yaml` (checked when it loads) |
| How the bot speaks of itself in Urdu, shown to the model | `messages.yaml`, `prompt_self_forms` (never said aloud) |
| The system prompt and the language notes | `src/voicebot/llm/prompts.py` |
| Which language was spoken | `src/voicebot/assistant/language.py` |
| Whether a reply is in one language, and in the bot's gender | `src/voicebot/reply_check.py` |
| How numbers, dates and codes are said | `src/voicebot/assistant/text_prep.py` |
| Recordings of the bot's own sentences | `data/voice_cache/` (not committed) |

The model is told who it is and which language to use, but it never chooses
either. `core.js` keeps the English text of three sentences (lost connection,
timed out, something went wrong), shown only if the page cannot fetch
`/v1/messages` at all.

### The profile

```yaml
agent:
  gender: female                 # female or male: the one switch
  names:
    female: {en: Layla, ur: لیلیٰ}
    male: {en: Omar, ur: عمر}
  voices:                        # one per gender and language
    female:
      en: {url: ..., model: qwen3-tts, gender: female, first_chunk_words: 5, ...}
      ur: {url: ..., model: k2-fsa/OmniVoice, voice: assistant-pk, gender: female, ...}
    male:
      en: {url: ..., model: k2-fsa/OmniVoice, voice: urdu_male5, gender: male, ...}
      ur: {url: ..., model: k2-fsa/OmniVoice, voice: urdu_male5, gender: male, ...}
  backup_voices:                 # optional, same gender, tried when the main voice fails
    female:
      en: {..., voice: assistant-en, gender: female}
      ur: {..., voice: assistant-ur, gender: female}
```

**To change the bot's gender**, change `agent.gender`. The voice in both
languages, the name, every fixed sentence, how the model is told to speak of
itself and which kept recordings may be played all follow. Recordings are
stored under the voice and gender that made them, so one for the other gender
is never played.

Each voice entry takes: `url`, `model`; `voice` (a voice the server has by
name) or `reference` + `reference_text` (a WAV to clone from, and exactly
what it says, sent with each request); `extra` (anything else the server
takes, e.g. `speed`, `seed`); `gender` (required); `label` (for logs); and the
piece settings in [section 7](#7-speaking-a-reply-and-speed).

**The service refuses to start**, saying why, when: a voice is of the other
gender from the key it is under (a female agent can never get a male voice or
backup); a voice states no gender; there are no voices or no name for the
agent's gender; a `reference` file is missing or has no `reference_text`;
`tts.en`/`tts.ur` are written as well as `agent.voices`; a server that lists
its voices does not list the configured one; or `messages.yaml` has a message
without both languages, a `{reference}` missing from one form, or a gendered
form in the wrong gender.

### The voices today

| Gender | Language | Voice | Source | Licence |
|---|---|---|---|---|
| female | English | qwen3-tts, its one built-in voice | shipped with the model | Qwen3-TTS licence (confirm for commercial use) |
| female | Urdu | OmniVoice `assistant-pk` | reference clip from Google FLEURS `ur_pk`, a real speaker | model CC-BY-NC; clip CC-BY-4.0, attribution required |
| male | English | OmniVoice `urdu_male5`, speaking English | synthetic | model CC-BY-NC |
| male | Urdu | OmniVoice `urdu_male5` | synthetic | model CC-BY-NC |
| female backup | English / Urdu | OmniVoice `assistant-en` / `assistant-ur` | synthetic | model CC-BY-NC |

qwen3-tts cannot do a male or accented voice as deployed, which is why both
male voices are on OmniVoice. No provider here offers an Arabic accent by
name; the documented route is cross-lingual cloning from a synthetic Arabic
reference (`reference:` above). Candidates were made and listened to; they
are in the git history (commit 991722a) if wanted again.

### Which language, and holding to it

1. **Detected every turn.** Spoken: both recognisers hear every utterance at
   once, and `choose()` weighs Urdu's grammar words in whisper's transcript,
   English spelled out in Urdu letters, Hindi script or Roman Urdu from
   qwen3-asr, and whisper's own verdict, which settles a tie. A grammar word
   whose English twin qwen3-asr heard (ہی for "hear", میں for "me") counts as
   English. Typed: by script, and Roman Urdu by its common words.
2. **Code-switching goes to the language the grammar is in.** "میرا checking
   account balance کیا ہے" is Urdu; "Please check my balance, shukriya" is
   English.
3. **A mid-call switch is followed**, because every turn is decided afresh.
4. **When unsure, the last confirmed language stands** — a typed "yes" or
   "$50", or a turn neither recogniser heard. The bot never asks "English or
   Urdu?": that question would itself mix the two.
5. **The language is a structured variable** (`Conversation.language`), set
   from the customer's words, never the reply's. The note beside the
   customer's latest message carries the strict rule ("Reply ONLY in
   English…", or the Urdu equivalent with "any English word goes in Urdu
   letters"), and it chooses the voice.
6. **Every reply is checked before it is spoken.** An English reply may have
   no Urdu or Hindi letters; an Urdu reply no English words, except names the
   bank holds in English and letters said as letters (PIN, ATM, OTP, SMS,
   PKR, USD, ID). A reply that fails is struck from the history and asked for
   again once with a stricter note; if it fails again, the bot says
   `reply_failed` in the right language and gender.

Measured on stored transcripts of 90 clips: 89 of 90 decided right.

### Gender: one voice, one way of speaking

The profile is set once, at startup; the default is female. Every fixed
sentence comes from `messages.yaml` in the profile's gender. The model is
told its name, gender and Urdu first-person forms, and what it writes is put
right before it is spoken: `reply_check.in_own_gender` turns the other
gender's first-person forms into the bot's own (سکتا ہوں/سکتی ہوں، رہا ہوں/رہی
ہوں، کروں گا/گی …), touching only forms that can be nothing but the bot
speaking of itself — "ٹرانسفر نہیں ہو سکا" stays as it is. Fallbacks never
switch gender: a backup voice must be of the agent's gender, a kept recording
is played only as the voice that made it, and the page never speaks in the
browser's voice.

### How it is said

`prepare_for_speech` turns a reply into what a voice can read; the page still
shows it as written.

| Written | English | Urdu |
|---|---|---|
| `$2,483.50` | two thousand four hundred eighty-three dollars and fifty cents | دو ہزار چار سو تراسی ڈالر اور پچاس سینٹ |
| `Rs 12,50,000` | one million two hundred fifty thousand rupees | بارہ لاکھ پچاس ہزار روپے |
| card ending `4417` | four four one seven | چار چار ایک سات |
| `0300-1234567` | zero three zero zero, one two three … | صفر تین صفر صفر، … |
| `12/10/2026` | the twelfth of October, two thousand twenty-six | بارہ اکتوبر دو ہزار چھبیس |
| `10:30`, `15%` | ten thirty, fifteen percent | دس بج کر تیس منٹ، پندرہ فیصد |

Urdu is kept in logical order, as typed, throughout; nothing reverses or
reshapes it. Card digits and references are read one at a time. Amounts,
payees, accounts and card digits are read back before anything moves.

---

## 6. Hands-free live voice

One click on the microphone starts a conversation: the microphone stays open,
the bot answers when you finish speaking, you can talk over it to stop it,
and it asks after a long silence. Another click ends it. Everything else is
the same bot — the tools and guards, both languages, the error handling, the
database — and the Yes and Cancel buttons, the PIN keypad and the screens
work during a live session. A browser without AudioWorklet and WebSockets
(no current one) says so, and typing still works.

### How it fits together

```
 browser                                            service (FastAPI)
 getUserMedia (echo cancellation, noise              /v1/live  (voicebot/live/routes.py)
   suppression, no automatic gain)                     │  binary: [playback level][PCM 16 kHz]
   │                                                    │          every 20 ms
 AudioWorklet live-capture.js                           │  JSON: start, stop, interrupted,
   input 0: microphone → 16 kHz, 20 ms ─── WebSocket ───┤        playback_done
   input 1: the bot's own playback → its level          ▼
                                                    LiveSession  (voicebot/live/session.py)
 live.js  LiveClient                                 audio thread: Silero VAD per 32 ms frame
   plays reply pieces back to back (Web Audio)         → near-voice gate (louder than the room?)
   stop_audio → silent at once, reports where          → echo gate, while the page plays
   reconnects with the same session                    → Endpointer, or the barge-in Endpointer
                                                       → the state machine
 mic.js  shows what the session is doing               → recognition started at 224 ms of silence
                                                     turn thread (one at a time):
                                                       interruption, or carry on? → the agent
                                                       → reply + piece 0 → pieces 1..n
```

### The state machine

The service holds the one true state; every change goes through
`LiveSession._to`, which refuses and counts anything the table does not allow.

```
    idle ──start──▶ listening ──speech──▶ user_speaking ──silence──▶ thinking
                       ▲   │                 ▲      ▲                  │
                       │   └─ long silence:  │      └── speech resumes ┘  same turn, asked
                       │      "still there?" │         before the reply   again with all of it
                       │                     │         is heard          ▼
                       └──── reply played ── agent_speaking ◀── reply ───┘
                                             └── speech (barge-in) ──▶ user_speaking
    any state ──stop, hang-up, limit, goodbye──▶ ended
```

Three kinds of thread: the connection's (only hands audio in and sends what
is queued), the session's audio thread (the VAD, the endpointers and every
state change), and one turn thread at a time. A turn may speak only if it is
still the current turn, still `thinking` and not cancelled — checked under
the lock a barge-in and a stop also take, so a replaced turn can never speak
over the one that replaced it. The page silences any reply still sounding
when the service says the caller is being heard, including a reply to a
button.

### Finding speech, and its end

Silero VAD (onnxruntime) scores each 32 ms frame, and the `Endpointer` turns
the scores into utterances.

- **Pausing mid-thought.** 400 ms of silence ends the turn and starts the
  answer, but speech that resumes before any of the reply is heard is the
  same turn: the first answer is cancelled and the turn asked again with
  everything said. A thinking pause costs nothing unless it is longer than the
  whole time to the first audio.
- **Recognising early.** Once the silence reaches 224 ms (what a turn keeps of
  it), the audio so far is exactly what the turn will be if nothing more is
  said, so recognition starts then; its words are used only if the turn ends
  with exactly that audio.
- **English and Urdu** go through the same VAD and settings.

### Noisy rooms

Silero finds any voice — someone across the room or a television is speech to
it as surely as the caller is. What tells them apart is loudness: the caller
is close to the microphone. So speech counts only if it is
`near_voice_margin_db` louder than the room; speech no louder than the room
is heard as silence, and speech in between (the caller's softer syllables)
neither counts nor breaks a count. The page asks for no automatic gain, which
turned the microphone up while the caller was quiet and brought far voices up
to theirs.

How loud the room is, is learnt on each call and never from the caller: for
the first ~2 s from everything heard (a room that talks all the time is never
anything but speech); then from what is heard between utterances, but not
what is 12 dB louder than the room or a loud sound that is no voice (a cough,
a door). The caller is the loudest voice and is followed by quiet; the room is
never taken to be within 12 dB of the loudest voice, so words said first or
most of the time cannot make it deaf to the caller. An utterance 12 dB below
the loudest was someone else, and is learnt as the room. Nothing is learnt
while the bot plays.

Measured offline (real Silero, recorded speech, a simulated page): once the caller has spoken, other people talking 15–25 dB
below them started no turn in 30 s (8–20 a minute without the gate), and the
caller was heard 6–8 times in 8 (1–4 without). It cannot tell apart a second
person as loud as the caller, and before the caller has said anything a voice
across the room may be taken for them. Each turn's metrics carry `room_db`
and `room_gated_frames`.

### Barge-in

Voice activity alone cannot tell the caller from the bot's own voice coming
back through the speaker. Three things decide:

1. **Loudness against the echo.** The capture worklet takes the bot's own
   playback as a second input, and every microphone chunk carries how loud
   the page was playing. The service learns how the bot comes back on this
   call (twice the median of the mic/playback ratio of frames it held back),
   and while the page plays, speech counts only if it is
   `barge_in_echo_margin_db` louder than that (6 dB more when the browser says
   it is not cancelling echo).
2. **Sure speech.** 0.8 for 300 ms, with hysteresis at 0.5.
3. **Stop at once, then let the words decide.** The page stops at once and
   reports exactly where. When the caller finishes: no words (a door, the
   television), a backchannel ("mm-hmm", "جی"), or the bot's own words near
   where it stopped → not an interruption, and the reply resumes from that
   point. Anything else → the rest of the reply is dropped and the words are
   the next turn, kept from their first frame.

Only what was heard goes on record: the history keeps the reply up to the
word reached, marked cut short, and the model is told in the note on the next
message. Measured offline: false stops fell from 5–14 per reply to 0–3 in 50
replies at −40 to −12 dB of echo; a real interruption stops it in ~0.45 s, 16
of 18 times (the misses are a caller starting in the reply's last half
second). With no echo cancellation at all (speakerphone), nothing makes
speakers usable — use a headset.

### Tuning

| Problem | Change |
|---|---|
| The bot cuts people off | Raise `live.end_silence_ms` (600–800). |
| Slow to answer | Lower `end_silence_ms` (300). Every 100 ms off is 100 ms off every turn. |
| Coughs or noise start turns | Raise `min_speech_ms` (250–300) or `threshold`. |
| Other people or a television start turns | Check `room_db`/`room_gated_frames`; raise `near_voice_margin_db` (8–10); keep the mic close. |
| The caller is not heard in a noisy room | Lower `near_voice_margin_db` (3–4), or speak closer. |
| The bot stops itself on speakers | Check `echo_cancellation` and `echo_return_db` in the metrics; raise `barge_in_echo_margin_db` (8–10). A headset ends it. |
| Hard to interrupt | Lower `barge_in_echo_margin_db` (3–4) or `barge_in_min_speech_ms` (250). Each brings back some false stops. |
| "Mm-hmm" stops the bot | Add the word to `backchannels`, or raise `backchannel_max_ms`. |

---

## 7. Speaking a reply, and speed

**The model's reply is streamed.** While it is still being written, the first
piece the voice would say is synthesised in the background; it is used only
if the reply that passes every guard starts with exactly that piece, so
nothing is spoken before all of it is checked.

**A reply is spoken a piece at a time**, because nothing is heard until the
voice has made a whole piece.

- **English (qwen3-tts):** the first piece is short (`first_chunk_words: 5`)
  and ends where a phrase does — within three words of the target, before a
  new clause, a verb or a preposition — never on "the", "your", "is". Later
  pieces grow by `chunk_growth` and are made while the first plays.
- **Urdu (OmniVoice):** every request costs ~1.8 s before any sound, so any
  join between pieces was a pause; each reply is said whole
  (`first_chunk_words: 0`, `rest_max_words: 60`), as one utterance with the
  voice's own pauses. Where the first sentence(s) play longer than the next
  part takes to make (`request_s`, `request_word_s`, `spoken_word_s`, as
  measured), they go alone and the reply starts sooner with no gap.
  `speed: 1.15` says the same in 13% less time, as clearly.

The system prompt asks for one or two short spoken sentences, under 20 words,
unless the customer asks for more. A new turn, or the caller leaving, cancels
what was left of the last reply. The bot's own sentences are recorded at
startup in the background and kept on disk (callers go first), so they can
still be said when a voice is down.

**Measured** (2 November 2026, after the turn is recognised; add ~0.5 s for
the end-of-speech wait and recognition): English 1.1–2.5 s to first audio;
Urdu 3.5–6.6 s, almost all of it the Urdu voice. A whole spoken turn with a
tool call adds one model round (~0.6–1.1 s). The voices' own cost is what is
left: getting under ~1.5 s needs a voice server that streams its first audio.

---

## 8. When something fails

The rule everything follows: **a caller always gets a true sentence.** Never
dead air, never a traceback, never a guessed figure, and never "please try
again" about money that has already moved.

### The error model

Every failure has an `ErrorCode` (`src/voicebot/errors.py`): `provider_timeout`,
`provider_unavailable` and `rate_limited` are retryable; `invalid_input`,
`auth_failure`, `database_error`, `tool_error`, `cancelled` and `internal`
are not. Every error carries
`code`, `component` (`llm`, `asr-en`, `tts-ur`, `bank`, a tool …), `call_id`,
`cause`, and for a 429 `retry_after`. What the customer hears is never the
error's message: it is chosen from the code and the situation, from
`messages.yaml`.

### Timeouts, retries and fallbacks

Every provider call goes through `call_with_retry`
(`src/voicebot/utils/resilience.py`): each attempt gets only what the
deadline has left; only retryable errors are retried, with jittered backoff
(0.2 s, 0.4 s … capped at 1 s); a Retry-After longer than the time left means
no retry; and each provider has a **circuit breaker** — 3 retryable failures
in a row open it for 20 s, then one call probes it. **No action with side
effects is ever retried**: the confirmation token is the idempotency key for
money, and a write that fails part-way is reported as *unknown*, never as
"nothing changed".

| Component | Bound | When it fails, the caller hears |
|---|---|---|
| **Model** | 12 s per request; 20 s per turn for every round together; 1 retry | `unavailable` — or `action_done`/`action_cancelled` if Yes or Cancel already acted. A backup model is asked first when configured. Conversation too long: trimmed and asked once more. Cut off by the token limit: up to its last whole sentence. Empty: `nothing_said`. |
| **Recognisers** | 6 s each, 1 retry, a breaker each | one down: the other decides alone. Both down: `speech_unavailable` (offers typing). |
| **Voices** | 12 s, 1 retry, a breaker each | the same-gender backup voice; then the bot's own recording of "my voice isn't working, the answer is on your screen". |
| **PostgreSQL** | connect 5 s; pooled connection 5 s; `statement_timeout` 5 s; `lock_timeout` 3 s | a lookup: `tool_failed` (never followed by a guess). A write that fails part-way: `outcome_unknown`, and figures read before it are treated as stale. |
| **SQLite** (the other engine) | busy timeout 5 s; one lock for every method | as PostgreSQL |
| **Audit log** | written after the turn, on a queue, 3 retries | nothing: the turn is done. A record still refused goes to `data/dead_letter/turns.jsonl`, is counted, and raises an alert. |
| **The page** | 45 s for a turn, 10 s for anything else | `timed_out` or `connection_lost`, shown **and spoken** in the bot's recorded voice. |

In a live session as well: speech that recognises to nothing is ignored if
short, "I didn't catch that" if it lasted 0.9 s or more; a dropped socket is
rejoined five times with back-off (0.3–5 s) before the bot says the
connection was lost; a refused, missing or unplugged microphone is said
(`microphone_refused`, `microphone_unavailable`, `microphone_lost`); silence
gets "Are you still there?" (`still_there`) and then a polite goodbye
(`idle_goodbye`); the session's time limit is said once and the call ends.
The tests check that no `live-…` thread is left after a stop, a hang-up, a
cancelled connection, or a 15-minute, 90-turn session.

### The conversation's own rules

- **One turn at a time.** A second request for the same conversation waits up
  to 2 s, then is answered `busy` — one transfer can never run twice.
- **The caller leaving** sets the turn's cancel token; it stops at its next
  safe point with the history left whole, logged as abandoned (`499`).
- **Repeated failures.** After 3 failed turns in a row, a case is opened for a
  colleague (`escalated`, with its reference), or `cannot_complete` (the
  helpline) if none can be opened.
- **Limits.** Warned (`session_warning`) 3 turns before 60 turns or at 90% of
  30 minutes, then closed (`session_ended`) and started afresh. The model has
  at most 4 rounds and 200 tokens a turn, cannot hand off twice, and cannot be
  forced to call a tool more than once a turn.
- **Nothing internal is said or shown.** Errors go to the log.

### Every fallback sentence

All of them are in `src/voicebot/messages.yaml`, in English and Urdu (and,
where Urdu marks it, female and male), and nowhere else; the file is checked
as it loads and the page fetches it through `GET /v1/messages`. Edit it and
restart to change what the bot says.

| Key | When |
|---|---|
| `unavailable` | The model could not be reached. |
| `nothing_said` | The model answered with nothing, or ran out of rounds. |
| `not_caught` | Nothing worth answering was heard. |
| `speech_unavailable` | The recognisers are down; offers typing. |
| `handed_over` | The customer asked for a person; carries the case reference. |
| `not_looked_up` | The model promised a lookup and never made one. |
| `unverified` | The guard stopped an unsourced figure with no real one to read. |
| `pin_needed` | Money is about to move and only the PIN is missing. |
| `tool_failed` | A lookup failed. |
| `outcome_unknown` | A write failed in a way that leaves it unknown whether it happened. |
| `action_done`, `action_cancelled` | Yes or Cancel acted, but the model could not be reached to say so. |
| `escalated`, `cannot_complete` | Several turns failed: passed to a colleague, or the helpline. |
| `session_warning`, `session_ended` | The conversation is near, or at, its limit. |
| `busy` | A second request arrived while the first was running. |
| `restarting` | The service is shutting down or restarting. |
| `internal` | Anything else. |
| `reply_failed` | A reply failed the language or gender check twice. |
| `voice_unavailable` | The voice failed; the answer is on the screen. |
| `connection_lost`, `timed_out` | Said by the page when it cannot reach the service, or a reply is too slow. |
| `microphone_refused`, `microphone_unavailable`, `microphone_lost` | The page's microphone failures. |
| `still_there`, `idle_goodbye` | A long silence in a live session. |
| `figure_one`, `figure_many`, `readback_on_screen` | Read-back wording the guards use. |
| `prompt_self_forms` | Shown to the model as the Urdu forms to speak of itself; never said. |

### Health, metrics, alerts and logs

- `GET /health` — liveness; answers whatever else is down.
- `GET /ready` — **503** while the database cannot be reached or the service
  is stopping. `checks` names each dependency: `database`, every provider's
  `circuit:<name>`, the `audit_queue`, `draining`. An open circuit leaves the
  service ready: it still answers, only less well.
- `GET /metrics` (Prometheus): turns, tool calls,
  `voicebot_guard_hits_total` (the one worth watching: a rising one means the
  model is reaching for figures it was not given),
  `voicebot_errors_total{component,code}`, `voicebot_circuit_open{provider}`,
  `voicebot_turns_in_flight`, `voicebot_audit_queue`,
  `voicebot_audit_dead_letters_total`, `voicebot_events_total{event}`.
- **Alerts** — when a circuit opens or a record goes to the dead-letter file:
  a CRITICAL `ALERT` line, and a POST to `VOICEBOT_ALERT_WEBHOOK` when set, at
  most once per kind every 5 minutes.
- **Logs** — every line names its call: `[call=<session id>]`. Every failure
  is one structured line (`error code=… component=… retryable=… attempt=…
  latency_ms=…`). What a customer said and the bot's replies are logged at
  **DEBUG only**; at INFO a turn is its shape. Every message is masked on its
  way out: email addresses, phone and card numbers (all but the last four
  digits), bearer tokens, `token=`, `pin=`, `password=`, `api_key=`. PINs are
  never logged, and a malformed request is never echoed back.

### Finding out why a call failed

1. **Get the call id** — the conversation's session id: in every turn the page
   receives (`call_id`, `session_id`), the audit log's `turns.session_id`, and
   the dead-letter file.
2. **Read its lines:** `./run.sh > voicebot.log 2>&1`, then
   `grep "call=<id>" voicebot.log`. `turn` lines give tools, fallback and
   timings; `error` lines the component, code, attempt and latency; `ALERT`
   lines an outage or a lost record.
3. **See what was said:** restart with `VOICEBOT_LOG_LEVEL=DEBUG` and
   reproduce (`heard=… replied=…`).
4. **Check the providers:** `/ready` shows open circuits; `voicebot_errors_total`
   shows which component fails, and how.
5. **Recover lost records:** each line of `data/dead_letter/turns.jsonl` is a
   turn record the database refused; replay them into `turns` once it is back
   (by hand — there is no replay command).

`setup` and `check` fail with one line and exit code 1; Ctrl+C exits 130; a
model download is retried once (not on 4xx), written atomically and checked
against its pinned hash.

---

## 9. Configuration

Three files, each checked when it loads (a bad value stops startup with the
reason): `configs/serve.yaml` for the service (defaults in
`src/voicebot/config.py`; `configs/app.yaml` is the all-fake default),
`configs/assistant.yaml` for the office services and who the bot is, and
`src/voicebot/messages.yaml` for every fixed sentence.

| Setting | Default | What it bounds |
|---|---|---|
| `llm.base_url`, `llm.model` | the office endpoint | the model |
| `llm.timeout_s`, `llm.max_retries` | 12, 1 | one request, and its retries |
| `llm.max_tokens`, `llm.max_tool_rounds` | 200, 4 | a reply's length, rounds in one turn |
| `llm.backup_base_url`, `llm.backup_model` | empty | a second model asked on an outage (both or neither) |
| `resilience.turn_budget_s` | 20 | every model round of one turn together |
| `resilience.retry_base_s`, `retry_cap_s` | 0.2, 1.0 | backoff between retries |
| `resilience.breaker_failures`, `breaker_cooldown_s` | 3, 20 | when a provider is skipped, and for how long |
| `database.engine` | `postgres` in serve.yaml | `postgres` or `sqlite` |
| `database.pool_size`, `pool_timeout_s` | 10, 5 | PostgreSQL connections |
| `database.statement_timeout_ms`, `lock_timeout_ms` | 5000, 3000 | a query, and a wait for a lock |
| `session.max_turns`, `max_minutes`, `warn_turns` | 60, 30, 3 | how long a conversation runs |
| `session.failure_limit`, `busy_wait_s` | 3, 2 | failed turns before a colleague takes over; a second request's wait |
| `session.ttl_s`, `max_sessions` | 1800, 1000 | when an idle conversation is forgotten |
| `audit.retries`, `queue_size`, `dead_letter` | 3, 1000, `data/dead_letter/turns.jsonl` | writing turn records |
| `asr.timeout_s`, `asr.retries` (assistant.yaml) | 6, 1 | each recogniser |
| `tts.timeout_s`, `tts.retries` (assistant.yaml) | 12, 1 | each voice |

The live session's settings (`live:` in serve.yaml):

| Setting | Default | What it does, and the trade |
|---|---|---|
| `threshold`, `neg_threshold` | 0.5, 0.35 | Speech starts above the first and carries on until below the second. |
| `min_speech_ms` | 200 | Shorter is a cough or a click. Longer ignores more noise but misses "yes". |
| `pre_roll_ms` | 300 | Audio kept from before speech was confirmed. |
| `end_silence_ms` | 400 | The silence that ends a turn. |
| `speculative_asr` | on | Recognition starts at 224 ms of silence. |
| `max_utterance_s` | 20 | A turn is cut here whatever happens. |
| `barge_in_threshold`, `barge_in_neg_threshold` | 0.8, 0.5 | While the bot speaks, speech must be surer. |
| `barge_in_min_speech_ms` | 300 | How much sure speech stops the bot. |
| `barge_in_echo_gate`, `barge_in_echo_margin_db`, `barge_in_echo_prior_db` | on, 6, −25 | The echo gate, its margin, and the echo assumed before it is measured. |
| `near_voice_gate`, `near_voice_margin_db` | on, 6 | The noisy-room gate, and how much louder than the room the caller must be. |
| `backchannel_max_ms`, `backchannels` | 800, a list | A short interruption of only these words is not one. |
| `idle_prompt_s`, `idle_end_s` | 20, 20 | "Are you still there?", then goodbye. |
| `max_session_minutes` | 30 | The longest a live session runs. |
| `playback_margin_s` | 3 | How long past a reply's audio before it is taken as finished. |
| `vad_model_path` | `models/silero/silero_vad.onnx` | Where `setup` puts the VAD model. |

**From the environment.** Any setting: `VOICEBOT__<SECTION>__<KEY>`, e.g.
`VOICEBOT__LLM__TIMEOUT_S=8` or `VOICEBOT__LIVE__END_SILENCE_MS=500`.

| Variable | Effect |
|---|---|
| `VOICEBOT_CONFIG` | which YAML file to load |
| `VOICEBOT_ASSISTANT_CONFIG` | another file instead of `configs/assistant.yaml` |
| `VOICEBOT_LOG_LEVEL` | `DEBUG`, `INFO` (default), `WARNING`, `ERROR` |
| `VOICEBOT_ALERT_WEBHOOK` | a URL to POST alerts to as JSON |
| `VOICEBOT_LLM_KEY` | the model's key, if one is ever needed (it accepts none) |
| `VOICEBOT_DB_*` | the PostgreSQL connection ([section 10](#10-the-database)) |

**At startup** the service refuses to start, with every reason at once, when
PostgreSQL is selected without `VOICEBOT_DB_USER`, the model's URL is not
http(s), a backup model is half configured, an endpoints file is missing, a
voice is wrong (section 5), or `messages.yaml` is incomplete.

---

## 10. The database

Eight tables in PostgreSQL, reached through psycopg and a connection pool —
no ORM. Money is integer cents throughout. `transactions` is a ledger: money
out is stored negative, and every balance change writes a row in the same
transaction as the update. Spending periods are a closed vocabulary (`today`,
`this week`, `last month` …), and an unrecognised one returns the valid ones.
`turns` is the audit log — what was said, which tools ran, whether the guard
fired and what it suppressed.

### Setting it up

Built and tested against PostgreSQL 18. The connection comes from five
variables, from the environment or `.env` (`cp .env.example .env`, then fill
in the password; `.env` is gitignored and no committed file holds one):

| Variable | If unset |
|---|---|
| `VOICEBOT_DB_HOST` | `localhost` |
| `VOICEBOT_DB_PORT` | `5432` |
| `VOICEBOT_DB_NAME` | `voicebot` |
| `VOICEBOT_DB_USER` | your login name |
| `VOICEBOT_DB_PASSWORD` | none |

The bank makes its own tables: on startup it applies any numbered file in
`src/voicebot/bank/migrations/` it has not applied before, in one
transaction, recorded in `schema_migrations`, and seeds the demo data if the
tables are empty. It does not make the database — the migration script does,
or once in `psql`:

```sql
CREATE DATABASE voicebot ENCODING 'UTF8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0;
```

`C` sorts text by its bytes, as SQLite does.

### Where things stand: SQLite and PostgreSQL

On 29 October 2026 the bank was copied from SQLite (`data/bank.db`) to
PostgreSQL: 344 rows in eight tables, every field compared with its source,
type included. `data/bank.db` was only ever read — its SHA-256
(`64f5d3b8…cbcf7`) is unchanged — and a backup taken first is in
`data/backups/`. **SQLite stays** in the code, the dependencies and the
configuration as the way back until the migration is confirmed.

How the types were carried: `INTEGER PRIMARY KEY AUTOINCREMENT` → `BIGINT
GENERATED BY DEFAULT AS IDENTITY` (ids copied, sequences moved past them);
money stays integer cents (`BIGINT`); `0/1` flags → `BOOLEAN`; `TEXT`
timestamps → `TIMESTAMP(0)` without time zone, unconverted; an empty
`pin_locked_until` → `NULL`; SQLite's hidden `rowid` → a `seq` column on
`disputes` and `handoffs`.

How it was proved: the source's integrity and foreign-key checks clean; a dry
run copied and verified everything, then rolled it back; the migration
matched 344 of 344 rows, a checksum per table, aggregates computed on each
side, every field and every constraint; a separate plain-SQL check agreed;
the full suite passed on PostgreSQL; and an end-to-end run on a migrated copy
moved money through the yes and the PIN, froze a card, changed a PIN and
reset, with no errors.

### Moving the SQLite data across

```bash
.venv/Scripts/python scripts/migrate_sqlite_to_postgres.py --dry-run   # all of it, then roll back
.venv/Scripts/python scripts/migrate_sqlite_to_postgres.py --report migration-report.md
```

It opens the file read-only (SHA-256 compared before and after), copies in
one transaction with every check inside it, and commits only if all pass. A
database that already matches is left alone; one holding anything else is
refused unless `--replace` is given (take a `pg_dump` first). `--source`
copies another file, such as a backup. Exit codes: 0 done, 1 refused or a
check failed, 2 could not connect.

### Going back to SQLite

Set `engine: sqlite` in `configs/serve.yaml` and restart; `./run.sh` then
reports `bank in SQLite: data/bank.db`. Nothing else changes, and
`python -m voicebot.bank` follows the configuration. Two things do not come
back: whatever was written in PostgreSQL after the switch stays there (no
script copies back), and the SQLite bank shares one connection between
threads, so under simultaneous requests it can charge a bill twice, let a
burst of wrong PINs past the lockout, or refuse a transfer that fits
(`tests/unit/test_bank_concurrency.py` shows each).

**When the migration is confirmed**, retiring SQLite would remove
`sqlite_bank.py`, `schema.sql`, the `sqlite3` code in `seed.py`, the `sqlite`
engine and the SQLite half of the test switch — after one decision: the
offline mode and the default test run use an in-memory SQLite bank today.
Until that word, nothing SQLite is touched.

### Looking after the demo bank

```bash
python -m voicebot.bank --show                          # balances and turn count
python -m voicebot.bank --deposit 500 --account checking # pay money in
python -m voicebot.bank --reset                          # opening balances, audit log kept
```

They act on the bank the server opens (`VOICEBOT_CONFIG`, or
`configs/serve.yaml`) and say which; `--config` picks another configuration,
`--path` names a SQLite file. Editing `bank/seed.py` does nothing to a
database that already exists: the seed runs once, on empty tables.

Known issues in both banks: timestamps mix local time (what Python writes)
and UTC (what the database fills in), copied as they were; and a reset
renumbers the ledger while disputes keep the old numbers.

---

## 11. The API and the live protocol

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat` | A turn, typed. |
| `WS /v1/live` | A hands-free conversation: microphone audio in; state, transcripts and reply audio out. |
| `POST /v1/confirm`, `POST /v1/cancel` | Agree to, or decline, the pending action. |
| `POST /v1/action` | Arm a card action from the Cards screen: the bank's read-back, settled through `/v1/confirm` like the spoken Yes. |
| `POST /v1/cards/{last4}/details`, `/pin/check`, `/pin` | Card details and PIN changes, behind the PIN. |
| `GET /v1/accounts`, `/v1/statement`, `/v1/cards`, `/v1/monthly` | The screens' data, with no turn taken. |
| `GET /v1/sessions/{id}/audio/{audio_id}/{n}` | The next piece of a reply's audio. |
| `GET /v1/messages`, `/v1/messages/{key}/audio` | The fallback sentences, and the bot's recordings of them. |
| `POST /v1/reset` | Restore the demo balances; keep the audit log. |
| `DELETE /v1/session/{id}` | Forget a conversation. |
| `GET /health`, `/ready`, `/metrics` | See [section 8](#health-metrics-alerts-and-logs). |

Interactive docs at `/docs`.

**The live protocol.** To the service: binary microphone chunks — with
`"capture": {"format": "level+pcm16"}` in `start`, each is a little-endian
float32 (how loud the page was playing the bot) followed by 16 kHz, 16-bit
mono PCM; and JSON `start` (`session_id`, `capture.settings`: what the
browser applied to the microphone), `stop`, `interrupted` (`audio_id`,
`piece`, `offset_ms`), `playback_done`, `ping`. From the service: `started`,
`state`, `transcript`, `reply` (the turn as `/v1/chat` returns it, less its
audio), `say`, `audio` (a header followed by a binary WAV piece),
`audio_end` (how many pieces), `audio_failed`, `stop_audio`,
`resume_audio`, `discard_audio`, `metrics` (per turn: endpointing,
recognition, model, voice, end of speech → first audio, barge-in, the echo
and room measured), `error` (with its sentence key), `ended`, `pong`.

---

## 12. Layout

| Path | Contents |
|---|---|
| `src/voicebot/llm/` | The model client (streamed), its fake, the tool schemas, the system prompt |
| `src/voicebot/agent/` | The model ⇄ tool loop, the executor that owns the confirmation gate, the guards |
| `src/voicebot/bank/` | The bank in PostgreSQL and in SQLite, the ledger, the audit log, the seed data, the migrations |
| `src/voicebot/assistant/` | The office services: both recognisers and the language decision, both voices, text made speakable, the VAD, who the bot is, `setup` and `check` |
| `src/voicebot/live/` | The hands-free session over a WebSocket: endpointing, the gates, barge-in, its state machine |
| `src/voicebot/asr/`, `src/voicebot/tts/` | The office recognisers and voices as the service's engines |
| `src/voicebot/pipeline/` | The turn: orchestrator, audio plumbing, per-stage timing, reply pieces |
| `src/voicebot/api/` | FastAPI app, routes, sessions, limits, audit, published payloads |
| `src/voicebot/api/static/` | The front end: `index.html`, one module per screen, one stylesheet each |
| `src/voicebot/messages.yaml` | Every fixed sentence, English and Urdu |
| `configs/` | `serve.yaml` (the real service), `app.yaml` (all fake), `assistant.yaml` (the office services) |
| `scripts/` | `migrate_sqlite_to_postgres.py` |
| `tests/` | Unit, integration and JavaScript tests |

Every engine sits behind an abstract base class with a registry, so swapping
a recogniser, a voice or the model endpoint is a config change. The front end
is ES modules loaded straight by the browser — no bundler, no build step.
`app.js` holds only the wiring; the work is in `core` (state and the API
client), `audio`, `conversation`, `cards`, `screens` and `mic`. `live.js` and
`live-capture.js` are classic scripts so they can be tested under Node.
`conversation.js` imports no screen: a screen asks to hear about a turn with
`onTurn`.

---

## 13. Tests and quality gates

```bash
.venv/Scripts/python -m ruff check src tests scripts     # lint
.venv/Scripts/python -m black --check src tests scripts  # formatting
.venv/Scripts/python -m mypy src scripts                 # types
.venv/Scripts/python -m pytest                           # 1252 tests
.venv/Scripts/python -m pytest --cov=voicebot            # with coverage
node --test tests/js/live.test.js tests/js/capture.test.js  # 20 tests: the live client
VOICEBOT_TEST_DATABASE=postgres .venv/Scripts/python -m pytest  # against PostgreSQL
.venv/Scripts/python -m pytest -m live                   # against the office services
```

All but the last run on every push (`.github/workflows/ci.yml`), the suite
against PostgreSQL on a server the job starts for itself. The default run uses
in-memory SQLite and needs no server; the PostgreSQL run uses its own
database (`voicebot_test`, or `VOICEBOT_TEST_DB_NAME`), a fresh schema per
test, refuses the app's own database, and is the only run with the
concurrency tests. `live` tests need the office network. Coverage is gated at
90%, across the SQLite and PostgreSQL runs together (add `--cov-append` to the
second). `pip install pre-commit && pre-commit install` runs the same checks
before a commit.

---

## 14. Known limitations and open decisions

**Limits, stated plainly:**

- **No arithmetic out loud** — the guard rejects any figure a tool did not
  return, including correct sums.
- **One currency** (USD); asked in pounds, it says so rather than converting.
- **Speed.** First audio is ~1.5–3 s in English and ~4–7 s in Urdu after you
  stop. The voices do not stream and take one request at a time; the floor is
  theirs.
- **One office network for everything** — the model, both recognisers and
  both voices. When it is unreachable every turn gets a fallback sentence. A
  backup model can be configured; a backup recogniser or voice server would
  need hosting. The male voices have no backup.
- **No authentication**, CORS allows `*`, and a session id from the client is
  accepted as it is: anyone who can reach the port can use the bot and the
  demo bank. The dead-letter file holds what customers said and should be
  protected like the database.
- **Sessions are in-process**: a restart forgets conversations (the money
  stays), and two instances would need a shared store (`SessionStore` is the
  seam).
- **The model's judgement is not tested.** The suite proves no model
  behaviour can move money without a confirmation or state an unsourced
  figure; it cannot prove the model picks the right tool. It sometimes asks
  something it should have looked up.
- **The noisy-room gate** cannot tell a second person as loud as the caller,
  and before the caller's first words a voice across the room may be taken
  for them. Speakerphones without echo cancellation are not usable; use a
  headset.
- **Recognition** is weakest on noisy, accented English (1 clip in 30 was
  taken for Urdu), and accuracy through a real microphone across accents is
  unmeasured; synthetic test speech is cleaner than a phone line.
- **The page's own labels are English** in an Urdu conversation; the
  conversation itself is in the caller's language.
- **Gender correction** covers the common Urdu first-person forms; a rarer
  construction in the wrong gender would be spoken as written.
- **The English voice** (a 0.6B base model) now and then says words that were
  not in its text, and nothing checks it.
- **Freezing a card with a replacement and a fraud report** is three writes;
  a failure part-way is reported as unknown, not undone.

**Needing a decision:**

1. **The model can confirm some actions itself.** Within one turn it can carry
   the token it was just given into its next round; for gated actions with no
   PIN (freezing or ordering a card, a dispute, a card control) that runs them
   without the customer's yes. Money cannot move this way — the PIN holds.
   The fix (honour a token only in a later turn) changes the gate's behaviour.
2. **Licences.** OmniVoice is CC-BY-NC (non-commercial) and runs five of the
   six voices, including the Urdu voice; `assistant-pk` is a real speaker
   needing attribution. A paying client needs a commercial licence or another
   provider.
3. **Speed.** A streaming voice server is the largest cut. A shorter Urdu
   reference clip would cut OmniVoice's per-request cost, at some risk to the
   voice.
4. **Explicit language requests.** "Answer me in English" in Urdu is answered
   in Urdu — the bot follows the language spoken. Honouring such a request
   for the rest of the call is an option.
5. **Retiring SQLite**, once the migration is confirmed ([section 10](#10-the-database)).
6. **Smaller:** a replay command for the dead-letter file; an alert webhook
   (`VOICEBOT_ALERT_WEBHOOK`); custom vocabulary on the recognisers, measured
   before it is kept.

---

## 15. History

- **v1** was the same product without a language model: a RoBERTa intent and
  slot model trained here, a finite-state dialogue manager and reply
  templates — 0.8963 test accuracy, 0.9241 macro-F1, and unable to invent a
  balance because every sentence was written by hand.
- **0.1.0 (22 Oct 2026)** — speech in, speech out, a language model answering
  through tools against a SQLite bank, with the confirmation gate and the
  numeric guard.
- **0.2.0 (27 Oct 2026)** — the Transactions, Cards and Monthly screens;
  moving money between your own accounts; a payee table; handing over to a
  person; disputes; lost and stolen cards; card controls; monthly
  commitments. The bot can no longer pay money in.
- **Since then** — the bank moved to PostgreSQL (SQLite kept as the way
  back); English and Urdu with the agent's gender and voices; error handling
  hardened end to end; hands-free live voice with barge-in, the echo gate and
  the noisy-room gate; the model's reply streamed and the Urdu voice made
  smooth; general-knowledge answers. The wake-word desktop assistant,
  push-to-talk, the browser voice and the local Whisper/VITS models were
  removed as unused. Earlier reports, plans and measurements are in the git
  history.

---

## 16. Licences

This project is MIT licensed — see `LICENSE`.

| Component | Licence |
|---|---|
| Silero VAD (run here, for the live session) | MIT |
| OmniVoice (office server: the Urdu voice, the male voices, the backups) | **CC-BY-NC — non-commercial** |
| `assistant-pk` reference clip (Google FLEURS `ur_pk`) | CC-BY-4.0, attribution required |
| qwen3-tts (office server: the female English voice) | Qwen3-TTS model licence — confirm for commercial use |
