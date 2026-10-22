# Voice Bot — a speech-to-speech banking assistant

Speak to it, and it speaks back. A cascaded voice assistant that handles
balances, transfers, bills, spending, cards and fraud reports — with an intent
and slot model trained here, and no large language model anywhere in the
answer path.

```
your voice -> VAD -> ASR -> normalise -> NLU (intent + slots) -> scope gate
                                                                     |
          its voice <- TTS <- templates <- dialogue manager <- mock bank
```

Everything runs locally: three open models, one FastAPI service, and a plain
HTML/CSS/JS front end with barge-in. No external AI service is called, and
nothing leaves the machine.

<!-- Save a screenshot of the running UI to docs/images/ui.png, then delete
     these two comment lines to show it.
![The interface, with the inspector panel open](docs/images/ui.png)
-->

---

## Results

The NLU stage is the part trained here. Full method, ablations and the two
measurement bugs that changed the conclusions are in
[`docs/experiments.md`](docs/experiments.md); what the model is for, and what
it is explicitly not for, is in [`docs/model_card.md`](docs/model_card.md).

| Model | Params | Test accuracy | Macro-F1 | In-scope accuracy | Slot F1 |
|---|---|---|---|---|---|
| TF-IDF + LinearSVC *(baseline)* | — | 0.8209 | 0.8639 | — | n/a |
| DistilBERT joint | 66M | 0.8807 | 0.9128 | 0.9410 | 0.5613 |
| **RoBERTa joint** *(shipped)* | **125M** | **0.8963** | **0.9241** | **0.9528** | **0.6866** |

**Out-of-scope gate**, threshold 0.93 tuned on validation data only:
**95% of out-of-scope questions declined correctly, 91% of in-scope questions
answered correctly and delivered.**

**Latency**, measured end to end through the browser:

| | T4 GPU | CPU |
|---|---|---|
| ASR (`whisper-small` / `whisper-base.en`) | 388 ms | 2.3 s |
| NLU | 12 ms | 0.3 s |
| TTS | 180 ms | 5 s |
| **Whole turn** | **~0.6 s** | **~8 s** |

---

## Why there is no LLM

An LLM asked for your balance will eventually produce a confident, plausible,
invented number, because producing plausible text is what it does. This bot
cannot: every number comes from the account store, and every sentence it can
say is a template in one readable file.

The model's job is narrow and measurable — classify the intent, tag the slots.
The answering is ordinary code:

| An LLM would | Here it is |
|---|---|
| Read what you meant | RoBERTa, two heads (`nlu/`) |
| Decide whether to attempt it | Confidence gate at 0.93 (`nlu/joint_engine.py`) |
| Remember the conversation | `dialogue/states.py` |
| Decide what to do next | `dialogue/manager.py` |
| Know the facts | `dialogue/mock_backend.py` |
| Write the sentence | `dialogue/responses.py` |

The cost is that it only does the seven things it was built for. For a
banking assistant, "I don't handle that" is the better failure.

---

## Quick start

### Run the tests — no GPU, no downloads, no network

```bash
pip install -e ".[dev]"
pytest
```

248 tests. Every engine has a fake implementation, so the whole suite — the
HTTP API included — runs offline in seconds.

### Run the service with the real models

```bash
pip install -e ".[dev,serve]"
VOICEBOT_CONFIG=configs/serve.yaml \
VOICEBOT_CHECKPOINT=checkpoints/roberta_oosplus \
uvicorn voicebot.api.app:app --port 8000
```

Open <http://localhost:8000>. The page is served by the same process as the
API, which is what lets the browser grant it a microphone.

Without `VOICEBOT_CONFIG` the service starts on the fake engines — useful for
working on the front end with no model downloads.

### On Google Colab

Colab is the practical way to train and to run this on a GPU. Mount Drive,
`pip install -e ".[dev,serve]"`, start the server in a background thread, and
open the port:

```python
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8000)"))
```

Open that address in a **new tab**, not an inline frame — an embedded frame is
not allowed to use the microphone.

---

## The API

| Endpoint | Purpose |
|---|---|
| `POST /v1/converse` | Audio in, audio out. One whole turn. |
| `POST /v1/chat` | The same turn, typed instead of spoken. |
| `POST /v1/transcribe` | ASR alone. |
| `POST /v1/understand` | NLU alone. |
| `POST /v1/speak` | TTS alone. |
| `DELETE /v1/session/{id}` | Forget a conversation. |
| `GET /health` | Liveness. Answers before the models load. |
| `GET /ready` | Readiness, naming the engines actually loaded. |
| `GET /metrics` | Prometheus text: turns, actions by type, live sessions. |

Each pipeline stage is exposed on its own so a client can inspect any one of
them in isolation — which is exactly what the inspector panel in the UI does.

Interactive docs at `/docs`.

Models load **once** in the application lifespan, not per request. Each
conversation holds **its own account state**, so one browser tab cannot spend
another's balance and a completed transfer persists across turns.

---

## Layout

| Path | Contents |
|---|---|
| `src/voicebot/asr/` | Whisper engine, and the normaliser that rewrites its output into the form the NLU was trained on |
| `src/voicebot/nlu/` | The joint intent + slot model, its dataset, BIO tagging, the SVM baseline |
| `src/voicebot/dialogue/` | State machine, slot specs, account store, reply templates |
| `src/voicebot/tts/` | VITS engine |
| `src/voicebot/pipeline/` | Orchestrator, audio plumbing, per-stage timing |
| `src/voicebot/api/` | FastAPI app, routes, sessions, published payload shapes |
| `src/voicebot/api/static/` | The front end: `index.html`, `app.css`, `app.js`, `vad.js` |
| `src/voicebot/training/` | Training and evaluation entry points |
| `configs/` | YAML run configuration, validated by pydantic at startup |
| `scripts/` | Data download; gate re-tuning without retraining |
| `tests/` | Unit and integration tests |
| `docs/experiments.md` | Method, results, ablations, measurement bugs |
| `docs/model_card.md` | What the trained model is, is for, and is not for |

Every engine sits behind an abstract base class with a registry, so swapping
Whisper for another ASR is a config change, and the fakes that make the test
suite offline implement the same interface.

---

## The front end

Four files, no framework.

- Records the microphone and converts WebM/Opus to 16 kHz mono PCM **in the
  browser**, so the server receives exactly the format it wants
- **Inspector panel** showing the transcript and its confidence, the predicted
  intent and its confidence, the extracted slots, the gate score and verdict,
  and milliseconds spent in each pipeline stage
- **Barge-in**: talk over the reply and it stops and listens, with a
  sensitivity slider and an off switch. Relies on the browser's echo
  cancellation, or the bot would interrupt itself
- **Yes / Cancel buttons** during confirmation, so a misheard "yes" cannot
  move money
- A text box for when the microphone is unavailable
- Works at phone width

---

## Training

```bash
# the shipped model
python -m voicebot.training.train_intent \
    --config configs/experiments/e1_roberta.yaml \
    --clinc data/raw/data/data_oos_plus.json \
    --output checkpoints/roberta_oosplus \
    --slot-examples 5000

# re-tune the out-of-scope gate by inference alone, ~1 min instead of 22
python scripts/retune_gate.py --checkpoint checkpoints/roberta_oosplus \
    --clinc data/raw/data/data_oos_plus.json
```

Intents come from CLINC150 (15,250 real utterances, 150 in-scope intents plus
`oos`). Slots come from a templated banking corpus generated here, split by
**template and by value** so no phrasing or value appears on both sides of the
split. That split is the difference between a slot F1 of 1.0000 and a
meaningful 0.6866 — see `docs/experiments.md`.

Every run writes `metrics.json` and `predictions.json` beside its checkpoint,
with the seed recorded.

---

## Quality gates

```bash
ruff check src tests     # lint
black --check src tests  # formatting
mypy src                 # types
pytest                   # 248 tests
```

---

## Known limitations

Stated plainly, because they change how the numbers should be read.

- **The slot corpus is synthetic.** CLINC150 carries no slot annotations, so
  the slot head trains on 18 generated templates. Slot F1 on real speech will
  be below 0.6866. The intent head is unaffected — it trains on real human
  utterances.
- **The slot tagger labels tokens in every utterance**, including ones with no
  slots, so it produces noise the dialogue manager discards. Visible in the
  inspector panel; harmless to the answer.
- **The gate threshold is tuned on 100 validation out-of-scope examples.** The
  value carries real sampling noise.
- **ASR quality drives everything downstream.** `whisper-base.en` on CPU
  mishears enough to break turns; `whisper-small` on a GPU does not. No
  dialogue logic repairs a word that was never heard.
- **English only.** All metrics above are text metrics; accuracy measured
  through a real speech front end is deferred to v2.
- **Sessions are in-process**, with a TTL and a cap. Correct for one process,
  wrong for two. `SessionStore` is an interface for exactly that reason.

---

## What's next

Specified in the design document and deliberately out of scope for v1: a
spoken benchmark (synthesising CLINC150 and measuring accuracy against word
error rate), ASR-robust training on perturbed transcripts, and a comparison of
out-of-scope scoring methods beyond a confidence threshold.

---

## Licences

Check these before any commercial use — they are not all permissive:

| Component | Licence |
|---|---|
| CLINC150 dataset | CC BY 3.0 |
| `openai/whisper-*` | MIT |
| `roberta-base` | MIT |
| `facebook/mms-tts-eng` | **CC BY-NC 4.0 — non-commercial** |

The TTS model is the binding constraint. Anything commercial would need a
permissively licensed voice; the engine registry makes that a config change.
