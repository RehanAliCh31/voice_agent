#!/usr/bin/env bash
#
# Start the voice bot.
#
#   ./run.sh           the real engines: the office recognisers and voices, in
#                      English and Urdu, and the model endpoint
#   ./run.sh --fake    no model, no endpoint, no downloads -- for front-end work
#   ./run.sh --test    run the test suite instead of the server
#
# Checks the things that actually go wrong before starting, because a server
# that boots and then fails on the first turn is harder to diagnose than one
# that refuses to boot with a reason.

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PYTHON=".venv/Scripts/python.exe"
[ -x "$PYTHON" ] || PYTHON=".venv/bin/python"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
die()  { printf '  \033[31mSTOP\033[0m  %s\n\n' "$*" >&2; exit 1; }

# --- the environment ---------------------------------------------------------

say "Checking the environment"

if [ ! -x "$PYTHON" ]; then
    die "No virtual environment. Create one first:
        py -3.12 -m venv .venv
        .venv/Scripts/python -m pip install -e \".[dev,serve,assistant]\" -c constraints.txt"
fi
ok "python $("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

# httpx and scipy are the assistant extra's: the office recognisers and voices
# are reached with one, and the page's audio is resampled with the other.
MISSING=$("$PYTHON" - <<'PY'
import importlib.util
needed = {
    "fastapi": "serve", "uvicorn": "serve",
    "httpx": "assistant", "scipy": "assistant",
}
print(" ".join(n for n in needed if importlib.util.find_spec(n) is None))
PY
)
if [ -n "$MISSING" ]; then
    die "Missing packages:$MISSING
        .venv/Scripts/python -m pip install -e \".[dev,serve,assistant]\" -c constraints.txt"
fi
ok "dependencies installed"

# --- --test ------------------------------------------------------------------

if [ "${1:-}" = "--test" ]; then
    say "Running the Python suite"
    "$PYTHON" -m pytest

    # The barge-in detector is the one piece of logic that cannot be tested
    # from Python, and the one whose failures were found by using it.
    if command -v node >/dev/null 2>&1; then
        say "Running the barge-in detector tests"
        node --test "tests/js/*.test.js" 2>&1 | grep -E "^(not ok|# (pass|fail))" || true
    else
        warn "node not installed; skipping the barge-in detector tests"
    fi
    exit 0
fi

# --- which configuration -----------------------------------------------------

FAKE=no
if [ "${1:-}" = "--fake" ]; then
    FAKE=yes
    say "Starting with fake engines"
    warn "No model and no speech recognition. The page works; the bot does not think."
    unset VOICEBOT_CONFIG
else
    CONFIG="configs/serve.yaml"
    [ -f "$CONFIG" ] || die "$CONFIG is missing."

    # The endpoint is the single most likely thing to be wrong: it lives on a
    # private network, so anything off that network cannot reach it at all.
    ENDPOINT=$("$PYTHON" -c "
from voicebot.config import load_config
print(load_config('$CONFIG').llm.base_url)
" 2>/dev/null || echo "")

    say "Checking the model endpoint"
    if [ -z "$ENDPOINT" ]; then
        warn "No llm.base_url in $CONFIG; the bot will not be able to think."
    elif curl -s -m 5 -o /dev/null "$ENDPOINT/models"; then
        ok "$ENDPOINT is reachable"
    else
        die "Cannot reach $ENDPOINT

        That address is on a private network. This machine has to be on that
        network to use it -- a laptop elsewhere, or a cloud notebook, cannot.

        To work on the interface without it:  ./run.sh --fake"
    fi

    # The bank is in PostgreSQL when the config says so. A wrong password or
    # a server that is not running should stop here, in the server's own
    # words, rather than as a crash on the first request. The last line the
    # check prints is its verdict; anything above it is logging.
    say "Checking the database"
    DB_CHECK=$("$PYTHON" - "$CONFIG" 2>&1 <<'PY' | tail -n 1
import importlib.util
import sys
from pathlib import Path

from voicebot.config import load_config

config = load_config(sys.argv[1])
if config.database.engine != "postgres":
    print(f"sqlite {config.database.path}")
    sys.exit(0)
if importlib.util.find_spec("psycopg") is None:
    print("missing-driver")
    sys.exit(0)

import psycopg

from voicebot.bank.postgres_bank import conninfo_from_env, describe
from voicebot.utils.env import load_env_file

load_env_file(Path(".env"))
info = conninfo_from_env()
where = describe(info)
try:
    psycopg.connect(info).close()
except psycopg.Error as exc:
    print(f"unreachable {where}: {str(exc).strip().splitlines()[-1]}")
    sys.exit(0)
print(f"postgres {where}")
PY
)
    case "$DB_CHECK" in
        sqlite*) ok "bank in SQLite: ${DB_CHECK#sqlite }" ;;
        postgres*) ok "bank in PostgreSQL: ${DB_CHECK#postgres }" ;;
        missing-driver) die "$CONFIG uses PostgreSQL, but psycopg is not installed:
        .venv/Scripts/python -m pip install -e \".[dev,serve]\" -c constraints.txt" ;;
        *) die "Cannot open the PostgreSQL bank: ${DB_CHECK#unreachable }

        Check VOICEBOT_DB_* in .env, and that PostgreSQL is running.
        To use the SQLite file instead, set database.engine: sqlite in $CONFIG." ;;
    esac

    # The office recognisers and voices, when the config uses them. Not a
    # reason to refuse to start: with them down the bot can still be typed to,
    # and still shows its replies. But it can neither hear nor speak, which is
    # better known now than found on the first question.
    say "Checking the speech services"
    SPEECH=$("$PYTHON" - "$CONFIG" 2>/dev/null <<'PY' || echo "error||"
import sys

import httpx

from voicebot.assistant.config import load_assistant_config
from voicebot.assistant.health import check_service, services
from voicebot.config import load_config

config = load_config(sys.argv[1])
used = False
with httpx.Client(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
    for kind, engine in (("ASR", config.asr), ("TTS", config.tts)):
        if engine.engine != "endpoints":
            continue
        used = True
        for name, url, model in services(load_assistant_config(engine.endpoints)):
            if name.endswith(kind):
                check = check_service(name, url, model, client)
                print(f"{'ok' if check.ok else 'down'}|{name}|{check.url}")
if not used:
    print("none||")
PY
)
    while IFS='|' read -r state name where; do
        case "$state" in
            ok) ok "$name at $where" ;;
            down) warn "$name is not answering at $where" ;;
            none) ok "none used: $CONFIG names no office recogniser or voice" ;;
            *) warn "could not check them; see configs/assistant.yaml" ;;
        esac
    done <<< "$SPEECH"
    export VOICEBOT_CONFIG="$CONFIG"
fi

# --- go ----------------------------------------------------------------------

say "Starting on http://localhost:$PORT"
printf '  Open the address in Chrome, then press the microphone and speak.\n'
[ "$FAKE" = "yes" ] || printf '  English or Urdu: the bot answers in the language it is asked in.\n'
printf '  Ctrl+C to stop.\n\n'

exec "$PYTHON" -m uvicorn voicebot.api.app:app --port "$PORT"
