#!/usr/bin/env bash
#
# Start the voice bot.
#
#   ./run.sh           the real engines: Whisper, the model endpoint, browser speech
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
        .venv/Scripts/python -m pip install -e \".[dev,serve]\""
fi
ok "python $("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

MISSING=$("$PYTHON" - <<'PY'
import importlib.util
needed = {"fastapi": "serve", "uvicorn": "serve", "torch": "core", "transformers": "core"}
print(" ".join(n for n in needed if importlib.util.find_spec(n) is None))
PY
)
if [ -n "$MISSING" ]; then
    die "Missing packages:$MISSING
        .venv/Scripts/python -m pip install -e \".[dev,serve]\""
fi
ok "dependencies installed"

# --- --test ------------------------------------------------------------------

if [ "${1:-}" = "--test" ]; then
    say "Running the test suite"
    exec "$PYTHON" -m pytest
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
    export VOICEBOT_CONFIG="$CONFIG"
fi

# --- go ----------------------------------------------------------------------

say "Starting on http://localhost:$PORT"
[ "$FAKE" = "yes" ] || printf '  Whisper loads on the first run and takes about 50 seconds.\n'
printf '  Open the address in Chrome, then press the microphone and speak.\n'
printf '  Ctrl+C to stop.\n\n'

exec "$PYTHON" -m uvicorn voicebot.api.app:app --port "$PORT"
