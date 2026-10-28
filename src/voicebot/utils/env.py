"""Reading a `.env` file into the environment.

Small enough to write rather than depend on: a dependency for twenty lines
is a dependency to keep up to date for twenty lines.

The rule that matters is that the environment wins. Anything already
exported is already decided, so a file somebody forgot they had cannot
quietly override what they typed -- and a server started with an explicit
`VOICEBOT_CONFIG=` keeps it.
"""

from __future__ import annotations

import os
from pathlib import Path

from voicebot.utils.logging import get_logger

log = get_logger("utils.env")

DEFAULT_ENV_FILE = Path(".env")


def load_env_file(path: Path | str = DEFAULT_ENV_FILE) -> None:
    """Put the settings in `path` into the environment, if it is there.

    A missing file is the usual case and is not an error: nothing here is
    required, and the committed defaults run with none of it.
    """
    file = Path(path)
    if not file.is_file():
        return

    loaded = []
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        # Split on the first only: a value may be a URL, and a URL has its
        # own equals signs in it.
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip("\"'")

        # An empty value means unset, not set-to-nothing. The example file
        # ships `VOICEBOT_LLM_KEY=` to name the variable, not to blank it.
        if not name or not value or name in os.environ:
            continue

        os.environ[name] = value
        loaded.append(name)

    if loaded:
        # Names only. The values are the reason the file exists.
        log.info("read %s: %s", file, ", ".join(loaded))
