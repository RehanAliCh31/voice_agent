"""Reading a `.env` file into the environment, and filling a config from it.

Small enough to write rather than depend on: a dependency for twenty lines
is a dependency to keep up to date for twenty lines.

The rule that matters is that the environment wins. Anything already
exported is already decided, so a file somebody forgot they had cannot
quietly override what they typed -- and a server started with an explicit
`VOICEBOT_CONFIG=` keeps it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from voicebot.utils.logging import get_logger

log = get_logger("utils.env")

DEFAULT_ENV_FILE = Path(".env")

# ${NAME}, or ${NAME:-default} when the committed file has a fallback.
_PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")


def read_env_file(path: Path | str = DEFAULT_ENV_FILE) -> dict[str, str]:
    """The settings in `path`, without touching the environment. Empty if absent."""
    file = Path(path)
    if not file.is_file():
        return {}

    values: dict[str, str] = {}
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
        if name and value:
            values[name] = value
    return values


def load_env_file(path: Path | str = DEFAULT_ENV_FILE) -> None:
    """Put the settings in `path` into the environment, if it is there.

    A missing file is the usual case and is not an error: nothing here is
    required, and the committed defaults run with none of it.
    """
    loaded = []
    for name, value in read_env_file(path).items():
        if name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)

    if loaded:
        # Names only. The values are the reason the file exists.
        log.info("read %s: %s", Path(path), ", ".join(loaded))


def expand(
    text: str,
    source: str = "config",
    env_file: Path | str = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> str:
    """`text` with each ${NAME} filled in: the environment, then `env_file`, then its default.

    Addresses that are not for publishing -- the office's servers -- are
    kept in .env and named in the committed config, so the config can be
    shared and the addresses cannot. The file is read here rather than
    loaded, so reading a config never changes the environment.
    """
    environ = os.environ if environ is None else environ
    from_file: dict[str, str] | None = None
    missing: list[str] = []

    def fill(match: re.Match[str]) -> str:
        nonlocal from_file
        name, default = match.group(1), match.group(2)
        if environ.get(name):
            return environ[name]
        if from_file is None:
            from_file = read_env_file(env_file)
        if name in from_file:
            return from_file[name]
        if default is not None:
            return default
        missing.append(name)
        return match.group(0)

    filled = _PLACEHOLDER.sub(fill, text)
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise ValueError(f"{source} needs {names}: set it in .env (see .env.example)")
    return filled
