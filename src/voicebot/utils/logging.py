"""A single place that configures logging, so modules never call basicConfig."""

from __future__ import annotations

import logging
import sys

_HANDLER_NAME = "voicebot-stdout"


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stdout at INFO level.

    Whether the handler is already attached is read off the logger itself, not
    a module-level flag. `logging` keeps its own registry that survives a
    module reload, so a flag would reset while the old handler stayed -- and
    every reload would add another, printing each line one more time. That is
    not hypothetical: reloading the package in a notebook does it immediately.
    """
    root = logging.getLogger("voicebot")
    if not any(getattr(h, "name", None) == _HANDLER_NAME for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.name = _HANDLER_NAME
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
    return logging.getLogger(f"voicebot.{name}")
