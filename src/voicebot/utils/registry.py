"""Name-to-class resolution for swappable engines.

A config saying `asr: whisper_small` resolves to a class here. This is what
lets experiments swap an implementation without touching the orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

_REGISTRY: dict[str, dict[str, type]] = {}

T = TypeVar("T", bound=type)


def register(kind: str, name: str) -> Callable[[T], T]:
    """Class decorator recording `cls` under (kind, name)."""

    def decorator(cls: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket:
            raise ValueError(f"{kind} engine {name!r} is already registered to {bucket[name]!r}")
        bucket[name] = cls
        return cls

    return decorator


def available(kind: str) -> list[str]:
    """Sorted names registered for `kind`."""
    return sorted(_REGISTRY.get(kind, {}))


def create(kind: str, name: str, **kwargs: Any) -> Any:
    """Instantiate the class registered under (kind, name)."""
    if kind not in _REGISTRY:
        raise KeyError(f"No engines registered for kind {kind!r}. Known kinds: {sorted(_REGISTRY)}")
    bucket = _REGISTRY[kind]
    if name not in bucket:
        raise KeyError(f"No {kind} engine named {name!r}. Available: {sorted(bucket)}")
    return bucket[name](**kwargs)
