"""The bot's own sentences for when something fails, read from messages.yaml.

The file is the one place they are kept, in English and Urdu, so changing a
sentence is an edit to that file and nothing else. It is checked as it is
read, and a problem with it stops the service starting rather than surfacing
as a missing sentence in the middle of a call.
"""

from __future__ import annotations

import string
from functools import lru_cache
from pathlib import Path

import yaml

LANGUAGES = ("en", "ur")
PATH = Path(__file__).with_name("messages.yaml")

# Messages that carry a value, and the one value each must carry.
_FIELDS: dict[str, set[str]] = {"handed_over": {"reference"}, "escalated": {"reference"}}


class MessagesError(ValueError):
    """messages.yaml is missing, malformed, or incomplete."""


def _fields(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def load(path: Path = PATH) -> dict[str, dict[str, str]]:
    """Read and check the messages. Raises MessagesError with what is wrong."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MessagesError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise MessagesError(f"{path} must map each message name to its sentences")

    problems: list[str] = []
    messages: dict[str, dict[str, str]] = {}
    for key, sentences in raw.items():
        if not isinstance(sentences, dict):
            problems.append(f"{key}: expected en and ur")
            continue
        for language in LANGUAGES:
            text = sentences.get(language)
            if not isinstance(text, str) or not text.strip():
                problems.append(f"{key}: no {language} sentence")
        if any(not isinstance(sentences.get(lang), str) for lang in LANGUAGES):
            continue
        wanted = _FIELDS.get(key, set())
        for language in LANGUAGES:
            found = _fields(sentences[language])
            if found != wanted:
                problems.append(
                    f"{key} ({language}): carries {sorted(found) or 'no values'}, "
                    f"should carry {sorted(wanted) or 'none'}"
                )
        messages[str(key)] = {lang: sentences[lang].strip() for lang in LANGUAGES}
    for key in _FIELDS:
        if key not in raw:
            problems.append(f"{key}: missing")
    if problems:
        raise MessagesError(f"{path}: " + "; ".join(problems))
    return messages


@lru_cache(maxsize=1)
def messages() -> dict[str, dict[str, str]]:
    return load()


def say(key: str, language: str | None = "en", **fields: str) -> str:
    """Message `key` in `language` (English for anything but Urdu)."""
    table = messages()
    if key not in table:
        raise KeyError(f"no message called {key!r} in {PATH.name}")
    text = table[key]["ur" if language == "ur" else "en"]
    return text.format(**fields) if fields else text


def english(key: str) -> str:
    return messages()[key]["en"]


def in_language(sentence: str, language: str | None) -> str:
    """One of the bot's own English sentences, in the customer's language.

    The English is the key, so a sentence that is not one of these -- a
    detail from the bank, say -- is said as it is rather than not at all.
    """
    if language != "ur":
        return sentence
    for sentences in messages().values():
        if sentences["en"] == sentence:
            return sentences["ur"]
    return sentence
