"""The bot's own sentences, read from messages.yaml, in the profile's gender.

The file is the one place they are kept, in English and Urdu, female and
male where Urdu tells them apart, so changing a sentence is an edit to that
file and nothing else. It is checked as it is read, and a problem with it
stops the service starting rather than surfacing as a missing sentence -- or
a sentence in the wrong gender -- in the middle of a call.
"""

from __future__ import annotations

import string
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from voicebot import profile
from voicebot.profile import GENDERS, Gender
from voicebot.reply_check import wrong_gender_forms

LANGUAGES = ("en", "ur")
PATH = Path(__file__).with_name("messages.yaml")
# Shown to the model, never said: not served to the page, not recorded.
PROMPT_PREFIX = "prompt_"

# Messages that carry a value, and the one value each must carry.
_FIELDS: dict[str, set[str]] = {"handed_over": {"reference"}, "escalated": {"reference"}}

# key -> language -> gender -> sentence. A sentence the same for both genders
# appears under both.
Table = dict[str, dict[str, dict[Gender, str]]]


class MessagesError(ValueError):
    """messages.yaml is missing, malformed, or incomplete."""


def _fields(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def _forms(key: str, language: str, value: Any, problems: list[str]) -> dict[Gender, str] | None:
    """One language's sentence as {female, male}, checked; None when it is unusable."""
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if language == "ur":
            for gender in GENDERS:
                # Given once, so said by either: it must not speak of the bot
                # as one gender or the other.
                if wrong_gender_forms(text, gender):
                    problems.append(
                        f"{key} ({language}): speaks of the bot as one gender; give female and male"
                    )
                    break
        return {"female": text, "male": text}
    if isinstance(value, dict) and set(value) == set(GENDERS):
        forms: dict[Gender, str] = {}
        for gender in GENDERS:
            text = value[gender]
            if not isinstance(text, str) or not text.strip():
                problems.append(f"{key} ({language}, {gender}): no sentence")
                return None
            forms[gender] = text.strip()
            if language == "ur" and wrong_gender_forms(forms[gender], gender):
                problems.append(
                    f"{key} ({language}, {gender}): speaks of the bot as the other gender"
                )
        return forms
    if isinstance(value, dict):
        problems.append(f"{key} ({language}): give one sentence, or exactly female and male")
    else:
        problems.append(f"{key}: no {language} sentence")
    return None


def load(path: Path = PATH) -> Table:
    """Read and check the messages. Raises MessagesError with what is wrong."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MessagesError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise MessagesError(f"{path} must map each message name to its sentences")

    problems: list[str] = []
    table: Table = {}
    for key, sentences in raw.items():
        if not isinstance(sentences, dict):
            problems.append(f"{key}: expected en and ur")
            continue
        found = {lang: _forms(str(key), lang, sentences.get(lang), problems) for lang in LANGUAGES}
        forms = {lang: f for lang, f in found.items() if f is not None}
        if len(forms) != len(LANGUAGES):
            continue
        wanted = _FIELDS.get(str(key), set())
        for language, by_gender in forms.items():
            for gender, text in by_gender.items():
                carried = _fields(text)
                if carried != wanted:
                    problems.append(
                        f"{key} ({language}, {gender}): carries {sorted(carried) or 'no values'}, "
                        f"should carry {sorted(wanted) or 'none'}"
                    )
        table[str(key)] = forms
    for key in _FIELDS:
        if key not in raw:
            problems.append(f"{key}: missing")
    if problems:
        raise MessagesError(f"{path}: " + "; ".join(dict.fromkeys(problems)))
    return table


@lru_cache(maxsize=1)
def table() -> Table:
    return load()


def messages(gender: Gender | None = None) -> dict[str, dict[str, str]]:
    """Every spoken message as {key: {en, ur}}, in `gender` (the profile's by default)."""
    gender = gender or profile.current().gender
    return {
        key: {lang: forms[lang][gender] for lang in LANGUAGES}
        for key, forms in table().items()
        if not key.startswith(PROMPT_PREFIX)
    }


def sentence(key: str, language: str | None, gender: Gender) -> str:
    """Message `key` as written for a bot of `gender`, its values not yet filled in."""
    entries = table()
    if key not in entries:
        raise KeyError(f"no message called {key!r} in {PATH.name}")
    return entries[key]["ur" if language == "ur" else "en"][gender]


def say(key: str, language: str | None = "en", **fields: str) -> str:
    """Message `key` in `language` (English for anything but Urdu), in the bot's gender."""
    text = sentence(key, language, profile.current().gender)
    return text.format(**fields) if fields else text


def english(key: str) -> str:
    return table()[key]["en"][profile.current().gender]


def in_language(sentence: str, language: str | None) -> str:
    """One of the bot's own English sentences, in the customer's language.

    The English is the key, so a sentence that is not one of these -- a
    detail from the bank, say -- is said as it is rather than not at all.
    """
    if language != "ur":
        return sentence
    gender = profile.current().gender
    for forms in table().values():
        if sentence in forms["en"].values():
            return forms["ur"][gender]
    return sentence
