"""The bot's fallback sentences: one file, both languages, checked as it loads."""

from __future__ import annotations

from pathlib import Path

import pytest

from voicebot import messages
from voicebot.llm import prompts


def test_every_message_has_both_languages() -> None:
    table = messages.load()
    assert table, "the shipped file must load"
    for key, sentences in table.items():
        assert sentences["en"].strip() and sentences["ur"].strip(), key


def test_the_names_the_code_uses_come_from_the_file() -> None:
    assert prompts.UNAVAILABLE == messages.english("unavailable")
    assert prompts.in_language(prompts.UNAVAILABLE, "ur") == messages.say("unavailable", "ur")
    assert prompts.in_language("a detail from the bank", "ur") == "a detail from the bank"


def test_a_reference_is_filled_in_in_either_language() -> None:
    assert "H-1234" in messages.say("escalated", "en", reference="H-1234")
    assert "H-1234" in messages.say("escalated", "ur", reference="H-1234")


def test_no_fallback_is_technical() -> None:
    # What the customer hears says what to do next, never what broke inside.
    words = ("exception", "error", "http", "timeout", "500", "traceback", "database")
    for key, sentences in messages.load().items():
        assert not any(word in sentences["en"].lower() for word in words), key


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "messages.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_missing_language_stops_the_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "handed_over: {en: 'ref {reference}', ur: 'ref {reference}'}\n"
        "escalated: {en: 'ref {reference}', ur: 'ref {reference}'}\n"
        "unavailable: {en: 'Sorry'}\n",
    )
    with pytest.raises(messages.MessagesError, match="unavailable: no ur sentence"):
        messages.load(path)


def test_a_missing_reference_stops_the_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "handed_over: {en: 'ref {reference}', ur: 'no reference here'}\n"
        "escalated: {en: 'ref {reference}', ur: 'ref {reference}'}\n",
    )
    with pytest.raises(messages.MessagesError, match=r"handed_over \(ur\)"):
        messages.load(path)


def test_an_unreadable_file_stops_the_load(tmp_path: Path) -> None:
    with pytest.raises(messages.MessagesError, match="cannot read"):
        messages.load(tmp_path / "missing.yaml")


def test_an_unknown_message_is_a_bug_not_silence() -> None:
    with pytest.raises(KeyError):
        messages.say("no_such_message")
