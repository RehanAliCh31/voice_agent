"""The bot's own sentences: one file, both languages, both genders, checked as it loads."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from voicebot import messages, profile
from voicebot.llm import prompts
from voicebot.profile import GENDERS, AgentProfile
from voicebot.reply_check import wrong_gender_forms


@pytest.fixture
def as_male() -> Iterator[None]:
    previous = profile.use(AgentProfile(gender="male", names={"en": "Omar", "ur": "عمر"}))
    yield
    profile.use(previous)


def test_every_message_has_both_languages_in_both_genders() -> None:
    table = messages.load()
    assert table, "the shipped file must load"
    for key, languages in table.items():
        for language in ("en", "ur"):
            for gender in GENDERS:
                assert languages[language][gender].strip(), (key, language, gender)


def test_no_urdu_sentence_speaks_of_the_bot_in_the_wrong_gender() -> None:
    for key, languages in messages.load().items():
        for gender in GENDERS:
            assert not wrong_gender_forms(languages["ur"][gender], gender), (key, gender)


def test_the_profiles_gender_chooses_the_sentence(as_male: None) -> None:
    assert messages.say("not_caught", "ur") == "میں سمجھ نہیں سکا۔ کیا آپ دوبارہ کہہ سکتے ہیں؟"
    assert messages.sentence("not_caught", "ur", "female").startswith("میں سمجھ نہیں سکی")
    # English is the same for both.
    assert messages.sentence("not_caught", "en", "male") == messages.sentence(
        "not_caught", "en", "female"
    )


def test_the_page_is_served_the_profiles_sentences_and_no_prompts(as_male: None) -> None:
    served = messages.messages()
    assert served["busy"]["ur"] == messages.sentence("busy", "ur", "male")
    assert not any(key.startswith(messages.PROMPT_PREFIX) for key in served)


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
    for key, languages in messages.load().items():
        if key.startswith(messages.PROMPT_PREFIX):
            continue
        assert not any(word in languages["en"]["female"].lower() for word in words), key


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "messages.yaml"
    path.write_text(text, encoding="utf-8")
    return path


REFERENCES = (
    "handed_over: {en: 'ref {reference}', ur: 'ref {reference}'}\n"
    "escalated: {en: 'ref {reference}', ur: 'ref {reference}'}\n"
)


def test_a_missing_language_stops_the_load(tmp_path: Path) -> None:
    path = _write(tmp_path, REFERENCES + "unavailable: {en: 'Sorry'}\n")
    with pytest.raises(messages.MessagesError, match="unavailable: no ur sentence"):
        messages.load(path)


def test_a_missing_reference_stops_the_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "handed_over: {en: 'ref {reference}', ur: 'no reference here'}\n"
        "escalated: {en: 'ref {reference}', ur: 'ref {reference}'}\n",
    )
    with pytest.raises(messages.MessagesError, match=r"handed_over \(ur, female\)"):
        messages.load(path)


def test_an_urdu_sentence_given_once_must_not_speak_as_one_gender(tmp_path: Path) -> None:
    path = _write(tmp_path, REFERENCES + "busy: {en: 'One moment', ur: 'میں دیکھ رہا ہوں'}\n")
    with pytest.raises(messages.MessagesError, match="give female and male"):
        messages.load(path)


def test_a_gendered_form_in_the_other_gender_stops_the_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        REFERENCES
        + "busy:\n  en: One moment\n  ur:\n"
        + "    female: میں دیکھ رہا ہوں\n    male: میں دیکھ رہا ہوں\n",
    )
    with pytest.raises(messages.MessagesError, match=r"busy \(ur, female\): speaks of the bot"):
        messages.load(path)


def test_a_gendered_sentence_needs_both_genders(tmp_path: Path) -> None:
    path = _write(
        tmp_path, REFERENCES + "busy:\n  en: One moment\n  ur:\n    female: میں دیکھ رہی ہوں\n"
    )
    with pytest.raises(messages.MessagesError, match="exactly female and male"):
        messages.load(path)


def test_an_unreadable_file_stops_the_load(tmp_path: Path) -> None:
    with pytest.raises(messages.MessagesError, match="cannot read"):
        messages.load(tmp_path / "missing.yaml")


def test_an_unknown_message_is_a_bug_not_silence() -> None:
    with pytest.raises(KeyError):
        messages.say("no_such_message")
