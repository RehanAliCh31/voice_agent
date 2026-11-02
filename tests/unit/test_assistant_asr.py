"""Recognition, language routing and commands: the decisions made on what was heard."""

from __future__ import annotations

import io
import wave

import httpx
import numpy as np
import pytest

from voicebot.assistant.asr_client import (
    AsrClient,
    AsrError,
    Transcript,
    is_phantom,
    normalise,
    to_wav,
)
from voicebot.assistant.commands import match_command
from voicebot.assistant.config import AsrEndpoint, load_assistant_config
from voicebot.assistant.language import (
    LanguageRouter,
    choose,
    has_devanagari,
    spoken_language,
    typed_language,
    urdu_share,
)

WHISPER = AsrEndpoint(
    url="http://ur.test/v1/audio/transcriptions",
    model="whisper-urdu",
    response_format="verbose_json",
)
QWEN = AsrEndpoint(url="http://en.test/v1/audio/transcriptions", model="qwen3-asr")


def _client(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- the WAV the recognisers are sent ---------------------------------------------


def test_the_wav_is_16k_mono_16_bit() -> None:
    data = to_wav(np.arange(1600, dtype=np.int16))
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (
            16000,
            1,
            2,
            1600,
        )


# --- the client -------------------------------------------------------------------


def test_verbose_json_gives_the_language_and_how_sure_whisper_was() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        assert b'name="model"' in body and b"whisper-urdu" in body and b"verbose_json" in body
        assert b'name="language"' not in body  # not told: it says which it heard
        return httpx.Response(
            200,
            json={
                "text": " آپ کا کارڈ بلاک ہے ",
                "language": "ur",
                "segments": [{"avg_logprob": -0.1}, {"avg_logprob": -0.3}],
            },
        )

    t = AsrClient(WHISPER, "ur", 5, 1, _client(handler)).transcribe(b"wav", None)
    assert t.text == "آپ کا کارڈ بلاک ہے" and t.language == "ur" and t.endpoint == "ur"
    assert t.avg_logprob == pytest.approx(-0.2)


def test_plain_json_has_text_and_nothing_else() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert b'name="language"' in request.content and b"en" in request.content
        return httpx.Response(200, json={"text": "Hello.", "usage": {}})

    t = AsrClient(QWEN, "en", 5, 1, _client(handler)).transcribe(b"wav", "en")
    assert (t.text, t.language, t.avg_logprob) == ("Hello.", None, None)


def test_a_timeout_is_retried_once() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"text": "ok"})

    assert AsrClient(QWEN, "en", 5, 1, _client(handler)).transcribe(b"w", "en").text == "ok"
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, content=b"not json"), "not JSON"),
        (httpx.Response(200, json={"no": "text"}), "no text"),
        (httpx.Response(503), "answered 503"),
    ],
)
def test_bad_answers_are_errors_after_the_retry(response: httpx.Response, message: str) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return response

    with pytest.raises(AsrError, match=message):
        AsrClient(QWEN, "en", 5, 1, _client(handler)).transcribe(b"w", "en")
    assert len(calls) == 2


def test_a_refused_request_is_not_retried() -> None:
    # A 400 is the request's fault and fails identically every time: sending
    # it again only makes the caller wait longer for the same answer.
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": "Invalid or unsupported audio file."})

    with pytest.raises(AsrError, match="answered 400") as raised:
        AsrClient(QWEN, "en", 5, 1, _client(handler)).transcribe(b"w", "en")
    assert len(calls) == 1 and not raised.value.retryable


# --- what the recognisers say about silence (measured) ------------------------------

PHANTOMS = ["i", "okay", "thank you", "ملتے ہیں"]


@pytest.mark.parametrize(
    ("text", "speech_ms", "logprob", "phantom"),
    [
        ("", 800, None, True),
        (" . ", 800, None, True),
        ("I.", 400, None, True),
        ("Okay.", 300, None, True),
        ("ملتے ہیں", 500, -0.3, True),
        ("Thank you.", 1200, None, False),  # long enough to have been said
        ("What is my balance?", 900, None, False),
        ("آپ کا بیلنس کیا ہے", 1000, -1.03, True),  # whisper itself doubts it
        ("آپ کا بیلنس کیا ہے", 1000, -0.1, False),
    ],
)
def test_phantoms(text: str, speech_ms: int, logprob: float | None, phantom: bool) -> None:
    t = Transcript(text, "ur", 0, avg_logprob=logprob)
    assert is_phantom(t, speech_ms, PHANTOMS, max_speech_ms=600, min_avg_logprob=-0.8) is phantom


def test_normalise_drops_urdu_punctuation_too() -> None:
    assert normalise("  Hello, THERE!  ") == "hello there"
    assert normalise("بس کرو۔") == "بس کرو"


# --- which language ----------------------------------------------------------------

EN_SPEECH = Transcript("Your request has been received.", "en", 300)
UR_SPEECH_BY_QWEN = Transcript("आपकी दरखास्त मौसूल हो गई हैं", "en", 300)  # measured
UR_SPEECH_BY_WHISPER = Transcript("آپ کی درخواست موصول ہو گئی ہے", "ur", 600, "ur", -0.08)
EN_SPEECH_BY_WHISPER = Transcript("Your request has been received", "ur", 200, "en", -0.21)


def test_script_shares() -> None:
    assert urdu_share("آپ کا ATM") == pytest.approx(4 / 7)  # four Urdu letters, three Latin
    assert urdu_share("123 ...") is None
    assert has_devanagari("आपकी") and not has_devanagari("آپ کی")


def test_the_language_of_some_text_is_its_script() -> None:
    assert spoken_language("میرا بیلنس کیا ہے؟") == "ur"
    assert spoken_language("What is my balance?") == "en"
    assert spoken_language("آپ کا ATM") == "ur"  # more Urdu letters than Latin
    # No letters at all: the conversation's language stands.
    assert spoken_language("$2,483.50", "ur") == "ur"


@pytest.mark.parametrize(
    ("typed", "language"),
    [
        # Urdu in Latin letters, the first two as typed into the page.
        ("kia ap muje sun skte hain?", "ur"),
        ("ap theak hain?", "ur"),
        ("mera checking balance kitna hai?", "ur"),
        ("savings account ka balance batao", "ur"),
        ("haan", "ur"),
        ("میرا بیلنس کیا ہے؟", "ur"),
        # The greeting, typed in Latin letters: it had no small words to count.
        ("assalam o alaikum", "ur"),
        ("Assalamualaikum!", "ur"),
        ("walaikum assalam, mera card block kar do", "ur"),
        # English with an Urdu courtesy in it is still English.
        ("tell me my balance, shukriya", "en"),
        # English, including English words Urdu also has.
        ("hey jarvis", "en"),
        ("Is the main branch open?", "en"),
        ("I want to pay for my Kia", "en"),
        ("What is my balance?", "en"),
    ],
)
def test_the_language_something_was_typed_in(typed: str, language: str) -> None:
    assert typed_language(typed) == language


@pytest.mark.parametrize("typed", ["$50", "yes", "Sara", "savings please"])
def test_a_word_or_two_keeps_the_conversations_language(typed: str) -> None:
    assert typed_language(typed, "ur") == "ur"
    assert typed_language(typed, "en") == "en"


def test_any_urdu_script_makes_it_urdu() -> None:
    assert typed_language("میرا balance") == "ur"
    assert typed_language("Thanks, شکریہ") == "ur"


def test_english_speech_goes_to_the_english_transcript() -> None:
    decision = choose(EN_SPEECH, EN_SPEECH_BY_WHISPER, "ur")
    assert (decision.language, decision.text) == ("en", EN_SPEECH.text)
    assert "English" in decision.reason


def test_urdu_speech_goes_to_whisper() -> None:
    decision = choose(UR_SPEECH_BY_QWEN, UR_SPEECH_BY_WHISPER, "en")
    assert (decision.language, decision.text) == ("ur", UR_SPEECH_BY_WHISPER.text)


def test_urdu_script_alone_decides_when_whisper_names_no_language() -> None:
    unnamed = Transcript(UR_SPEECH_BY_WHISPER.text, "ur", 600)
    assert choose(None, unnamed, "en").language == "ur"


def test_hindi_script_from_qwen_means_urdu_even_without_whisper() -> None:
    decision = choose(UR_SPEECH_BY_QWEN, None, "en")
    assert decision.language == "ur" and decision.text == UR_SPEECH_BY_QWEN.text


def test_when_only_whisper_heard_english_it_is_used() -> None:
    decision = choose(Transcript("", "en", 1), EN_SPEECH_BY_WHISPER, "ur")
    assert (decision.language, decision.text) == ("en", EN_SPEECH_BY_WHISPER.text)


def test_when_nobody_heard_anything_the_session_language_stands() -> None:
    decision = choose(Transcript("", "en", 1), Transcript("", "ur", 1), "ur")
    assert (decision.language, decision.text) == ("ur", "")


# What the two recognisers returned for the same clip, measured by
# scripts/benchmark.py: (qwen3-asr's text, whisper's text, whisper's verdict).
ENGLISH_THAT_WHISPER_CALLED_URDU = [
    # In Urdu letters, with English's small words spelled out.
    ("Can you show my transactions for last week?", "کین یو شو مائی ٹرانزیکشنز فور لاسٹ ویک"),
    ("Freeze the card ending 4417.", "فریز دہ کارڈ انڈنگ 4417"),
    ("Pay forty dollars for the internet bill.", "پی فورٹی دالرز فر دہ انترنیٹ بل"),
    ("Add Fatima Khan as a new pay.", "این عید فاطمہ کان ایز ای نیو پی"),
    ("Send the payment to Ayesha Siddiqui.", "سیند ای پیمنٹ ٹو عیشہ صدیقی"),
    ("Please check my balance. Shukria.", "پلیز چیک مائی بیلنس شکریہ"),
    # In Urdu letters, with none of Urdu's small words at all.
    ("Send two hundred and fifty dollars to Sarah.", "سینٹ 250 دولار تو سارا۔"),
    ("Transfer fifty dollars to Ali Raza.", "ٹرانسفیر ففتی دولار تو علی رازہ۔"),
    ("Is Osman Tarik on my list of pays?", "اس عثمان تاریخ عنمائل اسٹاپپیس"),
    # In English letters: called Urdu, written as English.
    ("Did Bilal Ahmed send me any money?", "Did Bilal Ahmed send me any money?"),
    # From a live test call: English words spelled as Urdu's own small words
    # -- "hear" as ہی, "me" as میں, "hey" as ہے -- and answered in Urdu.
    ("Hey, can you hear me?", "ہی، کن یو ہی مینگ؟"),
    ("Hey, can you hear me?", "ہے کہ یوں ہی میں؟"),
]

URDU_WITH_QWEN_GUESSING = [
    # qwen3-asr translated the Urdu into fluent English.
    ("I need to talk to someone.", "مجھے کسی انسان سے بات کرنی ہے۔"),
    ("I have saved my savings account balance.", "مرے سیونگز ایکاونٹ کا بیلنس کیا ہے۔"),
    ("My reference number is seven three nine two.", "میرا ریفرنس نمبر 7392 ہے۔"),
    ("Thank you. For today, just enough.", "شکریہ آج کے لیے بس اتنا ہی"),
    ("Fatma Khan will create a new household.", "فاطمہ خان کو نیا وصول کنندہ بنا دیں۔"),
    # Roman Urdu, and Hindi script.
    ("Kya Balal Ahmed ne, mujhe koi paise beje", "کیا بلال احمد نے مجھے کوئی پیسے بھیجے"),
    ("क्या मेरा बिजली का बलदा हो गया है?", "کیا میرا بجلی کا بل ادا ہو گیا ہے؟"),
    # Code-switched: English nouns, Urdu grammar.
    ("Please, Mary, last transaction, batay.", "پلیز میری لاسٹ ٹرانزیکشن بتائیں"),
    ("My checking account balance is high.", "میرا چیکنگ اکاؤنٹ بالنس کیا ہے۔"),
    # A goodbye with no grammar in it at all, from the held-out set.
    ("Very much, thank you, Huda Hafiz.", "بہت شکریہ خدا حافظ"),
    # Translated by qwen3-asr, with "or" and "me" in it: twins of اور and میں,
    # but کیا and ہے are Urdu all the same (from the live benchmark).
    (
        "Savings or checking account? Me, what's the difference?",
        "سیونگز اور چیکنگ اکاؤنٹ میں کیا فرق ہے؟",
    ),
]


@pytest.mark.parametrize(("qwen", "whisper"), ENGLISH_THAT_WHISPER_CALLED_URDU)
def test_english_whisper_called_urdu_is_english(qwen: str, whisper: str) -> None:
    decision = choose(Transcript(qwen, "en", 300), Transcript(whisper, "ur", 300, "ur"), "ur")
    assert (decision.language, decision.text) == ("en", qwen), decision.reason


@pytest.mark.parametrize(("qwen", "whisper"), URDU_WITH_QWEN_GUESSING)
def test_urdu_stays_urdu_whatever_qwen_wrote(qwen: str, whisper: str) -> None:
    decision = choose(Transcript(qwen, "en", 300), Transcript(whisper, "ur", 300, "ur"), "en")
    assert (decision.language, decision.text) == ("ur", whisper), decision.reason


def test_with_nothing_else_to_go_on_whispers_verdict_decides() -> None:
    # One word each, no grammar either way: the verdict is the only sign.
    for verdict in ("en", "ur"):
        decision = choose(Transcript("Okay.", "en", 1), Transcript("اوکے", "ur", 1, verdict), "en")
        assert decision.language == verdict


@pytest.mark.parametrize("conversation", ["en", "ur"])
def test_an_even_count_keeps_the_conversations_language(conversation: str) -> None:
    # Measured: "Thank you for your help", in an accent, came back with one
    # Urdu grammar word and one English word in Urdu letters. Uncertain, so
    # the language the conversation was already in stands.
    decision = choose(Transcript("Okay.", "en", 1), Transcript("اوکے", "ur", 1), conversation)  # type: ignore[arg-type]
    assert decision.language == conversation
    assert decision.text == ("اوکے" if conversation == "ur" else "Okay.")
    assert "even" in decision.reason


class _Fake:
    def __init__(self, result: Transcript | Exception) -> None:
        self.result = result
        self.languages: list[str | None] = []

    def transcribe(self, wav: bytes, language: str | None) -> Transcript:
        self.languages.append(language)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_auto_mode_asks_both_and_whisper_is_not_told_the_language() -> None:
    en, ur = _Fake(UR_SPEECH_BY_QWEN), _Fake(UR_SPEECH_BY_WHISPER)
    router = LanguageRouter("auto", en, ur)  # type: ignore[arg-type]
    decision = router.transcribe(b"wav", "en")
    router.close()
    assert decision.language == "ur" and en.languages == ["en"] and ur.languages == [None]


def test_auto_mode_carries_on_when_one_recogniser_is_down() -> None:
    router = LanguageRouter("auto", _Fake(AsrError("en down")), _Fake(UR_SPEECH_BY_WHISPER))  # type: ignore[arg-type]
    assert router.transcribe(b"wav", "en").language == "ur"


def test_auto_mode_fails_only_when_both_are_down() -> None:
    router = LanguageRouter("auto", _Fake(AsrError("en down")), _Fake(AsrError("ur down")))  # type: ignore[arg-type]
    with pytest.raises(AsrError, match="en down; ur down"):
        router.transcribe(b"wav", "en")


@pytest.mark.parametrize(("mode", "asked"), [("en", "en"), ("ur", "ur")])
def test_a_fixed_language_asks_only_its_own_recogniser(mode: str, asked: str) -> None:
    en, ur = _Fake(EN_SPEECH), _Fake(UR_SPEECH_BY_WHISPER)
    decision = LanguageRouter(mode, en, ur).transcribe(b"wav", "en")  # type: ignore[arg-type]
    assert decision.language == mode
    assert (en.languages, ur.languages) == (([asked], []) if mode == "en" else ([], [asked]))
    assert mode in decision.heard


# --- the commands --------------------------------------------------------------------

COMMANDS = load_assistant_config().session.commands


@pytest.mark.parametrize(
    ("said", "command"),
    [
        ("Stop.", "stop"),
        ("please stop", "stop"),
        ("Cancel!", "stop"),
        ("never mind", "stop"),
        ("Goodbye.", "goodbye"),
        ("ok bye", "goodbye"),
        ("thank you goodbye", "goodbye"),
        ("بس کرو۔", "stop"),
        ("رک جاؤ", "stop"),
        ("خدا حافظ", "goodbye"),
        ("جی اللہ حافظ", "goodbye"),
        ("بس شکریہ", "goodbye"),
        ("how do I stop a payment", None),
        ("stop my card", None),
        ("what is my balance", None),
        ("بس اتنا بتا دیں کہ میرا بیلنس کیا ہے", None),
        ("", None),
    ],
)
def test_commands_only_when_said_on_their_own(said: str, command: str | None) -> None:
    assert match_command(said, COMMANDS) == command
