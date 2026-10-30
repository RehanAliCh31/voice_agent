"""A reply before it is spoken: one language, and the bot's own gender."""

from __future__ import annotations

import pytest

from voicebot.reply_check import (
    in_own_gender,
    language_problems,
    latin_words,
    wrong_gender_forms,
)

# The bot speaking of itself, as a man, and the same as a woman.
FIRST_PERSON = [
    ("میں سمجھ نہیں سکا۔", "میں سمجھ نہیں سکی۔"),
    ("میں اسے استعمال نہیں کر سکا۔", "میں اسے استعمال نہیں کر سکی۔"),
    ("میں آپ کو ایک ساتھی سے ملا رہا ہوں۔", "میں آپ کو ایک ساتھی سے ملا رہی ہوں۔"),
    ("میں اسے دیکھنا چاہتا ہوں۔", "میں اسے دیکھنا چاہتی ہوں۔"),
    ("میں ابھی یہ چیک نہیں کر پا رہا۔", "میں ابھی یہ چیک نہیں کر پا رہی۔"),
    ("میں اس میں مدد نہیں کر سکتا۔", "میں اس میں مدد نہیں کر سکتی۔"),
    ("آپ کو بتاتا چلوں کہ وقت کم ہے۔", "آپ کو بتاتی چلوں کہ وقت کم ہے۔"),
    ("میں آپ کا بیلنس چیک کروں گا۔", "میں آپ کا بیلنس چیک کروں گی۔"),
    ("میں آپ کی مدد کر سکتا ہوں۔", "میں آپ کی مدد کر سکتی ہوں۔"),
    ("میں یہ کام کر چکا ہوں۔", "میں یہ کام کر چکی ہوں۔"),
]

# Not the bot speaking of itself: a thing, the customer, or no gender at all.
NOT_FIRST_PERSON = [
    "معذرت، اس وقت سروس سے رابطہ نہیں ہو پا رہا۔",
    "ٹرانسفر نہیں ہو سکا۔",
    "جواب نہیں مل سکا۔",
    "مجھے آپ کی آواز سننے میں دشواری ہو رہی ہے۔",
    "میں نے اسے منسوخ کر دیا ہے۔",
    "آپ کی ادائیگی ہو جائے گی۔",
    "کیا آپ دوبارہ کوشش کریں گے؟",
    "کیا آپ دوبارہ کہہ سکتے ہیں؟",
    "آپ کے سیونگز اکاؤنٹ میں $19,020.00 ہیں۔",
]


@pytest.mark.parametrize(("male", "female"), FIRST_PERSON)
def test_the_bots_own_verbs_are_put_in_its_gender(male: str, female: str) -> None:
    assert in_own_gender(male, "female") == (female, 1)
    assert in_own_gender(female, "male") == (male, 1)
    assert wrong_gender_forms(male, "female") == 1
    assert wrong_gender_forms(male, "male") == 0


@pytest.mark.parametrize(("male", "female"), FIRST_PERSON)
def test_a_reply_already_in_its_gender_is_left_alone(male: str, female: str) -> None:
    assert in_own_gender(female, "female") == (female, 0)
    assert in_own_gender(male, "male") == (male, 0)


@pytest.mark.parametrize("text", NOT_FIRST_PERSON)
def test_nothing_but_the_bot_speaking_of_itself_is_touched(text: str) -> None:
    for gender in ("female", "male"):
        assert in_own_gender(text, gender) == (text, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "language", "problem"),
    [
        # Measured: an Urdu turn answered in Latin letters, for the English voice.
        ("Checking account mein $2,483.50 hain.", "ur", "English letters"),
        ("آپ کے checking اکاؤنٹ میں $2,483.50 ہیں۔", "ur", "1 word(s) in English letters"),
        ("Your balance is $5. شکریہ", "en", "Urdu script in an English reply"),
        ("आपका बैलेंस $5 है", "ur", "Hindi script"),
    ],
)
def test_a_reply_in_the_wrong_language_or_two_is_caught(
    text: str, language: str, problem: str
) -> None:
    assert any(problem in p for p in language_problems(text, language))


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("آپ کے سیونگز اکاؤنٹ میں $19,020.00 ہیں۔", "ur"),
        ("اپنا PIN درج کریں، یا ATM استعمال کریں۔", "ur"),  # said as letters in Urdu too
        ("Your savings balance is $19,020.00.", "en"),
        ("$5", "en"),
    ],
)
def test_a_reply_in_one_language_passes(text: str, language: str) -> None:
    assert language_problems(text, language) == []


def test_names_from_the_banks_records_may_stay_in_english_letters() -> None:
    result = {"transactions": [{"merchant": "Netflix", "account": "checking"}]}
    allowed = latin_words([result, '{"payee": "Sara"}'])
    assert allowed == {"netflix", "sara"}  # names only: "checking" is said in Urdu
    assert language_problems("آپ کا Netflix کا بل $15.99 ہے۔", "ur", allowed) == []
    assert language_problems("آپ کا checking بیلنس", "ur", allowed)
