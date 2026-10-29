"""Text preparation: what the voices are given to read.

Both voices read digits wrongly (measured), so every number reaching them must
already be words. Every Urdu string here is in logical order, as typed.
"""

from __future__ import annotations

import pytest

from voicebot.assistant.text_prep import (
    SentenceStream,
    english_int,
    english_ordinal,
    numbers_to_words,
    prepare_for_speech,
    split_sentences,
    strip_foreign_script,
    strip_markup,
    urdu_int,
    urdu_number,
)

# --- English numbers -----------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (7, "seven"),
        (19, "nineteen"),
        (40, "forty"),
        (83, "eighty-three"),
        (100, "one hundred"),
        (2483, "two thousand four hundred eighty-three"),
        (1_250_000, "one million two hundred fifty thousand"),
        (-5, "minus five"),
    ],
)
def test_english_integers(n: int, words: str) -> None:
    assert english_int(n) == words


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (1, "first"),
        (2, "second"),
        (12, "twelfth"),
        (20, "twentieth"),
        (22, "twenty-second"),
        (100, "one hundredth"),
    ],
)
def test_english_ordinals(n: int, words: str) -> None:
    assert english_ordinal(n) == words


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("$2,483.50", "two thousand four hundred eighty-three dollars and fifty cents"),
        ("$1", "one dollar"),
        ("$0.01", "zero dollars and one cent"),
        ("Rs. 5,000", "five thousand rupees"),
        ("12.5%", "twelve point five percent"),
        ("at 10:30", "at ten thirty"),
        ("at 9:05", "at nine oh five"),
        ("at 3:00", "at three o'clock"),
        ("the card ending in 4417", "the card ending in four four one seven"),
        ("call 0300-1234567", "call zero three zero zero, one two three four five six seven"),
        ("account 1234567890", "account one two three four five six seven eight nine zero"),
        ("1250000 rupees", "one million two hundred fifty thousand rupees"),
        ("Rs 12,50,000", "one million two hundred fifty thousand rupees"),
        ("on the 22nd", "on the twenty-second"),
        ("expires 12/28", "expires twelve twenty-eight"),
        ("2.05 litres", "two point zero five litres"),
    ],
)
def test_english_numbers_become_words(text: str, spoken: str) -> None:
    assert numbers_to_words(text, "en") == spoken


# --- Urdu numbers --------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "صفر"),
        (9, "نو"),
        (19, "انیس"),
        (41, "اکتالیس"),
        (83, "تراسی"),
        (99, "ننانوے"),
        (100, "ایک سو"),
        (999, "نو سو ننانوے"),
        (2483, "دو ہزار چار سو تراسی"),
        (100_000, "ایک لاکھ"),
        (1_250_000, "بارہ لاکھ پچاس ہزار"),
        (10_000_000, "ایک کروڑ"),
        (3_000_000_007, "تین ارب سات"),
    ],
)
def test_urdu_integers(n: int, words: str) -> None:
    assert urdu_int(n) == words


def test_every_urdu_number_below_a_hundred_is_one_word_in_urdu_script() -> None:
    for n in range(100):
        word = urdu_int(n)
        assert " " not in word, n
        assert all("؀" <= c <= "ۿ" for c in word), n


def test_urdu_number_words_read_back_as_the_number() -> None:
    # The guard reads Urdu amounts with this, so it must read whatever the
    # bot itself would say.
    for n in [*range(2000), 19_020, 100_000, 1_250_000, 12_345_678, 3_000_000_007]:
        assert urdu_number(urdu_int(n)) == n, n
    assert urdu_number("سو") == 100


@pytest.mark.parametrize("words", ["", "دو ہزار ڈالر", "سوال"])
def test_words_that_are_not_a_number_read_as_none(words: str) -> None:
    assert urdu_number(words) is None


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("آپ کا بیلنس 2483 روپے ہے۔", "آپ کا بیلنس دو ہزار چار سو تراسی روپے ہے۔"),
        ("آپ کا بیلنس ۲۴۸۳ روپے ہے۔", "آپ کا بیلنس دو ہزار چار سو تراسی روپے ہے۔"),
        ("بیلنس 12,50,000 روپے", "بیلنس بارہ لاکھ پچاس ہزار روپے"),
        ("بیلنس ۱۲،۵۰،۰۰۰ روپے", "بیلنس بارہ لاکھ پچاس ہزار روپے"),
        ("بیلنس 1250000 روپے", "بیلنس بارہ لاکھ پچاس ہزار روپے"),
        ("$50 جمع ہوئے", "پچاس ڈالر جمع ہوئے"),
        ("شرح 12.5% ہے", "شرح بارہ اعشاریہ پانچ فیصد ہے"),
        ("وقت 10:30 ہے", "وقت دس بج کر تیس منٹ ہے"),
        ("وقت 4:00 ہے", "وقت چار بجے ہے"),
        ("کال 0300-1234567", "کال صفر تین صفر صفر، ایک دو تین چار پانچ چھ سات"),
    ],
)
def test_urdu_numbers_become_words(text: str, spoken: str) -> None:
    assert numbers_to_words(text, "ur") == spoken


# --- what is not speech --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("<think>let me see</think>Hello there.", "Hello there."),
        ("Hello.</think> Hello.", "Hello. Hello."),
        ("**Sure!** It is *easy*.", "Sure! It is easy."),
        ("See [our app](https://example.com/app) now.", "See our app now."),
        ("Visit https://example.com today 😊", "Visit today"),
        ("## Steps\n- Open the app\n- Tap Cards", "Steps. Open the app. Tap Cards."),
        ("1. First\n2. Second", "First. Second."),
        ("Use `code` here.", "Use code here."),
    ],
)
def test_markup_comes_out(text: str, spoken: str) -> None:
    assert strip_markup(text) == spoken


def test_urdu_list_lines_end_with_the_urdu_full_stop() -> None:
    assert strip_markup("- ایپ کھولیں\n- کارڈ دبائیں", "ur") == "ایپ کھولیں۔ کارڈ دبائیں۔"


def test_a_script_neither_voice_reads_is_removed_and_counted() -> None:
    text, dropped = strip_foreign_script("میرے پاس آپ کی ذاتی בנק کی معلومات نہیں")
    assert "בנק" not in text and dropped == 3
    assert text.startswith("میرے پاس")


def test_urdu_and_its_diacritics_and_latin_are_kept() -> None:
    text = "بَینک ATM café ۔"
    assert strip_foreign_script(text) == (text, 0)


def test_nothing_speakable_left_is_empty() -> None:
    assert prepare_for_speech("... 😊 **", "en") == ""
    assert prepare_for_speech("<think>only thinking</think>", "en") == ""


def test_speech_preparation_does_all_three() -> None:
    assert (
        prepare_for_speech("**Balance:** $5. See https://x.y", "en") == "Balance: five dollars. See"
    )


# --- sentences -----------------------------------------------------------------


def test_english_sentences_split_without_breaking_decimals_or_titles() -> None:
    assert split_sentences(
        "Hello Mr. Khan. Your rate is 2.5 percent! Is that ok? Yes... fine."
    ) == [
        "Hello Mr. Khan.",
        "Your rate is 2.5 percent!",
        "Is that ok?",
        "Yes...",
        "fine.",
    ]


def test_urdu_sentences_split_on_the_urdu_full_stop_and_question_mark() -> None:
    assert split_sentences(
        "آپ کا کارڈ بلاک کر دیا گیا ہے۔ کیا آپ کچھ اور جاننا چاہتے ہیں؟ شکریہ"
    ) == [
        "آپ کا کارڈ بلاک کر دیا گیا ہے۔",
        "کیا آپ کچھ اور جاننا چاہتے ہیں؟",
        "شکریہ",
    ]


def test_a_new_line_ends_a_sentence() -> None:
    assert split_sentences("First line\nSecond line") == ["First line", "Second line"]


def test_streamed_text_waits_to_know_a_full_stop_is_not_a_decimal_point() -> None:
    stream = SentenceStream()
    assert stream.feed("Hel") == []
    assert stream.feed("lo. It costs $2") == ["Hello."]
    assert stream.feed(".") == []
    assert stream.feed("50 now") == []
    assert stream.feed(". Next") == ["It costs $2.50 now."]
    assert stream.flush() == ["Next"]


def test_streamed_urdu_ends_a_sentence_at_once() -> None:
    stream = SentenceStream()
    assert stream.feed("آپ کا کارڈ بلاک ہے۔") == ["آپ کا کارڈ بلاک ہے۔"]


def test_a_sentence_that_runs_on_is_cut_at_a_comma() -> None:
    stream = SentenceStream(max_chars=40)
    out = stream.feed("This is a long opening clause, and it keeps going without stopping")
    assert out == ["This is a long opening clause,"]


def test_punctuation_alone_is_not_a_sentence() -> None:
    assert split_sentences("... !!") == []


def test_a_time_at_the_end_of_a_sentence_still_ends_it() -> None:
    # Measured: "p.m. However" did not split, and the first thing the voice
    # was given to read was two long sentences -- 8.6 s before it spoke.
    assert split_sentences("Open from 9 a.m. to 1 p.m. However, it varies.") == [
        "Open from 9 a.m. to 1 p.m.",
        "However, it varies.",
    ]


def test_a_title_before_a_name_never_ends_a_sentence() -> None:
    assert split_sentences("Ask Dr. Khan. He knows.") == ["Ask Dr. Khan.", "He knows."]


def test_the_first_chunk_goes_at_the_first_clause_break() -> None:
    stream = SentenceStream(first_words=8)
    assert stream.feed("Our branches are open, from Monday to Friday, ") == [
        "Our branches are open,"
    ]
    assert stream.feed("and on Saturdays too. Call us.") == [
        "from Monday to Friday, and on Saturdays too."
    ]


def test_the_first_chunk_goes_after_a_few_words_when_there_is_no_break() -> None:
    stream = SentenceStream(first_words=6)
    text = "بینک عام طور پر پیر سے جمعہ صبح دس بجے سے دوپہر چار بجے تک کھلا رہتا ہے۔"
    assert stream.feed(text) == [
        "بینک عام طور پر پیر سے",
        "جمعہ صبح دس بجے سے دوپہر چار بجے تک کھلا رہتا ہے۔",
    ]


def test_early_pieces_grow_so_each_is_ready_before_the_last_one_ends() -> None:
    stream = SentenceStream(first_words=4, growth=2)
    words = " ".join(f"w{n}" for n in range(40)) + " "
    pieces = [len(p.split()) for p in stream.feed(words)]
    # 4, 8 and 16 words; a fourth would be 32, more than an early piece may
    # be, so the rest waits for the end of its sentence.
    assert pieces == [4, 8, 16]
    assert [len(p.split()) for p in stream.flush()] == [12]


def test_without_growth_only_the_first_piece_is_cut_early() -> None:
    stream = SentenceStream(first_words=3)
    pieces = stream.feed("one two three four five six seven eight. ")
    assert pieces == ["one two three", "four five six seven eight."]


def test_a_short_reply_is_not_cut_early() -> None:
    stream = SentenceStream(first_words=8)
    assert stream.feed("Yes, of course. ") == ["Yes, of course."]


def test_numbers_with_commas_are_not_clause_breaks() -> None:
    stream = SentenceStream(first_words=20)
    assert stream.feed("You have 2,483 dollars in it ") == []
