"""Turning a reply into something a voice can say.

Three jobs. Take out what is not speech: reasoning, markdown, links, emoji,
and letters of a script neither voice reads (the model once wrote the Hebrew
"בנק" inside an Urdu sentence). Write numbers as words, because both voices
read digits wrongly -- measured: "4417" came back as "four hundred forty-one
seven", and Urdu digits as gibberish, while the same numbers spelled out
round-tripped exactly. And cut the reply into sentences as it streams in, so
the first one is spoken while the rest is still being written.

Urdu text is never reordered, reversed or reshaped here: it is stored and
sent in logical order, which is what the voice expects.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from voicebot.assistant.config import Language

# --- what is not speech --------------------------------------------------------

_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)
_STRAY_THINK = re.compile(r"</?think>")
_CODE_FENCE = re.compile(r"```.*?(?:```|$)", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"\S+@\S+\.\w+")
_EMPHASIS = re.compile(r"(\*\*|__|\*)(?=\S)(.+?)(?<=\S)\1")
_LINE_PREFIX = re.compile(r"^\s*(?:#{1,6}\s+|>\s?|[-*•▪◦]\s+|\d{1,2}[.)]\s+)")
_EMOJI = re.compile("[\U0001f000-\U0001faff\U00002600-\U000027bf\U00002b00-\U00002bff️‍⃣]")
_ARABIC_SCRIPT = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
_TERMINAL = {"en": ".", "ur": "۔"}
_ENDS_CLAUSE = tuple(".?!۔؟:;,،")


def _readable(char: str) -> bool:
    """Not a letter or mark at all, or one of Latin or the Arabic script."""
    if unicodedata.category(char)[0] not in "LM":
        return True
    code = ord(char)
    if code < 0x250:  # Latin, with its accented letters
        return True
    return any(low <= code <= high for low, high in _ARABIC_SCRIPT)


def strip_foreign_script(text: str) -> tuple[str, int]:
    """Letters (and their marks) neither voice can read removed, and how many."""
    kept = [c for c in text if _readable(c)]
    return "".join(kept), len(text) - len(kept)


def strip_markup(text: str, language: Language = "en") -> str:
    """Reasoning, markdown, links and emoji out; lines joined into sentences."""
    text = _THINK.sub(" ", text)
    text = _STRAY_THINK.sub(" ", text)
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _EMOJI.sub("", text)
    lines = [_LINE_PREFIX.sub("", raw).replace("|", ", ").strip(" ,") for raw in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) > 1:
        # A list item has no full stop of its own, and run together with the
        # next one it would be read as one long breathless sentence.
        lines = [
            line if line.endswith(_ENDS_CLAUSE) else line + _TERMINAL[language] for line in lines
        ]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


# --- numbers -------------------------------------------------------------------

_EN_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_EN_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_EN_ORDINAL = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}

# Urdu has a word of its own for every number to ninety-nine: there is no rule
# that builds "eighty-three" from "eighty" and "three", so they are listed.
_UR_0_99 = (
    "صفر ایک دو تین چار پانچ چھ سات آٹھ نو "
    "دس گیارہ بارہ تیرہ چودہ پندرہ سولہ سترہ اٹھارہ انیس "
    "بیس اکیس بائیس تئیس چوبیس پچیس چھبیس ستائیس اٹھائیس انتیس "
    "تیس اکتیس بتیس تینتیس چونتیس پینتیس چھتیس سینتیس اڑتیس انتالیس "
    "چالیس اکتالیس بیالیس تینتالیس چوالیس پینتالیس چھیالیس سینتالیس اڑتالیس انچاس "
    "پچاس اکیاون باون ترپن چون پچپن چھپن ستاون اٹھاون انسٹھ "
    "ساٹھ اکسٹھ باسٹھ ترسٹھ چونسٹھ پینسٹھ چھیاسٹھ سڑسٹھ اڑسٹھ انہتر "
    "ستر اکہتر بہتر تہتر چوہتر پچھتر چھہتر ستتر اٹھہتر اناسی "
    "اسی اکیاسی بیاسی تراسی چوراسی پچاسی چھیاسی ستاسی اٹھاسی نواسی "
    "نوے اکانوے بانوے ترانوے چورانوے پچانوے چھیانوے ستانوے اٹھانوے ننانوے"
).split()
assert len(_UR_0_99) == 100
# The South Asian scale: lakh is a hundred thousand, crore a hundred lakh.
_UR_SCALES = ((10**9, "ارب"), (10**7, "کروڑ"), (10**5, "لاکھ"), (1000, "ہزار"), (100, "سو"))
_UR_POINT = "اعشاریہ"
_UR_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def english_int(n: int) -> str:
    if n < 0:
        return "minus " + english_int(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _EN_TENS[tens] + (f"-{_EN_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_EN_ONES[hundreds]} hundred" + (f" {english_int(rest)}" if rest else "")
    for scale, name in _EN_SCALES:
        if n >= scale:
            high, rest = divmod(n, scale)
            return f"{english_int(high)} {name}" + (f" {english_int(rest)}" if rest else "")
    raise AssertionError("unreachable")


def english_ordinal(n: int) -> str:
    words = english_int(n)
    head, sep, last = words.rpartition("-" if "-" in words.split()[-1] else " ")
    if last in _EN_ORDINAL:
        last = _EN_ORDINAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head}{sep}{last}"


def urdu_int(n: int) -> str:
    if n < 0:
        return "منفی " + urdu_int(-n)
    if n < 100:
        return _UR_0_99[n]
    parts = []
    for scale, name in _UR_SCALES:
        if n >= scale:
            count, n = divmod(n, scale)
            # Only the scales above crore can hold more than ninety-nine.
            parts.append(f"{urdu_int(count)} {name}")
    if n:
        parts.append(_UR_0_99[n])
    return " ".join(parts)


_UR_VALUES = {word: n for n, word in enumerate(_UR_0_99)}
_UR_SCALE_VALUES = {name: scale for scale, name in _UR_SCALES}
# Every word an Urdu number is made of, for anything that has to find one.
URDU_NUMBER_WORDS = frozenset(_UR_VALUES) | frozenset(_UR_SCALE_VALUES)


def urdu_number(words: str) -> int | None:
    """The number some Urdu words say -- "دو ہزار چار سو تراسی" is 2483.

    None when any of the words is not part of a number.
    """
    total = current = 0
    seen = False
    for word in words.split():
        if word in _UR_VALUES:
            current += _UR_VALUES[word]
        elif word == "سو":
            current = (current or 1) * 100
        elif word in _UR_SCALE_VALUES:
            total += (current or 1) * _UR_SCALE_VALUES[word]
            current = 0
        else:
            return None
        seen = True
    return total + current if seen else None


_EN_MONTHS = (
    "January February March April May June July August September October November December"
).split()
_UR_MONTHS = "جنوری فروری مارچ اپریل مئی جون جولائی اگست ستمبر اکتوبر نومبر دسمبر".split()


def _digits(text: str, words: Callable[[int], str], separator: str = " ") -> str:
    return separator.join(words(int(d)) for d in text if d.isdigit())


def _to_int(text: str) -> int:
    return int(text.replace(",", ""))


class _Numbers:
    """The number patterns of one language, applied in a fixed order."""

    def __init__(self, language: Language) -> None:
        self.words = english_int if language == "en" else urdu_int
        self.language = language

    def _amount(self, whole: str, fraction: str | None) -> str:
        said = self.words(_to_int(whole))
        if fraction and int(fraction):
            point = "point" if self.language == "en" else _UR_POINT
            said += f" {point} " + _digits(fraction, self.words)
        return said

    def _money(self, match: re.Match[str], unit: str, cents_unit: str | None) -> str:
        whole, fraction = match.group("whole"), match.group("fraction")
        amount = _to_int(whole)
        if self.language == "en":
            name = unit if amount == 1 else unit + "s"
            said = f"{english_int(amount)} {name}"
            if fraction and cents_unit and int(fraction.ljust(2, "0")):
                cents = int(fraction.ljust(2, "0"))
                said += f" and {english_int(cents)} {cents_unit}{'' if cents == 1 else 's'}"
            return said
        said = f"{urdu_int(amount)} {unit}"
        if fraction and cents_unit and int(fraction.ljust(2, "0")):
            said += f" اور {urdu_int(int(fraction.ljust(2, '0')))} {cents_unit}"
        return said

    def _time(self, match: re.Match[str]) -> str:
        hours, minutes = int(match.group(1)), int(match.group(2))
        if self.language == "en":
            if minutes == 0:
                return f"{english_int(hours)} o'clock"
            if minutes < 10:
                return f"{english_int(hours)} oh {english_int(minutes)}"
            return f"{english_int(hours)} {english_int(minutes)}"
        if minutes == 0:
            return f"{urdu_int(hours)} بجے"
        return f"{urdu_int(hours)} بج کر {urdu_int(minutes)} منٹ"

    def _date(self, day: int, month: int, year: int, written: str) -> str:
        """A date as it is said: "the twelfth of October, two thousand twenty-six",
        "بارہ اکتوبر دو ہزار چھبیس". Anything that is not a real date is left as written."""
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return written
        if self.language == "en":
            return f"the {english_ordinal(day)} of {_EN_MONTHS[month - 1]}, {english_int(year)}"
        return f"{urdu_int(day)} {_UR_MONTHS[month - 1]} {urdu_int(year)}"

    def _code(self, match: re.Match[str]) -> str:
        """A reference like H-7392: its letters, then its digits one at a time."""
        pause = ", " if self.language == "en" else "، "
        return f"{' '.join(match.group(1))}{pause}{_digits(match.group(2), self.words)}"

    def apply(self, text: str) -> str:
        en = self.language == "en"
        number = rf"(?P<whole>{_GROUPED}|\d+)(?:\.(?P<fraction>\d{{1,2}}))?"
        text = re.sub(
            rf"\$\s?{number}",
            lambda m: self._money(m, "dollar" if en else "ڈالر", "cent" if en else "سینٹ"),
            text,
        )
        text = re.sub(
            rf"(?:\bRs\.?|\bPKR)\s?{number}",
            lambda m: self._money(m, "rupee" if en else "روپے", None),
            text,
        )
        text = re.sub(
            r"(\d+(?:\.\d+)?)\s?%",
            lambda m: self._amount(*m.group(1).partition(".")[::2])
            + (" percent" if en else " فیصد"),
            text,
        )
        # Dates before anything else reads their parts as numbers: day, month
        # and year, written day first as in Pakistan, or year first (ISO).
        text = re.sub(
            r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
            lambda m: self._date(int(m.group(3)), int(m.group(2)), int(m.group(1)), m.group(0)),
            text,
        )
        text = re.sub(
            r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
            lambda m: self._date(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0)),
            text,
        )
        # A reference (H-7392, D-1043) is read out to be written down.
        text = re.sub(r"\b([A-Z]{1,3})-(\d{3,})\b", self._code, text)
        # In Urdu, "بجے" after a time is part of what _time says already.
        time = r"\b(\d{1,2}):(\d{2})\b" + ("" if en else r"(?:\s*بجے)?")
        text = re.sub(time, self._time, text)
        text = re.sub(r"\+(?=\s?\d)", "plus " if en else "پلس ", text)
        # Phone, card and account numbers are read a digit at a time: in
        # groups when written in groups, and wherever seven or more digits run
        # together, or a number follows "ending in".
        pause = ", " if en else "، "
        text = re.sub(
            r"\b\d{2,}(?:[-\s]\d{2,})+\b",
            lambda m: pause.join(_digits(g, self.words) for g in re.split(r"[-\s]", m.group(0))),
            text,
        )
        text = re.sub(
            rf"\b\d{{7,}}\b(?!\s*{_CURRENCY_WORD})",
            lambda m: _digits(m.group(0), self.words),
            text,
        )
        if not en:
            # A card's last digits, read one at a time as in English: "آخری
            # ہندسے 4417", "آخر میں 4417", "4417 والا کارڈ". Not after "آخری"
            # alone: "آخری 10 ٹرانزیکشنز" is ten transactions.
            text = re.sub(
                r"((?:آخر\s+میں|ختم\s+ہونے\s+والے|ہندسے|ہندسوں)[^\d۔؟!]{0,25}?)(\d{2,6})\b",
                lambda m: f"{m.group(1)}{_digits(m.group(2), self.words)}",
                text,
            )
            text = re.sub(
                r"\b(\d{2,6})(?=\s+(?:والا|والے|والی)\s+کارڈ)",
                lambda m: _digits(m.group(1), self.words),
                text,
            )
        if en:
            text = re.sub(
                r"(?i)\b(ending(?: in| with)?|last (?:four|4)(?: digits)?(?: are| of)?)"
                r"\s+(\d{2,6})\b",
                lambda m: f"{m.group(1)} {_digits(m.group(2), self.words)}",
                text,
            )
            text = re.sub(
                r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: english_ordinal(int(m.group(1))), text
            )
            text = re.sub(
                r"\b(\d{1,2})/(\d{2,4})\b",
                lambda m: f"{english_int(int(m.group(1)))} {english_int(int(m.group(2)))}",
                text,
            )
        text = re.sub(
            rf"(?:{_GROUPED})(?:\.\d+)?|\d+\.\d+|\d+",
            lambda m: self._amount(*m.group(0).partition(".")[::2]),
            text,
        )
        return text


# Thousands grouped the international way (1,250,000) or the South Asian way
# (12,50,000), which is how a Pakistani amount is usually written.
_GROUPED = r"\d{1,2}(?:,\d{2})+,\d{3}|\d{1,3}(?:,\d{3})+"
# A long run of digits is an account or a phone number -- unless it is
# followed by a currency, when it is an amount written without commas.
_CURRENCY_WORD = r"(?:روپے|روپیہ|ڈالر|(?i:rupees?|dollars?|pkr|rs)\b)"


def numbers_to_words(text: str, language: Language) -> str:
    """Every number in `text` written out in `language`."""
    if language == "ur":
        # Urdu digits, and the Urdu comma used between groups of them.
        text = re.sub(r"(?<=\d)،(?=\d)", ",", text.translate(_UR_DIGITS))
    return _Numbers(language).apply(text)


def prepare_for_speech(text: str, language: Language) -> str:
    """What a voice should be given to read: speech, and only speech.

    Empty when nothing speakable is left.
    """
    text = strip_markup(text, language)
    text, _ = strip_foreign_script(text)
    text = numbers_to_words(text, language)
    text = re.sub(r"\s+", " ", text).strip()
    return text if any(c.isalpha() for c in text) else ""


# --- sentences -----------------------------------------------------------------

# A title is followed by a name, so it never ends a sentence. Any other
# abbreviation does when a capital letter follows: "until 1 p.m. However".
_TITLES = frozenset("mr mrs ms dr st jr sr prof".split())
_ABBREVIATIONS = frozenset("etc vs no approx dept e.g i.e a.m p.m u.s u.k".split())
_URDU_ENDERS = "۔؟"
_LATIN_ENDERS = ".?!"
_CLOSERS = "\"')]”’»"
_CLAUSE_BREAK = re.compile(r"[,،;؛:](?=\s)")
_WORD = re.compile(r"\S+(?=\s)")
_EARLY_MAX_WORDS = 24  # beyond this an early piece is no shorter than a sentence

# Where an early piece may end. Each piece is spoken as an utterance of its
# own, falling at its end, so one cut mid-phrase sounds broken: in a test call
# "Your checking account balance is" was said as a sentence, then the amount.
# An early piece ends within _EARLY_SLACK words of its target, at the best
# phrase break there: before a new clause, before a verb or a preposition,
# after an Urdu postposition or verb -- never on a word that cannot end a
# phrase, and never before one that cannot begin one.
_EARLY_SLACK = 3
_NEVER_LAST = frozenset(
    """
    the a an your my his her its our their this these those of to for with in on at by
    from about into onto and or but nor is are was were be been am will would can could
    should shall may might must has have had do does did not no very so than as if
    because which who whose that
    کا کی کے اور یا کہ جو جس جب اگر تو ایک یہ وہ اس ان بہت
    """.split()
)
_NEVER_FIRST = frozenset("کا کی کے میں سے پر کو نے تک ہے ہیں تھا تھی تھے گا گی گے ہوں ہو".split())
_STARTS_CLAUSE = frozenset(
    """
    and but or so because which who whose where when while if although though unless
    until since
    اور لیکن مگر کیونکہ تاکہ جو جس جب اگر کہ
    """.split()
)
_STARTS_PHRASE = frozenset(
    """
    to for with in on at by from about into after before during is are was were will
    would can could should has have had
    """.split()
)
_ENDS_PHRASE = frozenset(
    """
    میں سے پر کو نے تک ہے ہیں تھا تھی تھے گا گی گے ہوں ہو گیا گئی گئے رہا رہی رہے
    سکتا سکتی سکتے چاہیے
    """.split()
)
_ENDS_SENTENCE = tuple(".?!۔؟")


class SentenceStream:
    """Cuts streamed text into sentences as soon as each one is complete.

    A full stop only ends a sentence once the next character shows it is not
    a decimal point or an abbreviation, so a sentence ending exactly at the
    end of a chunk waits for one more chunk. The Urdu full stop and question
    mark are never anything else, so they end a sentence at once.

    `first_words` makes the first piece short, because nothing is heard until
    the voice has synthesised all of it: the first piece goes at its first
    clause break, or after that many words, whichever comes first. Measured:
    a long first sentence meant 8.6 s of silence before an English reply.

    `growth` keeps what follows from leaving a silence after it. With it,
    each early piece may be that many times longer than the one before, so
    it is synthesised while the previous one plays; a voice with a large
    cost per request (Urdu's) is better off without it, and goes straight
    to whole sentences after the first piece.

    `max_chars` bounds how long any later piece waits: a sentence that runs on
    is cut at its last comma, or failing that its last space.
    """

    def __init__(self, max_chars: int = 220, first_words: int = 0, growth: float = 0.0) -> None:
        self.max_chars = max_chars
        self.first_words = first_words
        self.growth = growth
        self._early = 0 if first_words > 0 else None  # index of the next early piece
        self._buffer = ""

    def feed(self, chunk: str) -> list[str]:
        self._buffer += chunk
        out: list[str] = []
        while True:
            cut = self._boundary(final=False)
            limit = self._early_limit()
            if limit is not None:
                early = self._early_cut(limit)
                if early is not None and (cut is None or early < cut):
                    cut = early
            if cut is None:
                break
            out.append(self._take(cut))
            if self._early is not None:
                self._early += 1
        if len(self._buffer) > self.max_chars:
            out.append(self._take(self._overflow_cut()))
        return [s for s in out if _speakable(s)]

    def _early_limit(self) -> int | None:
        """How many words the next early piece may have; None once they are over."""
        if self._early is None:
            return None
        if self._early == 0:
            return self.first_words
        if self.growth <= 1.0:
            return None
        limit = int(self.first_words * self.growth**self._early)
        return limit if limit <= _EARLY_MAX_WORDS else None

    def _early_cut(self, limit: int) -> int | None:
        text = self._buffer
        for mark in _CLAUSE_BREAK.finditer(text):
            if len(text[: mark.start()].split()) >= 3:
                return mark.end()
        words = list(_WORD.finditer(text))
        ends = [
            i for i, w in enumerate(words) if w.group().rstrip(_CLOSERS).endswith(_ENDS_SENTENCE)
        ]
        if ends and ends[0] + 1 <= limit:
            return None  # a sentence short enough to go whole: its end cuts it
        if not ends and len(words) <= limit + _EARLY_SLACK:
            return None  # what comes next decides where the phrase ends
        last = ends[0] if ends else len(words) - 1  # leave at least a word of the sentence
        best: tuple[float, int] | None = None
        for n in range(max(3, limit - _EARLY_SLACK), min(limit + _EARLY_SLACK, last) + 1):
            rank = _break_rank(words[n - 1].group(), words[n].group())
            if rank is None:
                continue
            score = rank - 0.5 * abs(n - limit)
            if best is None or score > best[0]:
                best = (score, n)
        if best is None:
            if ends and ends[0] + 1 <= limit + 2 * _EARLY_SLACK:
                return None  # nowhere to break it: a sentence this long goes whole
            return words[limit - 1].end() if len(words) >= limit else None
        return words[best[1] - 1].end()

    def flush(self) -> list[str]:
        out: list[str] = []
        while True:
            cut = self._boundary(final=True)
            if cut is None:
                break
            out.append(self._take(cut))
        out.append(self._take(len(self._buffer)))
        return [s for s in out if _speakable(s)]

    def _take(self, cut: int) -> str:
        piece, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return piece.strip()

    def _overflow_cut(self) -> int:
        window = self._buffer[: self.max_chars]
        for mark in ("، ", ", ", "; ", " "):
            at = window.rfind(mark)
            if at > self.max_chars // 3:
                return at + len(mark)
        return self.max_chars

    def _boundary(self, final: bool) -> int | None:
        text = self._buffer
        for i, char in enumerate(text):
            if char == "\n":
                if text[:i].strip():
                    return i + 1
                continue
            if char in _URDU_ENDERS:
                return self._past_closers(i + 1)
            if char not in _LATIN_ENDERS:
                continue
            after = self._past_closers(i + 1)
            if after >= len(text):
                if final:
                    return after
                return None  # wait: the next chunk decides
            if not text[after].isspace():
                continue  # "2.5", "e.g.x", "?!" -- not an end yet
            if char == ".":
                kind = self._abbreviation_before(i)
                if kind == "title":
                    continue
                if kind == "abbreviation":
                    following = text[after:].lstrip()
                    if not following:
                        if final:
                            return after
                        return None  # wait: a capital next would end it
                    if not following[0].isupper():
                        continue
            return after
        return None

    def _past_closers(self, i: int) -> int:
        text = self._buffer
        while i < len(text) and (text[i] in _CLOSERS or text[i] in _LATIN_ENDERS):
            i += 1
        return i

    def _abbreviation_before(self, dot: int) -> str | None:
        word = self._buffer[:dot].split()[-1:] or [""]
        token = word[0].lower().lstrip("(\"'")
        if token in _TITLES or (len(token) == 1 and token.isalpha()):
            return "title"  # an initial, like a title, comes before a name
        return "abbreviation" if token in _ABBREVIATIONS else None


def _break_rank(before: str, after: str) -> int | None:
    """How good a place to end a piece between two words is; None if it must not be."""
    a, b = _bare(before), _bare(after)
    if a in _NEVER_LAST or b in _NEVER_FIRST:
        return None
    if b in _STARTS_CLAUSE:
        return 3
    if b in _STARTS_PHRASE or a in _ENDS_PHRASE:
        return 2
    return 0


def _bare(word: str) -> str:
    return word.strip(".,;:!?\"'()[]،؛۔؟”’»").lower()


def _speakable(sentence: str) -> bool:
    return any(c.isalnum() for c in sentence)


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    stream = SentenceStream(max_chars)
    return stream.feed(text) + stream.flush()


_CLAUSE_END = re.compile(r"(?<=[,،;؛])\s+")


def split_long(piece: str, language: Language, max_words: int = 18) -> list[str]:
    """A piece too long to synthesise quickly, cut at its commas into pieces of
    at most `max_words` spoken words -- counted as said, numbers as words.

    Measured: the rest of a list of bills, short in figures, was 38 words
    spoken; qwen3-tts took 9.9 s to make it, near its time limit, and the
    caller heard 6.5 s of silence before it. A part with no comma to cut at
    is left whole.
    """
    if len(prepare_for_speech(piece, language).split()) <= max_words:
        return [piece]
    out: list[str] = []
    current = ""
    for part in _CLAUSE_END.split(piece):
        joined = f"{current} {part}".strip()
        if current and len(prepare_for_speech(joined, language).split()) > max_words:
            out.append(current)
            current = part
        else:
            current = joined
    if current:
        out.append(current)
    return out
