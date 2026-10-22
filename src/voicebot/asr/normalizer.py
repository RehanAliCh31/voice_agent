"""Bring ASR output back toward the text the NLU was trained on.

This stage is not cosmetic. Whisper writes what a transcriptionist would:
"fifty dollars" comes back as "$50", and "four four one seven" as "4-4-1-7".
The NLU never saw a dollar sign or a hyphenated digit run during training, so
without this step the slot tagger is being asked to generalise across a gap
nobody put in its training data.

Every rule here was written against observed Whisper output, not guessed.
"""

from __future__ import annotations

import re

# Spelled out so the mapping is visible and testable.
_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "dollars",
    "£": "pounds",
    "€": "euros",
    "₨": "rupees",
    "rs": "rupees",
}

# Hesitation noises Whisper transcribes faithfully and the NLU has never seen.
_FILLERS = frozenset({"um", "uh", "erm", "ah", "er", "hmm", "mm"})

_CURRENCY_AMOUNT = re.compile(r"([$£€₨])\s?([\d,]+(?:\.\d+)?)")
_DIGIT_RUN = re.compile(r"\b\d(?:[-\s]\d){2,}\b")
_MULTISPACE = re.compile(r"\s+")


def _expand_currency(text: str) -> str:
    """`$50` -> `50 dollars`, matching how the amounts were spoken."""

    def replace(match: re.Match[str]) -> str:
        symbol, number = match.group(1), match.group(2).replace(",", "")
        return f"{number} {_CURRENCY_SYMBOLS.get(symbol, '')}".strip()

    return _CURRENCY_AMOUNT.sub(replace, text)


def _join_digit_runs(text: str) -> str:
    """`4-4-1-7` -> `4417`.

    Whisper hyphenates digits dictated one at a time. A card number split
    across four tokens is four slot values as far as the tagger is concerned.
    """
    return _DIGIT_RUN.sub(lambda m: re.sub(r"[-\s]", "", m.group(0)), text)


def _drop_fillers(text: str) -> str:
    kept = [word for word in text.split() if word.strip(",.!?").lower() not in _FILLERS]
    return " ".join(kept)


def normalize(text: str) -> str:
    """Normalise one transcript for the NLU.

    Order matters: digit runs are joined before punctuation is stripped, or
    `4-4-1-7` would already have become `4 4 1 7` and lost its grouping.
    """
    if not text or not text.strip():
        return ""

    text = _expand_currency(text)
    text = _join_digit_runs(text)
    text = _drop_fillers(text)

    text = text.lower()
    text = text.replace("-", " ")
    # Keep digits, letters, apostrophes and spaces; drop the rest.
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = _MULTISPACE.sub(" ", text).strip()
    return text
