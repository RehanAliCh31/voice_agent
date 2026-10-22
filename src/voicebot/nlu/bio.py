"""BIO tag-sequence utilities.

Lives in `nlu` rather than `training` because inference needs it too: the
runtime engine decodes predicted tags into slots with exactly the same rules
the evaluation uses. Two implementations would eventually drift, and then the
metrics would stop measuring what the bot does.
"""

from __future__ import annotations

from collections.abc import Sequence

Entity = tuple[str, int, int]  # (slot type, start index, end index exclusive)


def extract_entities(tags: Sequence[str]) -> set[Entity]:
    """Collect entity spans from a BIO tag sequence.

    A `B-x` always opens a new entity. An `I-x` extends the open entity only
    when the types match; otherwise it opens one, which is how conlleval
    treats the malformed sequences a model can emit.
    """
    entities: set[Entity] = set()
    start: int | None = None
    current: str | None = None

    for index, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None and current is not None:
                entities.add((current, start, index))
            start, current = index, tag[2:]
        elif tag.startswith("I-"):
            if current is not None and tag[2:] == current:
                continue
            if start is not None and current is not None:
                entities.add((current, start, index))
            start, current = index, tag[2:]
        else:  # "O" or anything unrecognised closes the open entity
            if start is not None and current is not None:
                entities.add((current, start, index))
            start, current = None, None

    if start is not None and current is not None:
        entities.add((current, start, len(tags)))
    return entities


def ordered_entities(tags: Sequence[str]) -> list[Entity]:
    """The same entities, left to right. Inference needs the order."""
    return sorted(extract_entities(tags), key=lambda entity: entity[1])


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of whitespace-separated words, in order.

    The model works on word indices; a Slot carries character offsets so the
    caller can highlight the exact span of the original utterance.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in text.split():
        start = text.index(word, cursor)
        spans.append((start, start + len(word)))
        cursor = start + len(word)
    return spans
