"""Settings the whole suite shares.

One is about speed. The other keeps tests apart: circuit breakers and error
counters live for the whole process, the way they must in a server, so a test
that trips the model's breaker would otherwise fail the next test that calls
it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from voicebot.bank import sqlite_bank
from voicebot.utils import observability, resilience

# Every test opens its own bank, and every bank hashes a PIN for every card.
# At the real work factor that is about a minute and a half of the suite
# spent proving that PBKDF2 is slow, which is already known. Lowered here
# and nowhere else, so the code that ships keeps the cost it is meant to
# have -- the same trick Django plays with its password hashers.
sqlite_bank.PIN_ITERATIONS = 1_000


@pytest.fixture(autouse=True)
def _independent_breakers_and_counters() -> Iterator[None]:
    resilience.reset_breakers()
    observability.reset_counts()
    yield
    resilience.reset_breakers()
    observability.reset_counts()
