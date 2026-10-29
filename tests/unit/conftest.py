"""What the bank tests share.

One fixture, so each file that opens a bank gets the same one: its own
database, isolated from every other test. An in-memory SQLite bank by
default; a PostgreSQL bank in a schema of its own when the run is
VOICEBOT_TEST_DATABASE=postgres (see tests/support.py).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.support import fresh_bank
from voicebot.bank.base import Bank


@pytest.fixture
def bank() -> Iterator[Bank]:
    with fresh_bank() as opened:
        yield opened
