"""What the bank tests share.

One fixture, so each file that opens a bank gets the same one: its own
database, isolated from every other test. An in-memory SQLite bank by
default; a PostgreSQL bank in a schema of its own when the run is
VOICEBOT_TEST_DATABASE=postgres (see tests/support.py).

A test marked `mid_month` gets a bank whose clock reads support.MID_MONTH:
the mark for a test that says what a period holds, which on the calendar
depends on the day the suite runs.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.support import MID_MONTH, fresh_bank, pin_the_clock
from voicebot.bank.base import Bank


@pytest.fixture
def bank(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Bank]:
    # Before the bank opens, because opening it is what seeds it.
    if request.node.get_closest_marker("mid_month"):
        pin_the_clock(monkeypatch, MID_MONTH)
    with fresh_bank() as opened:
        yield opened
