"""What the bank tests share.

One fixture, so each file that opens a bank gets the same one: its own
`:memory:` database, isolated from every other test and from any file on
disk.
"""

from __future__ import annotations

import pytest

from voicebot.bank.sqlite_bank import SqliteBank


@pytest.fixture
def bank() -> SqliteBank:
    return SqliteBank()
