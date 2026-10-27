"""Settings the whole suite shares.

Only one so far, and it is about speed rather than behaviour.
"""

from __future__ import annotations

from voicebot.bank import sqlite_bank

# Every test opens its own bank, and every bank hashes a PIN for every card.
# At the real work factor that is about a minute and a half of the suite
# spent proving that PBKDF2 is slow, which is already known. Lowered here
# and nowhere else, so the code that ships keeps the cost it is meant to
# have -- the same trick Django plays with its password hashers.
sqlite_bank.PIN_ITERATIONS = 1_000
