"""What the bank promises when requests arrive at the same moment.

Each test lines several threads up on a barrier and lets them go together,
the way a server's thread pool hands it simultaneous requests.

They run against the PostgreSQL bank only. The SQLite bank shares one
connection between threads, and forced to run against it four of these
five fail: a bill charged twice, 18 guesses through a 3-guess lockout, a
duplicate dispute surfacing as a database error, and a transfer that fitted
refused with a transaction error. It is being retired rather than fixed,
and these tests are why the PostgreSQL bank pools its connections and
takes its read-then-write decisions under a row lock.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest

from tests import support
from voicebot.bank.base import PIN_TRIES, Bank, BankError

pytestmark = pytest.mark.skipif(
    not support.TESTING_POSTGRES,
    reason="the PostgreSQL bank's guarantees; run with VOICEBOT_TEST_DATABASE=postgres",
)


def all_at_once(count: int, work: Callable[[], Any]) -> list[Any]:
    """Run `work` on `count` threads released together. Exceptions are results."""
    gate = threading.Barrier(count)
    results: list[Any] = [None] * count

    def runner(slot: int) -> None:
        gate.wait()
        try:
            results[slot] = work()
        except Exception as exc:  # each outcome is what the test inspects
            results[slot] = exc

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_a_bill_paid_by_eight_requests_at_once_is_paid_once(bank: Bank) -> None:
    before = bank.balance("checking")
    amount = next(b for b in bank.bills() if b.name == "internet").amount_cents

    outcomes = all_at_once(8, lambda: bank.pay_bill("internet"))

    paid = [o for o in outcomes if o == amount]
    refused = [o for o in outcomes if isinstance(o, BankError)]
    assert len(paid) == 1
    assert len(refused) == 7
    assert all("already paid" in str(o) for o in refused)
    assert bank.balance("checking") == before - amount
    ledger = support.rows(bank, "SELECT cents FROM transactions WHERE merchant = 'internet'")
    assert [row["cents"] for row in ledger] == [-amount]


def test_a_burst_of_wrong_pins_gets_no_more_guesses_than_the_limit(bank: Bank) -> None:
    outcomes = all_at_once(10, lambda: bank.check_pin("4417", "0000"))

    judged_wrong = [o for o in outcomes if o is False]
    turned_away = [o for o in outcomes if isinstance(o, BankError)]
    assert len(judged_wrong) == PIN_TRIES
    assert len(turned_away) == 10 - PIN_TRIES
    assert bank.pin_locked("4417") is True
    card = support.rows(bank, "SELECT pin_failures FROM cards WHERE last4 = '4417'")[0]
    assert card["pin_failures"] == PIN_TRIES


def test_the_right_pin_in_a_burst_still_opens_the_card(bank: Bank) -> None:
    """Locking the row must not turn a correct PIN away when nothing is wrong."""
    outcomes = all_at_once(6, lambda: bank.check_pin("4417", "1234"))

    assert outcomes == [True] * 6
    assert bank.pin_locked("4417") is False


def test_one_charge_disputed_by_five_requests_at_once_is_disputed_once(bank: Bank) -> None:
    charge = next(t for t in bank.transactions(limit=20) if t.merchant == "netflix")

    outcomes = all_at_once(5, lambda: bank.open_dispute(charge.id, "not recognised"))

    opened = [o for o in outcomes if not isinstance(o, Exception)]
    refused = [o for o in outcomes if isinstance(o, BankError)]
    # A refusal the customer can hear, not a database error: the unique
    # constraint that stops a second dispute is translated, not leaked.
    assert len(opened) == 1
    assert len(refused) == 4
    assert all("already been disputed" in str(o) for o in refused)
    assert len(bank.disputes()) == 1


def test_simultaneous_transfers_never_overdraw(bank: Bank) -> None:
    """Two of these fit and one does not; exactly that must happen.

    Neither bank overdraws -- the check is the UPDATE itself in both. But the
    SQLite bank's shared connection also turns a transfer that fits into a
    transaction error when another lands at the same moment, so it refuses
    money the customer has. Here both transfers that fit go through.
    """
    balance = bank.balance("checking")
    share = balance // 3 + 1  # three of these cannot all fit

    outcomes = all_at_once(3, lambda: bank.transfer(share, "sarah", "checking"))

    assert sum(1 for o in outcomes if isinstance(o, int)) == 2
    assert sum(1 for o in outcomes if isinstance(o, BankError)) == 1
    assert bank.balance("checking") == balance - 2 * share
