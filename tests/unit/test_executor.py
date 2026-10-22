"""The tool executor, and the confirmation gate it owns.

The gate is the reason this file matters more than its size suggests. A model
can be argued out of its instructions; it cannot be argued out of Python. Every
test here that asserts a balance is unchanged is asserting that the gate held.
"""

from __future__ import annotations

from typing import Any

import pytest

from voicebot.agent.executor import ToolExecutor
from voicebot.bank.sqlite_bank import SqliteBank
from voicebot.llm.base import ToolCall
from voicebot.llm.tools import CONFIRMING, TOOL_SCHEMAS

TRANSFER = {"amount_cents": 5_000, "payee": "sarah", "from_account": "checking"}
CONFIRMING_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("transfer_money", TRANSFER),
    ("pay_bill", {"bill": "electricity"}),
    ("freeze_card", {"last4": "4417", "report_fraud": False}),
    ("order_card", {"account": "checking"}),
)


@pytest.fixture
def bank() -> SqliteBank:
    return SqliteBank()


@pytest.fixture
def executor(bank: SqliteBank) -> ToolExecutor:
    return ToolExecutor(bank)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


# --- the schemas -------------------------------------------------------------


def test_there_are_nine_well_formed_schemas() -> None:
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert len(TOOL_SCHEMAS) == 9
    assert len(names) == 9
    assert CONFIRMING <= names

    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        assert schema["type"] == "function"
        assert function["description"].strip()
        assert function["parameters"]["type"] == "object"


def test_the_discovery_tools_exist() -> None:
    # Without these the model has to interrogate the user for information the
    # bank already holds -- notably a card's last four digits.
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert {"list_accounts", "list_bills", "list_cards"} <= names


def test_get_spending_pins_the_period_vocabulary() -> None:
    spending = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "get_spending")
    assert "this week" in spending["function"]["parameters"]["properties"]["period"]["enum"]


def test_transfer_declares_an_integer_amount() -> None:
    transfer = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "transfer_money")
    amount = transfer["function"]["parameters"]["properties"]["amount_cents"]
    assert amount["type"] == "integer"
    assert "cents" in amount["description"].lower()


# --- reads execute straight away ---------------------------------------------


def test_a_read_tool_executes_immediately(executor: ToolExecutor) -> None:
    out = executor.execute(call("get_balance", account="savings"))
    assert out.result == {
        "account": "savings",
        "balance": "$19,020.00",
        "balance_cents": 1_902_000,
    }
    # The guard trusts this, and nothing else from the call.
    assert out.sourced_cents == (1_902_000,)


def test_list_cards_reports_the_digits_the_model_needs(executor: ToolExecutor) -> None:
    out = executor.execute(call("list_cards"))
    assert {card["last4"] for card in out.result["cards"]} == {"4417", "9021"}


# --- the confirmation gate ---------------------------------------------------


@pytest.mark.parametrize(("name", "arguments"), CONFIRMING_CALLS)
def test_every_confirming_tool_needs_confirmation_first(
    bank: SqliteBank, name: str, arguments: dict[str, Any]
) -> None:
    executor = ToolExecutor(bank)
    before = (bank.balance("checking"), len(bank.cards()), bank.bills())

    out = executor.execute(ToolCall(id="c1", name=name, arguments=arguments))

    assert out.result["status"] == "needs_confirmation"
    assert out.result["token"] and out.result["summary"]
    assert (bank.balance("checking"), len(bank.cards()), bank.bills()) == before


def test_the_second_call_with_the_token_executes(bank: SqliteBank, executor: ToolExecutor) -> None:
    token = executor.execute(call("transfer_money", **TRANSFER)).result["token"]
    out = executor.execute(call("transfer_money", **TRANSFER, confirmation_token=token))
    assert out.result["status"] == "done"
    assert bank.balance("checking") == 243_350


def test_a_wrong_token_is_refused(bank: SqliteBank, executor: ToolExecutor) -> None:
    executor.execute(call("transfer_money", **TRANSFER))
    out = executor.execute(call("transfer_money", **TRANSFER, confirmation_token="wrong"))
    assert "error" in out.result
    assert bank.balance("checking") == 248_350


def test_changed_arguments_invalidate_the_token(bank: SqliteBank, executor: ToolExecutor) -> None:
    # "actually make it eighty" must re-confirm, not silently move a different
    # amount under the old agreement.
    token = executor.execute(call("transfer_money", **TRANSFER)).result["token"]
    out = executor.execute(
        call("transfer_money", **{**TRANSFER, "amount_cents": 8_000}, confirmation_token=token)
    )
    assert out.result["status"] == "needs_confirmation"
    assert out.result["token"] != token
    assert bank.balance("checking") == 248_350


def test_an_expired_pending_action_is_refused(bank: SqliteBank) -> None:
    executor = ToolExecutor(bank, ttl_s=0.0)
    token = executor.execute(call("transfer_money", **TRANSFER)).result["token"]
    out = executor.execute(call("transfer_money", **TRANSFER, confirmation_token=token))
    assert "error" in out.result
    assert bank.balance("checking") == 248_350


def test_the_summary_reads_back_the_real_amount(executor: ToolExecutor) -> None:
    out = executor.execute(call("transfer_money", **TRANSFER))
    assert "$50.00" in out.result["summary"]
    assert "sarah" in out.result["summary"].lower()


# --- the buttons -------------------------------------------------------------


def test_confirm_executes_the_pending_action(bank: SqliteBank, executor: ToolExecutor) -> None:
    executor.execute(call("transfer_money", **TRANSFER))
    out = executor.confirm()
    assert out.result["status"] == "done"
    assert bank.balance("checking") == 243_350
    assert executor.pending is None


def test_cancel_discards_it(bank: SqliteBank, executor: ToolExecutor) -> None:
    executor.execute(call("transfer_money", **TRANSFER))
    assert executor.cancel().result["status"] == "cancelled"
    assert bank.balance("checking") == 248_350
    assert executor.pending is None


def test_confirm_with_nothing_pending_says_so_rather_than_raising(
    executor: ToolExecutor,
) -> None:
    # The user clicks Yes twice, or clicks it after a restart.
    out = executor.confirm()
    assert "error" in out.result


# --- argument validation -----------------------------------------------------


@pytest.mark.parametrize("amount", ["50.00", 50.5, None, -100, 0, True, [50]])
def test_a_wrongly_typed_amount_is_refused_not_rounded(
    bank: SqliteBank, executor: ToolExecutor, amount: Any
) -> None:
    out = executor.execute(
        call("transfer_money", amount_cents=amount, payee="sarah", from_account="checking")
    )
    assert "error" in out.result
    assert "hint" in out.result
    assert bank.balance("checking") == 248_350
    assert executor.pending is None


def test_a_missing_required_argument_is_refused(executor: ToolExecutor) -> None:
    out = executor.execute(call("transfer_money", amount_cents=5_000))
    assert "error" in out.result and "hint" in out.result


def test_an_unknown_tool_returns_an_error_rather_than_raising(executor: ToolExecutor) -> None:
    out = executor.execute(call("launch_rocket"))
    assert "error" in out.result


def test_a_bank_refusal_becomes_an_error_payload(executor: ToolExecutor) -> None:
    out = executor.execute(call("get_spending", period="october"))
    assert "error" in out.result
    assert "this week" in out.result["hint"]


def test_insufficient_funds_is_explained_not_raised(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    token = executor.execute(
        call("transfer_money", amount_cents=99_999_999, payee="s", from_account="checking")
    ).result["token"]
    out = executor.execute(
        call(
            "transfer_money",
            amount_cents=99_999_999,
            payee="s",
            from_account="checking",
            confirmation_token=token,
        )
    )
    assert "error" in out.result
    assert bank.balance("checking") == 248_350


def test_the_button_path_also_honours_the_expiry(bank: SqliteBank) -> None:
    """The Yes button is the path that actually moves money.

    The model path checked the TTL; this one did not, so a customer could
    return after half an hour and confirm a summary they no longer remembered.
    """
    executor = ToolExecutor(bank, ttl_s=0.0)
    executor.execute(call("transfer_money", **TRANSFER))

    out = executor.confirm()

    assert "error" in out.result
    assert "expired" in out.result["error"].lower()
    assert bank.balance("checking") == 248_350
    assert executor.pending is None


def test_a_live_pending_action_still_confirms(bank: SqliteBank) -> None:
    # The expiry check must not break the ordinary path.
    executor = ToolExecutor(bank, ttl_s=300.0)
    executor.execute(call("transfer_money", **TRANSFER))
    assert executor.confirm().result["status"] == "done"
    assert bank.balance("checking") == 243_350


def test_an_error_result_sources_no_figures(executor: ToolExecutor) -> None:
    # The whole of the guard bypass: an error echoes the model's argument,
    # so nothing in it may count as coming from the bank.
    out = executor.execute(call("get_balance", account="$31,415.00"))
    assert "error" in out.result
    assert out.sourced_cents == ()


def test_a_spending_result_is_not_labelled_with_an_unfiltered_merchant(
    executor: ToolExecutor,
) -> None:
    # A non-string merchant cannot narrow the query, so it must not label the
    # answer -- that would be a confident label on the wrong figure.
    out = executor.execute(call("get_spending", period="this week", merchant=5))
    assert "merchant" not in out.result
