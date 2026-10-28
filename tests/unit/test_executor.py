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
MOVE = {"amount_cents": 50_000, "from_account": "checking", "to_account": "savings"}
# The seed writes its spending rows first, so id 1 is a real charge.
DISPUTE = {"transaction_id": 1, "reason": "not recognised"}
CONTROL = {"last4": "4417", "control": "ecommerce", "enabled": False}
CONFIRMING_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("transfer_money", TRANSFER),
    ("pay_bill", {"bill": "electricity"}),
    ("freeze_card", {"last4": "4417", "report_fraud": False}),
    ("order_card", {"account": "checking"}),
    ("move_money", MOVE),
    ("dispute_transaction", DISPUTE),
    ("set_card_control", CONTROL),
)


@pytest.fixture
def executor(bank: SqliteBank) -> ToolExecutor:
    return ToolExecutor(bank)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


# --- the schemas -------------------------------------------------------------


def test_there_are_fifteen_well_formed_schemas() -> None:
    """The count is asserted on purpose: a new tool is a deliberate act.

    Every schema added is another choice the model has to get right, and the
    surface between it and a customer's money widening by one.
    """
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert len(TOOL_SCHEMAS) == 15
    assert len(names) == 15
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
        call("transfer_money", amount_cents=99_999_999, payee="sarah", from_account="checking")
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


def test_the_assistant_has_no_way_to_create_money() -> None:
    # Paying money in is an administrative act. A bank assistant that can do
    # it is not a bank assistant.
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert not {"deposit", "add_money", "credit", "top_up"} & names


def test_deposit_is_not_reachable_through_the_executor() -> None:
    out = ToolExecutor(SqliteBank()).execute(call("deposit", amount_cents=100_000))
    assert "error" in out.result


def test_a_bill_confirmation_reads_back_the_amount(executor: ToolExecutor) -> None:
    """The customer has to hear what they are agreeing to pay.

    The summary used to say only "Pay your electricity bill in full", so the
    amount in the model's sentence came from an earlier lookup -- and once
    that aged out of the recall window the guard suppressed the whole
    read-back.
    """
    out = executor.execute(call("pay_bill", bill="electricity"))
    assert out.result["status"] == "needs_confirmation"
    assert "$72.50" in out.result["summary"]
    assert 7_250 in out.sourced_cents


def test_a_card_confirmation_sources_no_amount(executor: ToolExecutor) -> None:
    # Freezing a card involves no money, so nothing is sourced by it.
    out = executor.execute(call("freeze_card", last4="4417"))
    assert out.result["status"] == "needs_confirmation"
    assert out.sourced_cents == ()


# --- reading the ledger ------------------------------------------------------


def test_list_transactions_reports_rows(executor: ToolExecutor) -> None:
    out = executor.execute(call("list_transactions", limit=3))

    rows = out.result["transactions"]
    assert len(rows) == 3
    assert {"merchant", "amount", "amount_cents", "direction", "occurred_at"} <= set(rows[0])


def test_transaction_amounts_are_sourced_so_they_can_be_spoken(
    executor: ToolExecutor,
) -> None:
    """Stored negative, spoken positive. The guard compares what is spoken."""
    out = executor.execute(call("list_transactions", merchant="netflix"))

    assert out.sourced_cents and all(cents > 0 for cents in out.sourced_cents)
    assert 1_099 in out.sourced_cents


def test_the_direction_is_stated_not_implied(executor: ToolExecutor) -> None:
    out = executor.execute(call("list_transactions", merchant="netflix"))
    assert out.result["transactions"][0]["direction"] == "out"


def test_reading_the_ledger_changes_nothing(bank: SqliteBank, executor: ToolExecutor) -> None:
    before = bank.balance("checking")
    executor.execute(call("list_transactions"))

    assert bank.balance("checking") == before
    assert executor.writes == 0
    assert executor.pending is None


def test_an_absurd_limit_is_clamped(executor: ToolExecutor) -> None:
    """Replies are spoken aloud; a hundred rows is not an answer."""
    out = executor.execute(call("list_transactions", limit=500))
    assert len(out.result["transactions"]) <= 20


def test_an_unknown_account_comes_back_as_a_sentence(executor: ToolExecutor) -> None:
    out = executor.execute(call("list_transactions", account="offshore"))
    assert "error" in out.result


# --- moving money between the customer's own accounts ------------------------


def test_a_move_is_read_back_with_both_accounts(bank: SqliteBank, executor: ToolExecutor) -> None:
    out = executor.execute(call("move_money", **MOVE))

    assert out.result["status"] == "needs_confirmation"
    assert "$500.00" in out.result["summary"]
    assert "checking" in out.result["summary"] and "savings" in out.result["summary"]
    assert bank.balance("checking") == 248_350


def test_a_confirmed_move_changes_both_balances(bank: SqliteBank, executor: ToolExecutor) -> None:
    token = executor.execute(call("move_money", **MOVE)).result["token"]

    out = executor.execute(call("move_money", **MOVE, confirmation_token=token))

    assert out.result["status"] == "done"
    assert bank.balance("checking") == 198_350
    assert bank.balance("savings") == 1_952_000


def test_a_move_sources_every_figure_it_reports(executor: ToolExecutor) -> None:
    token = executor.execute(call("move_money", **MOVE)).result["token"]

    out = executor.execute(call("move_money", **MOVE, confirmation_token=token))

    assert set(out.sourced_cents) == {50_000, 198_350, 1_952_000}


def test_a_move_counts_as_a_write(executor: ToolExecutor) -> None:
    before = executor.writes
    token = executor.execute(call("move_money", **MOVE)).result["token"]
    executor.execute(call("move_money", **MOVE, confirmation_token=token))

    assert executor.writes == before + 1


def test_moving_to_the_same_account_is_refused(executor: ToolExecutor) -> None:
    out = executor.execute(
        call("move_money", amount_cents=50_000, from_account="checking", to_account="checking")
    )
    assert "error" in out.result


def test_sending_to_an_account_name_comes_back_as_a_sentence(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    """The original bug, now a refusal the model can act on."""
    token = executor.execute(
        call("transfer_money", amount_cents=50_000, payee="savings", from_account="checking")
    ).result.get("token")
    out = executor.execute(
        call(
            "transfer_money",
            amount_cents=50_000,
            payee="savings",
            from_account="checking",
            confirmation_token=token,
        )
    )

    assert "error" in out.result
    assert "account" in out.result["error"]
    assert bank.balance("checking") == 248_350
    assert bank.balance("savings") == 1_902_000


def test_an_unknown_payee_never_becomes_pending(executor: ToolExecutor) -> None:
    """Refused before the read-back, not after the yes.

    Validation exists so a request that cannot succeed is never read back as
    though it will. Otherwise the customer agrees to paying Bob and only then
    learns there is no Bob.
    """
    out = executor.execute(
        call("transfer_money", amount_cents=10_000, payee="bob", from_account="checking")
    )

    assert "error" in out.result
    assert "sarah" in out.result["error"], "the refusal should say who can be paid"
    assert executor.pending is None


def test_an_account_named_as_a_payee_never_becomes_pending(executor: ToolExecutor) -> None:
    out = executor.execute(
        call("transfer_money", amount_cents=10_000, payee="savings", from_account="checking")
    )

    assert "error" in out.result
    assert "own account" in out.result["error"]
    assert executor.pending is None


# --- handing the conversation to a person ------------------------------------
#
# Deliberately not a confirming tool. Asking someone to confirm that they
# would like to speak to a human is the opposite of the point, and the one
# thing every source agreed on is that resisting the handoff costs more trust
# than it saves.


def test_handing_off_is_not_a_confirming_tool() -> None:
    assert "hand_off" not in CONFIRMING


def test_handing_off_returns_a_reference(executor: ToolExecutor) -> None:
    out = executor.execute(call("hand_off", reason="fraud", note="card used abroad"))

    assert out.result["status"] == "handed_off"
    assert out.result["reference"]
    assert out.result["reason"] == "fraud"


def test_handing_off_records_which_conversation_it_came_from(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    executor.session_id = "s-42"
    executor.turn_index = 7

    executor.execute(call("hand_off", reason="complaint"))

    case = bank.handoffs(session_id="s-42")[0]
    assert case.turn_index == 7
    assert case.reason == "complaint"


def test_handing_off_cancels_an_armed_transfer(bank: SqliteBank, executor: ToolExecutor) -> None:
    """A conversation that needs a person must not leave money one yes away."""
    executor.execute(call("transfer_money", **TRANSFER))
    assert executor.pending is not None

    executor.execute(call("hand_off", reason="fraud"))

    assert executor.pending is None
    assert bank.balance("checking") == 248_350


def test_handing_off_moves_no_money(executor: ToolExecutor) -> None:
    executor.execute(call("hand_off", reason="hardship"))
    assert executor.writes == 0


def test_an_unroutable_reason_comes_back_as_a_sentence(executor: ToolExecutor) -> None:
    out = executor.execute(call("hand_off", reason="because I said so"))

    assert "error" in out.result
    assert "fraud" in out.result["error"]


def test_a_listed_transaction_can_be_named_again(executor: ToolExecutor) -> None:
    """The id is what a dispute refers to; two $10.99 charges are not alike."""
    out = executor.execute(call("list_transactions", limit=3))

    rows = out.result["transactions"]
    assert all(isinstance(row["id"], int) for row in rows)


# --- disputing a charge ------------------------------------------------------


def _netflix_id(bank: SqliteBank) -> int:
    return bank.transactions(merchant="netflix")[0].id


def test_a_dispute_is_read_back_with_the_charge(bank: SqliteBank, executor: ToolExecutor) -> None:
    """The customer agrees to a specific charge, not to the idea of one."""
    out = executor.execute(
        call("dispute_transaction", transaction_id=_netflix_id(bank), reason="not recognised")
    )

    summary = out.result["summary"]
    assert "netflix" in summary.lower()
    assert "$10.99" in summary
    assert "not recognised" in summary


def test_a_confirmed_dispute_is_filed(bank: SqliteBank, executor: ToolExecutor) -> None:
    arguments = {"transaction_id": _netflix_id(bank), "reason": "not recognised"}
    token = executor.execute(call("dispute_transaction", **arguments)).result["token"]

    out = executor.execute(call("dispute_transaction", **arguments, confirmation_token=token))

    assert out.result["status"] == "filed"
    assert out.result["reference"]
    assert out.result["outcome"] == "open"
    assert len(bank.disputes()) == 1


def test_filing_a_dispute_moves_no_money(bank: SqliteBank, executor: ToolExecutor) -> None:
    arguments = {"transaction_id": _netflix_id(bank), "reason": "charged twice"}
    token = executor.execute(call("dispute_transaction", **arguments)).result["token"]
    before = bank.balance("checking")

    executor.execute(call("dispute_transaction", **arguments, confirmation_token=token))

    assert bank.balance("checking") == before
    assert executor.writes == 0


def test_the_disputed_amount_is_sourced(bank: SqliteBank, executor: ToolExecutor) -> None:
    arguments = {"transaction_id": _netflix_id(bank), "reason": "not recognised"}
    token = executor.execute(call("dispute_transaction", **arguments)).result["token"]

    out = executor.execute(call("dispute_transaction", **arguments, confirmation_token=token))

    assert 1_099 in out.sourced_cents


def test_a_charge_already_disputed_comes_back_as_a_sentence(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    bank.open_dispute(_netflix_id(bank), "not recognised")

    out = executor.execute(
        call("dispute_transaction", transaction_id=_netflix_id(bank), reason="charged twice")
    )

    assert "error" in out.result
    assert "already" in out.result["error"]


def test_an_unknown_charge_never_becomes_pending(executor: ToolExecutor) -> None:
    """Validated before the read-back, like every other impossible request."""
    out = executor.execute(
        call("dispute_transaction", transaction_id=999_999, reason="never arrived")
    )

    assert "error" in out.result
    assert executor.pending is None


# --- the bot cannot pay money in ---------------------------------------------
#
# Money arriving has to arrive from somewhere real -- a branch, a cash
# machine, a transfer from another bank. An assistant that can credit an
# account on request is not modelling a bank, it is printing money, so there
# is no tool for it and there is not meant to be one.
#
# Moving money between the customer's own accounts is a different thing and
# stays: nothing is created, it only changes sides.


def test_there_is_no_tool_for_paying_money_in() -> None:
    assert "deposit_money" not in {schema["function"]["name"] for schema in TOOL_SCHEMAS}


def test_asking_to_pay_money_in_does_nothing(bank: SqliteBank, executor: ToolExecutor) -> None:
    before = bank.balance("checking")

    out = executor.execute(
        call("deposit_money", amount_cents=25_000, account="checking", source="cheque")
    )

    assert "error" in out.result
    assert bank.balance("checking") == before


def test_moving_money_between_accounts_still_works(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    """The neighbouring feature, which this removal must not touch."""
    token = executor.execute(call("move_money", **MOVE)).result["token"]
    executor.execute(call("move_money", **MOVE, confirmation_token=token))

    assert bank.balance("checking") == 198_350
    assert bank.balance("savings") == 1_952_000


# --- what a card is allowed to do --------------------------------------------


def test_a_card_control_is_read_back_before_it_changes(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    out = executor.execute(call("set_card_control", **CONTROL))

    summary = out.result["summary"].lower()
    assert "ecommerce" in summary and "4417" in summary and "off" in summary
    assert next(c for c in bank.cards() if c.last4 == "4417").controls["ecommerce"] is True


def test_a_confirmed_card_control_changes_it(bank: SqliteBank, executor: ToolExecutor) -> None:
    token = executor.execute(call("set_card_control", **CONTROL)).result["token"]

    out = executor.execute(call("set_card_control", **CONTROL, confirmation_token=token))

    assert out.result["status"] == "done"
    assert next(c for c in bank.cards() if c.last4 == "4417").controls["ecommerce"] is False


def test_changing_a_card_control_moves_no_money(bank: SqliteBank, executor: ToolExecutor) -> None:
    before = bank.balance("checking")
    token = executor.execute(call("set_card_control", **CONTROL)).result["token"]
    executor.execute(call("set_card_control", **CONTROL, confirmation_token=token))

    assert bank.balance("checking") == before
    assert executor.writes == 0


def test_an_unknown_control_never_becomes_pending(executor: ToolExecutor) -> None:
    out = executor.execute(
        call("set_card_control", last4="4417", control="teleportation", enabled=False)
    )

    assert "error" in out.result
    assert executor.pending is None


def test_a_control_on_a_card_that_is_not_there_never_becomes_pending(
    executor: ToolExecutor,
) -> None:
    out = executor.execute(call("set_card_control", last4="0000", control="atm", enabled=False))

    assert "error" in out.result
    assert executor.pending is None


def test_list_cards_reports_what_the_screen_draws(executor: ToolExecutor) -> None:
    card = executor.execute(call("list_cards")).result["cards"][0]

    assert {"last4", "account", "frozen", "kind", "holder", "controls"} <= set(card)


def test_a_confirmed_card_order_issues_one(bank: SqliteBank, executor: ToolExecutor) -> None:
    """The gate tests stop at the read-back; this one goes through with it."""
    before = len(bank.cards())
    token = executor.execute(call("order_card", account="checking")).result["token"]

    out = executor.execute(call("order_card", account="checking", confirmation_token=token))

    assert out.result["status"] == "done"
    assert len(bank.cards()) == before + 1


# --- reporting a card lost or stolen -----------------------------------------
#
# One event, one confirmation. Freezing and replacing were separate tools, so
# losing a card meant agreeing twice to the same thing, and the replacement
# was ordered against an account rather than against the card it replaced.
#
# Stolen is not the same as lost. A theft is a fraud report, and a fraud
# report has to reach a person -- so it opens a case here rather than
# depending on the model to think of it.


def test_a_confirmed_freeze_actually_freezes(bank: SqliteBank, executor: ToolExecutor) -> None:
    """No test had ever let a freeze through: the gate tests stop at the
    read-back, which is what they are for."""
    token = executor.execute(call("freeze_card", last4="4417")).result["token"]

    executor.execute(call("freeze_card", last4="4417", confirmation_token=token))

    assert next(c for c in bank.cards() if c.last4 == "4417").frozen is True


def test_the_read_back_covers_the_whole_event(executor: ToolExecutor) -> None:
    out = executor.execute(
        call("freeze_card", last4="4417", report_fraud=True, order_replacement=True)
    )

    summary = out.result["summary"].lower()
    assert "4417" in summary
    assert "stolen" in summary
    assert "replacement" in summary


def test_losing_a_card_freezes_it_and_sends_another(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    arguments = {"last4": "4417", "order_replacement": True}
    token = executor.execute(call("freeze_card", **arguments)).result["token"]

    out = executor.execute(call("freeze_card", **arguments, confirmation_token=token))

    replacement = out.result["replacement"]
    assert replacement["last4"] != "4417"
    assert replacement["arrives_in_working_days"] == 5
    frozen = next(c for c in bank.cards() if c.last4 == "4417")
    issued = next(c for c in bank.cards() if c.last4 == replacement["last4"])
    assert frozen.frozen is True
    assert issued.frozen is False
    assert issued.account == frozen.account, "the replacement draws on the same account"


def test_freezing_without_asking_sends_no_card(bank: SqliteBank, executor: ToolExecutor) -> None:
    before = len(bank.cards())
    token = executor.execute(call("freeze_card", last4="4417")).result["token"]

    out = executor.execute(call("freeze_card", last4="4417", confirmation_token=token))

    assert out.result["replacement"] is None
    assert len(bank.cards()) == before


def test_a_stolen_card_opens_a_fraud_case(bank: SqliteBank, executor: ToolExecutor) -> None:
    """A theft has to reach a person, and not because the model remembered."""
    arguments = {"last4": "4417", "report_fraud": True}
    token = executor.execute(call("freeze_card", **arguments)).result["token"]

    out = executor.execute(call("freeze_card", **arguments, confirmation_token=token))

    cases = bank.handoffs()
    assert len(cases) == 1
    assert cases[0].reason == "fraud"
    assert out.result["reference"] == cases[0].reference


def test_a_card_merely_lost_opens_no_case(bank: SqliteBank, executor: ToolExecutor) -> None:
    arguments = {"last4": "4417", "order_replacement": True}
    token = executor.execute(call("freeze_card", **arguments)).result["token"]
    executor.execute(call("freeze_card", **arguments, confirmation_token=token))

    assert bank.handoffs() == ()


def test_a_frozen_card_is_still_only_four_digits(executor: ToolExecutor) -> None:
    arguments = {"last4": "4417", "order_replacement": True}
    token = executor.execute(call("freeze_card", **arguments)).result["token"]

    out = executor.execute(call("freeze_card", **arguments, confirmation_token=token))

    assert len(out.result["replacement"]["last4"]) == 4


# --- asking for the PIN without hearing it -----------------------------------
#
# The bot can ask for a PIN and cannot take one. This puts the keypad on the
# screen; what is typed there goes straight to be checked and never comes
# back through the conversation, the transcript or the audit log.


def test_asking_for_a_pin_opens_the_keypad(executor: ToolExecutor) -> None:
    out = executor.execute(call("ask_for_pin", last4="4417", purpose="details"))

    assert out.result["status"] == "pin_requested"
    assert out.result["last4"] == "4417"
    assert out.result["purpose"] == "details"


def test_asking_for_a_pin_is_not_a_confirming_tool() -> None:
    """Nothing happens yet, so there is nothing to agree to."""
    assert "ask_for_pin" not in CONFIRMING


def test_asking_for_a_pin_changes_nothing(bank: SqliteBank, executor: ToolExecutor) -> None:
    executor.execute(call("ask_for_pin", last4="4417", purpose="reset"))

    assert executor.writes == 0
    assert executor.pending is None
    assert bank.check_pin("4417", "1234") is True


def test_a_pin_cannot_be_asked_for_a_card_that_is_not_there(executor: ToolExecutor) -> None:
    out = executor.execute(call("ask_for_pin", last4="0000", purpose="details"))
    assert "error" in out.result


def test_a_pin_is_only_asked_for_something_it_is_needed_for(executor: ToolExecutor) -> None:
    out = executor.execute(call("ask_for_pin", last4="4417", purpose="just curious"))
    assert "error" in out.result


def test_the_tool_cannot_be_handed_a_pin(executor: ToolExecutor) -> None:
    """There is no parameter for one, so the model has nowhere to put it."""
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "ask_for_pin")
    properties = schema["function"]["parameters"]["properties"]

    assert "pin" not in properties
    assert not any("pin" == name for name in properties if name != "purpose")
