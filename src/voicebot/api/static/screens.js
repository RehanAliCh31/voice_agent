/* The screens that are read rather than talked to: the ledger beside the
 * conversation, the month of transactions, and what is committed every
 * month.
 *
 * Each draws from one call and owns nothing else. Switching between them
 * never interrupts a turn -- the conversation card keeps its state while
 * hidden, so a pending confirmation is still there, and still armed, when
 * the Assistant comes back.
 */

import { api, capitalise, dayLabel, el, momentOf, money, node, state } from "./core.js";
import { loadCards } from "./cards.js";

/**
 * Show one screen. Switching never interrupts a turn: the conversation
 * card keeps its state while hidden, so a pending confirmation is still
 * there, and still armed, when the Assistant comes back.
 */
function showView(name) {
  state.view = name;
  var tabs = el.screens.querySelectorAll(".screens__tab");
  Array.prototype.forEach.call(tabs, function (tab) {
    var chosen = tab.getAttribute("data-view") === name;
    var view = document.getElementById("view" + capitalise(tab.getAttribute("data-view")));
    if (chosen) {
      tab.setAttribute("aria-current", "page");
    } else {
      tab.removeAttribute("aria-current");
    }
    if (view) {
      view.hidden = !chosen;
    }
  });
  el.views.scrollTop = 0;
  if (name === "transactions") {
    loadStatement();
  }
  if (name === "cards") {
    loadCards();
  }
  if (name === "monthly") {
    loadMonthly();
  }
}

// --- the ledger --------------------------------------------------------------

/**
 * Draw the accounts from the turn that just happened, marking any whose
 * balance it changed. Showing which figure moved is the whole reason the
 * ledger sits beside the conversation.
 */
function renderAccounts(accounts) {
  el.accounts.replaceChildren();

  if (!accounts || !accounts.length) {
    var waiting = node("div", "account account--waiting");
    waiting.appendChild(node("span", "account__name", "—"));
    waiting.appendChild(node("span", "account__amount", "····"));
    el.accounts.appendChild(waiting);
    return;
  }

  accounts.forEach(function (account) {
    var before = state.balances[account.name];
    var change =
      typeof before === "number" ? account.balance_cents - before : 0;

    var card = node("div", "account");
    if (change !== 0) {
      card.classList.add("account--moved");
    }
    card.appendChild(node("span", "account__name", account.name));
    card.appendChild(node("span", "account__amount", account.balance));

    if (change !== 0) {
      var sign = change > 0 ? "+" : "−";
      card.appendChild(
        node("span", "account__delta", sign + money(Math.abs(change)) + " just now")
      );
    }

    el.accounts.appendChild(card);
    state.balances[account.name] = account.balance_cents;
  });
}

function renderBills(bills) {
  el.bills.replaceChildren();

  if (!bills || !bills.length) {
    el.bills.appendChild(node("p", "block__empty", "No bills on file."));
    return;
  }

  // Only what is still owed, and only the next few: the whole list is the
  // monthly screen's job, and nine rows here would crowd out the accounts.
  var owing = bills.filter(function (bill) {
    return !bill.paid;
  });
  var showing = owing.length ? owing : bills;
  var hidden = showing.length - 5;

  showing.slice(0, 5).forEach(function (bill) {
    var row = node("div", "bill");
    if (bill.paid) {
      row.classList.add("bill--paid");
    }
    row.appendChild(node("span", "bill__name", bill.name));
    row.appendChild(
      bill.paid
        ? node("span", "bill__paid", "paid")
        : node("span", "bill__amount", bill.amount)
    );
    el.bills.appendChild(row);
  });

  if (hidden > 0) {
    var more = node("button", "bills__more", hidden + " more this month");
    more.type = "button";
    more.addEventListener("click", function () {
      showView("monthly");
    });
    el.bills.appendChild(more);
  }
}

function loadLedger() {
  api
    .accounts()
    .then(function (ledger) {
      renderAccounts(ledger.accounts);
      renderBills(ledger.bills);
    })
    .catch(function () {
      // The ledger keeps its placeholder; the first turn will fill it.
    });
}

// --- the transactions screen --------------------------------------------------

function loadStatement() {
  api
    .statement()
    .then(renderStatement)
    .catch(function () {
      el.statement.replaceChildren(
        node("p", "block__empty", "Couldn't load your transactions just now.")
      );
    });
}

function renderStatement(data) {
  el.statementTotal.textContent = data.total_balance;
  el.statementMonth.textContent = data.month_label;
  el.statementIn.textContent = data.incoming;
  el.statementOut.textContent = data.outgoing;

  el.statement.replaceChildren();
  if (!data.rows.length) {
    el.statement.appendChild(node("p", "block__empty", "Nothing moved this month."));
    return;
  }

  var day = null;
  data.rows.forEach(function (row) {
    var moment = momentOf(row.occurred_at);
    var label = dayLabel(moment);
    if (label !== day) {
      day = label;
      el.statement.appendChild(node("p", "statement__day", label));
    }
    el.statement.appendChild(movementRow(row, moment));
  });
}

function movementRow(row, moment) {
  var movement = node("div", "movement movement--" + (row.direction === "in" ? "in" : "out"));

  // An arrow for a transfer, an initial for a shop: a "C" against a move
  // between Checking and Savings says nothing the row does not already.
  movement.appendChild(
    node(
      "span",
      "movement__mark",
      row.kind === "transfer"
        ? row.direction === "in"
          ? "↓"
          : "↑"
        : row.merchant.slice(0, 1)
    )
  );

  // A shop is named; an account is said as a direction. "Checking /
  // Savings" reads as neither one thing nor the other when both sides are
  // the customer's own.
  var incoming = row.direction === "in";
  var name =
    row.kind === "transfer"
      ? (incoming ? "From " : "To ") + row.merchant
      : row.merchant;

  var who = node("div", "movement__who");
  who.appendChild(node("span", "movement__name", name));
  who.appendChild(node("span", "movement__where", row.account));
  movement.appendChild(who);

  var side = node("div", "movement__side");
  side.appendChild(
    node(
      "span",
      "movement__time",
      moment.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })
    )
  );
  side.appendChild(
    node("span", "movement__amount", (row.direction === "in" ? "+" : "−") + row.amount)
  );
  movement.appendChild(side);

  return movement;
}

// --- the monthly screen --------------------------------------------------------

function loadMonthly() {
  api
    .monthly()
    .then(renderMonthly)
    .catch(function () {
      el.monthlyPlans.replaceChildren(
        node("p", "block__empty", "Couldn't load your monthly plan just now.")
      );
    });
}

function renderMonthly(data) {
  el.monthlyCommitted.textContent = data.committed;
  el.monthlyMonth.textContent = data.month_label;
  el.monthlyPaid.textContent = data.paid;
  el.monthlyLeft.textContent = data.remaining;

  var share = data.committed_cents
    ? Math.round((data.paid_cents / data.committed_cents) * 100)
    : 0;
  el.monthlyBar.style.width = share + "%";

  el.monthlyPlans.replaceChildren();
  if (!data.groups.length) {
    el.monthlyPlans.appendChild(node("p", "block__empty", "Nothing is owed this month."));
    return;
  }
  data.groups.forEach(function (group) {
    el.monthlyPlans.appendChild(planGroup(group));
  });
}

function planGroup(group) {
  var block = node("div", "plangroup");

  var head = node("div", "plangroup__head");
  head.appendChild(node("span", "plangroup__name", group.label));
  head.appendChild(node("span", "plangroup__total", group.total));
  block.appendChild(head);

  group.plans.forEach(function (plan) {
    var row = node("div", "plan" + (plan.paid ? " plan--paid" : ""));

    var what = node("div");
    what.appendChild(node("span", "plan__name", plan.name));
    what.appendChild(node("span", "plan__due", plan.due_label));
    row.appendChild(what);

    var side = node("div", "plan__side");
    side.appendChild(node("span", "plan__amount", plan.amount));
    side.appendChild(node("span", "plan__state", plan.paid ? "Paid" : "Due"));
    row.appendChild(side);

    block.appendChild(row);
  });

  return block;
}

export {
  loadLedger,
  loadMonthly,
  loadStatement,
  renderAccounts,
  renderBills,
  showView,
};
