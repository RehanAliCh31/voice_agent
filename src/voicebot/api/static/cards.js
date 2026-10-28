/* The cards screen, and the three dialogues that belong to it.
 *
 * A card is the one screen where looking is not free: the numbers printed
 * on it are behind a PIN and come down again on a timer, and freezing or
 * replacing one is asked for twice -- why, then are you sure -- because
 * neither can be undone from here.
 */

import { api, capitalise, el, node, state } from "./core.js";
import { addTurn, handleTurn } from "./conversation.js";

function loadCards() {
  api
    .cards()
    .then(function (data) {
      state.cards = data.cards;
      // Keep the card already being looked at when this is a refresh after
      // a turn, so freezing a card does not jump the screen to another one.
      var chosen = state.card;
      var known = data.cards.some(function (card) {
        return card.last4 === chosen;
      });
      state.card = known ? chosen : (data.cards[0] || {}).last4 || null;
      renderCards();
    })
    .catch(function () {
      el.cardPane.replaceChildren(
        node("p", "block__empty", "Couldn't load your cards just now.")
      );
    });
}

function renderCards() {
  el.cardPicker.replaceChildren();
  el.cardPane.replaceChildren();

  if (!state.cards.length) {
    el.cardPane.appendChild(node("p", "block__empty", "No cards on file."));
    return;
  }

  state.cards.forEach(function (card) {
    var tab = node("button", "cardpicker__tab");
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(card.last4 === state.card));
    tab.appendChild(node("span", null, capitalise(card.kind)));
    tab.appendChild(node("span", "cardpicker__digits", "••" + card.last4));
    tab.addEventListener("click", function () {
      state.card = card.last4;
      renderCards();
    });
    el.cardPicker.appendChild(tab);
  });

  var showing = state.cards.filter(function (card) {
    return card.last4 === state.card;
  })[0];
  if (showing) {
    el.cardPane.appendChild(cardFace(showing));
    el.cardPane.appendChild(cardActions(showing));
    el.cardPane.appendChild(cardControls(showing));
  }
}

/**
 * The two things that can be done to a card from here. Both go through the
 * same gate the spoken path uses: the button arms the action and the bank
 * says what it is, and only then is there anything to agree to.
 */
// Four tiles under the card. The icons are drawn here as fixed literals,
// which is why innerHTML is safe on this one line and nowhere else: no
// value from the server or the customer reaches it.
var TILE_ICONS = {
  details:
    '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
    ' stroke-width="1.8" stroke-linecap="round"><rect x="2" y="5" width="20" height="14"' +
    ' rx="2.5"/><path d="M2 10h20"/><circle cx="16" cy="15" r="2.4"/><path d="M18 17l2 2"/></svg>',
  freeze:
    '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
    ' stroke-width="1.8" stroke-linecap="round"><path d="M12 2v20M4 7l16 10M20 7L4 17"/></svg>',
  replace:
    '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
    ' stroke-width="1.8" stroke-linecap="round"><path d="M3 11a8 8 0 0 1 13.6-5.7L20 8"/>' +
    '<path d="M20 4v4h-4"/><path d="M21 13a8 8 0 0 1-13.6 5.7L4 16"/><path d="M4 20v-4h4"/></svg>',
  pin:
    '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
    ' stroke-width="1.8" stroke-linecap="round"><rect x="3" y="4" width="18" height="16"' +
    ' rx="2.5"/><circle cx="8" cy="10" r="1"/><circle cx="12" cy="10" r="1"/>' +
    '<circle cx="16" cy="10" r="1"/><path d="M8 15h8"/></svg>',
};

function tile(card, icon, label, extraClass, onClick, disabledWhy) {
  var button = node("button", "cardact" + (extraClass ? " " + extraClass : ""));
  button.type = "button";

  var mark = node("span", "cardact__icon");
  mark.setAttribute("aria-hidden", "true");
  mark.innerHTML = TILE_ICONS[icon];
  button.appendChild(mark);
  button.appendChild(node("span", null, label));

  if (disabledWhy) {
    button.disabled = true;
    button.title = disabledWhy;
    button.setAttribute("aria-label", label + " — " + disabledWhy);
  } else {
    button.addEventListener("click", onClick);
  }
  return button;
}

function cardActions(card) {
  var row = node("div", "cardacts");

  row.appendChild(
    tile(card, "details", "View details", null, function () {
      openPin(card, "details");
    })
  );
  row.appendChild(
    tile(
      card,
      "freeze",
      card.frozen ? "Frozen" : "Freeze card",
      "cardact--stop",
      function () {
        openAsk(card, "freeze");
      },
      card.frozen ? "this card is already frozen" : null
    )
  );
  row.appendChild(
    tile(card, "replace", "Replace card", null, function () {
      openAsk(card, "replace");
    })
  );
  // Deliberately dead rather than quietly doing nothing: a tile that looks
  // live and is not is worse than one that says it is not ready.
  row.appendChild(
    tile(card, "pin", "Reset PIN", null, function () {
      openPin(card, "reset");
    })
  );

  return row;
}

// --- the card itself -------------------------------------------------------

function cardFace(card) {
  var face = node("div", "cardface" + (card.frozen ? " cardface--frozen" : ""));

  var top = node("div", "cardface__top");
  top.appendChild(node("span", "cardface__bank", "Voice Banking"));
  top.appendChild(node("span", "cardface__kind", card.kind));
  face.appendChild(top);

  // Dots and four digits, because dots and four digits is all the bank
  // has. A full number on screen would be one the database never held.
  face.appendChild(node("div", "cardface__number", card.masked));

  var foot = node("div", "cardface__foot");
  var who = node("div");
  who.appendChild(node("span", "cardface__holder", card.holder));
  who.appendChild(node("span", "cardface__account", card.account));
  foot.appendChild(who);
  foot.appendChild(node("span", "cardface__state", card.frozen ? "Frozen" : "Active"));
  face.appendChild(foot);

  return face;
}

function cardControls(card) {
  // A frozen card allows nothing, whatever its switches say. They are left
  // showing their real values rather than flipped to off, because the
  // columns have not changed and unfreezing restores exactly this -- but
  // they are shown as not in force, so the screen is not telling the
  // customer that online purchases work on a card that is dead.
  var controls = node("div", "controls" + (card.frozen ? " controls--frozen" : ""));

  card.controls.forEach(function (control) {
    var row = node("div", "control");
    var words = node("div");
    words.appendChild(node("span", "control__name", control.label));
    words.appendChild(node("span", "control__detail", control.detail));
    row.appendChild(words);

    var state_ = node("span", "control__state");
    state_.setAttribute("data-on", String(control.enabled));
    state_.setAttribute("role", "img");
    state_.setAttribute(
      "aria-label",
      control.label + (control.enabled ? ": allowed" : ": not allowed")
    );
    row.appendChild(state_);

    controls.appendChild(row);
  });

  controls.appendChild(
    node(
      "p",
      "controls__hint",
      card.frozen
        ? "This card is frozen, so none of these are in force. They come " +
          "back as they are when it is unfrozen."
        : "Ask to change one — say “turn off online purchases on my card”. " +
          "Every change goes through the same confirmation as moving money."
    )
  );
  return controls;
}

// --- what is printed on the card --------------------------------------------

function detailPair(key, value, wide, mono) {
  var pair = node("div", "details__pair" + (wide ? " details__pair--wide" : ""));
  pair.appendChild(node("dt", "details__key", key));
  pair.appendChild(node("dd", "details__value" + (mono ? " details__value--mono" : ""), value));
  return pair;
}

function showDetails(details) {
      el.detailsList.replaceChildren(
        detailPair("Name", details.holder, true),
        detailPair("Card type", details.card_type),
        detailPair("Status", details.status),
        detailPair("Card number", details.number, true, true),
        detailPair("Valid from", details.valid_from, false, true),
        detailPair("Valid to", details.valid_to, false, true),
        detailPair("Security code", details.security_code, false, true)
      );
      el.cardDetails.showModal();
      el.detailsClose.focus();
      countDown(details.expires_in_seconds);
}

/** Close the details when the code they show has run out. */
function countDown(seconds) {
  clearInterval(state.detailsTimer);
  var left = seconds;

  function show() {
    el.detailsClock.textContent = left;
    el.detailsClock.setAttribute("data-low", String(left <= 15));
  }

  show();
  state.detailsTimer = setInterval(function () {
    left -= 1;
    if (left <= 0) {
      closeDetails();
      return;
    }
    show();
  }, 1000);
}

function closeDetails() {
  clearInterval(state.detailsTimer);
  state.detailsTimer = null;
  if (el.cardDetails.open) {
    el.cardDetails.close();
  }
}

// --- the PIN ----------------------------------------------------------------

//
// Asked for here and never spoken. A PIN said out loud would be
// transcribed, carried to the model and written to the audit log, which is
// the one thing a card secret must never be. It goes from this box
// straight to be checked, and the box is emptied however the dialogue is
// shut.

var PIN_ASKS = {
  details: {
    title: "Enter your PIN",
    why: "This card's details are behind your PIN.",
  },
  reset: {
    title: "Change your PIN",
    why: "Start with the PIN you have now.",
  },
  payment: {
    title: "Approve this payment",
    why: "Enter the PIN for the account the money leaves.",
  },
};

/** The loaded card if the screen has one, or just its four digits. */
function cardNamed(last4) {
  return (
    state.cards.filter(function (card) {
      return card.last4 === last4;
    })[0] || { last4: last4 }
  );
}

function openPin(card, purpose, note) {
  state.pin = { card: card, purpose: purpose };
  el.pinTitle.textContent = PIN_ASKS[purpose].title;
  // The bank's own sentence when it sent one, so what is being approved is
  // on the dialogue that asks for the PIN and not only on the bar behind it.
  el.pinWhy.textContent = note || PIN_ASKS[purpose].why;
  forgetPins();
  el.pinStepCurrent.hidden = false;
  el.pinStepNew.hidden = true;
  el.pinError.hidden = true;
  el.pinGo.textContent = "Continue";
  el.pinGo.disabled = false;
  el.pinAsk.showModal();
  el.pinCurrent.focus();
}

/** Empty every box that has held a PIN. */
function forgetPins() {
  el.pinCurrent.value = "";
  el.pinNew.value = "";
  el.pinAgain.value = "";
}

function closePin() {
  forgetPins();
  state.pin = null;
  if (el.pinAsk.open) {
    el.pinAsk.close();
  }
}

function pinFailed(message) {
  el.pinError.textContent = message;
  el.pinError.hidden = false;
  el.pinGo.disabled = false;
}

function onPinGo() {
  if (!state.pin) {
    return;
  }
  el.pinError.hidden = true;

  if (!el.pinStepNew.hidden) {
    changePin();
    return;
  }

  var pin = el.pinCurrent.value.trim();
  if (pin.length !== 4) {
    pinFailed("A PIN is four digits.");
    return;
  }

  el.pinGo.disabled = true;
  var card = state.pin.card;

  if (state.pin.purpose === "details") {
    // One step: the PIN buys the details, so there is nothing in between.
    api
      .details(card.last4, pin)
      .then(function (details) {
        closePin();
        showDetails(details);
      })
      .catch(function (error) {
        pinFailed(error.message);
      });
    return;
  }

  if (state.pin.purpose === "payment") {
    // The PIN travels with the agreement: one call approves and performs, so
    // there is no moment where the bank has been told yes, been given the
    // PIN, and is still waiting to be told to go ahead.
    api
      .confirm(pin)
      .then(function (turn) {
        closePin();
        handleTurn(turn);
      })
      .catch(function (error) {
        // The action is still pending, so a wrong PIN costs another try and
        // not the agreement. Cancelling here leaves the bar to press again.
        pinFailed(error.message);
      });
    return;
  }

  api
    .checkPin(card.last4, pin)
    .then(function () {
      state.pin.current = pin;
      el.pinCurrent.value = "";
      el.pinStepCurrent.hidden = true;
      el.pinStepNew.hidden = false;
      el.pinWhy.textContent = "Choose a new PIN, then enter it again.";
      el.pinGo.textContent = "Change PIN";
      el.pinGo.disabled = false;
      el.pinNew.focus();
    })
    .catch(function (error) {
      pinFailed(error.message);
    });
}

function changePin() {
  var next = el.pinNew.value.trim();
  var again = el.pinAgain.value.trim();

  if (next.length !== 4) {
    pinFailed("A new PIN is four digits.");
    return;
  }
  if (next !== again) {
    // The server is sent one new PIN, so this is the only place the two
    // can be compared -- which is the whole reason for asking twice.
    pinFailed("Those two do not match.");
    el.pinAgain.value = "";
    el.pinAgain.focus();
    return;
  }

  el.pinGo.disabled = true;
  var last4 = state.pin.card.last4;
  api
    .changePin(last4, state.pin.current, next)
    .then(function () {
      closePin();
      addTurn("bot", "The PIN on the card ending " + last4 + " has been changed.", {
        notice: true,
      });
    })
    .catch(function (error) {
      pinFailed(error.message);
    });
}

// --- asking before acting ----------------------------------------------------

var ASKS = {
  freeze: {
    title: "Freeze this card",
    tool: "freeze_card",
    args: function (card) {
      return { last4: card.last4 };
    },
  },
  replace: {
    // Replacing stops the old card as well as sending a new one, which is
    // what replacing a card means: two cards on one account is not it.
    title: "Replace this card",
    tool: "freeze_card",
    args: function (card) {
      return { last4: card.last4, order_replacement: true };
    },
  },
};

function openAsk(card, kind) {
  state.ask = { card: card, kind: kind };
  el.askTitle.textContent = ASKS[kind].title;
  el.askReason.value = "";
  el.askSummary.textContent = "";
  el.askError.hidden = true;
  el.askWhy.hidden = false;
  el.askSure.hidden = true;
  el.askGo.textContent = "Done";
  el.askGo.disabled = false;
  el.cardAsk.showModal();
  el.askReason.focus();
}

function closeAsk() {
  state.ask = null;
  if (el.cardAsk.open) {
    el.cardAsk.close();
  }
}

function askFailed(message) {
  el.askError.textContent = message;
  el.askError.hidden = false;
  el.askGo.disabled = false;
}

/** Step one asks why; step two shows what the bank says will happen. */
function onAskGo() {
  if (!state.ask) {
    return;
  }
  el.askError.hidden = true;

  if (!el.askSure.hidden) {
    settleAsk();
    return;
  }

  var reason = el.askReason.value.trim();
  if (!reason) {
    askFailed("Say why, so it can be written down.");
    el.askReason.focus();
    return;
  }

  var ask = ASKS[state.ask.kind];
  el.askGo.disabled = true;
  api
    .arm(ask.tool, ask.args(state.ask.card), reason)
    .then(function (armed) {
      state.sessionId = armed.session_id;
      if (!armed.armed) {
        askFailed(armed.error || "That could not be done.");
        return;
      }
      // The bank's own sentence, not one written here, so what is agreed
      // to is exactly what will happen.
      el.askSummary.textContent = armed.summary + "?";
      el.askWhy.hidden = true;
      el.askSure.hidden = false;
      el.askGo.textContent = "Yes, confirm";
      el.askGo.disabled = false;
      el.askGo.focus();
    })
    .catch(function (error) {
      askFailed(error.message);
    });
}

function settleAsk() {
  el.askGo.disabled = true;
  el.askGo.textContent = "Working…";
  // The same endpoint the spoken Yes button uses, so the model is told
  // what happened and its next answer knows the card is frozen.
  api
    .confirm()
    .then(function (turn) {
      closeAsk();
      handleTurn(turn);
      loadCards();
    })
    .catch(function (error) {
      el.askGo.textContent = "Yes, confirm";
      askFailed(error.message);
    });
}

export {
  cardNamed,
  closeAsk,
  closeDetails,
  closePin,
  forgetPins,
  loadCards,
  onAskGo,
  onPinGo,
  openPin,
};
