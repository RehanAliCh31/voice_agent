/* Voice Banking -- the client.
 *
 * One turn is: record, convert to the WAV the API expects, POST it, draw
 * what comes back, speak it. Everything the customer sees is drawn from
 * that one answer, so the balances beside the conversation and the sentence
 * the bot just said always describe the same moment.
 *
 * This file is the wiring and nothing else: it finds the elements once,
 * hangs the listeners on them, says which screens want to hear about a
 * turn, and starts. The work itself is in the modules it imports.
 */

import { abortTurn, api, el, loadSentences, state } from "./core.js";
import { loadVoices, stopPlayback } from "./audio.js";
import { fail, onTurn, resolvePending, send, setPhase, setStatus } from "./conversation.js";
import {
  cardNamed,
  closeAsk,
  closeDetails,
  closePin,
  forgetPins,
  loadCards,
  onAskGo,
  onPinGo,
  openPin,
} from "./cards.js";
import {
  loadLedger,
  loadMonthly,
  loadStatement,
  renderAccounts,
  renderBills,
  showView,
} from "./screens.js";
import { onBargeIn, onMicClick } from "./mic.js";

// The screens that redraw after a turn. Saying it here, once, is what lets
// the conversation stay ignorant of them: a new screen is a line in this
// list rather than an edit inside handleTurn.
onTurn(function (turn) {
  // null means the ledger could not be read with this turn: what is on
  // screen stays, rather than being cleared as if the accounts were gone.
  if (turn.accounts != null) {
    renderAccounts(turn.accounts);
  }
  if (turn.bills != null) {
    renderBills(turn.bills);
  }
  if (state.view === "monthly") {
    // A turn can pay one of them.
    loadMonthly();
  }
  if (state.view === "cards") {
    // A turn can freeze a card or change what it allows.
    loadCards();
  }
  if (state.view === "transactions") {
    // The turn may have moved money, and a statement left as it was would
    // disagree with the balances drawn a line above it.
    loadStatement();
  }
  if (turn.pin_request) {
    // The bot asked for a PIN, which means it put the keypad here rather
    // than listening for one. No screen change: the dialogue is above
    // whatever is being read, and the conversation stays where it was.
    openPin(
      cardNamed(turn.pin_request.last4),
      turn.pin_request.purpose,
      turn.pin_request.summary
    );
  }
});

// --- wiring --------------------------------------------------------------------

function onSay(event) {
  event.preventDefault();
  var text = el.sayInput.value.trim();
  if (!text) {
    return;
  }
  el.sayInput.value = "";
  send(text);
}

function onReset() {
  // Resets the conversation AND the demo balances. With one shared bank,
  // a drained checking account would otherwise outlive the conversation
  // that drained it, and every later transfer would fail for want of funds.
  // A turn still being waited for is abandoned first: its answer belongs to
  // the conversation being thrown away.
  abortTurn();
  api
    .reset()
    .catch(function () {})
    .finally(function () {
      return api.endSession().catch(function () {}).finally(function () {
        state.sessionId = null;
        state.balances = {};
        state.lastTotalMs = null;
        stopPlayback();
        el.conversation.replaceChildren(el.empty);
        el.confirm.hidden = true;
        setPhase("idle");
        setStatus("Ready", "Tap the microphone to start a turn", "");
        loadLedger();
      });
    });
}

// Whatever no handler caught: logged, and a turn left half-way is ended
// rather than left spinning.
window.addEventListener("unhandledrejection", function (event) {
  console.error("unhandled", event.reason);
  if (state.phase === "thinking") {
    fail(event.reason);
  }
});

function boot() {
  loadVoices();
  loadSentences();
  if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }

  // Which engines are loaded was a builder's question, not a customer's.
  // Whether the server answers at all still matters, and the dock is where
  // that is said, so the check stays and only the chips went.
  api.ready().then(loadLedger, function () {
    setStatus("No server", "The API did not answer /ready", "error");
  });
}

document.addEventListener("DOMContentLoaded", function () {
  [
    "accounts", "bills",
    "conversation", "empty", "confirm", "confirmYes", "confirmNo",
    "micButton", "statusLabel", "statusHint", "levelFill",
    "sayForm", "sayInput", "resetButton",
    "vadToggle", "vadSensitivity", "vadValue", "vadNote",
    "screens", "statement", "statementTotal", "statementMonth",
    "statementIn", "statementOut", "cardPicker", "cardPane",
    "monthlyPlans", "monthlyCommitted", "monthlyMonth", "monthlyPaid",
    "monthlyLeft", "monthlyBar",
    "cardAsk", "askTitle", "askWhy", "askReason", "askSure", "askSummary",
    "askError", "askCancel", "askGo",
    "cardDetails", "detailsList", "detailsClock", "detailsClose",
    "pinAsk", "pinTitle", "pinWhy", "pinStepCurrent", "pinCurrent",
    "pinStepNew", "pinNew", "pinAgain", "pinError", "pinCancel", "pinGo",
  ].forEach(function (id) {
    el[id] = document.getElementById(id);
  });

  el.views = document.getElementById("views");
  el.confirmText = document.querySelector(".confirm__text");
  el.bargein = document.querySelector(".bargein");

  state.vad = new window.VoiceActivity({
    onSpeech: onBargeIn,
    onLevel: function (rms, trigger) {
      // While listening, the meter is just loudness. While the bot is
      // speaking, it is loudness against the line barge-in has to clear --
      // which moves with the room -- so you can see why it did or did not
      // fire, and set the slider by looking rather than by guessing.
      if (state.phase === "listening") {
        el.levelFill.style.width = Math.min(100, Math.round(rms * 400)) + "%";
        el.levelFill.removeAttribute("data-over");
      } else if (state.phase === "speaking") {
        var share = trigger > 0 ? rms / trigger : 0;
        el.levelFill.style.width = Math.min(100, Math.round(share * 90)) + "%";
        el.levelFill.setAttribute("data-over", share >= 1 ? "true" : "false");
      }
    },
  });

  el.pinGo.addEventListener("click", onPinGo);
  el.pinCancel.addEventListener("click", closePin);
  // However it is shut -- button, Escape, backdrop -- the boxes are
  // emptied, so nothing is left on the page holding a PIN.
  el.pinAsk.addEventListener("close", function () {
    forgetPins();
    state.pin = null;
  });
  [el.pinCurrent, el.pinNew, el.pinAgain].forEach(function (box) {
    box.addEventListener("keydown", function (event) {
      if (event.key === "Enter") {
        event.preventDefault();
        onPinGo();
      }
    });
  });

  el.detailsClose.addEventListener("click", closeDetails);
  // Escape closes it natively; the timer has to stop either way.
  el.cardDetails.addEventListener("close", function () {
    clearInterval(state.detailsTimer);
    state.detailsTimer = null;
  });

  el.askGo.addEventListener("click", onAskGo);
  el.askCancel.addEventListener("click", closeAsk);
  // Escape and the backdrop close it natively; this keeps our own state
  // in step with the dialogue however it was shut.
  el.cardAsk.addEventListener("close", function () {
    state.ask = null;
  });

  el.screens.addEventListener("click", function (event) {
    var tab = event.target.closest(".screens__tab");
    if (tab) {
      showView(tab.getAttribute("data-view"));
    }
  });

  el.micButton.addEventListener("click", onMicClick);
  el.sayForm.addEventListener("submit", onSay);
  el.resetButton.addEventListener("click", onReset);

  el.confirmYes.addEventListener("click", function () {
    el.confirm.hidden = true;
    resolvePending(api.confirm());
  });
  el.confirmNo.addEventListener("click", function () {
    el.confirm.hidden = true;
    resolvePending(api.cancel());
  });

  el.vadToggle.addEventListener("change", function () {
    state.vad.enabled = el.vadToggle.checked;
    el.bargein.setAttribute("data-enabled", String(el.vadToggle.checked));
    el.vadNote.textContent = el.vadToggle.checked
      ? "Speak while the bot is talking and it will stop and listen."
      : "The bot will finish speaking before it listens again.";
  });

  el.vadSensitivity.addEventListener("input", function () {
    el.vadValue.textContent = el.vadSensitivity.value;
    state.vad.setSensitivity(el.vadSensitivity.value);
  });

  boot();
});
