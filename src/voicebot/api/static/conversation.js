/* What the customer reads, what the status line says, and what happens
 * when a turn comes back.
 *
 * No text here reaches the page as markup. Every element is built with
 * createElement and textContent, because a transcript is the customer's own
 * words read back and the reply is a language model's output -- neither is
 * trusted markup.
 */

import { api, el, explain, failureKey, node, state } from "./core.js";
import { base64ToBlob, play, playAll, stopPlayback } from "./audio.js";

function clearEmptyState() {
  if (el.empty && el.empty.parentNode) {
    el.empty.remove();
  }
}

function addTurn(who, text, options) {
  clearEmptyState();
  options = options || {};

  var turn = node("div", "turn turn--" + (who === "you" ? "you" : "bot"));
  if (options.error) {
    turn.classList.add("turn--error");
  }
  if (options.notice) {
    turn.classList.add("turn--notice");
  }

  // dir=auto: the browser takes the direction from the first letter, so an
  // Urdu turn runs right to left and an English one left to right, in the
  // same conversation.
  var said = node("p", "turn__text", text);
  said.dir = "auto";
  turn.appendChild(said);

  if (options.guardHit) {
    var flag = node("div", "turn__flag", "figure read back from your account");
    flag.title =
      "The model's own wording stated a figure no tool returned, so the " +
      "stored value was spoken instead.";
    turn.appendChild(flag);
  }

  if (options.caseRef) {
    // Shown as its own line rather than left inside the sentence: this is
    // the one thing the customer may need to write down. The label says
    // which kind it is, because a dispute and a handoff mean different
    // things -- one filed a claim, the other ended the bot's part.
    var case_ = node("div", "turn__case");
    case_.appendChild(node("span", "turn__case-label", options.caseLabel));
    case_.appendChild(node("span", "turn__case-ref", options.caseRef));
    turn.appendChild(case_);
  }

  if (options.audioBlobs) {
    var row = node("div", "turn__audio");
    var replay = node("button", "replay");
    replay.type = "button";
    replay.setAttribute("aria-label", "Replay this reply");
    replay.innerHTML =
      '<svg width="10" height="10" viewBox="0 0 12 12" aria-hidden="true">' +
      '<path d="M2 1.5l8 4.5-8 4.5z" fill="currentColor"></path></svg>';
    replay.addEventListener("click", function () {
      stopPlayback();
      playAll(options.audioBlobs);
    });
    row.appendChild(replay);
    turn.appendChild(row);
  }

  el.conversation.appendChild(turn);
  el.conversation.scrollTop = el.conversation.scrollHeight;
  return turn;
}

function showThinking() {
  clearEmptyState();
  var turn = node("div", "turn turn--bot");
  turn.id = "thinkingTurn";

  var dots = node("div", "thinking");
  dots.appendChild(node("span"));
  dots.appendChild(node("span"));
  dots.appendChild(node("span"));
  turn.appendChild(dots);

  el.conversation.appendChild(turn);
  el.conversation.scrollTop = el.conversation.scrollHeight;
}

function hideThinking() {
  var turn = document.getElementById("thinkingTurn");
  if (turn) {
    turn.remove();
  }
}

// --- status ----------------------------------------------------------------

function setStatus(label, hint, tone) {
  el.statusLabel.textContent = label;
  el.statusLabel.setAttribute("data-tone", tone || "");
  el.statusHint.textContent = hint;
}

function setPhase(phase) {
  state.phase = phase;
  el.micButton.setAttribute("data-state", phase);
  el.micButton.setAttribute(
    "aria-label",
    phase === "listening" ? "Stop recording" : "Start recording"
  );
}

function idle() {
  setPhase("idle");
  if (state.lastTotalMs) {
    setStatus(
      "Answered in " + Math.round(state.lastTotalMs) + " ms",
      "Tap to start another turn",
      "done"
    );
  } else {
    setStatus("Ready", "Tap the microphone to start a conversation", "");
  }
}

/**
 * Something failed. Say so -- shown and spoken, in the language of the
 * conversation -- with the service's own words when it sent some, and never
 * the browser's ("Failed to fetch") or a status code.
 */
function fail(error) {
  hideThinking();
  var kind = error && error.kind;
  if (kind === "aborted") {
    // The caller started over; there is nothing to report.
    idle();
    return;
  }
  var message = typeof error === "string" ? error : explain(error);
  var key = typeof error === "string" ? null : failureKey(error);
  setStatus("Something went wrong", message, "error");
  addTurn("bot", message, { error: true });
  // Said as well as shown: someone listening rather than looking must not be
  // left in silence. In the bot's own voice, recorded in advance.
  var clip = key && state.clips[key + ":" + state.language];
  if (clip) {
    setPhase("speaking");
    play(clip).then(idle, idle);
  } else {
    setPhase("idle");
  }
}

/**
 * Play a reply that arrives a piece at a time: the first piece came with the
 * turn, and each next one is asked for while the one before it plays. Every
 * piece is added to `blobs`, so the replay button plays the whole reply. A
 * piece the voice could not make means it failed partway: the bot says so in
 * its own recorded voice, and the rest is on the screen. A reply the service
 * stopped (a newer turn, a hang-up) just stops.
 */
function playReply(first, turn, blobs) {
  var total = turn.audio_more || 0;
  var generation = state.playGeneration;
  var asked = [];

  function piece(n) {
    if (!asked[n]) {
      asked[n] = api.replyAudio(turn.session_id, turn.audio_id, n).catch(function () {
        return null;
      });
    }
    return asked[n];
  }

  function from(n, blob) {
    if (n < total) {
      piece(n + 1); // made while this one plays
    }
    return play(blob).then(function () {
      if (generation !== state.playGeneration || n >= total) {
        return undefined;
      }
      return piece(n + 1).then(function (next) {
        if (generation !== state.playGeneration || next === "stopped") {
          return undefined;
        }
        if (!next) {
          var clip = state.clips["voice_unavailable:" + state.language];
          return clip ? play(clip) : undefined;
        }
        blobs.push(next);
        return from(n + 1, next);
      });
    });
  }

  return from(0, first);
}

// --- a turn ------------------------------------------------------------------

// Redrawn when a turn lands. The conversation does not know which screens
// exist -- they say so themselves by registering here, which keeps this
// module free of them and the dependency pointing one way.
var listeners = [];

/** Hear about every turn after the conversation has been drawn. */
function onTurn(listener) {
  listeners.push(listener);
}

/** The read-back the customer is agreeing to, taken from the pending tool. */
function pendingSummary(turn) {
  var calls = turn.tool_calls || [];
  for (var i = calls.length - 1; i >= 0; i -= 1) {
    var result = calls[i].result || {};
    if (result.status === "needs_confirmation" && result.summary) {
      return result.summary + "?";
    }
  }
  return "This moves money. Say yes, or choose below.";
}

function handleTurn(turn) {
  var blob = turn.audio_b64 ? base64ToBlob(turn.audio_b64, "audio/wav") : null;
  var blobs = blob ? [blob] : null;
  showTurn(turn, blobs);

  if (blob) {
    setPhase("speaking");
    setStatus("Speaking", "Talk over it to interrupt", "busy");
    playReply(blob, turn, blobs).then(idle, idle);
  } else {
    idle();
  }
}

/**
 * Draw a turn -- the words, the screens, the Yes and Cancel buttons -- and
 * play nothing. The live session plays its replies itself, as they stream.
 * Returns the reply's text.
 */
function showTurn(turn, blobs) {
  state.sessionId = turn.session_id;
  state.lastTotalMs = (turn.latency_ms || {}).total || null;
  hideThinking();

  if (turn.text) {
    addTurn("you", turn.text);
  }

  var reply = turn.reply || "";
  // What the next failure is said in.
  state.language = /[\u0600-\u06FF]/.test(reply) ? "ur" : "en";

  addTurn("bot", reply, {
    audioBlobs: blobs || null,
    guardHit: Boolean(turn.guard_hit),
    caseRef: turn.handed_off || turn.dispute_filed || null,
    caseLabel: turn.handed_off ? "Passed to a specialist" : "Dispute filed",
  });

  // Drawn after the conversation, so whatever a screen is showing and the
  // sentence the bot just spoke come from the same answer and describe the
  // same moment. Which screens exist is not this module's business: each one
  // asks to be told, and the dependency only ever runs one way.
  listeners.forEach(function (listener) {
    listener(turn);
  });

  // An action waits on a yes. Buttons remove any doubt about whether the
  // word was heard correctly, which matters more here than anywhere else --
  // and they act on the pending action directly, not through the model.
  el.confirm.hidden = !turn.awaiting_confirmation;
  if (turn.awaiting_confirmation) {
    el.confirmText.textContent = pendingSummary(turn);
  }
  return reply;
}

/** The Yes or Cancel button. Rendered exactly like any other turn. */
function resolvePending(promise) {
  showThinking();
  setStatus("Working", "", "busy");
  return promise.then(handleTurn, fail);
}

function send(text) {
  if (state.phase === "thinking") {
    // One turn at a time: the service would only say it is still busy.
    return;
  }
  stopPlayback();
  setPhase("thinking");
  setStatus("Processing", "Thinking", "busy");
  addTurn("you", text);
  showThinking();

  api
    .chat(text)
    .then(function (turn) {
      // The text is already on screen; do not echo it twice.
      turn.text = "";
      handleTurn(turn);
    })
    .catch(fail);
}

export {
  addTurn,
  fail,
  handleTurn,
  hideThinking,
  showTurn,
  idle,
  onTurn,
  resolvePending,
  send,
  setPhase,
  setStatus,
  showThinking,
};
