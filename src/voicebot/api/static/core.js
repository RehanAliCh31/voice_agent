/* Shared ground: what the page knows, what it is pointing at, and how it
 * talks to the server.
 *
 * `state` is the page's own memory -- which screen is up, which card is
 * open, the session the server gave us. `el` is filled once on load so an
 * id is never looked up twice. The rest is small enough to have no opinions
 * of its own, which is why it can sit under everything else.
 */

var state = {
  phase: "idle", // idle | listening | thinking | speaking
  sessionId: null,
  stream: null,
  recorder: null,
  chunks: [],
  playing: null,
  vad: null,
  voice: null,
  toldAboutVoices: false,
  balances: {}, // account name -> cents, as of the last turn
  lastTotalMs: null,
  view: "assistant",
  cards: [],
  card: null, // the card being looked at, by last four
  ask: null, // the card action being asked about, if any
  detailsTimer: null, // counts the card details off the screen
  pin: null, // the card and purpose the PIN dialogue is open for
};

var el = {};

// --- the API ------------------------------------------------------------

var api = {
  ready: function () {
    return fetch("/ready").then(function (response) {
      if (!response.ok) {
        throw new Error("not ready");
      }
      return response.json();
    });
  },

  converse: function (wavBlob) {
    var form = new FormData();
    form.append("audio", wavBlob, "turn.wav");
    if (state.sessionId) {
      form.append("session_id", state.sessionId);
    }
    return request("/v1/converse", { method: "POST", body: form });
  },

  chat: function (text) {
    return request("/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, session_id: state.sessionId }),
    });
  },

  confirm: function () {
    return request("/v1/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
  },

  cancel: function () {
    return request("/v1/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
  },

  accounts: function () {
    return request("/v1/accounts", { method: "GET" });
  },

  statement: function () {
    return request("/v1/statement", { method: "GET" });
  },

  cards: function () {
    return request("/v1/cards", { method: "GET" });
  },

  monthly: function () {
    return request("/v1/monthly", { method: "GET" });
  },

  // Three requests that carry a PIN. It is sent to be checked and held
  // nowhere afterwards: not in state, not in the conversation, not in a
  // log line.
  details: function (last4, pin) {
    return post("/v1/cards/" + encodeURIComponent(last4) + "/details", { pin: pin });
  },

  checkPin: function (last4, pin) {
    return post("/v1/cards/" + encodeURIComponent(last4) + "/pin/check", { pin: pin });
  },

  changePin: function (last4, current, next) {
    return post("/v1/cards/" + encodeURIComponent(last4) + "/pin", {
      current_pin: current,
      new_pin: next,
    });
  },

  arm: function (tool, args, reason) {
    return request("/v1/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tool: tool,
        arguments: args,
        reason: reason,
        session_id: state.sessionId,
      }),
    });
  },

  reset: function () {
    return fetch("/v1/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
  },

  endSession: function () {
    if (!state.sessionId) {
      return Promise.resolve();
    }
    return fetch("/v1/session/" + encodeURIComponent(state.sessionId), {
      method: "DELETE",
    });
  },
};

function post(url, body) {
  return request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function request(url, options) {
  return fetch(url, options).then(function (response) {
    return response.json().then(
      function (body) {
        if (!response.ok) {
          throw new Error(body.detail || "The server returned " + response.status);
        }
        return body;
      },
      function () {
        throw new Error("The server returned " + response.status);
      }
    );
  });
}

// --- small helpers ---------------------------------------------------------

function node(tag, className, text) {
  var element = document.createElement(tag);
  if (className) {
    element.className = className;
  }
  if (text !== undefined && text !== null) {
    element.textContent = text;
  }
  return element;
}

function money(cents) {
  return "$" + (cents / 100).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function capitalise(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

/** A stamp like "2026-10-27 10:15:00" as a local Date. */
function momentOf(stamp) {
  return new Date(String(stamp || "").replace(" ", "T"));
}

/**
 * The heading a movement sits under. Today and yesterday are named because
 * that is how someone thinks about a charge they are trying to place; older
 * ones get their date.
 */
function dayLabel(moment) {
  // Both ends normalised to midnight first. Comparing the raw gap counts
  // calendar days wrong whenever the clock disagrees with the date: a
  // charge at 1am yesterday is 23 hours old and still yesterday.
  var midnight = new Date();
  midnight.setHours(0, 0, 0, 0);
  var that = new Date(moment.getTime());
  that.setHours(0, 0, 0, 0);

  var days = Math.round((midnight - that) / 86400000);
  if (days <= 0) {
    return "Today";
  }
  if (days === 1) {
    return "Yesterday";
  }
  return moment.toLocaleDateString(undefined, {
    day: "numeric",
    month: "long",
    year: moment.getFullYear() === midnight.getFullYear() ? undefined : "numeric",
  });
}

export { api, capitalise, dayLabel, el, momentOf, money, node, state };
