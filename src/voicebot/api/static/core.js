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
  language: "en", // the language of the last reply: what a failure is said in
  turnRequest: null, // the turn being waited for, so it can be abandoned
  playGeneration: 0, // moved on by every interruption: a reply still playing stops
  clips: {}, // the bot's own recordings of its failure sentences, "key:language" -> Blob
  // Whether the page speaks with the browser's own voice. Only when the
  // service says so: with the office voices it never does, because the
  // browser's voice may be the other gender and has no Urdu.
  clientVoice: false,
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
    return request("/v1/converse", { method: "POST", body: form }, TURN);
  },

  chat: function (text) {
    return request(
      "/v1/chat",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, session_id: state.sessionId }),
      },
      TURN
    );
  },

  messages: function () {
    return request("/v1/messages", { method: "GET" });
  },

  voice: function () {
    return request("/v1/voice", { method: "GET" });
  },

  /** A WAV the service has, as a Blob; null for 204 or 404. */
  audio: function (url) {
    return fetch(url, { method: "GET" }).then(function (response) {
      if (response.status === 204 || response.status === 404) {
        return null;
      }
      if (!response.ok) {
        throw failure("server", null, response.status);
      }
      return response.blob();
    });
  },

  /**
   * Piece `n` (from 1) of a reply's audio: a Blob; null when the voice failed
   * on it (204); "stopped" when the reply was stopped or replaced (404, 410).
   */
  replyAudio: function (sessionId, audioId, n) {
    var url =
      "/v1/sessions/" + encodeURIComponent(sessionId) + "/audio/" +
      encodeURIComponent(audioId) + "/" + n;
    return fetch(url, { method: "GET" }).then(function (response) {
      if (response.status === 204) {
        return null;
      }
      if (response.status === 404 || response.status === 410) {
        return "stopped";
      }
      if (!response.ok) {
        throw failure("server", null, response.status);
      }
      return response.blob();
    });
  },

  // The PIN is left out of the first press. Whether one is needed at all is
  // the server's answer to that press, and a PIN sent before it is asked for
  // is a PIN sent somewhere it may not be wanted.
  confirm: function (pin) {
    var body = { session_id: state.sessionId };
    if (pin) {
      body.pin = pin;
    }
    return request(
      "/v1/confirm",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      TURN
    );
  },

  cancel: function () {
    return request(
      "/v1/cancel",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId }),
      },
      TURN
    );
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
    return post("/v1/reset", { session_id: state.sessionId });
  },

  endSession: function () {
    if (!state.sessionId) {
      return Promise.resolve();
    }
    return request("/v1/session/" + encodeURIComponent(state.sessionId), {
      method: "DELETE",
    });
  },
};

// --- how long to wait ----------------------------------------------------
//
// A turn is given 20 s of model by the service, plus hearing and speaking;
// past 45 s it is not coming, and the caller is told rather than left with a
// spinner. Anything else is a quick read.
var TURN = { timeoutMs: 45000, turn: true };
var REQUEST_TIMEOUT_MS = 10000;

/** Abandon the turn being waited for: the caller has started over. */
function abortTurn() {
  if (state.turnRequest) {
    state.turnRequest.abort("abandoned");
  }
}

// --- what the page says when the service cannot -------------------------
//
// The service's own sentences, from its messages.yaml, are fetched on load
// and replace these. These two are kept here only for the moment the service
// cannot be reached at all -- which is exactly when the page cannot ask it.
var sentences = {
  connection_lost: {
    en: "Sorry — I've lost the connection. Please check your network and try again.",
  },
  timed_out: { en: "Sorry — that took too long. Please try again." },
  internal: { en: "Sorry — something went wrong on my side. Please try that again." },
};

function sentence(key, language) {
  var found = sentences[key] || sentences.internal;
  return (language === "ur" && found.ur) || found.en;
}

/** Which of the bot's own sentences a failed request is told in. */
function failureKey(error) {
  if (error && error.key) {
    return error.key; // the failure names its own sentence
  }
  var kind = error && error.kind;
  if (kind === "timeout") {
    return "timed_out";
  }
  if (kind === "network") {
    return "connection_lost";
  }
  return error && error.message ? null : "internal";
}

/**
 * What to tell the customer about a failed request: the service's own
 * sentence when it sent one, or the right one for a timeout or a lost
 * connection -- never an empty box or the browser's wording.
 */
function explain(error) {
  var key = failureKey(error);
  return key ? sentence(key, state.language) : error.message;
}

// The sentences the page may have to say when the service cannot, recorded
// by the service in the bot's own voice and kept here in advance -- a lost
// connection is exactly when they cannot be fetched.
var KEPT_CLIPS = [
  "connection_lost",
  "timed_out",
  "internal",
  "voice_unavailable",
  "speech_unavailable",
  "microphone_refused",
  "microphone_unavailable",
  "microphone_lost",
];

function loadClips() {
  KEPT_CLIPS.forEach(function (key) {
    ["en", "ur"].forEach(function (language) {
      api
        .audio("/v1/messages/" + key + "/audio?language=" + language)
        .then(function (blob) {
          if (blob) {
            state.clips[key + ":" + language] = blob;
          }
        })
        .catch(function () {
          // Not recorded yet, or the service is down: asked again next load.
        });
    });
  });
}

function loadSentences() {
  api
    .voice()
    .then(function (voice) {
      state.clientVoice = Boolean(voice && voice.speaks_client_side);
    })
    .catch(function () {
      // Unknown: the browser's voice stays unused, which is never wrong.
    });
  loadClips();
  return api
    .messages()
    .then(function (table) {
      Object.keys(table).forEach(function (key) {
        sentences[key] = table[key];
      });
    })
    .catch(function () {
      // The built-in ones stand; the next load will try again.
    });
}

function post(url, body) {
  return request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/**
 * A request that cannot wait for ever, and whose failure says what kind it
 * was: "timeout", "network" (the service could not be reached), "aborted"
 * (the page gave up on it on purpose), or "server" with the service's own
 * safe sentence as the message, when it sent one.
 */
function request(url, options, how) {
  how = how || {};
  var controller = new AbortController();
  var timer = setTimeout(function () {
    controller.abort("timeout");
  }, how.timeoutMs || REQUEST_TIMEOUT_MS);
  if (how.turn) {
    state.turnRequest = controller;
  }

  return fetch(url, Object.assign({}, options, { signal: controller.signal }))
    .then(
      function (response) {
        return response.json().then(
          function (body) {
            if (!response.ok) {
              // Only a sentence the service wrote for people. A 422 lists
              // fields, which are for developers, not for the page.
              var said = body && typeof body.detail === "string" ? body.detail : null;
              throw failure("server", said, response.status);
            }
            return body;
          },
          function () {
            throw failure("server", null, response.status);
          }
        );
      },
      function () {
        if (controller.signal.aborted) {
          throw failure(controller.signal.reason === "timeout" ? "timeout" : "aborted");
        }
        throw failure("network");
      }
    )
    .finally(function () {
      clearTimeout(timer);
      if (state.turnRequest === controller) {
        state.turnRequest = null;
      }
    });
}

function failure(kind, message, status) {
  var error = new Error(message || "");
  error.kind = kind;
  error.status = status || 0;
  return error;
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

export {
  abortTurn,
  api,
  capitalise,
  dayLabel,
  el,
  explain,
  failureKey,
  loadSentences,
  momentOf,
  money,
  node,
  sentence,
  state,
};
