/* The microphone: the hands-free live session (live.js).
 *
 * One click starts a conversation the bot hears and answers on its own, and
 * that it stops talking in when spoken over; one click ends it. This module
 * only shows what the session is doing.
 */

import { el, state } from "./core.js";
import { play, stopPlayback } from "./audio.js";
import {
  addTurn,
  fail,
  hideThinking,
  idle,
  setPhase,
  setStatus,
  showThinking,
  showTurn,
} from "./conversation.js";

// --- the live session ------------------------------------------------------------

// What the button and the status line say in each of the session's states.
var LIVE_STATUS = {
  connecting: ["thinking", "Connecting", "Opening the microphone"],
  reconnecting: ["thinking", "Reconnecting", "The connection dropped; joining again"],
  listening: ["listening", "Listening", "Speak whenever you like · tap to end"],
  user_speaking: ["hearing", "Hearing you", "Pause when you have finished"],
  thinking: ["thinking", "Thinking", "Tap to end the conversation"],
  agent_speaking: ["speaking", "Speaking", "Talk over it to interrupt"],
};

var live = null;

function liveSupported() {
  return Boolean(
    window.WebSocket &&
      window.AudioWorkletNode &&
      window.LiveClient &&
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia
  );
}

function liveClient() {
  if (live) {
    return live;
  }
  var Ctx = window.AudioContext || window.webkitAudioContext;
  live = new window.LiveClient({
    url: (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/v1/live",
    deps: {
      WebSocket: window.WebSocket,
      getUserMedia: function (constraints) {
        return navigator.mediaDevices.getUserMedia(constraints);
      },
      createAudioContext: function () {
        return new Ctx();
      },
      createWorkletNode: function (context, name, options) {
        return new AudioWorkletNode(context, name, options);
      },
      workletUrl: "live-capture.js",
      now: function () {
        return window.performance.now();
      },
      setTimeout: window.setTimeout.bind(window),
      clearTimeout: window.clearTimeout.bind(window),
    },
    on: { state: onLiveState, event: onLiveEvent, level: onLiveLevel, error: onLiveError },
  });
  return live;
}

function onLiveState(name) {
  if (name === "idle") {
    state.live = false;
    state.route = null;
    el.micButton.setAttribute("aria-label", "Start a conversation");
    el.levelFill.style.width = "0%";
    hideThinking();
    idle();
    return;
  }
  state.live = true;
  state.route = function (element) {
    if (live) {
      live.route(element);
    }
  };
  if (name === "user_speaking" || name === "agent_speaking") {
    // A reply to a button (Yes, Cancel, the keypad) plays through the page's
    // own player, which the live session does not know of. The caller being
    // heard, or the session speaking, stops it: the bot never talks over the
    // caller, or over itself.
    stopPlayback();
  }
  var shown = LIVE_STATUS[name];
  if (!shown) {
    return;
  }
  setPhase(shown[0]);
  setStatus(shown[1], shown[2], name === "listening" ? "" : "busy");
  el.micButton.setAttribute("aria-label", "End the conversation");
  if (name === "thinking") {
    showThinking();
  } else {
    hideThinking();
  }
}

function onLiveEvent(event) {
  switch (event.type) {
    case "started":
      state.sessionId = event.session_id;
      break;
    case "transcript":
      hideThinking();
      addTurn("you", event.text);
      break;
    case "reply":
      // The words were shown as they were heard; the reply is drawn, and
      // the live session plays its audio as it arrives.
      showTurn(Object.assign({}, event, { text: "" }));
      break;
    case "say":
      hideThinking();
      addTurn("bot", event.reply);
      break;
    case "metrics":
      state.lastTotalMs = event.end_of_speech_to_first_audio_ms;
      break;
    case "audio_failed":
      // The voice failed part way: the rest is on the screen, and the bot
      // says so in its own recorded voice.
      var clip = state.clips["voice_unavailable:" + state.language];
      if (clip) {
        play(clip);
      }
      break;
    case "error":
      fail(event.key ? { kind: "service", key: event.key } : event.message);
      break;
    default:
      break;
  }
}

function onLiveLevel(level) {
  el.levelFill.style.width = Math.min(100, Math.round(level * 400)) + "%";
}

// Each in the conversation's language, said in the bot's own recorded voice.
var LIVE_ERRORS = {
  connection_lost: "connection_lost",
  microphone_refused: "microphone_refused",
  microphone_lost: "microphone_lost",
};

function onLiveError(kind) {
  fail({ kind: kind, key: LIVE_ERRORS[kind] || "microphone_unavailable" });
}

/** End the live conversation, if one is running: on reset, or when the page goes. */
function stopLive(reason) {
  if (live) {
    live.stop(reason || "stopped");
  }
}

/** A tab back in view: wake the live session's audio if the browser paused it. */
function wakeLive() {
  if (live) {
    live.wake();
  }
}

function onMicClick() {
  if (liveSupported()) {
    var client = liveClient();
    if (client.phase === "idle") {
      stopPlayback();
      client.sessionId = state.sessionId;
    }
    client.toggle();
    return;
  }
  // A browser without the live session's audio, or a page that is not
  // secure, has no microphone to offer.
  fail("This browser cannot hold a voice conversation here. You can still type below.");
}

export { onMicClick, stopLive, wakeLive };
