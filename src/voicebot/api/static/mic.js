/* The microphone: the hands-free live session, and push-to-talk where the
 * browser cannot do the live one.
 *
 * Live (live.js): one click starts a conversation the bot hears and answers
 * on its own, and that it stops talking in when spoken over; one click ends
 * it. This module only shows what the session is doing.
 *
 * Push-to-talk: tap to record, tap again to send. The stream is asked for
 * once and kept, because asking again mid-sentence is a permission prompt
 * in front of somebody who is already talking.
 */

import { api, el, state } from "./core.js";
import { play, stopPlayback, toWav } from "./audio.js";
import {
  addTurn,
  fail,
  handleTurn,
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
    // The page's own barge-in detector is for push-to-talk; it comes back.
    if (state.vad) {
      state.vad.enabled = el.vadToggle.checked;
    }
    el.micButton.setAttribute("aria-label", "Start a conversation");
    el.levelFill.style.width = "0%";
    hideThinking();
    idle();
    return;
  }
  state.live = true;
  if (state.vad) {
    state.vad.enabled = false; // the service decides barge-in now
  }
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
      fail(event.message);
      break;
    default:
      break;
  }
}

function onLiveLevel(level) {
  el.levelFill.style.width = Math.min(100, Math.round(level * 400)) + "%";
}

function onLiveError(kind) {
  if (kind === "connection_lost") {
    fail({ kind: "network" });
  } else if (kind === "microphone_refused") {
    fail("The microphone was refused. You can still type below.");
  } else if (kind === "microphone_lost") {
    fail("The microphone was disconnected. Tap to start again.");
  } else {
    fail("The microphone could not be opened. You can still type below.");
  }
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

// --- push-to-talk ------------------------------------------------------------------

function ensureStream() {
  if (state.stream) {
    return Promise.resolve(state.stream);
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // Not a secure page, or a browser without it: a rejection, not a throw
    // that escapes the handler waiting for one.
    return Promise.reject(new Error("no microphone here"));
  }

  return navigator.mediaDevices
    .getUserMedia({
      audio: {
        // Without echo cancellation the bot hears itself through the
        // speakers and barges in on its own reply.
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    })
    .then(function (stream) {
      state.stream = stream;
      state.vad.connect(stream);
      return stream;
    });
}

function startRecording() {
  stopPlayback();

  return ensureStream().then(function (stream) {
    state.chunks = [];
    state.recorder = new MediaRecorder(stream);
    state.recorder.addEventListener("dataavailable", function (event) {
      if (event.data && event.data.size) {
        state.chunks.push(event.data);
      }
    });
    state.recorder.start();
    setPhase("listening");
    setStatus("Listening", "Speak now — tap again when you finish", "busy");
  });
}

function stopRecording() {
  return new Promise(function (resolve) {
    var recorder = state.recorder;
    if (!recorder || recorder.state === "inactive") {
      resolve(null);
      return;
    }
    recorder.addEventListener("stop", function () {
      resolve(new Blob(state.chunks, { type: recorder.mimeType }));
    });
    // A recorder that fails never fires "stop"; without this the turn waits
    // for ever.
    recorder.addEventListener("error", function () {
      resolve(null);
    });
    try {
      recorder.stop();
    } catch (error) {
      resolve(null);
    }
  });
}

function finishTurn() {
  setPhase("thinking");
  setStatus("Processing", "Transcribing, then thinking", "busy");
  el.levelFill.style.width = "0%";

  return stopRecording()
    .then(function (blob) {
      if (!blob || !blob.size) {
        throw new Error("Nothing was recorded. Check the microphone.");
      }
      showThinking();
      return toWav(blob);
    })
    .then(api.converse)
    .then(handleTurn)
    .catch(fail);
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
  if (state.phase === "thinking") {
    // One turn at a time. A second one sent now would only be told the
    // service is still answering the first.
    return;
  }
  if (state.phase === "listening") {
    finishTurn();
    return;
  }
  startRecording().catch(function () {
    fail("The microphone was refused. You can still type below.");
  });
}

function onBargeIn() {
  stopPlayback();
  startRecording().catch(function () {
    idle();
  });
}

export { onBargeIn, onMicClick, stopLive, wakeLive };
