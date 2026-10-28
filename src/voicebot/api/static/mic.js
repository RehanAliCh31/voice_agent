/* The microphone: holding the stream, recording a turn, and stopping the
 * bot by speaking over it.
 *
 * The stream is asked for once and kept, because asking again mid-sentence
 * is a permission prompt in front of somebody who is already talking.
 */

import { api, el, state } from "./core.js";
import { stopPlayback, toWav } from "./audio.js";
import {
  fail,
  handleTurn,
  idle,
  setPhase,
  setStatus,
  showThinking,
} from "./conversation.js";

function ensureStream() {
  if (state.stream) {
    return Promise.resolve(state.stream);
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
    recorder.stop();
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
    .catch(function (error) {
      fail(error.message);
    });
}

function onMicClick() {
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

export { onBargeIn, onMicClick };
