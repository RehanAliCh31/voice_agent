/* Sound in, and sound out.
 *
 * The microphone gives WebM; the API wants 16 kHz mono PCM, so everything
 * recorded passes through `toWav` before it leaves the page. A reply comes
 * back either as audio to play or as words for the browser to say, and
 * either way it arms the barge-in detector for as long as it lasts.
 */

import { state } from "./core.js";

var TARGET_RATE = 16000;

/** Encode float samples as the 16-bit PCM WAV the API decodes. */
function encodeWav(samples, sampleRate) {
  var buffer = new ArrayBuffer(44 + samples.length * 2);
  var view = new DataView(buffer);

  function ascii(offset, text) {
    for (var i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  }

  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM header length
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // channels: mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // bytes per second
  view.setUint16(32, 2, true); // bytes per frame
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);

  for (var i = 0; i < samples.length; i += 1) {
    // Clip first: above 1.0 would wrap to a large negative value and
    // arrive at the server as a loud click.
    var sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, sample * 32767, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * MediaRecorder hands us Opus in a WebM container; the API wants 16 kHz
 * mono PCM. Rendering through an OfflineAudioContext does the decode, the
 * downmix and the resample in one pass.
 */
function toWav(blob) {
  var Ctx = window.AudioContext || window.webkitAudioContext;

  return blob.arrayBuffer().then(function (raw) {
    var context = new Ctx();
    return context.decodeAudioData(raw).then(function (decoded) {
      context.close();
      var frames = Math.ceil(decoded.duration * TARGET_RATE);
      var offline = new OfflineAudioContext(1, Math.max(frames, 1), TARGET_RATE);
      var source = offline.createBufferSource();
      source.buffer = decoded;
      source.connect(offline.destination);
      source.start();
      return offline.startRendering().then(function (rendered) {
        return encodeWav(rendered.getChannelData(0), TARGET_RATE);
      });
    });
  });
}

function base64ToBlob(base64, type) {
  var binary = atob(base64);
  var bytes = new Uint8Array(binary.length);
  for (var i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: type });
}

function stopPlayback() {
  if (state.playing) {
    state.playing.pause();
    URL.revokeObjectURL(state.playing.src);
    state.playing = null;
  }
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
}

/** True when this browser can actually speak. Some have no voices at all. */
function canSpeak() {
  if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {
    return false;
  }
  var voices = window.speechSynthesis.getVoices();
  return !voices || voices.length > 0;
}

/** Speak text with the browser's own synthesiser, arming barge-in. */
function speak(text) {
  stopPlayback();

  return new Promise(function (resolve) {
    if (!text || !canSpeak()) {
      resolve();
      return;
    }

    var utterance = new SpeechSynthesisUtterance(text);
    if (state.voice) {
      utterance.voice = state.voice;
    }
    utterance.rate = 1.05;

    var settled = false;
    function finish() {
      if (settled) {
        return;
      }
      settled = true;
      if (state.vad) {
        state.vad.disarm();
      }
      resolve();
    }

    utterance.onstart = function () {
      if (state.vad) {
        state.vad.arm();
      }
    };
    utterance.onend = finish;
    utterance.onerror = finish;

    window.speechSynthesis.speak(utterance);
  });
}

/** Play the bot's reply, arming barge-in for as long as it lasts. */
function play(blob) {
  stopPlayback();

  return new Promise(function (resolve) {
    var audio = new Audio(URL.createObjectURL(blob));
    state.playing = audio;

    function finish() {
      if (state.vad) {
        state.vad.disarm();
      }
      if (state.playing === audio) {
        URL.revokeObjectURL(audio.src);
        state.playing = null;
      }
      resolve();
    }

    audio.addEventListener("ended", finish);
    audio.addEventListener("error", finish);

    audio.play().then(
      function () {
        if (state.vad) {
          state.vad.arm();
        }
      },
      function () {
        // Autoplay was refused until the user interacts. Not fatal: the
        // reply is on screen and has a replay button.
        finish();
      }
    );
  });
}

/** Pick a voice once. Chrome populates the list asynchronously. */
function loadVoices() {
  if (!window.speechSynthesis) {
    return;
  }
  var voices = window.speechSynthesis.getVoices();
  if (!voices.length) {
    return;
  }
  state.voice =
    voices.filter(function (v) {
      return v.localService && /^en/i.test(v.lang);
    })[0] ||
    voices.filter(function (v) {
      return /^en/i.test(v.lang);
    })[0] ||
    voices[0];
}

/**
 * Fill the ledger before the first turn, so the page opens showing the
 * customer's money rather than placeholders. A plain read: it takes no
 * turn, spends no session, and never reaches the model.
 */

export { base64ToBlob, canSpeak, loadVoices, play, speak, stopPlayback, toWav };
