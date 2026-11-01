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
      context.close().catch(function () {});
      var frames = Math.ceil(decoded.duration * TARGET_RATE);
      var offline = new OfflineAudioContext(1, Math.max(frames, 1), TARGET_RATE);
      var source = offline.createBufferSource();
      source.buffer = decoded;
      source.connect(offline.destination);
      source.start();
      return offline.startRendering().then(function (rendered) {
        return encodeWav(rendered.getChannelData(0), TARGET_RATE);
      });
    }, function (error) {
      // Closed on failure too: a context left open on every bad recording
      // is how a page runs out of audio contexts.
      context.close().catch(function () {});
      throw error;
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

/** Stop what is playing now, and every piece of the reply still to come. */
function stopPlayback() {
  state.playGeneration += 1;
  halt();
}

/** Stop what is playing now; a reply going on to its next piece carries on. */
function halt() {
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
  halt();

  return new Promise(function (resolve) {
    if (!text || !canSpeak()) {
      resolve();
      return;
    }

    // Urdu goes to an Urdu voice, or to none: read by an English voice it is
    // noise, and the words are on the screen anyway.
    var urdu = /[\u0600-\u06FF]/.test(text);
    var voice = urdu ? urduVoice() : state.voice;
    if (urdu && !voice) {
      resolve();
      return;
    }

    var utterance = new SpeechSynthesisUtterance(text);
    if (voice) {
      utterance.voice = voice;
      utterance.lang = voice.lang;
    }
    utterance.rate = 1.05;

    var settled = false;
    // Some browsers never fire onend -- after a cancel, or a voice that
    // failed to load -- and the page would say "Speaking" for ever.
    var watchdog = setTimeout(finish, Math.max(5000, text.length * 120));
    function finish() {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(watchdog);
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
  halt();

  return new Promise(function (resolve) {
    var audio = new Audio(URL.createObjectURL(blob));
    state.playing = audio;
    if (state.route) {
      state.route(audio); // through the live session's output, while there is one
    }

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
function urduVoice() {
  if (!window.speechSynthesis) {
    return null;
  }
  return (
    window.speechSynthesis.getVoices().filter(function (v) {
      return /^ur/i.test(v.lang);
    })[0] || null
  );
}

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

/** Play several clips one after another; an interruption stops the rest. */
function playAll(blobs) {
  var generation = state.playGeneration;
  return blobs.reduce(function (chain, blob) {
    return chain.then(function () {
      return generation === state.playGeneration ? play(blob) : undefined;
    });
  }, Promise.resolve());
}

export { base64ToBlob, canSpeak, loadVoices, play, playAll, speak, stopPlayback, toWav };
