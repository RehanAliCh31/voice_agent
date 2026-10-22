/* Voice Bot -- the client.
 *
 * One turn is: record, convert to the WAV the API expects, POST it to
 * /v1/converse, render what came back, play the reply. While the reply is
 * playing the barge-in detector is armed, so talking over the bot stops it
 * and starts the next turn.
 *
 * Every element is built with createElement and textContent, never
 * innerHTML. A transcript is user input echoed back, and the model's own
 * output is not trusted markup either.
 */

(function () {
  "use strict";

  var TARGET_RATE = 16000;

  var state = {
    phase: "idle", // idle | listening | thinking | speaking
    sessionId: null,
    stream: null,
    recorder: null,
    chunks: [],
    playing: null,
    vad: null,
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

  // --- audio --------------------------------------------------------------

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

  // --- rendering ----------------------------------------------------------

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

  function clearEmptyState() {
    if (el.empty && el.empty.parentNode) {
      el.empty.remove();
    }
  }

  function addTurn(who, text, options) {
    clearEmptyState();
    options = options || {};

    var turn = node("div", "turn turn--" + (who === "YOU" ? "you" : "bot"));
    if (options.error) {
      turn.classList.add("turn--error");
    }
    turn.appendChild(node("div", "turn__who", who));

    var body = node("div", "turn__body");
    body.appendChild(node("p", "turn__text", text));

    if (options.guardHit) {
      var flag = node("div", "turn__flag");
      flag.textContent = "figure read back from your account";
      flag.title =
        "The model's own wording stated a figure no tool returned, so the " +
        "stored value was spoken instead.";
      body.appendChild(flag);
    }

    if (options.audioBlob || options.replayText) {
      var row = node("div", "turn__audio");

      var replay = node("button", "replay");
      replay.type = "button";
      replay.setAttribute("aria-label", "Replay this reply");
      replay.innerHTML =
        '<svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true">' +
        '<path d="M2 1.5l8 4.5-8 4.5z" fill="currentColor"></path></svg>';
      replay.addEventListener("click", function () {
        if (options.audioBlob) {
          play(options.audioBlob);
        } else {
          speak(options.replayText);
        }
      });

      row.appendChild(replay);
      body.appendChild(row);
    }

    turn.appendChild(body);
    el.conversation.appendChild(turn);
    el.conversation.scrollTop = el.conversation.scrollHeight;
    return turn;
  }

  function showThinking() {
    clearEmptyState();
    var turn = node("div", "turn turn--bot");
    turn.id = "thinkingTurn";
    turn.appendChild(node("div", "turn__who", "BOT"));

    var dots = node("div", "thinking");
    dots.appendChild(node("span"));
    dots.appendChild(node("span"));
    dots.appendChild(node("span"));

    var body = node("div", "turn__body");
    body.appendChild(dots);
    turn.appendChild(body);

    el.conversation.appendChild(turn);
    el.conversation.scrollTop = el.conversation.scrollHeight;
  }

  function hideThinking() {
    var turn = document.getElementById("thinkingTurn");
    if (turn) {
      turn.remove();
    }
  }

  function formatClock(seconds) {
    var whole = Math.round(seconds);
    return Math.floor(whole / 60) + ":" + String(whole % 60).padStart(2, "0");
  }

  function setStatus(label, hint, tone) {
    el.statusLabel.textContent = label;
    el.statusLabel.setAttribute("data-tone", tone || "");
    el.statusHint.textContent = hint;
  }

  function setPhase(phase) {
    state.phase = phase;
    el.mic.setAttribute("data-state", phase);
    el.micButton.setAttribute(
      "aria-label",
      phase === "listening" ? "Stop recording" : "Start recording"
    );
  }

  // --- the inspector ------------------------------------------------------

  /** The inspector before anything has happened: every field blank, not zero. */
  function resetInspector() {
    el.asrText.textContent = "awaiting audio";
    el.asrText.classList.add("transcript--idle");
    el.asrConf.textContent = "conf —";

    renderToolCalls([]);

    el.guardVerdict.textContent = "idle";
    el.guardVerdict.removeAttribute("data-verdict");
    el.guardNote.textContent = "Nothing spoken yet.";

    renderLatency({});
  }

  function renderInspector(turn) {
    if (!turn) {
      resetInspector();
      return;
    }

    var transcript = turn.transcript || {};
    el.asrText.textContent = transcript.text || "(nothing heard)";
    el.asrText.classList.toggle("transcript--idle", !transcript.text);
    el.asrConf.textContent =
      typeof transcript.confidence === "number"
        ? "conf " + transcript.confidence.toFixed(2)
        : "conf —";

    renderToolCalls(turn.tool_calls || []);
    renderGuard(turn);
    renderLatency(turn.latency_ms || {});
  }

  /** What the model actually did: which tools, with what, and what came back. */
  function renderToolCalls(calls) {
    el.tools.replaceChildren();

    if (!calls.length) {
      var blank = node("div", "tool");
      blank.appendChild(node("div", "tool__name tool__name--idle", "no tools called"));
      blank.appendChild(
        node("div", "tool__result", "The model answered without looking anything up.")
      );
      el.tools.appendChild(blank);
      return;
    }

    calls.forEach(function (call) {
      var row = node("div", "tool");
      row.appendChild(node("div", "tool__name", call.name));

      var args = call.arguments || {};
      var argText = Object.keys(args)
        .filter(function (key) {
          return key !== "confirmation_token";
        })
        .map(function (key) {
          return key + ": " + args[key];
        })
        .join(", ");
      if (argText) {
        row.appendChild(node("div", "tool__args", argText));
      }

      var result = call.result || {};
      row.appendChild(node("div", "tool__result", summariseResult(result)));
      if (result.error) {
        row.classList.add("tool--error");
      }
      el.tools.appendChild(row);
    });
  }

  /** One readable line from a tool's JSON result. */
  function summariseResult(result) {
    if (result.error) {
      return result.error;
    }
    if (result.status === "needs_confirmation") {
      return "waiting: " + result.summary;
    }
    var parts = Object.keys(result)
      .filter(function (key) {
        return key !== "token" && key.indexOf("_cents") === -1 && typeof result[key] !== "object";
      })
      .map(function (key) {
        return key + ": " + result[key];
      });
    return parts.length ? parts.join(", ") : "ok";
  }

  /** Whether the reply the user heard is the one the model wrote. */
  function renderGuard(turn) {
    var hit = Boolean(turn.guard_hit);
    el.guardVerdict.textContent = hit ? "suppressed" : "verified";
    el.guardVerdict.setAttribute("data-verdict", hit ? "out" : "in");
    el.guardNote.textContent = hit
      ? "A figure in the reply came from no tool. The bot read back the stored value instead."
      : "Every figure spoken came from the database.";
  }

  function renderLatency(latency) {
    el.stages.replaceChildren();

    var total = latency.total || 0;
    el.latencyTotal.textContent = total ? Math.round(total) + " ms" : "—";
    el.latencyTotal.setAttribute("data-filled", total ? "true" : "false");

    Object.keys(latency)
      .filter(function (name) {
        return name !== "total";
      })
      .forEach(function (name) {
        var ms = latency[name];
        var share = total ? ms / total : 0;

        var row = node("div", "stage-row");
        row.appendChild(node("span", "stage-row__name", name));

        var track = node("span", "stage-row__track");
        var fill = node("span", "stage-row__fill");
        fill.style.width = Math.round(share * 100) + "%";
        fill.setAttribute("data-heavy", share > 0.2 ? "true" : "false");
        track.appendChild(fill);
        row.appendChild(track);

        row.appendChild(node("span", "stage-row__ms", Math.round(ms) + " ms"));
        el.stages.appendChild(row);
      });
  }

  // --- a turn -------------------------------------------------------------

  /** The Yes or Cancel button. Rendered exactly like any other turn. */
  function resolvePending(promise) {
    showThinking();
    setStatus("Working", "", "busy");
    return promise.then(handleTurn, function (error) {
      hideThinking();
      addTurn("BOT", "Sorry — that didn't go through: " + error.message, { error: true });
      idle();
    });
  }

  function handleTurn(turn) {
    state.sessionId = turn.session_id;
    hideThinking();
    renderInspector(turn);

    if (turn.text) {
      addTurn("YOU", turn.text);
    }

    var blob = turn.audio_b64 ? base64ToBlob(turn.audio_b64, "audio/wav") : null;
    var reply = turn.reply || "";

    addTurn("BOT", reply, {
      audioBlob: blob,
      replayText: turn.speak_client_side ? reply : null,
      guardHit: Boolean(turn.guard_hit),
    });

    // An action waits on a yes. Buttons remove any doubt about whether the
    // word was heard correctly, which matters more here than anywhere else --
    // and they act on the pending action directly, not through the model.
    el.confirm.hidden = !turn.awaiting_confirmation;

    if (blob) {
      setPhase("speaking");
      setStatus("Speaking", bargeHint(), "busy");
      play(blob).then(idle);
    } else if (turn.speak_client_side && canSpeak()) {
      setPhase("speaking");
      setStatus("Speaking", bargeHint(), "busy");
      speak(reply).then(idle);
    } else {
      idle();
    }
  }

  function bargeHint() {
    return el.vadToggle.checked
      ? "Talk over it to interrupt"
      : "Barge-in is off — wait for it to finish";
  }

  function idle() {
    setPhase("idle");
    var total = el.latencyTotal.textContent;
    if (total && total !== "—") {
      setStatus("Answered in " + total, "Tap to start another turn", "done");
    } else {
      setStatus("Ready", "Tap the microphone to start a turn", "");
    }
  }

  function fail(message) {
    hideThinking();
    setPhase("idle");
    setStatus("Something went wrong", message, "error");
    addTurn("BOT", message, { error: true });
  }

  // --- recording ----------------------------------------------------------

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
    setStatus("Processing", "Transcribing, then reading the intent", "busy");
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

  // --- wiring -------------------------------------------------------------

  function onMicClick() {
    if (state.phase === "listening") {
      finishTurn();
      return;
    }
    startRecording().catch(function () {
      fail("The microphone was refused. You can still type below.");
    });
  }

  function onSay(event) {
    event.preventDefault();
    var text = el.sayInput.value.trim();
    if (!text) {
      return;
    }
    el.sayInput.value = "";
    send(text);
  }

  function send(text) {
    stopPlayback();
    setPhase("thinking");
    setStatus("Processing", "Reading the intent", "busy");
    addTurn("YOU", text);
    showThinking();

    api
      .chat(text)
      .then(function (turn) {
        // The text is already on screen; do not echo it twice.
        turn.text = "";
        handleTurn(turn);
      })
      .catch(function (error) {
        fail(error.message);
      });
  }

  function onReset() {
    // Resets the conversation AND the demo balances. With one shared bank,
    // a drained checking account would otherwise outlive the conversation
    // that drained it, and every later transfer would fail for want of funds.
    api.reset()
      .catch(function () {})
      .finally(function () {
        return api.endSession().finally(function () {
          state.sessionId = null;
          stopPlayback();
          el.conversation.replaceChildren(el.empty);
          el.confirm.hidden = true;
          resetInspector();
          setPhase("idle");
          setStatus("Ready", "Tap the microphone to start a turn", "");
        });
      });
  }

  function onBargeIn() {
    stopPlayback();
    startRecording().catch(function () {
      idle();
    });
  }

  function describeEngine(name) {
    return String(name || "")
      .replace(/(ASR|NLU|TTS)?Engine$/, "")
      .toLowerCase();
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

  function boot() {
    loadVoices();
    if (window.speechSynthesis) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    api.ready().then(
      function (info) {
        el.chipAsr.textContent = describeEngine(info.asr);
        el.chipNlu.textContent = describeEngine(info.llm);
        el.chipTts.textContent = describeEngine(info.tts);
        el.asrModel.textContent = describeEngine(info.asr);
        el.chipStatus.textContent = info.device + " ready";
        el.chipStatus.className = "chip chip--ready";
      },
      function () {
        el.chipStatus.textContent = "server down";
        el.chipStatus.className = "chip chip--down";
        setStatus("No server", "The API did not answer /ready", "error");
      }
    );
  }

  document.addEventListener("DOMContentLoaded", function () {
    [
      "conversation", "empty", "confirm", "confirmYes", "confirmNo",
      "tools", "guardVerdict", "guardNote",
      "mic", "micButton", "statusLabel", "statusHint", "levelFill",
      "sayForm", "sayInput", "resetButton",
      "vadToggle", "vadSensitivity", "vadValue", "vadNote",
      "chipAsr", "chipNlu", "chipTts", "chipStatus",
      "asrText", "asrConf", "asrModel",
      "latencyTotal", "stages",
    ].forEach(function (id) {
      el[id] = document.getElementById(id);
    });

    el.mic = document.querySelector(".mic");
    el.bargein = document.querySelector(".bargein");

    state.vad = new window.VoiceActivity({
      onSpeech: onBargeIn,
      onLevel: function (rms) {
        if (state.phase === "listening") {
          el.levelFill.style.width = Math.min(100, Math.round(rms * 400)) + "%";
        }
      },
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

    renderInspector({});
    boot();
  });
})();
