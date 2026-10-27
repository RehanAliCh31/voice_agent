/* Voice Banking -- the client.
 *
 * One turn is: record, convert to the WAV the API expects, POST it to
 * /v1/converse, render what came back, speak the reply. While the reply is
 * playing the barge-in detector is armed, so talking over the bot stops it
 * and starts the next turn.
 *
 * ONE SET OF FACTS. Every turn's answer carries the account state as it
 * stands after that turn, and the page draws the ledger from it. Nothing
 * fetches balances separately, so the figure in a card and the figure the
 * bot just spoke can never disagree.
 *
 * No text ever reaches the DOM as markup: every element is built with
 * createElement and textContent. A transcript is user input echoed back, and
 * the model's own output is not trusted markup either. The one innerHTML in
 * this file writes a fixed SVG literal with no interpolation.
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
    voice: null,
    toldAboutVoices: false,
    balances: {}, // account name -> cents, as of the last turn
    lastTotalMs: null,
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

  // --- small helpers ------------------------------------------------------

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

  // --- the ledger ---------------------------------------------------------

  /**
   * Draw the accounts from the turn that just happened, marking any whose
   * balance it changed. Showing which figure moved is the whole reason the
   * ledger sits beside the conversation.
   */
  function renderAccounts(accounts) {
    el.accounts.replaceChildren();

    if (!accounts || !accounts.length) {
      var waiting = node("div", "account account--waiting");
      waiting.appendChild(node("span", "account__name", "—"));
      waiting.appendChild(node("span", "account__amount", "····"));
      el.accounts.appendChild(waiting);
      return;
    }

    accounts.forEach(function (account) {
      var before = state.balances[account.name];
      var change =
        typeof before === "number" ? account.balance_cents - before : 0;

      var card = node("div", "account");
      if (change !== 0) {
        card.classList.add("account--moved");
      }
      card.appendChild(node("span", "account__name", account.name));
      card.appendChild(node("span", "account__amount", account.balance));

      if (change !== 0) {
        var sign = change > 0 ? "+" : "−";
        card.appendChild(
          node("span", "account__delta", sign + money(Math.abs(change)) + " just now")
        );
      }

      el.accounts.appendChild(card);
      state.balances[account.name] = account.balance_cents;
    });
  }

  function renderBills(bills) {
    el.bills.replaceChildren();

    if (!bills || !bills.length) {
      el.bills.appendChild(node("p", "block__empty", "No bills on file."));
      return;
    }

    bills.forEach(function (bill) {
      var row = node("div", "bill");
      if (bill.paid) {
        row.classList.add("bill--paid");
      }
      row.appendChild(node("span", "bill__name", bill.name));
      row.appendChild(
        bill.paid
          ? node("span", "bill__paid", "paid")
          : node("span", "bill__amount", bill.amount)
      );
      el.bills.appendChild(row);
    });
  }

  // --- the conversation ---------------------------------------------------

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

    turn.appendChild(node("p", "turn__text", text));

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

    if (options.audioBlob || options.replayText) {
      var row = node("div", "turn__audio");
      var replay = node("button", "replay");
      replay.type = "button";
      replay.setAttribute("aria-label", "Replay this reply");
      replay.innerHTML =
        '<svg width="10" height="10" viewBox="0 0 12 12" aria-hidden="true">' +
        '<path d="M2 1.5l8 4.5-8 4.5z" fill="currentColor"></path></svg>';
      replay.addEventListener("click", function () {
        if (options.audioBlob) {
          play(options.audioBlob);
        } else {
          speak(options.replayText);
        }
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

  /** Say once that this browser cannot speak, rather than seeming broken. */
  function noteSilence() {
    if (state.toldAboutVoices) {
      return;
    }
    state.toldAboutVoices = true;
    addTurn(
      "bot",
      "This browser has no speech voices installed, so replies are shown but " +
        "not spoken. Everything else works.",
      { notice: true }
    );
  }

  // --- status -------------------------------------------------------------

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

  function bargeHint() {
    return el.vadToggle.checked
      ? "Talk over it to interrupt"
      : "Barge-in is off — wait for it to finish";
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
      setStatus("Ready", "Tap the microphone to start a turn", "");
    }
  }

  function fail(message) {
    hideThinking();
    setPhase("idle");
    setStatus("Something went wrong", message, "error");
    addTurn("bot", message, { error: true });
  }

  // --- a turn -------------------------------------------------------------

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
    state.sessionId = turn.session_id;
    state.lastTotalMs = (turn.latency_ms || {}).total || null;
    hideThinking();

    if (turn.text) {
      addTurn("you", turn.text);
    }

    var blob = turn.audio_b64 ? base64ToBlob(turn.audio_b64, "audio/wav") : null;
    var reply = turn.reply || "";

    addTurn("bot", reply, {
      audioBlob: blob,
      replayText: turn.speak_client_side && canSpeak() ? reply : null,
      guardHit: Boolean(turn.guard_hit),
      caseRef: turn.handed_off || turn.dispute_filed || null,
      caseLabel: turn.handed_off ? "Passed to a specialist" : "Dispute filed",
    });

    // Drawn after the conversation, so the ledger and the sentence the bot
    // just spoke come from the same answer and describe the same moment.
    renderAccounts(turn.accounts);
    renderBills(turn.bills);

    // An action waits on a yes. Buttons remove any doubt about whether the
    // word was heard correctly, which matters more here than anywhere else --
    // and they act on the pending action directly, not through the model.
    el.confirm.hidden = !turn.awaiting_confirmation;
    if (turn.awaiting_confirmation) {
      el.confirmText.textContent = pendingSummary(turn);
    }

    if (blob) {
      setPhase("speaking");
      setStatus("Speaking", bargeHint(), "busy");
      play(blob).then(idle);
    } else if (turn.speak_client_side && canSpeak()) {
      setPhase("speaking");
      setStatus("Speaking", bargeHint(), "busy");
      speak(reply).then(idle);
    } else {
      if (turn.speak_client_side) {
        noteSilence();
      }
      idle();
    }
  }

  /** The Yes or Cancel button. Rendered exactly like any other turn. */
  function resolvePending(promise) {
    showThinking();
    setStatus("Working", "", "busy");
    return promise.then(handleTurn, function (error) {
      hideThinking();
      addTurn("bot", "Sorry — that didn't go through: " + error.message, { error: true });
      idle();
    });
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
      .catch(function (error) {
        fail(error.message);
      });
  }

  function onReset() {
    // Resets the conversation AND the demo balances. With one shared bank,
    // a drained checking account would otherwise outlive the conversation
    // that drained it, and every later transfer would fail for want of funds.
    api
      .reset()
      .catch(function () {})
      .finally(function () {
        return api.endSession().finally(function () {
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

  function onBargeIn() {
    stopPlayback();
    startRecording().catch(function () {
      idle();
    });
  }

  function describeEngine(name) {
    return String(name || "")
      .replace(/(ASR|LLM|TTS)?(Engine|Client)$/, "")
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

  /**
   * Fill the ledger before the first turn, so the page opens showing the
   * customer's money rather than placeholders. A plain read: it takes no
   * turn, spends no session, and never reaches the model.
   */
  function loadLedger() {
    api
      .accounts()
      .then(function (ledger) {
        renderAccounts(ledger.accounts);
        renderBills(ledger.bills);
      })
      .catch(function () {
        // The ledger keeps its placeholder; the first turn will fill it.
      });
  }

  function boot() {
    loadVoices();
    if (window.speechSynthesis) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    api.ready().then(
      function (info) {
        el.chipAsr.textContent = describeEngine(info.asr);
        el.chipLlm.textContent = describeEngine(info.llm);
        el.chipTts.textContent = describeEngine(info.tts);
        el.chipStatus.textContent = info.device + " ready";
        el.chipStatus.className = "chip chip--ready";
        loadLedger();
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
      "accounts", "bills",
      "conversation", "empty", "confirm", "confirmYes", "confirmNo",
      "micButton", "statusLabel", "statusHint", "levelFill",
      "sayForm", "sayInput", "resetButton",
      "vadToggle", "vadSensitivity", "vadValue", "vadNote",
      "chipAsr", "chipLlm", "chipTts", "chipStatus",
    ].forEach(function (id) {
      el[id] = document.getElementById(id);
    });

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
})();
