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
    view: "assistant",
    cards: [],
    card: null, // the card being looked at, by last four
    ask: null, // the card action being asked about, if any
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

    // Only what is still owed, and only the next few: the whole list is the
    // monthly screen's job, and nine rows here would crowd out the accounts.
    var owing = bills.filter(function (bill) {
      return !bill.paid;
    });
    var showing = owing.length ? owing : bills;
    var hidden = showing.length - 5;

    showing.slice(0, 5).forEach(function (bill) {
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

    if (hidden > 0) {
      var more = node("button", "bills__more", hidden + " more this month");
      more.type = "button";
      more.addEventListener("click", function () {
        showView("monthly");
      });
      el.bills.appendChild(more);
    }
  }

  // --- the screens ----------------------------------------------------------

  /**
   * Show one screen. The dock and any pending confirmation live outside the
   * screens, so switching never interrupts a turn or loses a yes that is
   * still waiting to be given.
   */
  function showView(name) {
    state.view = name;
    var tabs = el.screens.querySelectorAll(".screens__tab");
    Array.prototype.forEach.call(tabs, function (tab) {
      var chosen = tab.getAttribute("data-view") === name;
      var view = document.getElementById("view" + capitalise(tab.getAttribute("data-view")));
      if (chosen) {
        tab.setAttribute("aria-current", "page");
      } else {
        tab.removeAttribute("aria-current");
      }
      if (view) {
        view.hidden = !chosen;
      }
    });
    // The microphone belongs to the conversation. On a screen that is being
    // read rather than talked to it is clutter, and the status line under it
    // would be describing a turn that is not on screen.
    el.dock.hidden = name !== "assistant";
    el.views.scrollTop = 0;
    if (name === "transactions") {
      loadStatement();
    }
    if (name === "cards") {
      loadCards();
    }
    if (name === "monthly") {
      loadMonthly();
    }
  }

  function capitalise(word) {
    return word.charAt(0).toUpperCase() + word.slice(1);
  }

  // --- the transactions screen ------------------------------------------------

  function loadStatement() {
    api
      .statement()
      .then(renderStatement)
      .catch(function () {
        el.statement.replaceChildren(
          node("p", "block__empty", "Couldn't load your transactions just now.")
        );
      });
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

  function renderStatement(data) {
    el.statementTotal.textContent = data.total_balance;
    el.statementMonth.textContent = data.month_label;
    el.statementIn.textContent = data.incoming;
    el.statementOut.textContent = data.outgoing;

    el.statement.replaceChildren();
    if (!data.rows.length) {
      el.statement.appendChild(node("p", "block__empty", "Nothing moved this month."));
      return;
    }

    var day = null;
    data.rows.forEach(function (row) {
      var moment = momentOf(row.occurred_at);
      var label = dayLabel(moment);
      if (label !== day) {
        day = label;
        el.statement.appendChild(node("p", "statement__day", label));
      }
      el.statement.appendChild(movementRow(row, moment));
    });
  }

  function movementRow(row, moment) {
    var movement = node("div", "movement movement--" + (row.direction === "in" ? "in" : "out"));

    // An arrow for a transfer, an initial for a shop: a "C" against a move
    // between Checking and Savings says nothing the row does not already.
    movement.appendChild(
      node(
        "span",
        "movement__mark",
        row.kind === "transfer"
          ? row.direction === "in"
            ? "↓"
            : "↑"
          : row.merchant.slice(0, 1)
      )
    );

    // A shop is named; an account is said as a direction. "Checking /
    // Savings" reads as neither one thing nor the other when both sides are
    // the customer's own.
    var incoming = row.direction === "in";
    var name =
      row.kind === "transfer"
        ? (incoming ? "From " : "To ") + row.merchant
        : row.merchant;

    var who = node("div", "movement__who");
    who.appendChild(node("span", "movement__name", name));
    who.appendChild(node("span", "movement__where", row.account));
    movement.appendChild(who);

    var side = node("div", "movement__side");
    side.appendChild(
      node(
        "span",
        "movement__time",
        moment.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })
      )
    );
    side.appendChild(
      node("span", "movement__amount", (row.direction === "in" ? "+" : "−") + row.amount)
    );
    movement.appendChild(side);

    return movement;
  }

  // --- the cards screen -------------------------------------------------------

  function loadCards() {
    api
      .cards()
      .then(function (data) {
        state.cards = data.cards;
        // Keep the card already being looked at when this is a refresh after
        // a turn, so freezing a card does not jump the screen to another one.
        var chosen = state.card;
        var known = data.cards.some(function (card) {
          return card.last4 === chosen;
        });
        state.card = known ? chosen : (data.cards[0] || {}).last4 || null;
        renderCards();
      })
      .catch(function () {
        el.cardPane.replaceChildren(
          node("p", "block__empty", "Couldn't load your cards just now.")
        );
      });
  }

  function renderCards() {
    el.cardPicker.replaceChildren();
    el.cardPane.replaceChildren();

    if (!state.cards.length) {
      el.cardPane.appendChild(node("p", "block__empty", "No cards on file."));
      return;
    }

    state.cards.forEach(function (card) {
      var tab = node("button", "cardpicker__tab");
      tab.type = "button";
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(card.last4 === state.card));
      tab.appendChild(node("span", null, capitalise(card.kind)));
      tab.appendChild(node("span", "cardpicker__digits", "••" + card.last4));
      tab.addEventListener("click", function () {
        state.card = card.last4;
        renderCards();
      });
      el.cardPicker.appendChild(tab);
    });

    var showing = state.cards.filter(function (card) {
      return card.last4 === state.card;
    })[0];
    if (showing) {
      el.cardPane.appendChild(cardFace(showing));
      el.cardPane.appendChild(cardActions(showing));
      el.cardPane.appendChild(cardControls(showing));
    }
  }

  /**
   * The two things that can be done to a card from here. Both go through the
   * same gate the spoken path uses: the button arms the action and the bank
   * says what it is, and only then is there anything to agree to.
   */
  function cardActions(card) {
    var row = node("div", "cardacts");

    var freeze = node("button", "cardact cardact--stop", card.frozen ? "Already frozen" : "Freeze card");
    freeze.type = "button";
    freeze.disabled = card.frozen;
    freeze.addEventListener("click", function () {
      openAsk(card, "freeze");
    });
    row.appendChild(freeze);

    var replace = node("button", "cardact", "Replace card");
    replace.type = "button";
    replace.addEventListener("click", function () {
      openAsk(card, "replace");
    });
    row.appendChild(replace);

    return row;
  }

  // --- asking before acting -----------------------------------------------

  var ASKS = {
    freeze: {
      title: "Freeze this card",
      tool: "freeze_card",
      args: function (card) {
        return { last4: card.last4 };
      },
    },
    replace: {
      // Replacing stops the old card as well as sending a new one, which is
      // what replacing a card means: two cards on one account is not it.
      title: "Replace this card",
      tool: "freeze_card",
      args: function (card) {
        return { last4: card.last4, order_replacement: true };
      },
    },
  };

  function openAsk(card, kind) {
    state.ask = { card: card, kind: kind };
    el.askTitle.textContent = ASKS[kind].title;
    el.askReason.value = "";
    el.askSummary.textContent = "";
    el.askError.hidden = true;
    el.askWhy.hidden = false;
    el.askSure.hidden = true;
    el.askGo.textContent = "Done";
    el.askGo.disabled = false;
    el.cardAsk.showModal();
    el.askReason.focus();
  }

  function closeAsk() {
    state.ask = null;
    if (el.cardAsk.open) {
      el.cardAsk.close();
    }
  }

  function askFailed(message) {
    el.askError.textContent = message;
    el.askError.hidden = false;
    el.askGo.disabled = false;
  }

  /** Step one asks why; step two shows what the bank says will happen. */
  function onAskGo() {
    if (!state.ask) {
      return;
    }
    el.askError.hidden = true;

    if (!el.askSure.hidden) {
      settleAsk();
      return;
    }

    var reason = el.askReason.value.trim();
    if (!reason) {
      askFailed("Say why, so it can be written down.");
      el.askReason.focus();
      return;
    }

    var ask = ASKS[state.ask.kind];
    el.askGo.disabled = true;
    api
      .arm(ask.tool, ask.args(state.ask.card), reason)
      .then(function (armed) {
        state.sessionId = armed.session_id;
        if (!armed.armed) {
          askFailed(armed.error || "That could not be done.");
          return;
        }
        // The bank's own sentence, not one written here, so what is agreed
        // to is exactly what will happen.
        el.askSummary.textContent = armed.summary + "?";
        el.askWhy.hidden = true;
        el.askSure.hidden = false;
        el.askGo.textContent = "Yes, confirm";
        el.askGo.disabled = false;
        el.askGo.focus();
      })
      .catch(function (error) {
        askFailed(error.message);
      });
  }

  function settleAsk() {
    el.askGo.disabled = true;
    el.askGo.textContent = "Working…";
    // The same endpoint the spoken Yes button uses, so the model is told
    // what happened and its next answer knows the card is frozen.
    api
      .confirm()
      .then(function (turn) {
        closeAsk();
        handleTurn(turn);
        loadCards();
      })
      .catch(function (error) {
        el.askGo.textContent = "Yes, confirm";
        askFailed(error.message);
      });
  }

  // --- the monthly screen

  function cardFace(card) {
    var face = node("div", "cardface" + (card.frozen ? " cardface--frozen" : ""));

    var top = node("div", "cardface__top");
    top.appendChild(node("span", "cardface__bank", "Voice Banking"));
    top.appendChild(node("span", "cardface__kind", card.kind));
    face.appendChild(top);

    // Dots and four digits, because dots and four digits is all the bank
    // has. A full number on screen would be one the database never held.
    face.appendChild(node("div", "cardface__number", card.masked));

    var foot = node("div", "cardface__foot");
    var who = node("div");
    who.appendChild(node("span", "cardface__holder", card.holder));
    who.appendChild(node("span", "cardface__account", card.account));
    foot.appendChild(who);
    foot.appendChild(node("span", "cardface__state", card.frozen ? "Frozen" : "Active"));
    face.appendChild(foot);

    return face;
  }

  function cardControls(card) {
    // A frozen card allows nothing, whatever its switches say. They are left
    // showing their real values rather than flipped to off, because the
    // columns have not changed and unfreezing restores exactly this -- but
    // they are shown as not in force, so the screen is not telling the
    // customer that online purchases work on a card that is dead.
    var controls = node("div", "controls" + (card.frozen ? " controls--frozen" : ""));

    card.controls.forEach(function (control) {
      var row = node("div", "control");
      var words = node("div");
      words.appendChild(node("span", "control__name", control.label));
      words.appendChild(node("span", "control__detail", control.detail));
      row.appendChild(words);

      var state_ = node("span", "control__state");
      state_.setAttribute("data-on", String(control.enabled));
      state_.setAttribute("role", "img");
      state_.setAttribute(
        "aria-label",
        control.label + (control.enabled ? ": allowed" : ": not allowed")
      );
      row.appendChild(state_);

      controls.appendChild(row);
    });

    controls.appendChild(
      node(
        "p",
        "controls__hint",
        card.frozen
          ? "This card is frozen, so none of these are in force. They come " +
            "back as they are when it is unfrozen."
          : "Ask to change one — say “turn off online purchases on my card”. " +
            "Every change goes through the same confirmation as moving money."
      )
    );
    return controls;
  }

  // --- the monthly screen -----------------------------------------------------

  function loadMonthly() {
    api
      .monthly()
      .then(renderMonthly)
      .catch(function () {
        el.monthlyPlans.replaceChildren(
          node("p", "block__empty", "Couldn't load your monthly plan just now.")
        );
      });
  }

  function renderMonthly(data) {
    el.monthlyCommitted.textContent = data.committed;
    el.monthlyMonth.textContent = data.month_label;
    el.monthlyPaid.textContent = data.paid;
    el.monthlyLeft.textContent = data.remaining;

    var share = data.committed_cents
      ? Math.round((data.paid_cents / data.committed_cents) * 100)
      : 0;
    el.monthlyBar.style.width = share + "%";

    el.monthlyPlans.replaceChildren();
    if (!data.groups.length) {
      el.monthlyPlans.appendChild(node("p", "block__empty", "Nothing is owed this month."));
      return;
    }
    data.groups.forEach(function (group) {
      el.monthlyPlans.appendChild(planGroup(group));
    });
  }

  function planGroup(group) {
    var block = node("div", "plangroup");

    var head = node("div", "plangroup__head");
    head.appendChild(node("span", "plangroup__name", group.label));
    head.appendChild(node("span", "plangroup__total", group.total));
    block.appendChild(head);

    group.plans.forEach(function (plan) {
      var row = node("div", "plan" + (plan.paid ? " plan--paid" : ""));

      var what = node("div");
      what.appendChild(node("span", "plan__name", plan.name));
      what.appendChild(node("span", "plan__due", plan.due_label));
      row.appendChild(what);

      var side = node("div", "plan__side");
      side.appendChild(node("span", "plan__amount", plan.amount));
      side.appendChild(node("span", "plan__state", plan.paid ? "Paid" : "Due"));
      row.appendChild(side);

      block.appendChild(row);
    });

    return block;
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
    if (state.view === "monthly") {
      // A turn can pay one of them.
      loadMonthly();
    }
    if (state.view === "cards") {
      // A turn can freeze a card or change what it allows.
      loadCards();
    }
    if (state.view === "transactions") {
      // The turn may have moved money, and a statement left as it was would
      // disagree with the balances drawn a line above it.
      loadStatement();
    }

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

    // Which engines are loaded was a builder's question, not a customer's.
    // Whether the server answers at all still matters, and the dock is where
    // that is said, so the check stays and only the chips went.
    api.ready().then(loadLedger, function () {
      setStatus("No server", "The API did not answer /ready", "error");
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    [
      "accounts", "bills",
      "conversation", "empty", "confirm", "confirmYes", "confirmNo",
      "micButton", "statusLabel", "statusHint", "levelFill",
      "sayForm", "sayInput", "resetButton",
      "vadToggle", "vadSensitivity", "vadValue", "vadNote",
      "screens", "statement", "statementTotal", "statementMonth",
      "statementIn", "statementOut", "cardPicker", "cardPane",
      "monthlyPlans", "monthlyCommitted", "monthlyMonth", "monthlyPaid",
      "monthlyLeft", "monthlyBar", "dock",
      "cardAsk", "askTitle", "askWhy", "askReason", "askSure", "askSummary",
      "askError", "askCancel", "askGo",
    ].forEach(function (id) {
      el[id] = document.getElementById(id);
    });

    el.views = document.getElementById("views");
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

    el.askGo.addEventListener("click", onAskGo);
    el.askCancel.addEventListener("click", closeAsk);
    // Escape and the backdrop close it natively; this keeps our own state
    // in step with the dialogue however it was shut.
    el.cardAsk.addEventListener("close", function () {
      state.ask = null;
    });

    el.screens.addEventListener("click", function (event) {
      var tab = event.target.closest(".screens__tab");
      if (tab) {
        showView(tab.getAttribute("data-view"));
      }
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
