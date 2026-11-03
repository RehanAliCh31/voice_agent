/* The hands-free live session, in the page.
 *
 * One click starts it: the microphone opens and stays open, and is streamed
 * to the service over a WebSocket, 16 kHz and 20 ms at a time (live-capture
 * runs on the audio thread). The service decides when the caller has
 * finished, answers, and sends the reply back a piece at a time; this plays
 * the pieces back to back through Web Audio, so it can stop them at once and
 * say exactly where it stopped. One click ends it: microphone released,
 * sounds stopped, socket closed, everything let go.
 *
 * The service is the one source of truth for the conversation's state; the
 * page only shows it. What the page owns is its own phase -- idle, starting,
 * live, stopping -- so a second click while starting cannot start twice, and
 * a click while stopping does nothing.
 *
 * The browser's echo cancellation, noise suppression and gain control are
 * asked for: without the first the service hears the bot through the
 * speakers. What the browser actually applied is told to the service, and
 * each chunk of microphone audio goes with how loud the bot was playing
 * meanwhile, for the service's echo gate. A socket that drops is reopened,
 * a few times, with the same conversation; a microphone that is unplugged is
 * opened again.
 *
 * Everything the browser provides is passed in (`deps`), so the tests drive
 * this with stand-ins (tests/js/live.test.js).
 */

(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.LiveClient = factory().LiveClient;
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Waits between attempts to reopen a dropped socket; after the last, the
  // session ends and says so.
  var RECONNECT_MS = [300, 800, 1600, 3200, 5000];

  // Each microphone chunk is led by the playback loudness, as a float32.
  var FRAMED = "level+pcm16";
  // What the browser applied to the microphone, told to the service.
  var SETTINGS = ["echoCancellation", "noiseSuppression", "autoGainControl", "sampleRate"];

  // No automatic gain: it turns the microphone up while the caller is
  // quiet, so a voice across the room comes in as loud as theirs, and the
  // service tells the caller from the room by how loud each is.
  var CONSTRAINTS = {
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,
      channelCount: 1,
    },
  };

  function LiveClient(options) {
    this.url = options.url;
    this.sessionId = options.sessionId || null;
    this.deps = options.deps;
    this.on = options.on || {};
    this.phase = "idle"; // idle | starting | live | stopping
    this.serverState = "idle";
    this.socket = null;
    this.stream = null;
    this.context = null;
    this.source = null;
    this.node = null;
    this.output = null;
    this.attempt = 0;
    this.reconnectTimer = null;
    this.header = null;
    this.replies = {};
    this.active = null; // { reply, source, index, startedAt }
    this.framesSent = 0;
    // Moved on by every start and stop: a microphone that opens for a start
    // that has since been stopped is released, never used.
    this.generation = 0;
  }

  // --- one click starts, one click stops -------------------------------------

  LiveClient.prototype.toggle = function () {
    if (this.phase === "idle") {
      return this.start();
    }
    if (this.phase === "live" || this.phase === "starting") {
      this.stop("the caller stopped");
    }
    return Promise.resolve();
  };

  LiveClient.prototype.start = function () {
    if (this.phase !== "idle") {
      return Promise.resolve();
    }
    var self = this;
    var generation = ++this.generation;
    this.phase = "starting";
    this.state("connecting");
    return this.openMicrophone(generation).then(
      function (opened) {
        if (!opened) {
          return; // stopped (and perhaps started again) while it opened
        }
        self.phase = "live";
        self.attempt = 0;
        self.connect();
      },
      function (error) {
        if (generation !== self.generation) {
          return;
        }
        self.cleanup();
        self.phase = "idle";
        var refused = error && (error.name === "NotAllowedError" || error.name === "SecurityError");
        self.error(refused ? "microphone_refused" : "microphone_failed", error);
        self.state("idle");
      }
    );
  };

  LiveClient.prototype.stop = function (reason) {
    if (this.phase === "idle" || this.phase === "stopping") {
      return;
    }
    this.phase = "stopping";
    this.generation += 1;
    if (this.socket && this.socket.readyState === 1) {
      try {
        this.socket.send(JSON.stringify({ type: "stop", reason: reason || "stopped" }));
      } catch (error) {
        // Closing anyway, just below.
      }
    }
    this.cleanup();
    this.phase = "idle";
    this.state("idle");
  };

  // --- the microphone ------------------------------------------------------------

  /** Resolves true once the microphone is open and wired; false if this start was overtaken. */
  LiveClient.prototype.openMicrophone = function (generation) {
    var self = this;
    var deps = this.deps;
    var current = function () {
      return generation === self.generation && self.phase === "starting";
    };
    return deps.getUserMedia(CONSTRAINTS).then(function (stream) {
      if (!current()) {
        release(stream);
        return false;
      }
      self.stream = stream;
      self.context = deps.createAudioContext();
      return self.context.audioWorklet.addModule(deps.workletUrl).then(function () {
        if (!current()) {
          return false; // cleanup, run by the stop, has the stream and context
        }
        // Input 0 is the microphone; input 1 the bot's own playback, so the
        // worklet can say how loud it was over each chunk.
        self.node = deps.createWorkletNode(self.context, "live-capture", {
          numberOfInputs: 2,
          processorOptions: { targetRate: 16000, chunk: 320 },
        });
        self.node.port.onmessage = function (event) {
          self.onFrame(event.data);
        };
        // The worklet runs only while something downstream pulls on it:
        // a silent gain to the speakers keeps it running.
        var hush = self.context.createGain();
        hush.gain.value = 0;
        self.node.connect(hush);
        hush.connect(self.context.destination);
        self.output = self.context.createGain();
        self.output.connect(self.context.destination);
        self.output.connect(self.node, 0, 1);
        self.attach(stream);
        return true;
      });
    });
  };

  function release(stream) {
    stream.getTracks().forEach(function (track) {
      track.onended = null;
      track.stop();
    });
  }

  LiveClient.prototype.attach = function (stream) {
    var self = this;
    if (this.source) {
      this.source.disconnect();
    }
    this.source = this.context.createMediaStreamSource(stream);
    this.source.connect(this.node);
    stream.getAudioTracks().forEach(function (track) {
      // Unplugged, or taken by another device: opened again.
      track.onended = function () {
        self.reopenMicrophone();
      };
    });
  };

  LiveClient.prototype.reopenMicrophone = function () {
    var self = this;
    if (this.phase !== "live") {
      return Promise.resolve();
    }
    return this.deps.getUserMedia(CONSTRAINTS).then(
      function (stream) {
        if (self.phase !== "live") {
          stream.getTracks().forEach(function (track) {
            track.stop();
          });
          return;
        }
        if (self.stream) {
          self.stream.getTracks().forEach(function (track) {
            track.stop();
          });
        }
        self.stream = stream;
        self.attach(stream);
      },
      function (error) {
        self.error("microphone_lost", error);
        self.stop("the microphone was lost");
      }
    );
  };

  /** A tab back in view: a context the browser suspended is woken. */
  LiveClient.prototype.wake = function () {
    if (this.context && this.context.state === "suspended" && this.context.resume) {
      this.context.resume();
    }
  };

  LiveClient.prototype.onFrame = function (data) {
    if (this.on.level) {
      this.on.level(data.level);
    }
    if (this.phase === "live" && this.socket && this.socket.readyState === 1) {
      var pcm = new Uint8Array(data.pcm);
      var frame = new ArrayBuffer(4 + pcm.byteLength);
      new DataView(frame).setFloat32(0, data.far || 0, true);
      new Uint8Array(frame, 4).set(pcm);
      this.socket.send(frame);
      this.framesSent += 1;
    }
  };

  /** What the browser says it applied to the microphone: echo cancellation above all. */
  LiveClient.prototype.micSettings = function () {
    var track = this.stream && this.stream.getAudioTracks()[0];
    var applied = track && track.getSettings ? track.getSettings() : {};
    var settings = {};
    SETTINGS.forEach(function (name) {
      if (applied[name] !== undefined) {
        settings[name] = applied[name];
      }
    });
    return settings;
  };

  // --- the socket ------------------------------------------------------------------

  LiveClient.prototype.connect = function () {
    var self = this;
    var socket = new this.deps.WebSocket(this.url);
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    socket.onopen = function () {
      socket.send(
        JSON.stringify({
          type: "start",
          session_id: self.sessionId,
          capture: { format: FRAMED, settings: self.micSettings() },
        })
      );
    };
    socket.onmessage = function (event) {
      self.onMessage(event.data);
    };
    socket.onclose = function () {
      if (self.socket === socket) {
        self.socket = null;
        self.dropped();
      }
    };
    socket.onerror = function () {
      // onclose follows, and decides.
    };
  };

  LiveClient.prototype.dropped = function () {
    if (this.phase !== "live") {
      return;
    }
    this.stopSound();
    if (this.attempt >= RECONNECT_MS.length) {
      this.error("connection_lost");
      this.stop("the connection was lost");
      return;
    }
    var self = this;
    var wait = RECONNECT_MS[this.attempt];
    this.attempt += 1;
    this.state("reconnecting");
    this.reconnectTimer = this.deps.setTimeout(function () {
      self.reconnectTimer = null;
      if (self.phase === "live") {
        self.connect();
      }
    }, wait);
  };

  LiveClient.prototype.send = function (message) {
    if (this.socket && this.socket.readyState === 1) {
      this.socket.send(JSON.stringify(message));
    }
  };

  LiveClient.prototype.onMessage = function (data) {
    if (typeof data !== "string") {
      this.onAudio(data);
      return;
    }
    var message;
    try {
      message = JSON.parse(data);
    } catch (error) {
      return;
    }
    switch (message.type) {
      case "started":
        this.sessionId = message.session_id;
        this.attempt = 0;
        break;
      case "state":
        this.serverState = message.state;
        if (message.state === "user_speaking" && this.active) {
          // The service is hearing the caller and has not said stop: a reply
          // it had taken for finished is still sounding. It never talks over
          // the caller, so it stops here too.
          this.stopSound();
        }
        if (message.state !== "ended") {
          this.state(message.state);
        }
        break;
      case "audio":
        this.header = message;
        return;
      case "audio_end":
        this.ended(message.audio_id, message.pieces);
        break;
      case "stop_audio":
        this.pause(message.audio_id);
        break;
      case "resume_audio":
        this.resume(message.audio_id);
        break;
      case "discard_audio":
        this.discard(message.audio_id);
        break;
      case "ended":
        if (this.phase === "live") {
          this.cleanup();
          this.phase = "idle";
          this.state("idle");
        }
        break;
      default:
        break;
    }
    if (this.on.event) {
      this.on.event(message);
    }
  };

  // --- the reply's audio -----------------------------------------------------------

  LiveClient.prototype.reply = function (id) {
    if (!this.replies[id]) {
      this.replies[id] = {
        id: id,
        pieces: [],
        received: 0, // pieces whose audio has come, decoded or not
        decoding: 0,
        expected: null,
        next: 0,
        offset: 0,
        paused: false,
      };
    }
    return this.replies[id];
  };

  LiveClient.prototype.onAudio = function (bytes) {
    var header = this.header;
    this.header = null;
    if (!header || !this.context) {
      return;
    }
    var self = this;
    var reply = this.reply(header.audio_id);
    // Counted now, not when decoded: the reply's end can arrive while this
    // piece is still decoding, and must not be taken for the end of the sound.
    reply.received = Math.max(reply.received, header.piece + 1);
    reply.decoding += 1;
    this.context.decodeAudioData(bytes).then(
      function (buffer) {
        reply.decoding -= 1;
        if (self.replies[reply.id] !== reply) {
          return; // discarded while it was decoding
        }
        reply.pieces[header.piece] = buffer;
        self.play(reply);
      },
      function () {
        // Unreadable: it is skipped, and the next piece plays.
        reply.decoding -= 1;
        reply.pieces[header.piece] = null;
        self.play(reply);
      }
    );
  };

  /** Every piece has been sent: `count` of them, as the service says. */
  LiveClient.prototype.ended = function (id, count) {
    var reply = this.reply(id);
    reply.expected = Math.max(count || 0, reply.received, reply.next);
    this.play(reply);
  };

  LiveClient.prototype.play = function (reply) {
    if (reply.paused || this.active || !this.context) {
      return;
    }
    var buffer = reply.pieces[reply.next];
    while (!buffer) {
      if (buffer === undefined) {
        // Not here: on its way, or still decoding -- unless the reply's end
        // has come and nothing is decoding, when it is done, or the piece was
        // never sent.
        if (reply.expected === null || reply.decoding > 0) {
          return;
        }
        if (reply.next >= reply.expected) {
          delete this.replies[reply.id];
          this.send({ type: "playback_done", audio_id: reply.id });
          return;
        }
      }
      reply.next += 1; // could not be decoded, or never sent: the next one plays
      buffer = reply.pieces[reply.next];
    }
    var self = this;
    var source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.output);
    var offset = reply.offset;
    reply.offset = 0;
    var active = {
      reply: reply,
      source: source,
      index: reply.next,
      startedAt: this.context.currentTime - offset,
    };
    this.active = active;
    source.onended = function () {
      if (self.active !== active) {
        return; // stopped on purpose
      }
      self.active = null;
      reply.next = active.index + 1;
      self.play(reply);
    };
    source.start(0, offset);
  };

  /** Barge-in: silent at once, and tell the service exactly where it stopped. */
  LiveClient.prototype.pause = function (id) {
    var received = this.deps.now();
    var reply = this.replies[id];
    var piece = 0;
    var offsetMs = 0;
    if (reply) {
      reply.paused = true;
      if (this.active && this.active.reply === reply) {
        piece = this.active.index;
        offsetMs = Math.max(0, (this.context.currentTime - this.active.startedAt) * 1000);
      } else {
        piece = reply.next;
      }
      reply.next = piece;
      reply.offset = offsetMs / 1000;
    }
    this.stopSound();
    this.send({
      type: "interrupted",
      audio_id: id,
      piece: piece,
      offset_ms: Math.round(offsetMs),
      stop_ms: Math.round(this.deps.now() - received),
    });
  };

  /** It was only a "mm-hmm": carry on from exactly there. */
  LiveClient.prototype.resume = function (id) {
    var reply = this.replies[id];
    if (!reply) {
      return;
    }
    reply.paused = false;
    this.play(reply);
  };

  LiveClient.prototype.discard = function (id) {
    if (this.active && this.active.reply.id === id) {
      this.stopSound();
    }
    delete this.replies[id];
  };

  LiveClient.prototype.stopSound = function () {
    var active = this.active;
    this.active = null;
    if (active) {
      try {
        active.source.stop();
      } catch (error) {
        // Already stopped.
      }
      active.source.disconnect();
    }
  };

  /**
   * Another of the bot's sounds -- a reply to a button, played by the page's
   * own player -- through the same output as the live replies, so the
   * service hears how loud it is and does not take its echo for the caller.
   */
  LiveClient.prototype.route = function (element) {
    if (!this.context || !this.output || !this.context.createMediaElementSource) {
      return;
    }
    try {
      this.context.createMediaElementSource(element).connect(this.output);
    } catch (error) {
      // Already routed, or a browser that refuses: it plays as it was.
    }
  };

  /** True while any of the bot's audio is sounding. */
  LiveClient.prototype.sounding = function () {
    return Boolean(this.active);
  };

  // --- letting go of everything ---------------------------------------------------

  LiveClient.prototype.cleanup = function () {
    if (this.reconnectTimer !== null) {
      this.deps.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopSound();
    this.replies = {};
    this.header = null;
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node = null;
    }
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(function (track) {
        track.onended = null;
        track.stop();
      });
      this.stream = null;
    }
    if (this.context) {
      var context = this.context;
      this.context = null;
      this.output = null;
      context.close().catch(function () {});
    }
    if (this.socket) {
      var socket = this.socket;
      this.socket = null;
      socket.onclose = null;
      socket.onmessage = null;
      try {
        socket.close(1000);
      } catch (error) {
        // Already closed.
      }
    }
  };

  LiveClient.prototype.state = function (name) {
    if (this.on.state) {
      this.on.state(name);
    }
  };

  LiveClient.prototype.error = function (kind, cause) {
    if (this.on.error) {
      this.on.error(kind, cause);
    }
  };

  return { LiveClient: LiveClient, RECONNECT_MS: RECONNECT_MS };
});
