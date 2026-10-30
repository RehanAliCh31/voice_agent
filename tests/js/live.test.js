/* The live session's page side, driven by stand-ins for the browser.
 *
 *   node --test tests/js
 *
 * No microphone, speaker or socket here: each is a small fake that records
 * what was asked of it, so the tests can check that a click starts once,
 * that stopping lets go of everything, that a barge-in silences the reply at
 * once and says where, and that a dropped connection is reopened.
 */

const test = require("node:test");
const assert = require("node:assert");

const { LiveClient, RECONNECT_MS } = require("../../src/voicebot/api/static/live.js");

class FakeSocket {
  constructor(url) {
    this.url = url;
    this.sent = [];
    this.readyState = 0;
    this.closedWith = null;
    FakeSocket.made.push(this);
  }
  open() {
    this.readyState = 1;
    this.onopen && this.onopen();
  }
  send(data) {
    this.sent.push(data);
  }
  receive(message) {
    this.onmessage && this.onmessage({ data: typeof message === "string" ? message : message });
  }
  event(body) {
    this.receive(JSON.stringify(body));
  }
  close(code) {
    this.closedWith = code;
    this.readyState = 3;
  }
  drop() {
    this.readyState = 3;
    this.onclose && this.onclose({ code: 1006 });
  }
  json() {
    return this.sent.filter((d) => typeof d === "string").map((d) => JSON.parse(d));
  }
}
FakeSocket.made = [];

function harness({ denied = false } = {}) {
  FakeSocket.made = [];
  const log = {
    states: [],
    errors: [],
    events: [],
    timers: [],
    tracks: [],
    sources: [],
    connections: [],
  };
  const node = (name) => ({
    name,
    connect(to, output, input) {
      log.connections.push({ from: name, to: to && to.name, output, input });
    },
    disconnect() {},
  });
  const context = {
    currentTime: 0,
    state: "running",
    closed: false,
    destination: { name: "speakers" },
    audioWorklet: { addModule: () => Promise.resolve() },
    createGain: () => Object.assign(node("gain"), { gain: { value: 1 } }),
    createMediaStreamSource: () => ({ connect() {}, disconnect() {} }),
    decodeAudioData: (bytes) => Promise.resolve({ duration: bytes.byteLength / 1000 }),
    createBufferSource() {
      const source = {
        started: null,
        stopped: false,
        connect() {},
        disconnect() {},
        start(when, offset) {
          this.started = offset;
        },
        stop() {
          this.stopped = true;
        },
      };
      log.sources.push(source);
      return source;
    },
    close() {
      this.closed = true;
      return Promise.resolve();
    },
    resume() {
      this.state = "running";
    },
  };
  const mic = () => {
    const track = {
      stopped: false,
      onended: null,
      stop() {
        this.stopped = true;
      },
      // What the browser says it applied -- and an identifier that stays here.
      getSettings: () => ({
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
        sampleRate: 48000,
        deviceId: "a-device",
      }),
    };
    log.tracks.push(track);
    return { getTracks: () => [track], getAudioTracks: () => [track] };
  };
  const deps = {
    WebSocket: FakeSocket,
    getUserMedia: (constraints) => {
      log.constraints = constraints;
      log.micAsks = (log.micAsks || 0) + 1;
      return denied
        ? Promise.reject(Object.assign(new Error("denied"), { name: "NotAllowedError" }))
        : Promise.resolve(mic());
    },
    createAudioContext: () => context,
    createWorkletNode: (context, name, options) => {
      log.workletOptions = options;
      return Object.assign(node("capture"), { port: { onmessage: null } });
    },
    workletUrl: "live-capture.js",
    now: () => Date.now(),
    setTimeout: (fn, ms) => {
      log.timers.push({ fn, ms });
      return log.timers.length;
    },
    clearTimeout: () => {},
  };
  const client = new LiveClient({
    url: "ws://test/v1/live",
    deps,
    on: {
      state: (s) => log.states.push(s),
      error: (kind) => log.errors.push(kind),
      event: (e) => log.events.push(e),
    },
  });
  return { client, context, log };
}

const tick = () => new Promise((resolve) => setImmediate(resolve));

async function live(h) {
  await h.client.toggle();
  const socket = FakeSocket.made[FakeSocket.made.length - 1];
  socket.open();
  socket.event({ type: "started", session_id: "s1" });
  socket.event({ type: "state", state: "listening" });
  return socket;
}

async function replyOf(socket, id, pieces) {
  for (let i = 0; i < pieces; i += 1) {
    socket.event({ type: "audio", audio_id: id, piece: i, seconds: 1, last: i === pieces - 1 });
    socket.receive(new ArrayBuffer(1000));
  }
  await tick();
  await tick();
}

// --- one click starts, one click stops -------------------------------------------

test("one click opens the microphone with echo cancellation and starts the session", async () => {
  const h = harness();
  const socket = await live(h);
  assert.deepStrictEqual(h.log.constraints.audio, {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    channelCount: 1,
  });
  // The start says what the browser applied to the microphone, and that
  // every chunk comes with how loud the bot was playing.
  assert.deepStrictEqual(socket.json()[0], {
    type: "start",
    session_id: null,
    capture: {
      format: "level+pcm16",
      settings: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
        sampleRate: 48000,
      },
    },
  });
  // The bot's own playback is the worklet's second input.
  assert.strictEqual(h.log.workletOptions.numberOfInputs, 2);
  assert.ok(h.log.connections.some((c) => c.to === "capture" && c.input === 1));
  // The microphone goes straight to the socket, as binary, led by the playback level.
  h.client.onFrame({ pcm: new Int16Array([1, -1]).buffer, level: 0.1, far: 0.25 });
  const frame = socket.sent.find((d) => d instanceof ArrayBuffer);
  assert.strictEqual(frame.byteLength, 4 + 4);
  assert.ok(Math.abs(new DataView(frame).getFloat32(0, true) - 0.25) < 1e-6);
  assert.deepStrictEqual(Array.from(new Int16Array(frame.slice(4))), [1, -1]);
  assert.deepStrictEqual(h.log.states, ["connecting", "listening"]);
});

test("a second click stops it and lets go of everything", async () => {
  const h = harness();
  const socket = await live(h);
  await replyOf(socket, "a1", 1);
  assert.ok(h.client.sounding());
  h.client.toggle();
  assert.deepStrictEqual(socket.json().pop(), { type: "stop", reason: "the caller stopped" });
  assert.strictEqual(socket.closedWith, 1000);
  assert.ok(h.log.tracks.every((t) => t.stopped), "the microphone is released");
  assert.ok(h.context.closed, "the audio context is closed");
  assert.ok(h.log.sources.every((s) => s.stopped), "nothing keeps playing");
  assert.strictEqual(h.client.sounding(), false);
  assert.strictEqual(h.client.phase, "idle");
  // Nothing more is sent after the stop.
  const before = socket.sent.length;
  h.client.onFrame({ pcm: new ArrayBuffer(640), level: 0.1 });
  assert.strictEqual(socket.sent.length, before);
});

test("rapid clicks start one session or none, never two", async () => {
  const h = harness();
  const first = h.client.toggle(); // starting
  h.client.toggle(); // stop, while still starting
  h.client.toggle(); // ignored: still stopping the first? no -- idle now, so starts again
  await first;
  await tick();
  assert.ok(FakeSocket.made.length <= 1, "at most one socket");
  assert.ok(h.log.micAsks <= 2);
  // Whatever happened, a microphone opened for a stopped session is released.
  const open = h.log.tracks.filter((t) => !t.stopped).length;
  assert.ok(open <= 1);
  h.client.stop();
  assert.ok(h.log.tracks.every((t) => t.stopped));
});

test("a refused microphone says so and starts nothing", async () => {
  const h = harness({ denied: true });
  await h.client.toggle();
  assert.deepStrictEqual(h.log.errors, ["microphone_refused"]);
  assert.strictEqual(FakeSocket.made.length, 0);
  assert.strictEqual(h.client.phase, "idle");
  assert.deepStrictEqual(h.log.states, ["connecting", "idle"]);
});

// --- the connection ---------------------------------------------------------------

test("a dropped connection is reopened, with the same conversation", async () => {
  const h = harness();
  const socket = await live(h);
  socket.drop();
  assert.strictEqual(h.log.states[h.log.states.length - 1], "reconnecting");
  assert.strictEqual(h.log.timers[0].ms, RECONNECT_MS[0]);
  h.log.timers[0].fn();
  const again = FakeSocket.made[1];
  again.open();
  const start = again.json()[0];
  assert.strictEqual(start.type, "start");
  assert.strictEqual(start.session_id, "s1");
});

test("a connection that will not come back ends the session and says why", async () => {
  const h = harness();
  let socket = await live(h);
  for (let i = 0; i < RECONNECT_MS.length; i += 1) {
    socket.drop();
    h.log.timers[h.log.timers.length - 1].fn();
    socket = FakeSocket.made[FakeSocket.made.length - 1];
  }
  socket.drop();
  assert.deepStrictEqual(h.log.errors, ["connection_lost"]);
  assert.strictEqual(h.client.phase, "idle");
  assert.ok(h.log.tracks.every((t) => t.stopped));
});

test("the service ending the session -- a goodbye, a limit -- lets go here too", async () => {
  const h = harness();
  const socket = await live(h);
  socket.event({ type: "ended", reason: "said goodbye" });
  assert.strictEqual(h.client.phase, "idle");
  assert.ok(h.log.tracks.every((t) => t.stopped) && h.context.closed);
});

// --- the reply, and barge-in ----------------------------------------------------------

test("a reply's pieces play one after another, then the service is told", async () => {
  const h = harness();
  const socket = await live(h);
  await replyOf(socket, "a1", 2);
  socket.event({ type: "audio_end", audio_id: "a1" });
  assert.strictEqual(h.log.sources.length, 1, "one at a time");
  h.log.sources[0].onended();
  assert.strictEqual(h.log.sources.length, 2);
  h.log.sources[1].onended();
  assert.deepStrictEqual(socket.json().pop(), { type: "playback_done", audio_id: "a1" });
});

test("a last piece that comes after a gap is played, not dropped", async () => {
  const h = harness();
  const socket = await live(h);
  const done = () => socket.json().some((m) => m.type === "playback_done");
  // The first piece arrives and plays to its end before the next is made...
  await replyOf(socket, "a1", 1);
  h.log.sources[0].onended();
  // ...then the last comes, and the reply's end right behind it, while the
  // last piece is still being decoded.
  socket.event({ type: "audio", audio_id: "a1", piece: 1, seconds: 1, last: true });
  socket.receive(new ArrayBuffer(1000));
  socket.event({ type: "audio_end", audio_id: "a1", pieces: 2 });
  assert.strictEqual(done(), false, "not finished: a piece is still to play");
  await tick();
  assert.strictEqual(h.log.sources.length, 2, "the last piece is played");
  assert.strictEqual(done(), false);
  h.log.sources[1].onended();
  assert.strictEqual(done(), true);
});

test("a piece that never came is skipped, not waited for", async () => {
  const h = harness();
  const socket = await live(h);
  // The first piece's voice failed; the second was made.
  socket.event({ type: "audio", audio_id: "a1", piece: 1, seconds: 1, last: true });
  socket.receive(new ArrayBuffer(1000));
  socket.event({ type: "audio_end", audio_id: "a1", pieces: 1 });
  await tick();
  assert.strictEqual(h.log.sources.length, 1, "the piece there is played");
  h.log.sources[0].onended();
  assert.ok(socket.json().some((m) => m.type === "playback_done"));
});

test("barge-in silences the reply at once and says exactly where it stopped", async () => {
  const h = harness();
  const socket = await live(h);
  await replyOf(socket, "a1", 2);
  h.log.sources[0].onended(); // the first piece played through
  h.context.currentTime = 10;
  const playing = h.log.sources[1];
  h.context.currentTime = 10.6; // 600 ms into the second piece
  socket.event({ type: "stop_audio", audio_id: "a1", reason: "barge_in" });
  assert.ok(playing.stopped, "stopped at once");
  assert.strictEqual(h.client.sounding(), false);
  const report = socket.json().pop();
  assert.strictEqual(report.type, "interrupted");
  assert.strictEqual(report.piece, 1);
  assert.ok(report.stop_ms >= 0 && report.stop_ms < 50, "no waiting before it stops");
});

test("after a mm-hmm the reply carries on from where it stopped", async () => {
  const h = harness();
  const socket = await live(h);
  await replyOf(socket, "a1", 1);
  h.context.currentTime = 0.4;
  socket.event({ type: "stop_audio", audio_id: "a1" });
  socket.event({ type: "resume_audio", audio_id: "a1" });
  const resumed = h.log.sources[h.log.sources.length - 1];
  assert.notStrictEqual(resumed, h.log.sources[0]);
  assert.ok(Math.abs(resumed.started - 0.4) < 1e-9, "from 400 ms in");
});

test("a reply let go of is never heard", async () => {
  const h = harness();
  const socket = await live(h);
  await replyOf(socket, "a1", 1);
  socket.event({ type: "stop_audio", audio_id: "a1" });
  socket.event({ type: "discard_audio", audio_id: "a1" });
  socket.event({ type: "resume_audio", audio_id: "a1" });
  assert.strictEqual(h.log.sources.length, 1);
  assert.strictEqual(h.client.sounding(), false);
});

test("two barge-ins in a row leave nothing playing that should not be", async () => {
  const h = harness();
  const socket = await live(h);
  await replyOf(socket, "a1", 1);
  socket.event({ type: "stop_audio", audio_id: "a1" });
  socket.event({ type: "discard_audio", audio_id: "a1" });
  await replyOf(socket, "a2", 1);
  socket.event({ type: "stop_audio", audio_id: "a2" });
  socket.event({ type: "discard_audio", audio_id: "a2" });
  assert.strictEqual(h.client.sounding(), false);
  assert.ok(h.log.sources.every((s) => s.stopped));
});

test("stopping while listening, thinking or speaking leaves no sound and no microphone", async () => {
  for (const where of ["listening", "user_speaking", "thinking", "agent_speaking"]) {
    const h = harness();
    const socket = await live(h);
    socket.event({ type: "state", state: where });
    if (where === "agent_speaking") {
      await replyOf(socket, "a1", 2);
    }
    h.client.toggle();
    assert.strictEqual(h.client.sounding(), false, where);
    assert.ok(h.log.tracks.every((t) => t.stopped), where);
    assert.strictEqual(h.client.phase, "idle", where);
    // Audio that arrives after the stop is not played.
    await replyOf(socket, "late", 1);
    assert.strictEqual(h.client.sounding(), false, where);
  }
});

test("an unplugged microphone is opened again", async () => {
  const h = harness();
  await live(h);
  const first = h.log.tracks[0];
  await first.onended();
  assert.strictEqual(h.log.tracks.length, 2);
  assert.ok(first.stopped);
  assert.strictEqual(h.client.phase, "live");
});

test("the caller being heard silences the bot even without a stop", async () => {
  const h = harness();
  const socket = await live(h);
  await replyOf(socket, "a1", 1);
  assert.ok(h.client.sounding());
  socket.event({ type: "state", state: "user_speaking" });
  assert.strictEqual(h.client.sounding(), false);
});
