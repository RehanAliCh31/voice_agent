/* The live session's microphone worklet, run outside a browser.
 *
 *   node --test tests/js/capture.test.js
 *
 * It needs only the two globals an AudioWorklet scope gives it -- its base
 * class and registerProcessor -- and the context's sample rate.
 */

const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const SOURCE = fs.readFileSync(
  path.join(__dirname, "../../src/voicebot/api/static/live-capture.js"),
  "utf8"
);

function capture() {
  const posted = [];
  let Processor = null;
  vm.runInNewContext(SOURCE, {
    sampleRate: 48000,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = { postMessage: (message) => posted.push(message) };
      }
    },
    registerProcessor: (name, processor) => {
      Processor = processor;
    },
  });
  return { node: new Processor({ processorOptions: { targetRate: 16000, chunk: 320 } }), posted };
}

// 128 frames a render quantum at 48 kHz; 8 of them make one 20 ms chunk at 16 kHz.
const QUANTA = 8;

test("each 20 ms of microphone goes with how loud the bot played over it", () => {
  const { node, posted } = capture();
  const mic = new Float32Array(128).fill(0.1);
  const playback = new Float32Array(128).fill(0.5);
  for (let i = 0; i < QUANTA; i += 1) {
    node.process([[mic], [playback]]);
  }
  assert.strictEqual(posted.length, 1);
  assert.strictEqual(new Int16Array(posted[0].pcm).length, 320);
  assert.ok(Math.abs(posted[0].level - 0.1) < 1e-3);
  assert.ok(Math.abs(posted[0].far - 0.5) < 1e-6);
});

test("nothing playing is a playback level of nothing", () => {
  const { node, posted } = capture();
  const mic = new Float32Array(128).fill(0.1);
  for (let i = 0; i < QUANTA; i += 1) {
    node.process([[mic], []]); // no playback: the input has no channels
  }
  assert.strictEqual(posted.length, 1);
  assert.strictEqual(posted[0].far, 0);
});
