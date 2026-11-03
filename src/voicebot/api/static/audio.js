/* Sound out: the bot's replies, and its own recorded sentences.
 *
 * The live session plays what it streams itself; this plays a reply to a
 * typed turn or a button, through the live session's output while there is
 * one.
 */

import { state } from "./core.js";

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
      if (state.playing === audio) {
        URL.revokeObjectURL(audio.src);
        state.playing = null;
      }
      resolve();
    }

    audio.addEventListener("ended", finish);
    audio.addEventListener("error", finish);

    audio.play().then(
      function () {},
      function () {
        // Autoplay was refused until the user interacts. Not fatal: the
        // reply is on screen and has a replay button.
        finish();
      }
    );
  });
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

export { base64ToBlob, play, playAll, stopPlayback };
