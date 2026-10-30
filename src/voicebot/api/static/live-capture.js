/* The microphone for the live session: 16 kHz, 16-bit, 20 ms at a time.
 *
 * Runs on the audio rendering thread as an AudioWorklet, so capture never
 * waits on the page. The context runs at the device's rate (usually 48 kHz);
 * each output sample is the average of the input samples it spans -- a box
 * filter, enough to keep speech clean on its way down to 16 kHz -- and every
 * 320 of them (20 ms) are posted to the page, with their loudness for the
 * meter. The buffer is transferred, not copied.
 *
 * The second input is the bot's own playback, as the speakers are given it.
 * How loud it was over the same 20 ms goes with each chunk: the service's
 * echo gate tells the caller from the bot coming back through the
 * microphone by it.
 */

class LiveCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    var given = (options && options.processorOptions) || {};
    this.ratio = sampleRate / (given.targetRate || 16000);
    this.size = given.chunk || 320;
    this.chunk = new Int16Array(this.size);
    this.filled = 0;
    this.sum = 0;
    this.count = 0;
    this.step = 0;
    this.energy = 0;
    this.farEnergy = 0;
    this.farCount = 0;
  }

  emit(value) {
    var clamped = Math.max(-1, Math.min(1, value));
    this.chunk[this.filled] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    this.energy += clamped * clamped;
    this.filled += 1;
    if (this.filled === this.size) {
      var level = Math.sqrt(this.energy / this.size);
      var far = this.farCount ? Math.sqrt(this.farEnergy / this.farCount) : 0;
      this.port.postMessage({ pcm: this.chunk.buffer, level: level, far: far }, [this.chunk.buffer]);
      this.chunk = new Int16Array(this.size);
      this.filled = 0;
      this.energy = 0;
      this.farEnergy = 0;
      this.farCount = 0;
    }
  }

  process(inputs) {
    var input = inputs[0] && inputs[0][0];
    if (!input) {
      return true;
    }
    // Nothing playing is no channels at all, not a channel of zeros.
    var far = inputs[1] && inputs[1][0];
    for (var i = 0; i < input.length; i += 1) {
      if (far) {
        this.farEnergy += far[i] * far[i];
      }
      this.farCount += 1;
      this.sum += input[i];
      this.count += 1;
      this.step += 1;
      if (this.step >= this.ratio) {
        this.step -= this.ratio;
        this.emit(this.sum / this.count);
        this.sum = 0;
        this.count = 0;
      }
    }
    return true;
  }
}

registerProcessor("live-capture", LiveCapture);
