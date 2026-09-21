import assert from 'node:assert/strict';
import {AudioCapture, encodeWav, recordingToWav} from '../src/audioCapture.js';

const tick = () => new Promise(resolve => setImmediate(resolve));
const defer = () => {let resolve; const promise = new Promise(done => {resolve = done;}); return {promise, resolve};};
function stream() {const track = {stopped: false, stop() {this.stopped = true;}}; return {track, getTracks: () => [track]};}
class Recorder {
  static isTypeSupported(type) {return type.startsWith('audio/webm');}
  constructor() {this.state = 'inactive'; this.mimeType = 'audio/webm';}
  start() {this.state = 'recording';}
  stop() {
    assert.equal(this.state, 'recording'); this.state = 'inactive';
    queueMicrotask(() => {this.ondataavailable({data: new Blob(['test'])}); this.onstop();});
  }
}
function harness(overrides = {}) {
  const states = [], completed = [], errors = [], streams = [];
  let now = 0, timer;
  const capture = new AudioCapture({
    Recorder, mediaDevices: {getUserMedia: async () => {const value = stream(); streams.push(value); return value;}},
    encode: async blob => {assert.equal(blob.size, 4); return encodeWav(new Float32Array(16000));},
    now: () => now, every: callback => {timer = callback; return 1;}, clear: () => {timer = null;},
    onState: state => states.push(state), onComplete: (...args) => completed.push(args), onError: error => errors.push(error),
    ...overrides,
  });
  return {capture, states, completed, errors, streams, advance: seconds => {now = seconds * 1000; timer?.();}};
}
const reference = {minimum: 3, maximum: 10};
const recording = {minimum: 0.5, maximum: 30};

const success = harness();
await success.capture.start('reference', reference);
await success.capture.start('recording', recording);
assert.equal(success.streams.length, 1, 'only one input may capture at a time');
success.advance(9.9);
await tick();
assert.equal(success.completed[0][0], 'reference', 'reference goes to its own input');
assert.equal(success.completed[0][1].type, 'audio/wav');
assert.ok(success.streams[0].track.stopped, 'automatic stop releases microphone');
assert.equal(success.states.at(-1).phase, 'idle');
await success.capture.start('recording', recording);
success.advance(38);
assert.equal(success.states.at(-1).phase, 'recording', 'recordings have a separate 30-second limit');
success.capture.stop(); await tick();
assert.equal(success.completed[1][0], 'recording');
assert.ok(success.streams[1].track.stopped);

const pending = defer(), late = stream();
const cancelled = harness({mediaDevices: {getUserMedia: () => pending.promise}});
const requesting = cancelled.capture.start('reference', reference);
cancelled.capture.cancel(); pending.resolve(late); await requesting;
assert.ok(late.track.stopped, 'late permission must not leave the mic open');
assert.equal(cancelled.completed.length, 0);

const conversion = defer();
const obsolete = harness({encode: () => conversion.promise});
await obsolete.capture.start('reference', reference);
obsolete.capture.stop(); await tick();
obsolete.capture.cancel();
conversion.resolve(new Blob(['old'])); await tick();
assert.equal(obsolete.completed.length, 0, 'discarded conversion cannot overwrite the input');

const discarded = harness();
await discarded.capture.start('recording', recording);
discarded.capture.cancel(); await tick();
assert.ok(discarded.streams[0].track.stopped);
assert.equal(discarded.completed.length, 0, 'discard keeps the previous input');

const denied = harness({mediaDevices: {getUserMedia: async () => {const error = new Error(); error.name = 'NotAllowedError'; throw error;}}});
await denied.capture.start('reference', reference);
assert.match(denied.errors[0], /denied/);
assert.equal(denied.capture.session, null);
const broken = harness({encode: async () => {throw new Error('Record at least 3 seconds');}});
await broken.capture.start('reference', reference); broken.capture.stop(); await tick();
assert.match(broken.errors[0], /3 seconds/);
assert.ok(broken.streams[0].track.stopped);
assert.equal(broken.completed.length, 0);

const pcm = new DataView(await encodeWav(new Float32Array([-2, -0.5, 0, 0.5, 2])).arrayBuffer());
assert.equal(pcm.getUint32(24, true), 16000);
assert.equal(pcm.getUint16(22, true), 1);
assert.equal(pcm.getUint16(34, true), 16);
assert.equal(pcm.getUint32(40, true), 10);
assert.deepEqual([44, 46, 48, 50, 52].map(offset => pcm.getInt16(offset, true)), [-32768, -16384, 0, 16384, 32767]);

let closed = false, decodedDuration = 2, renderFrames;
globalThis.AudioContext = class {
  async decodeAudioData() {return {duration: decodedDuration};}
  async close() {closed = true;}
};
globalThis.OfflineAudioContext = class {
  constructor(channels, frames, rate) {assert.equal(channels, 1); assert.equal(rate, 16000); renderFrames = frames;}
  createBufferSource() {return {connect() {}, start() {}};}
  async startRendering() {return {getChannelData: () => new Float32Array(renderFrames)};}
};
await assert.rejects(recordingToWav(new Blob(['clip']), reference), /at least 3 seconds/);
assert.ok(closed, 'decoder is released even when a reference is too short');
decodedDuration = 10.4;
const capped = await recordingToWav(new Blob(['clip']), reference);
assert.equal(renderFrames, 160000, 'encoder trims duration to backend maximum');
assert.equal(capped.size, 320044);
console.log('Recording checks passed: input targeting, limits, permissions, cancellation, resource cleanup, WAV encoding and minimum length.');
