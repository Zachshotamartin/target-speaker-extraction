import assert from 'node:assert/strict';
import {captionsSrt, editedDuration, editedToSource, editedWords, editHistory, editPlan,
  initialEdits, renderEdit, sourceToEdited, transcriptWords, wavBytes} from '../src/audioEdit.js';

const words = [{start: 0.2, end: 0.8, text: 'Hello'}, {start: 1, end: 1.5, text: 'unwanted'},
  {start: 1.6, end: 2, text: 'world.'}, {start: 3, end: 3.5, text: 'Again.'}];
const full = editPlan(words, [], 4);
assert.deepEqual(full, [{start: 0, end: 4, outputStart: 0, outputEnd: 4}]);
const cut = editPlan(words, [1], 4);
assert.equal(cut.length, 2);
assert.ok(Math.abs(editedDuration(cut) - 3.42) < 1e-9);
assert.ok(Math.abs(sourceToEdited(1.6, cut) - 1.02) < 1e-9);
assert.ok(Math.abs(editedToSource(1.02, cut) - 1.6) < 1e-9);
assert.equal(sourceToEdited(1.2, cut), 0.96, 'deleted source time maps to next cut seam');
assert.equal(editedToSource(0.96, cut), 1.54, 'cut seam maps to next retained sample');
const captions = editedWords(words, [1], cut);
assert.equal(captions.map(w => w.text).join(' '), 'Hello world. Again.');
assert.match(captionsSrt(captions), /00:00:00,200 --> 00:00:01,420\nHello world\./);
assert.match(captionsSrt(captions), /00:00:02,420 --> 00:00:02,920\nAgain\./);
assert.ok(!captionsSrt(captions).includes('unwanted'));

// Keep only two middle words trims both ends, preserves the gap between them.
const only = editPlan(words, [0, 3], 4);
assert.equal(only.length, 1);
assert.ok(Math.abs(only[0].start - 0.96) < 1e-9);
assert.ok(Math.abs(only[0].end - 2.04) < 1e-9);
const empty = editPlan(words, [0, 1, 2, 3], 4);
assert.deepEqual(empty, []);
assert.deepEqual(editedWords(words, [0, 1, 2, 3], empty), []);
assert.equal(captionsSrt([]), '');
assert.deepEqual(editPlan([], [], 0), []);
assert.deepEqual(editPlan([], [], 2), [{start: 0, end: 2, outputStart: 0, outputEnd: 2}]);

// Every remaining word is inside the edited duration under every deletion mask.
for (let mask = 0; mask < 16; mask++) {
  const removed = words.flatMap((_, index) => mask & (1 << index) ? [index] : []);
  const plan = editPlan(words, removed, 4);
  for (const word of editedWords(words, removed, plan)) {
    assert.ok(word.start >= 0 && word.end <= editedDuration(plan) + 1e-10);
    assert.ok(word.end > word.start);
  }
  for (const clip of plan) {
    const center = (clip.start + clip.end) / 2;
    assert.ok(Math.abs(editedToSource(sourceToEdited(center, plan), plan) - center) < 1e-10);
  }
}

let history = editHistory(initialEdits, {type: 'set', removed: [1, 1]});
assert.equal(editHistory(initialEdits, {type: 'undo'}), initialEdits);
assert.equal(editHistory(history, {type: 'set', removed: [1]}), history, 'no-op keeps playback unchanged');
history = editHistory(history, {type: 'set', removed: [1, 2]});
history = editHistory(history, {type: 'undo'});
assert.deepEqual(history.removed, [1]);
history = editHistory(history, {type: 'redo'});
assert.deepEqual(history.removed, [1, 2]);
history = editHistory(history, {type: 'set', removed: []});
history = editHistory(history, {type: 'undo'});
assert.deepEqual(history.removed, [1, 2], 'reset itself is reversible');
history = editHistory(history, {type: 'set', removed: [3]});
assert.deepEqual(history.future, [], 'new edit drops abandoned redo branch');

const normalized = transcriptWords({duration: 4, segments: [{attribution: 'uncertain', words: [
  ...words, {start: NaN, end: 3, text: 'invalid'}, {start: 3.3, end: 5, text: 'clamped'},
]}]});
assert.equal(normalized.length, 5);
assert.deepEqual(normalized.at(-1), {start: 3.5, end: 4, text: 'clamped', attribution: 'uncertain'});
assert.equal(transcriptWords({duration: 4, comparison: {one_voice: {words}}}).length, 4);

const rate = 1000;
const source = Float32Array.from({length: 4000}, (_, index) => Math.sin(index * 0.02));
assert.deepEqual(renderEdit([source], rate, full)[0], source, 'no edits preserve every sample');
const samples = renderEdit([source], rate, cut)[0];
assert.equal(samples.length, 3420);
assert.ok(samples[959] === 0, 'fade out at cut');
assert.ok(samples[960] === 0, 'fade in at join');
assert.equal(samples[1000], source[1580]);
const encoded = new DataView(wavBytes([samples, samples], rate));
assert.equal(encoded.getUint32(24, true), 1000);
assert.equal(encoded.getUint16(22, true), 2);
assert.equal(encoded.getUint32(40, true), 3420 * 4);
assert.equal(encoded.getInt16(44 + 1000 * 4, true), Math.round(samples[1000] * (samples[1000] < 0 ? 32768 : 32767)));
assert.throws(() => wavBytes([new Float32Array()], rate), /No audio/);
const clipped = new DataView(wavBytes([new Float32Array([-2, 2, NaN])], rate));
assert.equal(clipped.getInt16(44, true), -32768);
assert.equal(clipped.getInt16(46, true), 32767);
assert.equal(clipped.getInt16(48, true), 0);
console.log('Audio editor: cut timelines, captions, history, timestamps, fades and PCM exports passed.');
