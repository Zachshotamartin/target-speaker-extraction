import assert from 'node:assert/strict';
import {comparisonRows} from '../src/comparisonRows.js';

const pair = {
  raw: {words: [{start: 0, text: ' Hello'}, {start: 4.9, text: ' there.'}, {start: 5, text: ' Another'}, {start: 9, text: ' speaker.'}]},
  one_voice: {words: [{start: 1, text: ' Keep'}, {start: 5.1, text: ' my'}, {start: 10, text: ' voice.'}]},
};
const rows = comparisonRows(pair, 12);
assert.deepEqual(rows, [
  {start: 0, end: 5, original: 'Hello there.', extracted: 'Keep'},
  {start: 5, end: 10, original: 'Another speaker.', extracted: 'my'},
  {start: 10, end: 12, original: '', extracted: 'voice.'},
]);
assert.equal(rows.map(row => row.original).filter(Boolean).join(' '), 'Hello there. Another speaker.');
assert.equal(rows.map(row => row.extracted).join(' '), 'Keep my voice.');
assert.deepEqual(comparisonRows({raw: {text: 'Legacy transcript'}, one_voice: {words: []}}, 3), [
  {start: 0, end: 3, original: 'Legacy transcript', extracted: ''},
]);
assert.equal(comparisonRows(null, 0).length, 1);
assert.equal(comparisonRows({raw: {words: [{start: 5, text: ' End'}]}}, 5)[0].original, 'End');
console.log('Comparison rows: shared time windows, boundaries, empty speech and text preservation passed.');
