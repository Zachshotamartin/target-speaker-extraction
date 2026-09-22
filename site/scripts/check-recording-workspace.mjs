import assert from 'node:assert/strict';
import {emptyEdit, naturalPlan, correctedWords, historyStep, suggestions} from '../src/naturalEdits.js';
import {editedWords, renderEdit} from '../src/audioEdit.js';
const words = [
  {start: 0, end: .5, text: 'Hello', attribution: 'uncertain'},
  {start: 1, end: 1.5, text: 'um', attribution: 'accepted'},
  {start: 2, end: 2.5, text: 'world', attribution: 'accepted'},
  {start: 5, end: 6, text: 'again', attribution: 'accepted'},
];
const removed = {...emptyEdit, removed: [1], padding: .3};
const plan = naturalPlan(words, removed, 7);
assert.deepEqual(plan.map(c => [c.start, c.end]), [[0,.8],[1.7,7]]);
assert.ok(plan.every(c => c.end <= 1 || c.start >= 1.5), 'Deleted speech never leaks through padding');
const custom = naturalPlan(words, {...removed, customCuts: [{start: 3, end: 4}, {start: 3.5, end: 4.5}]}, 7);
assert.deepEqual(custom.map(c => [c.start,c.end]), [[0,.8],[1.7,3],[4.5,7]]);
assert.ok(Math.abs(editedWords(words, removed.removed, custom).at(-1).start - 2.6) < 1e-9);
const edited = correctedWords(words, {...emptyEdit, corrections: {0: 'Hi'}, decisions: {0: 'accepted'}});
assert.equal(edited[0].text, 'Hi'); assert.equal(words[0].text, 'Hello');
assert.deepEqual(naturalPlan(edited, emptyEdit, 7), naturalPlan(words, emptyEdit, 7), 'Text/speaker corrections do not change audio');
let history = {past: [], present: emptyEdit, future: []};
history = historyStep(history, {value: removed});
history = historyStep(history, {value: {customCuts: [{start: 3,end: 4}]}});
assert.equal(historyStep(history, {type:'undo'}).present.customCuts.length, 0);
assert.deepEqual(historyStep(historyStep(history, {type:'undo'}), {type:'redo'}).present, history.present);
assert.deepEqual(historyStep(historyStep(history, {type:'reset'}), {type:'undo'}).present, history.present);
assert.equal(suggestions(words).filter(s => s.cut).length, 1);
assert.deepEqual(suggestions(words).find(s => s.ids)?.ids, [1]);
const restored = structuredClone({edits: history.present, recording: new Blob(['audio']), tracks: []});
assert.equal(await restored.recording.text(), 'audio');
assert.deepEqual(restored.edits, history.present);
const signal = new Float32Array(700).fill(.2);
assert.equal(renderEdit([signal],100,custom)[0].length,460);
console.log('Recording workspace: natural cuts, breath padding, corrections, history, captions and serialization passed.');
const {indexedDB} = await import('fake-indexeddb');
globalThis.indexedDB = indexedDB;
const {projects} = await import('../src/recordingProjects.js');
const project = {id:'test',name:'Saved recording',updated:1,recording:new Blob(['private audio']),edits:history.present,tracks:[{audio:new Blob(['isolated']),result:{duration:7}}]};
await projects('save',project);
assert.equal((await projects('list'))[0].tracks,1);
const reopened=await projects('get','test');
assert.equal(await reopened.recording.text(),'private audio');
assert.equal(await reopened.tracks[0].audio.text(),'isolated');
assert.deepEqual(reopened.edits,history.present);
await projects('delete','test'); assert.equal((await projects('list')).length,0);
console.log('IndexedDB project blobs, edit persistence, library metadata and deletion passed.');
