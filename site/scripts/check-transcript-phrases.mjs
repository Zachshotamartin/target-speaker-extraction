import assert from 'node:assert/strict';
import {transcriptPhrases} from '../src/transcriptPhrases.js';

const words = [
  {start:0,end:.2,text:'Keep',attribution:'accepted'},
  {start:.2,end:.6,text:'these',attribution:'accepted'},
  {start:.6,end:1,text:'words.',attribution:'accepted'},
  {start:2,end:2.4,text:'Another',attribution:'accepted'},
  {start:2.4,end:3,text:'passage.',attribution:'accepted'},
];
const original = structuredClone(words);
const phrases = transcriptPhrases(words);
assert.deepEqual(words,original,'Displaying passages must not mutate word timings used for editing/exports.');
assert.deepEqual(phrases.map(p=>[p.start,p.end,p.text]),[[0,1,'Keep these words.'],[2,3,'Another passage.']]);
assert.equal(phrases.map(p=>p.text).join(' '),words.map(w=>w.text).join(' '),'Every supplied word stays in order.');
assert.deepEqual(transcriptPhrases([]),[]);
assert.equal(transcriptPhrases([{start:0,end:2,text:'Already a full sentence.'}])[0].text,'Already a full sentence.');
const long = Array.from({length:80},(_,index)=>({start:index*.2,end:(index+1)*.2,text:`word${index}`}));
assert.ok(transcriptPhrases(long).length>1,'Long output is broken into readable passages.');
assert.equal(transcriptPhrases(long).map(p=>p.text).join(' '),long.map(w=>w.text).join(' '));
console.log('Transcript passages preserve text, source timings and input data.');
