import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {createHmac, randomUUID} from 'node:crypto';
import {Buffer} from 'node:buffer';
import process from 'node:process';
import transcription, {sessionOwner} from '../server/transcription.mjs';
import {comparisonRows} from '../src/comparisonRows.js';

const secret = 'test-key-only', owner = randomUUID();
const value = `${owner}.${Date.now() + 60000}`;
const cookie = `ov_session=${value}.${createHmac('sha256', secret).update(value).digest('hex')}`;
assert.equal(sessionOwner(cookie, secret), owner);
assert.equal(sessionOwner(cookie, 'wrong-key'), null);
assert.equal(sessionOwner(cookie, secret, Date.now() + 120000), null);
assert.equal(sessionOwner('ov_session=broken', secret), null);
assert.deepEqual(comparisonRows({raw:{words:[{start:0,text:' Hello'},{start:5.1,text:' world'}]},one_voice:{text:'Fallback'}},8), [
  {start:0,end:5,original:'Hello',extracted:'Fallback'}, {start:5,end:8,original:'world',extracted:''},
]);

const oldBase = process.env.ONE_VOICE_TRANSCRIPTION_URL, oldKey = process.env.ONE_VOICE_TRANSCRIPTION_TOKEN;
let observed;
const upstream = createServer(async (req,res) => {
  const chunks=[];for await(const chunk of req)chunks.push(chunk);
  observed={url:req.url, method:req.method, headers:req.headers, body:Buffer.concat(chunks)};
  res.setHeader('Content-Type','application/json');
  res.end(JSON.stringify(req.url==='/health'?{ready:true}:{id:owner,status:'queued'}));
});
const proxy = createServer(transcription);
await new Promise(resolve=>upstream.listen(0,'127.0.0.1',resolve));
await new Promise(resolve=>proxy.listen(0,'127.0.0.1',resolve));
const base=`http://127.0.0.1:${proxy.address().port}/api/poc?route=`;
try {
  delete process.env.ONE_VOICE_TRANSCRIPTION_URL;
  assert.equal((await fetch(base+'/health')).status,503);
  process.env.ONE_VOICE_TRANSCRIPTION_URL=`http://127.0.0.1:${upstream.address().port}`;
  process.env.ONE_VOICE_TRANSCRIPTION_TOKEN=secret;
  assert.equal((await (await fetch(base+'/health')).json()).processing_location,'local');
  assert.equal((await fetch(base+'/health?url=http://attacker')).status,404);
  assert.equal((await fetch(base+'/unknown')).status,404);
  assert.equal((await fetch(base+'/health&route=/transcriptions')).status,404);
  assert.equal((await fetch(base+'/health&extra=value')).status,404);
  assert.equal((await fetch(base+'/health',{method:'POST'})).status,405);
  assert.equal((await fetch(base+'/health',{headers:{Origin:'https://attacker.example'}})).status,403);
  assert.equal((await fetch(base+'/transcriptions/'+owner)).status,404);
  assert.equal((await fetch(base+'/transcriptions/'+owner,{headers:{Cookie:'ov_session=forged'}})).status,404);
  const form=new FormData();form.append('reference',new Blob(['reference']),'ref.wav');form.append('mixture',new Blob(['mixture']),'mix.wav');
  const response=await fetch(base+'/transcriptions',{method:'POST',body:form});
  assert.equal(response.status,200);
  const session=response.headers.get('set-cookie');
  assert.ok(session.includes('HttpOnly; SameSite=Strict'));
  assert.ok(sessionOwner(session,secret));
  assert.equal(observed.headers['x-onevoice-owner'],sessionOwner(session,secret));
  assert.equal(observed.headers.authorization,`Bearer ${secret}`);
  assert.ok(observed.body.includes(Buffer.from('mixture')));
  assert.equal((await fetch(base+'/transcriptions/'+owner,{headers:{Cookie:session}})).status,200);
  assert.equal(observed.url,'/transcriptions/'+owner);
  assert.equal((await fetch(base+'/transcriptions',{method:'POST',body:new Uint8Array(4*1024*1024+1),headers:{'Content-Type':'multipart/form-data; boundary=test'}})).status,413);
} finally {
  if(oldBase===undefined)delete process.env.ONE_VOICE_TRANSCRIPTION_URL;else process.env.ONE_VOICE_TRANSCRIPTION_URL=oldBase;
  if(oldKey===undefined)delete process.env.ONE_VOICE_TRANSCRIPTION_TOKEN;else process.env.ONE_VOICE_TRANSCRIPTION_TOKEN=oldKey;
  proxy.closeAllConnections();upstream.closeAllConnections();await Promise.all([new Promise(resolve=>proxy.close(resolve)),new Promise(resolve=>upstream.close(resolve))]);
}

console.log('Transcription session ownership, route boundaries, and bounded transport verified.');
