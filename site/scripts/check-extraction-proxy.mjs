import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import process from 'node:process';
import handler from '../server/one-voice.mjs';

test('audio proxy bounds requests and forwards multipart bytes without exposing its token', async () => {
  const original = process.env.ONE_VOICE_SERVICE_URL, token = process.env.ONE_VOICE_SERVICE_TOKEN;
  const upstream = http.createServer(async(req,res) => {
    assert.equal(req.headers.authorization, 'Bearer test-secret');
    if(req.url === '/model') {res.setHeader('Content-Type','application/json');res.end('{"ready":true}');return;}
    const chunks=[];for await(const chunk of req)chunks.push(chunk);
    assert.equal(Buffer.concat(chunks).toString(),'multipart bytes');
    res.setHeader('Content-Type','audio/wav');res.setHeader('X-Checkpoint-SHA256','test-checkpoint');res.end('RIFF');
  });
  const server = http.createServer(handler);
  upstream.listen(0,'127.0.0.1');server.listen(0,'127.0.0.1');
  await Promise.all([once(upstream,'listening'),once(server,'listening')]);
  const base=`http://127.0.0.1:${server.address().port}/api/one-voice`;
  try {
    delete process.env.ONE_VOICE_SERVICE_URL;
    assert.equal((await fetch(base+'/model')).status,503);
    process.env.ONE_VOICE_SERVICE_URL=`http://127.0.0.1:${upstream.address().port}`;
    process.env.ONE_VOICE_SERVICE_TOKEN='test-secret';
    assert.deepEqual(await (await fetch(base+'/model')).json(),{ready:true});
    assert.equal((await fetch(base+'/extract',{method:'POST',headers:{Origin:'https://example.com'}})).status,403);
    assert.equal((await fetch(base+'/extract',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).status,415);
    const result=await fetch(base+'/extract',{method:'POST',headers:{'Content-Type':'multipart/form-data; boundary=test'},body:'multipart bytes'});
    assert.equal(await result.text(),'RIFF');assert.equal(result.headers.get('x-checkpoint-sha256'),'test-checkpoint');assert.equal(result.headers.get('cache-control'),'no-store');
    assert.equal((await fetch(base+'/extract',{method:'POST',headers:{'Content-Type':'multipart/form-data; boundary=test'},body:Buffer.alloc(4*1024*1024+1)})).status,413);
  } finally {
    server.closeAllConnections();upstream.closeAllConnections();server.close();upstream.close();
    if(original === undefined)delete process.env.ONE_VOICE_SERVICE_URL;else process.env.ONE_VOICE_SERVICE_URL=original;
    if(token === undefined)delete process.env.ONE_VOICE_SERVICE_TOKEN;else process.env.ONE_VOICE_SERVICE_TOKEN=token;
  }
});
import { Buffer } from 'node:buffer';
