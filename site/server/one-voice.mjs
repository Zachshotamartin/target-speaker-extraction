import process from 'node:process';
const LIMIT = 4 * 1024 * 1024;

export default async function oneVoice(req, res) {
  const action = new URL(req.url, 'http://localhost').pathname.split('/').filter(Boolean).at(-1);
  res.setHeader('Cache-Control', 'no-store');
  const fail = (code, detail) => {res.statusCode = code; res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({detail}));};
  if (!['model', 'extract'].includes(action)) return fail(404, 'Not found');
  if (req.method !== (action === 'model' ? 'GET' : 'POST')) return fail(405, 'Method not allowed');
  const origin = req.headers.origin;
  const publicOrigin = process.env.ONE_VOICE_PUBLIC_ORIGIN;
  const local = !process.env.VERCEL && /^(localhost|127\.0\.0\.1):\d+$/.test(String(req.headers.host || ''));
  const allowed = ['https://one-voice.vercel.app', publicOrigin, ...(local ? [`http://${req.headers.host}`] : [])];
  if ((origin && !allowed.includes(origin)) || req.headers['sec-fetch-site'] === 'cross-site') return fail(403, 'Use the upload tool on the OneVoice website.');
  const base = process.env.ONE_VOICE_SERVICE_URL;
  const secret = process.env.ONE_VOICE_SERVICE_TOKEN;
  if (!base || (!local && !secret)) return fail(503, 'Uploads are not available yet. You can listen to the prepared examples above.');
  try {
    const url = new URL(base);
    if (!['http:', 'https:'].includes(url.protocol) || (!local && url.protocol !== 'https:') || url.username || url.password || url.search || url.hash) throw new Error();
  } catch {return fail(503, 'The extraction service is not configured.');}
  try {
    let body;
    if (action === 'extract') {
      if (!String(req.headers['content-type']).startsWith('multipart/form-data;')) return fail(415, 'Choose two audio files.');
      const chunks = []; let size = 0;
      for await (const chunk of req) {size += chunk.length; if (size > LIMIT) return fail(413, 'Combined upload must be under 4 MiB.'); chunks.push(chunk);}
      body = Buffer.concat(chunks);
    }
    const upstream = await fetch(`${base.replace(/\/$/, '')}/${action}`, {method: req.method, body, redirect: 'error', headers: {...(body ? {'Content-Type': req.headers['content-type']} : {}), ...(secret ? {Authorization: `Bearer ${secret}`} : {})}, signal: AbortSignal.timeout(110000)});
    let size = 0; const chunks = [];
    for await (const chunk of upstream.body || []) {size += chunk.length; if (size > 8 * 1024 * 1024) throw new Error(); chunks.push(Buffer.from(chunk));}
    res.statusCode = upstream.status;
    res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/json');
    for (const name of ['x-checkpoint-sha256','content-disposition']) if (upstream.headers.has(name)) res.setHeader(name, upstream.headers.get(name));
    res.end(Buffer.concat(chunks));
  } catch { fail(503, 'The extraction service may be waking up. Please try again shortly.'); }
}
